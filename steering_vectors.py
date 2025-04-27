# %%
import json
import torch
from nnsight import LanguageModel
import numpy as np
# %%
def load_annotated_chain(file_path):
    """Load annotated chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_annotations(annotated_text):
    """Extract all annotations and their text from an annotated chain"""
    annotations = []
    
    # Find all annotation segments
    current_pos = 0
    while True:
        # Find the next category tag
        start_tag_pos = annotated_text.find('[\"', current_pos)
        if start_tag_pos == -1:
            break
            
        end_tag_pos = annotated_text.find('\"]', start_tag_pos)
        if end_tag_pos == -1:
            break
            
        # Extract the category
        category = annotated_text[start_tag_pos+2:end_tag_pos]
        
        # Skip if this is an end-section tag
        if category == "end-section":
            current_pos = end_tag_pos + 2
            continue
            
        # Find the corresponding end-section tag
        start_text_pos = end_tag_pos + 2
        end_section_tag = annotated_text.find('[\"end-section\"]', start_text_pos)
        
        # Extract the text between category tag and end-section tag
        if end_section_tag != -1:
            text = annotated_text[start_text_pos:end_section_tag].strip()
            annotations.append((category, text))
            current_pos = end_section_tag + 15  # Length of "[\"end-section\"]"
        else:
            # If no end-section tag, move to the next position
            current_pos = end_tag_pos + 2
    
    return annotations

def process_chain(model, chain):
    """
    Process a chain to format it with user/assistant tags and track annotation indices.
    
    Args:
        model: Language model with tokenizer
        chain: Dictionary containing problem and annotated_chain
        
    Returns:
        tuple: (tokenized_full_text, annotation_indices)
            where annotation_indices is a dict mapping categories to lists of (start, end) token pairs
    """
    # Format problem with user/assistant tags
    problem = chain["problem"]
    formatted_problem = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize the formatted problem
    tokenized_problem = model.tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]
    
    # Extract all annotations from the annotated chain
    annotations = extract_annotations(chain["annotated_chain"])
    
    # Track token indices for each category
    annotation_indices = {}
    
    # Current token position
    current_token_pos = len(tokenized_problem)
    
    # Full tokenized text (starting with the tokenized problem)
    full_tokens = tokenized_problem.tolist()
    
    # Process each annotation
    for i, (category, text) in enumerate(annotations):
        if i > 0:
            text = " " + text
        # Tokenize this text segment
        segment_tokens = model.tokenizer.encode(text, add_special_tokens=False)
        
        # Record start and end token indices
        start_idx = current_token_pos
        end_idx = start_idx + len(segment_tokens) - 1
        
        # Add to annotation indices for this category
        if category not in annotation_indices:
            annotation_indices[category] = []
        annotation_indices[category].append((start_idx, end_idx))
        
        # Add segment tokens to full tokens
        full_tokens.extend(segment_tokens)
        
        # Update current token position
        current_token_pos += len(segment_tokens)
    
    # Convert full tokens back to tensor
    tokenized_full_text = torch.tensor(full_tokens)
    
    return tokenized_full_text, annotation_indices

# %%
# Create an iterator for processing multiple chains
def process_chains_iterator(model, chains):
    """
    Process multiple chains and yield tokenized text with annotation indices for each.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries containing problem and annotated_chain
        
    Yields:
        tuple: (chain_id, tokenized_full_text, annotation_indices)
            where annotation_indices is a dict mapping categories to lists of (start, end) token pairs
    """
    for i, chain in enumerate(chains):
        # Process this chain
        tokenized_text, annotation_indices = process_chain(model, chain)
        
        # Yield the results along with chain identifier (task_id if available, otherwise index)
        chain_id = chain.get('task_id', f'chain_{i}')
        yield tokenized_text, annotation_indices

# %%
chains = load_annotated_chain("annotated_chains/all_annotated_chains.json")
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)

processed_chains = process_chains_iterator(model, chains)

# %%
# test the iterator, it should return a tokenized text and a dictionary of annotation indices
tokens, indices = next(processed_chains)

for category, index_tuples in indices.items():
    print(category)
    for start, end in index_tuples:
        token_segment = tokens[start-1:end+1]
        print("\t" + model.tokenizer.decode(token_segment))
# %%
from collections import defaultdict
from tqdm import tqdm

processed_chains = process_chains_iterator(model, chains)

layer_of_interest = 10
activations = defaultdict(list)

acc = 0
with torch.inference_mode():
    for tokens, indices in tqdm(processed_chains):
        # need to decode tokens back to text, so we can use model.trace
        text = model.tokenizer.decode(tokens)
        tokens2 = model.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
        assert torch.equal(tokens, tokens2)
        with model.trace(text) as tracer:
            # layer_activations = model.model.embed_tokens.output.save()
            layer_activations = model.model.layers[layer_of_interest].output[0].save()
        torch.cuda.empty_cache()
        for category, index_tuples in indices.items():
            if category not in ["backtracking", "uncertainty-estimation", "initializing", "deduction", "example-testing", "adding-knowledge"]:
            # if category not in ["backtracking"]:
                print(f"Found spurious category: {category}, skipping...")
                continue
            for start, end in index_tuples:
                # token_segment = tokens[start-1:end+1]
                # end = start
                start = start - 12
                end = start + 4
                # print(model.tokenizer.decode(tokens[start-1:end+1]))
                # start = np.random.randint(0, len(tokens))
                # end = start + 5
                activations[category].append(layer_activations[0, start-1:end+1].float().cpu())
    
        # my kernel crashes if I run it for more than 28 prompts... I don't know why :(
        # acc += 1
        # if acc > 80:
        #     break

# %%
overall_mean = torch.zeros(4096, dtype=torch.float64)
num_activations = 0
for _, layer_activations in activations.items():
    for la in layer_activations:
        overall_mean += la.to(torch.float64).sum(dim=0)
        num_activations += la.shape[0]
overall_mean /= num_activations
overall_mean = overall_mean.to(torch.float32)

# %%
# calculate mean vectors for each category
# mean_vectors = {}
# for category, layer_activations in activations.items():
#     print(category)
#     means = [la.mean(dim=0) for la in layer_activations]
#     mean = torch.stack(means, dim=0).mean(dim=0)
#     mean_vectors[category] = mean
# print(mean_vectors["backtracking"].shape)

mean_vectors = {}
for category, layer_activations in activations.items():
    category_mean = torch.zeros(4096, dtype=torch.float64)
    num_activations = 0
    for la in layer_activations:
        category_mean += la.to(torch.float64).sum(dim=0)
        num_activations += la.shape[0]
    category_mean /= num_activations
    mean_vectors[category] = category_mean.to(torch.float32)

# %%
# compute difference-of-means vectors for each category
steering_vectors = {}
for category, mean_vector in mean_vectors.items():
    steering_vectors[category] = mean_vectors[category] - overall_mean
    # steering_vectors[category] = torch.randn(4096)
    # steering_vectors[category] = overall_mean  # test effect of adding mean to activations

entropy_direction = torch.load("entropy_direction.pt")
steering_vectors["entropy"] = entropy_direction
# %%
# free memory
import gc
del(activations)
gc.collect()
torch.cuda.empty_cache()

# %%
# compute cosine similarity matrix between all steering vectors
import numpy as np
import plotly.graph_objects as go

# # Get all categories
categories = list(steering_vectors.keys())
n_categories = len(categories)

# Initialize similarity matrix
similarity_matrix = np.zeros((n_categories, n_categories))

# Compute cosine similarity for each pair
for i, cat1 in enumerate(categories): # new
    for j, cat2 in enumerate(categories): # original
        vec1 = steering_vectors[cat1]
        vec2 = steering_vectors[cat2]
        similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        similarity_matrix[i, j] = similarity

# Plot the similarity matrix with plotly
fig = go.Figure(data=go.Heatmap(
    z=similarity_matrix,
    x=categories,
    y=categories,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-1,
    zmax=1,
    colorbar=dict(title='Cosine Similarity')
))

# add x axis label
fig.update_layout(
    xaxis_title='Offset Steering Vectors',
    yaxis_title='Original Steering Vectors'
)

fig.update_layout(
    title='Cosine Similarity Between Steering Vectors',
    xaxis=dict(tickangle=45),
    height=600,
    width=700
)

fig.show()


# %%
for cat, vec in steering_vectors.items():
    print(cat, vec.norm())

steering_vectors = {k: v / v.norm() for k, v in steering_vectors.items()}

for cat, vec in steering_vectors.items():
    print(cat, vec.norm())

# %%
# Compute the rank of the subspace spanned by the steering vectors
# Stack all steering vectors into a matrix
# First get the dimension of vectors
vec_dim = next(iter(steering_vectors.values())).shape[0]

# Create a matrix where each row is a steering vector
matrix = torch.zeros((len(steering_vectors), vec_dim))
for i, category in enumerate(steering_vectors.keys()):
    matrix[i] = steering_vectors[category]

# Compute the rank using SVD
u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

# Count non-zero singular values (with some tolerance for numerical precision)
tolerance = 1e-10
rank = (s > tolerance).sum().item()

print(f"Number of steering vectors: {len(steering_vectors)}")
print(f"Rank of the subspace: {rank}")
print(f"Singular values: {s[:10]}")  # Print first 10 singular values

# Compute and display explained variance of singular values
# Square the singular values
s_squared = s ** 2

# Calculate the total variance
total_variance = s_squared.sum()

# Calculate the proportion of variance explained by each singular value
explained_variance = s_squared / total_variance

# Calculate cumulative explained variance
cumulative_variance = torch.cumsum(explained_variance, dim=0)

# Print explained variance for each singular value
print("\nExplained variance by singular value:")
for i, (sv, var, cum_var) in enumerate(zip(s, explained_variance, cumulative_variance)):
    print(f"SV {i+1}: {sv:.6f} - Explained variance: {var*100:.2f}% - Cumulative: {cum_var*100:.2f}%")

# Create a plot of explained variance
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'Singular Value Index': range(1, len(s)+1),
    'Explained Variance': explained_variance.detach().cpu().numpy() * 100,
    'Cumulative Variance': cumulative_variance.detach().cpu().numpy() * 100
})

fig = px.bar(df, x='Singular Value Index', y='Explained Variance',
             title='Explained Variance by Singular Value',
             labels={'Explained Variance': 'Explained Variance (%)'})

fig.add_scatter(x=df['Singular Value Index'], y=df['Cumulative Variance'],
                mode='lines+markers', name='Cumulative Variance (%)')
fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))

fig.show()

# %%
# Project steering vectors onto the subspace defined by the first two singular values
projection_matrix = vh[:2].T  # First two right singular vectors as columns

# Project each steering vector onto this subspace
projected_vectors = {}
for category, vector in steering_vectors.items():
    # Project the vector onto the subspace
    projection = torch.matmul(vector, projection_matrix)
    projected_vectors[category] = projection

# Create a scatter plot of the projections
fig = go.Figure()

# Add lines from origin to each point
for category, projection in projected_vectors.items():
    x, y = projection[0].item(), projection[1].item()
    
    # Calculate norm (magnitude) of the projected vector
    norm = torch.norm(projection).item()
    
    # Add line from origin to point
    fig.add_trace(go.Scatter(
        x=[0, x],
        y=[0, y],
        mode='lines',
        name=f"{category} vector",
        line=dict(color="gray", width=2, dash='solid'),
        showlegend=False
    ))
    
    # Add norm label at the midpoint of the line
    fig.add_trace(go.Scatter(
        x=[x/2],
        y=[y/2],
        mode='text',
        text=[f"{norm:.3f}"],
        textposition="middle center",
        textfont=dict(size=10, color="black"),
        showlegend=False
    ))
    
    # Add point and label
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers+text',
        name=category,
        text=[category],
        textposition="top center",
        marker=dict(size=10)
    ))

# Add origin point
fig.add_trace(go.Scatter(
    x=[0],
    y=[0],
    mode='markers',
    name='Origin',
    marker=dict(color='black', size=8)
))

fig.update_layout(
    title='Steering Vectors Projected onto First Two Principal Components',
    xaxis_title='First Principal Component',
    yaxis_title='Second Principal Component',
    height=600,
    width=700
)

fig.show()

# %%
# Project steering vectors onto the subspace defined by the first three singular values
projection_matrix_3d = vh[:3].T  # First three right singular vectors as columns

# Project each steering vector onto this 3D subspace
projected_vectors_3d = {}
for category, vector in steering_vectors.items():
    # Project the vector onto the subspace
    projection = torch.matmul(vector, projection_matrix_3d)
    projected_vectors_3d[category] = projection

# Create a 3D scatter plot of the projections
fig_3d = go.Figure()

# Add lines from origin to each point
for category, projection in projected_vectors_3d.items():
    x, y, z = projection[0].item(), projection[1].item(), projection[2].item()
    
    # Calculate norm (magnitude) of the projected vector
    norm = torch.norm(projection).item()
    
    # Add line from origin to point
    fig_3d.add_trace(go.Scatter3d(
        x=[0, x],
        y=[0, y],
        z=[0, z],
        mode='lines',
        name=f"{category} vector",
        line=dict(width=4, color="gray"),
        showlegend=False
    ))
    
    # Add norm label at the midpoint of the line
    fig_3d.add_trace(go.Scatter3d(
        x=[x/2],
        y=[y/2],
        z=[z/2],
        mode='text',
        text=[f"{norm:.3f}"],
        textposition="middle center",
        textfont=dict(size=10, color="black"),
        showlegend=False
    ))
    
    # Add point and label
    fig_3d.add_trace(go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers+text',
        name=category,
        text=[category],
        textposition="top center",
        marker=dict(size=8)
    ))

# Add origin point
fig_3d.add_trace(go.Scatter3d(
    x=[0],
    y=[0],
    z=[0],
    mode='markers',
    name='Origin',
    marker=dict(color='black', size=6)
))

fig_3d.update_layout(
    title='Steering Vectors Projected onto First Three Principal Components',
    scene=dict(
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        zaxis_title='Third Principal Component'
    ),
    height=700,
    width=800
)

fig_3d.show()


# %%
# apply 10x backtracking steering vector
with torch.inference_mode():
    with model.generate(text, max_new_tokens=128) as tracer:
        with model.model.layers.all():
            model.model.layers[31].output[0][:] += 30 * steering_vectors["entropy"].detach()
            out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))
# %%
# apply 5x backtracking steering vector
with model.generate(text, max_new_tokens=128) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] += 12 * steering_vectors["backtracking"].detach()
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))

# %%
# test effect of adding mean to activations
with model.generate(text, max_new_tokens=128) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] += 3 * overall_mean.detach()
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))

# %%
# test effect of increasing magnitude of activations
with model.generate(text, max_new_tokens=128) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] *= 8
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))


# %%
# test effect of adding gaussian noise to activations
with model.generate(text, max_new_tokens=128) as tracer:
    with model.model.layers.all():
        activation = model.model.layers[layer_of_interest].output[0]
        noise = torch.randn_like(activation)
        noise = noise / noise.norm(dim=-1, keepdim=True)
        activation += 12 * noise
        model.model.layers[layer_of_interest].output[0][:] = activation
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))




# %%
# apply 5x initializing steering vector
with model.generate(text, max_new_tokens=64) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] += 5 * steering_vectors["initializing"].detach()
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))
# %%
import random

# get random chain from dataset
with open("reasoning_chains/all_reasoning_chains.json", "r") as f:
    original_chains = json.load(f)

random_chain = random.choice(original_chains)

# %%
print(random_chain)

# %%
# get activations for random chain
formatted_text = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": random_chain["problem"]},
    {"role": "assistant", "content": random_chain["reasoning_chain"]}],
    tokenize=False,
    add_generation_prompt=True,
)
# %%
with torch.inference_mode():
    with model.trace(formatted_text) as tracer:
        layer_activations = model.model.layers[layer_of_interest].output[0].squeeze().save()

print(layer_activations.shape)

# %%
# get cosine similarity between activations and steering vectors
cosine_similarities = {}

for category, steering_vector in steering_vectors.items():
    cosine_similarities[category] = torch.nn.functional.cosine_similarity(layer_activations.cpu(), steering_vector, dim=1).float()

# %%
# get dot products between activations and steering vectors
dot_products = {}

for category, steering_vector in steering_vectors.items():
    # Compute dot product between each activation vector and the steering vector
    dot_products[category] = torch.matmul(layer_activations.cpu().float(), steering_vector).float()

# %%
import plotly.express as px
import pandas as pd
# plot each category's cosine similarity as a multi-line plot
# print tokens on the x-axis
tokens = model.tokenizer.encode(formatted_text)
token_text = [model.tokenizer.decode(t) for t in tokens]
print(len(token_text))

# create a dataframe with the cosine similarities
df = pd.DataFrame(cosine_similarities)
df["token"] = token_text
df["position"] = range(len(token_text))  # Add position column

# Use position for x-axis but display token text as labels
fig = px.line(df, x="position", y=list(cosine_similarities.keys()), hover_data=["token"])
fig.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(token_text))),
    ticktext=token_text
)
fig.show()

# %%
def create_token_visualization(token_text, similarities, category="backtracking", threshold=0.4, color="blue"):
    """
    Create HTML visualization of tokens highlighted based on cosine similarity.
    
    Args:
        token_text: List of tokens
        similarities: Dictionary of cosine similarities
        category: Which similarity category to use for highlighting
        threshold: Values below this threshold will be set to zero
        color: Color for highlighting ("blue", "orange", "red", "green")
    
    Returns:
        HTML string with highlighted tokens
    """
    # Get the specific category's similarities
    similarity_values = similarities[category]
    
    # Find maximum similarity for normalization
    max_sim = float(max(similarity_values))
    
    # Define color RGB values
    color_map = {
        "blue": "0,0,255",
        "orange": "255,165,0",
        "red": "255,0,0",
        "green": "0,128,0"
    }
    rgb = color_map.get(color, "0,0,255")  # Default to blue if color not found
    
    # Create HTML with token highlighting - white background and black text for dark mode
    html = "<div style='font-family:monospace; line-height:1.5; background-color:white; color:black; padding:10px;'>"
    
    for i, token in enumerate(token_text):
        # Get similarity value (between -1 and 1)
        sim = float(similarity_values[i])
        
        # Apply threshold - values below threshold are set to zero
        if sim < threshold:
            sim = 0
        else:
            # Normalize only non-zero values
            if max_sim > 0:
                sim = sim / max_sim
        
        # Create span with background color
        html += f"<span style='background-color:rgba({rgb},{max(0, sim):.3f})'>{token}</span>"
    
    html += "</div>"
    
    return html

# Test the visualization function
from IPython.display import display, HTML

# Save to file
html_viz = create_token_visualization(
    token_text,
    cosine_similarities,
    category="uncertainty-estimation",
    threshold=0.0,
    color="orange"
)
with open("token_visualization.html", "w") as f:
    f.write(html_viz)

# Display in the interactive window
display(HTML(html_viz))

# %%
def create_token_visualization_dot_product(token_text, dot_products, category="backtracking", threshold=0, color="blue"):
    """
    Create HTML visualization of tokens highlighted based on dot product.
    
    Args:
        token_text: List of tokens
        dot_products: Dictionary of dot products
        category: Which dot product category to use for highlighting
        threshold: Values below this threshold will be set to zero
        color: Color for highlighting ("blue", "orange", "red", "green")
    
    Returns:
        HTML string with highlighted tokens
    """
    # Get the specific category's dot products
    similarity_values = dot_products[category]
    
    # Find maximum absolute dot product for normalization
    # max_abs_value = float(max(abs(similarity_values)))
    max_abs_value = 2
    
    # Define color RGB values
    color_map = {
        "blue": "0,0,255",
        "orange": "255,165,0",
        "red": "255,0,0",
        "green": "0,128,0"
    }
    rgb = color_map.get(color, "0,0,255")  # Default to blue if color not found
    
    # Create HTML with token highlighting - white background and black text for dark mode
    html = "<div style='font-family:monospace; line-height:1.5; background-color:white; color:black; padding:10px;'>"
    
    for i, token in enumerate(token_text):
        # Get dot product value
        dp_value = float(similarity_values[i])
        
        # Apply threshold - values below threshold are set to zero
        if abs(dp_value) < threshold:
            intensity = 0
        else:
            # Normalize to [0,1] for visualization intensity
            if max_abs_value > 0:
                intensity = abs(dp_value) / max_abs_value
            else:
                intensity = 0
        
        # Create span with background color
        html += f"<span style='background-color:rgba({rgb},{intensity:.3f})'>{token}</span>"
    
    html += "</div>"
    
    return html

# Test the dot product visualization function
html_viz_dot = create_token_visualization_dot_product(
    token_text,
    dot_products,
    category="uncertainty-estimation",
    threshold=0.3,
    color="green"
)
with open("token_visualization_dot_product.html", "w") as f:
    f.write(html_viz_dot)

# Display in the interactive window
display(HTML(html_viz_dot))

# %%
# Compare dot products and cosine similarity for a selected category
category = "uncertainty-estimation"
comparison_df = pd.DataFrame({
    "position": range(len(token_text)),
    "token": token_text,
    "dot_product": dot_products[category].numpy(),
    "cosine_similarity": cosine_similarities[category].numpy()
})

# Create comparison plot
fig = px.line(comparison_df, x="position", y=["dot_product", "cosine_similarity"], 
              hover_data=["token"], title=f"Comparison of Dot Product vs Cosine Similarity for '{category}'")
fig.update_xaxes(
    tickmode='array',
    tickvals=list(range(0, len(token_text), 20)),  # Show every 20th token label to avoid crowding
    ticktext=[token_text[i] for i in range(0, len(token_text), 20)]
)
fig.show()

# %%






# test_prompt = "What is the largest prime factor of 1011?"
test_prompt = "What is the third tallest building in NYC?"
messages = [
    {"role": "user", "content": test_prompt}
]
text = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    add_special_tokens=False,
)
# %%
with model.generate(text, max_new_tokens=256, do_sample=False) as tracer:
    out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))

# %%
wait_idx = 160
partial_generation = model.tokenizer.decode(out[0][:wait_idx+1])
print(partial_generation)

start = wait_idx - 13
end = start + 6
print("-"*100)
print(model.tokenizer.decode(out[0][start:end]))
print("-"*100)
partial_input = model.tokenizer.decode(out[0][:end], skip_special_tokens=True)
print(partial_input)


# %%
with model.generate(partial_input, max_new_tokens=128, do_sample=False) as tracer:
    out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))

# %%
with model.generate(partial_input, max_new_tokens=128, do_sample=False) as tracer:
    acts = model.model.layers[layer_of_interest].output[0][:, start:end]
    noise = torch.randn_like(acts)
    # model.model.layers[layer_of_interest].output[0][:, start:end] -= (100 * steering_vectors["backtracking"].detach()) + (100 * steering_vectors["uncertainty-estimation"].detach())
    model.model.layers[layer_of_interest].output[0][:, start:end] += 20 * noise

    # target_acts = model.model.layers[layer_of_interest].output[0][:, start:end]
    # # target_acts = model.model.layers[layer_of_interest].output[0][:].save()
    # steer_vec = steering_vectors["backtracking"].detach()
    
    # # Project out the component in the direction of the steering vector
    # # Formula: act_proj = act - (act·vec)/(vec·vec) * vec
    # # Since steering vectors are normalized, (vec·vec) = 1
    # # dot_products = torch.matmul(target_acts.float(), steer_vec)
    # print(target_acts.shape)
    # print(steer_vec.shape)
    # dot_products = target_acts.float() @ steer_vec
    # projection = dot_products.unsqueeze(-1) * steer_vec
    # projection.save()
    
    # model.model.layers[layer_of_interest].output[0][:, start:end] = target_acts - 1000 * projection
    # # model.model.layers[layer_of_interest].output[0][:] = target_acts - 100 * projection
    out = model.generator.output.save()

print(out.shape)
# print(acts.shape)

print(model.tokenizer.decode(out[0]))

# %%