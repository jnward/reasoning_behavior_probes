# %%
import json
import torch
from nnsight import LanguageModel
import numpy as np
from transformers import AutoTokenizer
import os

os.environ["HF_TOKEN"] = "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"

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

def process_chain(tokenizer, chain):
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
    formatted_problem = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize the formatted problem
    tokenized_problem = tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]
    
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
        segment_tokens = tokenizer.encode(text, add_special_tokens=False)
        
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
def process_chains_iterator(tokenizer, chains):
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
        tokenized_text, annotation_indices = process_chain(tokenizer, chain)
        
        # Yield the results along with chain identifier (task_id if available, otherwise index)
        chain_id = chain.get('task_id', f'chain_{i}')
        yield tokenized_text, annotation_indices


# %%
chains = load_annotated_chain("annotated_chains/all_annotated_chains.json")
# model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)
base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=torch.bfloat16)
finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

processed_chains = process_chains_iterator(finetune_tokenizer, chains)

# %%
# test the iterator, it should return a tokenized text and a dictionary of annotation indices
tokens, indices = next(processed_chains)

for category, index_tuples in indices.items():
    print(category)
    for start, end in index_tuples:
        token_segment = tokens[start-1:end+1]
        print("\t" + finetune_tokenizer.decode(token_segment))
# %%
from collections import defaultdict
from tqdm import tqdm

processed_chains = process_chains_iterator(finetune_tokenizer, chains)

# %%
layer_of_interest = 10
activations = defaultdict(list)

def convert_to_base_tokens(tokens: torch.Tensor):
    """
    Convert r1 tokens to base tokens. Only works for Llama tokenizers.
    """
    # patch_token = 77627 # ` ############`
    patch_token = 27370 # ` ####`
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens


# %%
acc = 0
with torch.inference_mode():
    for ft_tokens, indices in tqdm(processed_chains):
        # need to decode tokens back to text, so we can use model.trace
        text = finetune_tokenizer.decode(ft_tokens)
        ft_tokens2 = finetune_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
        assert torch.equal(ft_tokens, ft_tokens2)

        base_tokens = convert_to_base_tokens(ft_tokens)
        base_text = base_tokenizer.decode(base_tokens)
        base_tokens2 = base_tokenizer.encode(base_text, add_special_tokens=False, return_tensors="pt")[0]
        if not torch.equal(base_tokens, base_tokens2):
            print("mismatch, skipping...")
            continue

        with base_model.trace(base_text) as tracer:
            # layer_activations = model.model.embed_tokens.output.save()
            layer_activations = base_model.model.layers[layer_of_interest].output[0].save()
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

# entropy_direction = torch.load("l10_entropy_direction.pt")
# steering_vectors["entropy"] = entropy_direction

# %%
# free memory
import gc
del(activations)
gc.collect()
torch.cuda.empty_cache()

# %%
# save steering vectors
torch.save(steering_vectors, "base_steering_vectors.pt")


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


fig.update_layout(
    title='Cosine Similarity Between Base Steering Vectors',
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
del base_model
gc.collect()
torch.cuda.empty_cache()


# %%
# model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)
model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=torch.bfloat16)

# %%
test_prompt = "What is the third tallest building in NYC?"
messages = [
    {"role": "user", "content": test_prompt}
]
text = finetune_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    add_special_tokens=False,
)
ft_tokens = finetune_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
base_tokens = convert_to_base_tokens(ft_tokens)
base_text = base_tokenizer.decode(base_tokens)

assert torch.equal(base_tokens, base_tokenizer.encode(base_text, add_special_tokens=False, return_tensors="pt")[0])

# %%
print(text)
print(base_text)

with torch.inference_mode():
    with model.generate(base_text + "Ok, so the user is asking that I find the third tallest building in NYC. I need to", max_new_tokens=128) as tracer:
        with model.model.layers.all():
            model.model.layers[layer_of_interest].output[0][:] += 12 * steering_vectors["backtracking"].detach()
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)

# %%
# apply 10x backtracking steering vector
with torch.inference_mode():
    with model.generate(text, max_new_tokens=128) as tracer:
        with model.model.layers.all():
            model.model.layers[layer_of_interest].output[0][:] += 12 * steering_vectors["backtracking"].detach()
            out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))
