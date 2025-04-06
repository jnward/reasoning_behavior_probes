# %%
import json
import torch
from nnsight import LanguageModel

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
    formatted_problem = f"<｜User｜>{problem}<｜Assistant｜><think>\n"
    
    # Tokenize the formatted problem
    tokenized_problem = model.tokenizer.encode(formatted_problem, return_tensors="pt")[0]
    
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
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

processed_chains = process_chains_iterator(model, chains)

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
for tokens, indices in tqdm(processed_chains):
    # need to decode tokens back to text, so we can use model.trace
    text = model.tokenizer.decode(tokens)
    tokens2 = model.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    assert torch.equal(tokens, tokens2)
    with model.trace(text) as tracer:
        layer_activations = model.model.layers[layer_of_interest].output[0].save()
    print(layer_activations.shape)
    for category, index_tuples in indices.items():
        for start, end in index_tuples:
            # token_segment = tokens[start-1:end+1]
            activations[category].append(layer_activations[0, start-1:end+1])
    
    # my kernel crashes if I run it for more than 28 prompts... I don't know wny :()
    acc += 1
    if acc > 28:
        break

# %%
overall_mean = 0
num_activations = 0
for _, layer_activations in activations.items():
    for la in layer_activations:
        overall_mean += la.sum(dim=0)
        num_activations += la.shape[0]
overall_mean /= num_activations
print(overall_mean.shape)

# %%
# calculate mean vectors for each category
mean_vectors = {}
for category, layer_activations in activations.items():
    print(category)
    means = [la.mean(dim=0) for la in layer_activations]
    mean = torch.stack(means, dim=0).mean(dim=0)
    mean_vectors[category] = mean
print(mean_vectors["backtracking"].shape)

# %%
# compute difference-of-means vectors for each category
steering_vectors = {}
for category, mean_vector in mean_vectors.items():
    steering_vectors[category] = mean_vectors[category] - overall_mean

# %%
test_prompt = "What is 2 + 2?"
text = f"<｜User｜>{test_prompt}<｜Assistant｜><think>\nOkay, so the user is asking what 2 + 2 is."
# %%
with model.generate(text, max_new_tokens=64) as tracer:
    out = model.generator.output.save()
# %%
print(model.tokenizer.decode(out[0]))

# %%
# apply 5x backtracking steering vector
with model.generate(text, max_new_tokens=64) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] += 5 * steering_vectors["backtracking"].detach()
        out = model.generator.output.save()

print(model.tokenizer.decode(out[0]))
# %%
# apply 10x uncertainty-estimation steering vector
with model.generate(text, max_new_tokens=64) as tracer:
    with model.model.layers.all():
        model.model.layers[layer_of_interest].output[0][:] += 10 * steering_vectors["uncertainty-estimation"].detach()
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
text = f"Problem: {random_chain['problem']} 
Reasoning: {random_chain['reasoning_chain']}"
print(text)
# %%
with model.trace(text) as tracer:
    layer_activations = model.model.layers[layer_of_interest].output[0].squeeze().save()

print(layer_activations.shape)

# %%
# get cosine similarity between activations and steering vectors
cosine_similarities = {}

for category, steering_vector in steering_vectors.items():
    cosine_similarities[category] = torch.nn.functional.cosine_similarity(layer_activations, steering_vector, dim=1).detach()

# %%
import plotly.express as px
import pandas as pd
# plot each category's cosine similarity as a multi-line plot
# print tokens on the x-axis
tokens = model.tokenizer.encode(text)
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

# %%
def create_token_visualization(token_text, similarities, category="backtracking", threshold=0.25, color="blue"):
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
    threshold=0.05,
    color="orange"
)
with open("token_visualization.html", "w") as f:
    f.write(html_viz)

# Display in the interactive window
display(HTML(html_viz))

# %%
