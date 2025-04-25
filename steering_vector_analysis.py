import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from nnsight import LanguageModel
from collections import defaultdict
from tqdm import tqdm
import os
import argparse

def load_annotated_chain(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_annotations(annotated_text):
    annotations = []
    
    current_pos = 0
    while True:
        start_tag_pos = annotated_text.find('[\"', current_pos)
        if start_tag_pos == -1:
            break
            
        end_tag_pos = annotated_text.find('\"]', start_tag_pos)
        if end_tag_pos == -1:
            break
            
        category = annotated_text[start_tag_pos+2:end_tag_pos]
        
        if category == "end-section":
            current_pos = end_tag_pos + 2
            continue
            
        start_text_pos = end_tag_pos + 2
        end_section_tag = annotated_text.find('[\"end-section\"]', start_text_pos)
        
        if end_section_tag != -1:
            text = annotated_text[start_text_pos:end_section_tag].strip()
            annotations.append((category, text))
            current_pos = end_section_tag + 15
        else:
            current_pos = end_tag_pos + 2
    
    return annotations

def process_chain(model, chain):
    problem = chain["problem"]
    formatted_problem = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    tokenized_problem = model.tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]
    
    annotations = extract_annotations(chain["annotated_chain"])
    
    annotation_indices = {}
    
    current_token_pos = len(tokenized_problem)
    
    full_tokens = tokenized_problem.tolist()
    
    for i, (category, text) in enumerate(annotations):
        if i > 0:
            text = " " + text
        segment_tokens = model.tokenizer.encode(text, add_special_tokens=False)
        
        start_idx = current_token_pos
        end_idx = start_idx + len(segment_tokens) - 1
        
        if category not in annotation_indices:
            annotation_indices[category] = []
        annotation_indices[category].append((start_idx, end_idx))
        
        full_tokens.extend(segment_tokens)
        
        current_token_pos += len(segment_tokens)
    
    tokenized_full_text = torch.tensor(full_tokens)
    
    return tokenized_full_text, annotation_indices

def process_chains_iterator(model, chains):
    for i, chain in enumerate(chains):
        tokenized_text, annotation_indices = process_chain(model, chain)
        
        chain_id = chain.get('task_id', f'chain_{i}')
        yield tokenized_text, annotation_indices

def analyze_steering_vectors(layer_number, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", chains_file="annotated_chains/all_annotated_chains.json", output_dir="steering_vector_plots"):
    """
    Analyze steering vectors for a specific layer and generate plots
    
    Args:
        layer_number: Layer to analyze
        model_name: Name of the model to use
        chains_file: Path to annotated chains file
        output_dir: Directory to save plots
    """
    print(f"Analyzing steering vectors for layer {layer_number}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and chains
    model = LanguageModel(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
    chains = load_annotated_chain(chains_file)
    processed_chains = process_chains_iterator(model, chains)
    
    # Extract activations
    activations = defaultdict(list)
    
    with torch.inference_mode():
        for tokens, indices in tqdm(processed_chains):
            text = model.tokenizer.decode(tokens)
            tokens2 = model.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
            assert torch.equal(tokens, tokens2)
            with model.trace(text) as tracer:
                if layer_number == -1:
                    layer_activations = model.model.embed_tokens.output.save()
                else:
                    layer_activations = model.model.layers[layer_number].output[0].save()
            torch.cuda.empty_cache()
            for category, index_tuples in indices.items():
                if category not in ["backtracking", "uncertainty-estimation", "initializing", "deduction", "example-testing", "adding-knowledge"]:
                    continue
                for start, end in index_tuples:
                    activations[category].append(layer_activations[0, start-1:end+1].float().cpu())
    
    # Calculate overall mean
    overall_mean = torch.zeros(model.config.hidden_size, dtype=torch.float64)
    num_activations = 0
    for _, layer_activations in activations.items():
        for la in layer_activations:
            overall_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
    overall_mean /= num_activations
    overall_mean = overall_mean.to(torch.float32)
    
    # Calculate mean vectors for each category
    mean_vectors = {}
    for category, layer_activations in activations.items():
        category_mean = torch.zeros(model.config.hidden_size, dtype=torch.float64)
        num_activations = 0
        for la in layer_activations:
            category_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
        category_mean /= num_activations
        mean_vectors[category] = category_mean.to(torch.float32)
    
    # Compute steering vectors
    steering_vectors = {}
    for category, mean_vector in mean_vectors.items():
        steering_vectors[category] = mean_vectors[category] - overall_mean
    
    # Normalize steering vectors
    steering_vectors = {k: v / v.norm() for k, v in steering_vectors.items()}
    
    # Free memory
    del activations
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create cosine similarity matrix plot
    plot_cosine_similarity(steering_vectors, layer_number, output_dir)
    
    # Create PCA plots
    plot_pca_analysis(steering_vectors, layer_number, output_dir)
    
    return steering_vectors

def plot_cosine_similarity(steering_vectors, layer_number, output_dir):
    """
    Plot cosine similarity matrix between all steering vectors
    
    Args:
        steering_vectors: Dictionary of steering vectors
        layer_number: Layer number for plot title
        output_dir: Directory to save plots
    """
    categories = list(steering_vectors.keys())
    n_categories = len(categories)
    
    similarity_matrix = np.zeros((n_categories, n_categories))
    
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            vec1 = steering_vectors[cat1]
            vec2 = steering_vectors[cat2]
            similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
            similarity_matrix[i, j] = similarity
    
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
        title=f'Layer {layer_number}: Cosine Similarity Between Steering Vectors',
        xaxis=dict(tickangle=45),
        height=600,
        width=700
    )
    
    # Save to file as PNG with high DPI
    filename = os.path.join(output_dir, f'layer_{layer_number}_cosine_similarity.png')
    fig.write_image(filename, scale=4)  # Higher scale for better resolution
    
    # No interactive display in script mode

def plot_pca_analysis(steering_vectors, layer_number, output_dir):
    """
    Perform PCA on steering vectors and create plots
    
    Args:
        steering_vectors: Dictionary of steering vectors
        layer_number: Layer number for plot titles
        output_dir: Directory to save plots
    """
    # Get vector dimension
    vec_dim = next(iter(steering_vectors.values())).shape[0]
    
    # Create matrix of vectors
    matrix = torch.zeros((len(steering_vectors), vec_dim))
    for i, category in enumerate(steering_vectors.keys()):
        matrix[i] = steering_vectors[category]
    
    # Compute SVD
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    
    # Compute explained variance
    s_squared = s ** 2
    total_variance = s_squared.sum()
    explained_variance = s_squared / total_variance
    cumulative_variance = torch.cumsum(explained_variance, dim=0)
    
    # Create explained variance plot
    df = pd.DataFrame({
        'Singular Value Index': range(1, len(s)+1),
        'Explained Variance': explained_variance.detach().cpu().numpy() * 100,
        'Cumulative Variance': cumulative_variance.detach().cpu().numpy() * 100
    })
    
    fig = px.bar(df, x='Singular Value Index', y='Explained Variance',
                 title=f'Layer {layer_number}: Explained Variance by Singular Value',
                 labels={'Explained Variance': 'Explained Variance (%)'})
    
    fig.add_scatter(x=df['Singular Value Index'], y=df['Cumulative Variance'],
                    mode='lines+markers', name='Cumulative Variance (%)')
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    
    # Save to file as PNG with high DPI
    filename = os.path.join(output_dir, f'layer_{layer_number}_explained_variance.png')
    fig.write_image(filename, scale=4)  # Higher scale for better resolution
    
    # No interactive display in script mode
    
    # Create 2D projection plot
    plot_2d_projection(steering_vectors, vh[:2].T, layer_number, output_dir)

def plot_2d_projection(steering_vectors, projection_matrix, layer_number, output_dir):
    """
    Create a 2D scatter plot of steering vectors projected onto first two principal components
    
    Args:
        steering_vectors: Dictionary of steering vectors
        projection_matrix: Matrix to project vectors onto 2D space
        layer_number: Layer number for plot title
        output_dir: Directory to save plot
    """
    # Project vectors
    projected_vectors = {}
    for category, vector in steering_vectors.items():
        projection = torch.matmul(vector, projection_matrix)
        projected_vectors[category] = projection
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add lines from origin to points
    for category, projection in projected_vectors.items():
        x, y = projection[0].item(), projection[1].item()
        
        # Calculate norm of projected vector
        norm = torch.norm(projection).item()
        
        # Add line from origin
        fig.add_trace(go.Scatter(
            x=[0, x],
            y=[0, y],
            mode='lines',
            name=f"{category} vector",
            line=dict(color="gray", width=2, dash='solid'),
            showlegend=False
        ))
        
        # Add norm label
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
    
    # Add origin
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='Origin',
        marker=dict(color='black', size=8)
    ))
    
    fig.update_layout(
        title=f'Layer {layer_number}: Steering Vectors Projected onto First Two Principal Components',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        height=600,
        width=700
    )
    
    # Save to file as PNG with high DPI
    filename = os.path.join(output_dir, f'layer_{layer_number}_2d_projection.png')
    fig.write_image(filename, scale=4)  # Higher scale for better resolution
    
    # No interactive display in script mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze steering vectors for a specific layer and generate plots")
    parser.add_argument("layer", type=int, help="Layer number to analyze")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--chains", type=str, default="annotated_chains/all_annotated_chains.json", help="Path to annotated chains file")
    parser.add_argument("--output", type=str, default="steering_vector_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    analyze_steering_vectors(
        layer_number=args.layer,
        model_name=args.model,
        chains_file=args.chains,
        output_dir=args.output
    )
