import json
import os
import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nnsight import LanguageModel

# Constants
ANNOTATED_CHAINS_DIR = "annotated_chains"
BEHAVIOR_CATEGORIES = [
    "initializing", 
    "deduction", 
    "adding-knowledge", 
    "example-testing", 
    "uncertainty-estimation", 
    "backtracking"
]
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

def load_annotated_chains(file_path):
    """Load annotated chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_annotated_chain(chain):
    """Parse an annotated chain to extract text segments for each behavior category"""
    annotated_text = chain["annotated_chain"]
    parsed_segments = {cat: [] for cat in BEHAVIOR_CATEGORIES}
    
    for category in BEHAVIOR_CATEGORIES:
        # Split by category tag
        parts = annotated_text.split(f'["{category}"]')
        # Skip the first part (before the first occurrence)
        for i in range(1, len(parts)):
            if '["end-section"]' in parts[i]:
                # Extract text between category tag and end-section tag
                segment = parts[i].split('["end-section"]')[0].strip()
                parsed_segments[category].append(segment)
    
    return parsed_segments

def get_representative_examples(chains, category, num_examples=10):
    """Get representative text examples for a specific behavior category"""
    all_segments = []
    
    for chain in chains:
        parsed = parse_annotated_chain(chain)
        all_segments.extend(parsed[category])
    
    # Sort by length and take medium-length examples
    sorted_segments = sorted(all_segments, key=len)
    middle_index = len(sorted_segments) // 2
    start_index = max(0, middle_index - num_examples // 2)
    
    # Return a subset of representative examples
    examples = sorted_segments[start_index:start_index + num_examples]
    
    # Make sure we don't have empty examples
    return [ex for ex in examples if ex.strip()]

def tokenize_segments(model, segments):
    """Tokenize text segments for model processing"""
    return [model.tokenizer.encode(segment, return_tensors="pt") for segment in segments]

def extract_attributions(model, positive_examples, negative_examples, layer_range=None):
    """
    Extract attribution scores for model activations using nnsight.
    
    Args:
        model: The language model to analyze
        positive_examples: List of tokenized examples for the target behavior
        negative_examples: List of tokenized examples for contrasting behaviors
        layer_range: Range of model layers to analyze (default: all layers)
    
    Returns:
        Dictionary of attribution scores per layer
    """
    if layer_range is None:
        # Determine total number of layers for the model
        if hasattr(model.model, "layers"):
            layer_range = range(len(model.model.layers))
        else:
            layer_range = range(len(model.transformer.h))
    
    print(f"Extracting attributions for layers {layer_range.start} to {layer_range.stop-1}")
    
    # Store attributions for each layer
    layer_attributions = {}
    
    # Process each layer separately to avoid memory issues
    for layer_idx in tqdm(layer_range, desc="Processing layers"):
        # Collect activation means for positive examples
        pos_layer_activations = []
        
        for example in tqdm(positive_examples, desc=f"Processing positive examples for layer {layer_idx}", leave=False):
            try:
                # Print some debugging info
                print(f"Processing positive example shape: {example.shape}")
                
                # Simple forward pass to get activations
                with torch.no_grad():
                    outputs = model(example)
                    hidden_states = outputs.hidden_states[layer_idx]
                    # Use the last token representation
                    last_token_hidden = hidden_states[:, -1, :]
                    pos_layer_activations.append(last_token_hidden.detach().cpu())
            except Exception as e:
                print(f"Error processing positive example: {e}")
                # Try alternative approach with tracing
                try:
                    with model.trace() as tracer:
                        # Include output_hidden_states
                        with tracer.invoke(example, output_hidden_states=True) as invoker:
                            # For DeepSeek models, try to access hidden states
                            if layer_idx == 0:  # Only print for first layer
                                print(f"Invoker keys: {dir(invoker)}")
                            
                            # Try different ways to access layer outputs
                            if hasattr(model.model, "layers"):
                                # Try to access self attention output
                                result = model.model.layers[layer_idx].self_attn.output.save()
                                # Alternative: try to access MLP output 
                                mlp_result = model.model.layers[layer_idx].mlp.output.save()
                    
                    # Check if we got any results
                    if 'result' in locals() and result is not None and hasattr(result, "value"):
                        pos_layer_activations.append(result.value[:, -1, :].detach().cpu())
                        print(f"Successfully retrieved self_attn output for layer {layer_idx}")
                    elif 'mlp_result' in locals() and mlp_result is not None and hasattr(mlp_result, "value"):
                        pos_layer_activations.append(mlp_result.value[:, -1, :].detach().cpu())
                        print(f"Successfully retrieved mlp output for layer {layer_idx}")
                except Exception as nested_e:
                    print(f"Both approaches failed: {nested_e}")
                    continue
        
        if not pos_layer_activations:
            print(f"Warning: No positive activations collected for layer {layer_idx}")
            continue
        
        # Calculate mean of positive activations
        pos_mean = torch.mean(torch.stack(pos_layer_activations), dim=0)
        print(f"Positive mean shape: {pos_mean.shape}")
            
        # Collect activation means for negative examples
        neg_layer_activations = []
        
        for example in tqdm(negative_examples, desc=f"Processing negative examples for layer {layer_idx}", leave=False):
            try:
                # Simple forward pass to get activations
                with torch.no_grad():
                    outputs = model(example)
                    hidden_states = outputs.hidden_states[layer_idx]
                    # Use the last token representation
                    last_token_hidden = hidden_states[:, -1, :]
                    neg_layer_activations.append(last_token_hidden.detach().cpu())
            except Exception as e:
                print(f"Error processing negative example: {e}")
                # Try alternative approach with tracing
                try:
                    with model.trace() as tracer:
                        # Include output_hidden_states
                        with tracer.invoke(example, output_hidden_states=True) as invoker:
                            # Try different ways to access layer outputs
                            if hasattr(model.model, "layers"):
                                # Try to access self attention output
                                result = model.model.layers[layer_idx].self_attn.output.save()
                                # Alternative: try to access MLP output 
                                mlp_result = model.model.layers[layer_idx].mlp.output.save()
                    
                    # Check if we got any results
                    if 'result' in locals() and result is not None and hasattr(result, "value"):
                        neg_layer_activations.append(result.value[:, -1, :].detach().cpu())
                    elif 'mlp_result' in locals() and mlp_result is not None and hasattr(mlp_result, "value"):
                        neg_layer_activations.append(mlp_result.value[:, -1, :].detach().cpu())
                except Exception as nested_e:
                    print(f"Both approaches failed: {nested_e}")
                    continue
        
        if not neg_layer_activations:
            print(f"Warning: No negative activations collected for layer {layer_idx}")
            continue
        
        # Calculate mean of negative activations
        neg_mean = torch.mean(torch.stack(neg_layer_activations), dim=0)
        print(f"Negative mean shape: {neg_mean.shape}")
            
        # The steering vector is the difference between positive and negative means
        steering_vector = pos_mean - neg_mean
        layer_attributions[layer_idx] = steering_vector
    
    return layer_attributions

def save_steering_vectors(vectors, category, output_dir="steering_vectors"):
    """Save extracted steering vectors to disk"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{category}_steering_vector.pt")
    
    # Save as PyTorch tensors
    torch.save(vectors, output_path)
    print(f"Saved steering vector for '{category}' to {output_path}")

def visualize_attributions(attributions, category, output_dir="visualizations"):
    """Visualize attribution magnitudes across layers"""
    os.makedirs(output_dir, exist_ok=True)
    
    layer_indices = list(attributions.keys())
    attribution_magnitudes = [torch.norm(attr, p=2).item() for attr in attributions.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(layer_indices, attribution_magnitudes)
    plt.xlabel("Layer")
    plt.ylabel("Attribution Magnitude (L2 Norm)")
    plt.title(f"Attribution Magnitude for '{category}' Behavior")
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"{category}_attributions.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")

def extract_steering_vector_for_category(category, model, all_chains):
    """Extract steering vector for a specific behavior category"""
    print(f"\n=== Extracting steering vector for '{category}' ===")
    
    # Get positive examples (examples of the target category)
    positive_examples = get_representative_examples(all_chains, category)
    print(f"Selected {len(positive_examples)} representative examples for '{category}'")
    
    # Get negative examples (examples from other categories)
    other_categories = [cat for cat in BEHAVIOR_CATEGORIES if cat != category]
    negative_examples = []
    for other_cat in other_categories:
        examples = get_representative_examples(all_chains, other_cat, num_examples=2)
        negative_examples.extend(examples)
    print(f"Selected {len(negative_examples)} examples from other categories for contrast")
    
    # Tokenize examples
    tokenized_positive = tokenize_segments(model, positive_examples)
    tokenized_negative = tokenize_segments(model, negative_examples)
    
    # Extract attributions
    attributions = extract_attributions(model, tokenized_positive, tokenized_negative)
    
    # Save the steering vector
    save_steering_vectors(attributions, category)
    
    # Visualize results
    visualize_attributions(attributions, category)
    
    return attributions

def main():
    """Main function to extract steering vectors for all behavior categories"""
    # Load the model
    print(f"Loading model {MODEL_ID}...")
    model = LanguageModel(MODEL_ID, device_map="auto")
    
    # Load all annotated chains
    all_chains_path = os.path.join(ANNOTATED_CHAINS_DIR, "all_annotated_chains.json")
    all_chains = load_annotated_chains(all_chains_path)
    print(f"Loaded {len(all_chains)} annotated reasoning chains")
    
    # Extract steering vectors for each category
    for category in BEHAVIOR_CATEGORIES:
        extract_steering_vector_for_category(category, model, all_chains)
    
    print("\nSteering vector extraction complete!")

if __name__ == "__main__":
    main()
