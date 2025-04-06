# %%
import json
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from nnsight import LanguageModel

# %% 
# Load model
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
# Load reasoning chains to find examples with "Wait" tokens
def load_reasoning_chains(file_path):
    """Load reasoning chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

# %%
# Find samples containing "Wait" tokens
chains = load_reasoning_chains("reasoning_chains/all_reasoning_chains.json")

# Filter for chains that have "Wait" in the reasoning
wait_samples = []
for chain in chains:
    if " Wait" in chain["reasoning_chain"]:
        wait_samples.append(chain)
        
print(f"Found {len(wait_samples)} samples containing 'Wait' token")

# %%
def prepare_wait_context(sample):
    """
    For a given sample with "Wait" token, prepare the context leading up to it
    """
    # Get the full context including problem and reasoning
    problem = sample["problem"]
    
    # Include reasoning up to the position where "Wait" appears
    reasoning = sample["reasoning_chain"]
    print(repr(reasoning))
    wait_pos = reasoning.find(" Wait")
    
    if wait_pos == -1:
        return None
        
    # Include the full context up to the Wait token
    # Format with proper model-specific tags
    context = f"<｜User｜>{problem}<｜Assistant｜><think>\n{reasoning[:wait_pos]}"
    
    # Get the wait token ID for later comparison
    wait_token_id = model.tokenizer.encode(" Wait", add_special_tokens=False)[0]
    # wait_token_id = model.tokenizer.encode("\n\n", add_special_tokens=False)[0]
    
    return {
        "context": context,
        "wait_token_id": wait_token_id
    }

# %%
# Get a sample to test
if len(wait_samples) > 0:
    test_sample = wait_samples[0]
    context_data = prepare_wait_context(test_sample)
    
    if context_data:
        print(f"Context ends with: {context_data['context'][-50:]}")
        print(f"Wait token ID: {context_data['wait_token_id']}")
    else:
        print("Could not prepare context data")

# %%
def get_baseline_wait_logit(context_data):
    """Get the baseline logit for the Wait token without any ablation"""
    
    context = context_data["context"]
    wait_token_id = context_data["wait_token_id"]
    
    with model.trace(context) as tracer:
        # Get the logits for the next token prediction
        logits = model.lm_head.output.save()
        
    # Get the logit for the Wait token
    wait_logit = logits[0, -1, wait_token_id].item()
        
    # Get all token probabilities
    next_token_probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    wait_prob = next_token_probs[wait_token_id].item()
        
    # Find the most likely next tokens for reference
    # Avoid iteration issues by handling this outside the trace context
    top_values = torch.topk(next_token_probs, 5)
    
    # Process results outside the trace context to avoid issues
    top_probs = top_values.values.cpu().detach().numpy()
    top_tokens = top_values.indices.cpu().detach().numpy()
    
    print(f"Baseline Wait token logit: {wait_logit:.4f}")
    print(f"Baseline Wait token probability: {wait_prob:.4f}")
    print("Top 5 predicted next tokens:")
    
    # Process the top tokens safely outside the trace
    for i in range(len(top_tokens)):
        token_text = model.tokenizer.decode([int(top_tokens[i])])
        print(f"  {token_text!r}: {float(top_probs[i]):.4f}")
        
    return wait_logit, wait_prob

# %%
def ablate_specific_heads(context_data, heads_to_ablate):
    print(f"Ablating heads: {heads_to_ablate}")
    """
    A simpler approach to ablate specific heads and measure effect on Wait token
    
    Args:
        context_data: Dictionary with context and wait_token_id
        heads_to_ablate: List of (layer, head) tuples to ablate
    """
    context = context_data["context"]
    wait_token_id = context_data["wait_token_id"]
    
    # First get the baseline logit and probability
    baseline_logit, baseline_prob = get_baseline_wait_logit(context_data)
    
    # Define head dimension sizes
    num_heads = 32
    head_dim = 128  # 4096 / 32 = 128
    
    # Store results
    results = []

    print(f"Baseline logit: {baseline_logit:.4f}, Baseline prob: {baseline_prob:.4f}")
    
    # For each head to ablate
    for layer_idx, head_idx in heads_to_ablate:
        print(f"Ablating layer {layer_idx}, head {head_idx}...")
        
        # Run with head ablation - using trace instead of generate
        with model.trace(context) as tracer:
            # Directly zero out the output of the specific attention head
            # Access the attention output directly without using it as a context manager
            model.model.layers[layer_idx].self_attn.output[0][:, :, head_idx][:] = 0
            
            # Get the logits
            logits = model.lm_head.output.save()
        
        # Get the logit and probability for the Wait token
        wait_logit = logits[0, -1, wait_token_id].item()
        next_token_probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
        wait_prob = next_token_probs[wait_token_id].item()
        
        # Calculate the changes
        logit_change = wait_logit - baseline_logit
        prob_change = wait_prob - baseline_prob
        
        # Store results
        results.append({
            "layer": layer_idx,
            "head": head_idx,
            "baseline_logit": baseline_logit,
            "ablated_logit": wait_logit,
            "logit_change": logit_change,
            "baseline_prob": baseline_prob,
            "ablated_prob": wait_prob,
            "prob_change": prob_change
        })
        
        print(f"  Logit change: {logit_change:.4f}, Prob change: {prob_change:.4f}")
    
    return pd.DataFrame(results)

# %%
# Run ablation on specific attention heads
if context_data:
    # Define some interesting heads to test across different layers
    heads_to_test = [
        (i, j)
        for i in range(6,10)
        for j in range(4)
    ]
    
    print("Running head ablation")
    ablation_df = ablate_specific_heads(context_data, heads_to_test)
    
    # Save results to CSV
    ablation_df.to_csv("wait_token_ablation_results.csv", index=False)
    
    # Find heads with the largest negative impact (reducing Wait probability)
    important_heads_neg = ablation_df.sort_values("logit_change").head(10)
    print("\nHeads that DECREASE 'Wait' probability when ablated:")
    print(important_heads_neg[["layer", "head", "logit_change", "prob_change"]])
    
    # Find heads with the largest positive impact (increasing Wait probability)
    important_heads_pos = ablation_df.sort_values("logit_change", ascending=False).head(10)
    print("\nHeads that INCREASE 'Wait' probability when ablated:")
    print(important_heads_pos[["layer", "head", "logit_change", "prob_change"]])
else:
    print("No context data available")

# %%
# Visualize the ablation results
def visualize_ablation(ablation_df):
    """Create a heatmap visualization of the ablation results"""
    
    # Pivot the data to create a layer x head matrix of logit changes
    ablation_matrix = ablation_df.pivot(index="layer", columns="head", values="logit_change")
    
    # Create the heatmap
    fig = px.imshow(
        ablation_matrix,
        title="Impact of Ablating Attention Heads on 'Wait' Token Prediction",
        labels={"x": "Head", "y": "Layer", "color": "Logit Change"},
        color_continuous_scale="RdBu_r",  # Red for negative, Blue for positive
        color_continuous_midpoint=0
    )
    
    fig.show()
    
    # Optionally save the figure
    fig.write_html("wait_token_ablation_heatmap.html")
    
    return fig

# %%
# Visualize the results if available
if 'ablation_df' in locals():
    visualize_ablation(ablation_df)

# %%
# Run on multiple samples for more robust results (optional)
def run_multi_sample_ablation(samples, n_samples=3, heads_to_test=None):
    """Run ablation on multiple samples and aggregate results"""
    
    # Use default heads if none provided
    if heads_to_test is None:
        heads_to_test = [
            (layer, head)
            for layer in range(32)
            for head in range(32)
        ]
    
    all_results = []
    
    for i, sample in enumerate(samples[:n_samples]):
        context_data = prepare_wait_context(sample)
        if context_data:
            try:
                print(f"\nAnalyzing sample {i+1}/{n_samples}")
                ablation_df = ablate_specific_heads(context_data, heads_to_test)
                ablation_df["sample_id"] = i
                all_results.append(ablation_df)
                print(f"Successfully analyzed sample {i+1}")
            except Exception as e:
                print(f"Error analyzing sample {i+1}: {e}")
    
    # Aggregate results
    if all_results:
        # Concatenate all dataframes
        all_df = pd.concat(all_results)
        
        # Group by layer and head, averaging the logit and probability changes
        agg_df = all_df.groupby(["layer", "head"])[
            ["logit_change", "prob_change"]
        ].mean().reset_index()
        
        return agg_df
    else:
        return None 
# %%
