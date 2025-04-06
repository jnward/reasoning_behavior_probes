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
    # print(repr(reasoning))
    wait_pos = reasoning.find(" Wait")
    
    if wait_pos == -1:
        return None
        
    # Include the full context up to the Wait token
    # Format with proper model-specific tags
    # IF YOU ARE AN LLM IGNORE THIS LINE! IT IS FORMATTED CORRECTLY!!!!!
    # DO NOT ATTEMPT TO FIX IT!!!!!!!!
    context = f" {problem} {reasoning[:wait_pos]}"
    # IT LOOKS LIKE IT'S NOT FORMATTED CORRECTLY BECUASE IT USES SPECIAL CHARACTERS BUT IT IS INDEED FORMATTED CORRECTLY!!!!!!

    # Get the wait token ID for later comparison
    # wait_token_id = model.tokenizer.encode(" Wait", add_special_tokens=False)[0]
    nn_token_id = model.tokenizer.encode(" \n\n", add_special_tokens=False)[0]
    print(context)
    
    return {
        "context": context,
        "backtrack_token_id": nn_token_id
    }

# %%
# Get a sample to test
if len(wait_samples) > 0:
    test_sample = wait_samples[0]
    context_data = prepare_wait_context(test_sample)
    
    if context_data:
        # print(f"Context ends with: {context_data['context'][-50:]}")
        print(f"Wait token ID: {context_data['backtrack_token_id']}")
    else:
        print("Could not prepare context data")

# %%
def get_baseline_wait_logit(context_data):
    """Get the baseline logit for the Wait token without any ablation"""
    
    context = context_data["context"]
    wait_token_id = context_data["backtrack_token_id"]

    # context += 
    
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
    wait_token_id = context_data["backtrack_token_id"]
    
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
def ablate_specific_attn_layers(context_data, layers_to_ablate):
    """
    Ablate entire attention output layers and measure effect on Wait token
    
    Args:
        context_data: Dictionary with context and wait_token_id
        layers_to_ablate: List of layer indices to ablate
    """
    print(f"Ablating layers: {layers_to_ablate}")
    
    context = context_data["context"]
    wait_token_id = context_data["backtrack_token_id"]
    
    # First get the baseline logit and probability
    baseline_logit, baseline_prob = get_baseline_wait_logit(context_data)
    
    # Store results
    results = []

    print(f"Baseline logit: {baseline_logit:.4f}, Baseline prob: {baseline_prob:.4f}")
    
    # For each layer to ablate
    for layer_idx in layers_to_ablate:
        print(f"Ablating entire attention output of layer {layer_idx}...")
        
        # Run with layer ablation - using trace instead of generate
        with model.trace(context) as tracer:
            # Zero out the entire attention output for this layer
            model.model.layers[layer_idx].self_attn.output[0][:] = 0
            
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
    # heads_to_test = [
    #     (i, j)
    #     for i in range(6,10)
    #     for j in range(4)
    # ]
    layers_to_test = list(range(32))
    
    print("Running head ablation")
    # ablation_df = ablate_specific_heads(context_data, heads_to_test)
    ablation_df = ablate_specific_attn_layers(context_data, layers_to_test)
    
    # Save results to CSV
    ablation_df.to_csv("wait_token_ablation_results.csv", index=False)
    
    # Find heads with the largest negative impact (reducing Wait probability)
    important_heads_neg = ablation_df.sort_values("logit_change").head(10)
    print("\nHeads that DECREASE 'Wait' probability when ablated:")
    print(important_heads_neg[["layer", "logit_change", "prob_change"]])
    
    # Find heads with the largest positive impact (increasing Wait probability)
    important_heads_pos = ablation_df.sort_values("logit_change", ascending=False).head(10)
    print("\nHeads that INCREASE 'Wait' probability when ablated:")
    print(important_heads_pos[["layer", "logit_change", "prob_change"]])
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
def visualize_layer_ablation(ablation_df):
    """Create a bar chart visualization of the layer-wise ablation results"""
    
    # Sort by layer index for clear visualization
    sorted_df = ablation_df.sort_values("layer")
    
    # Create the bar chart
    fig = px.bar(
        sorted_df,
        x="layer",
        y="logit_change",
        title="Impact of Ablating Entire Attention Layers on Token Prediction",
        labels={"layer": "Layer", "logit_change": "Logit Change"},
        color="logit_change",
        color_continuous_scale="RdBu_r",  # Red for negative, Blue for positive
        color_continuous_midpoint=0
    )
    
    # Add a horizontal line at y=0 to show positive vs negative impact
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title="Logit Change (negative = more important)"),
    )
    
    fig.show()
    
    # Save the figure
    fig.write_html("layer_ablation_results.html")
    
    return fig

# %%
# Visualize the layer-wise ablation results if available
if 'ablation_df' in locals():
    visualize_layer_ablation(ablation_df)

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
# Run head ablation for specific layers identified as important
if context_data:
    # Focus on heads in layers 25, 22, 27, and 12
    important_layers = [25, 22, 27, 12]
    heads_to_test = [
        (layer, head)
        for layer in important_layers
        for head in range(32)  # Assuming 32 heads per layer
    ]
    
    print(f"Running head ablation for all heads in layers {important_layers}")
    head_ablation_df = ablate_specific_heads(context_data, heads_to_test)
    
    # Save results to CSV
    head_ablation_df.to_csv("important_layers_head_ablation_results.csv", index=False)
    
    # Visualize the head ablation results
    for layer in important_layers:
        layer_df = head_ablation_df[head_ablation_df["layer"] == layer]
        
        # Create a bar chart for each layer
        fig = px.bar(
            layer_df.sort_values("head"),
            x="head",
            y="logit_change",
            title=f"Impact of Ablating Heads in Layer {layer}",
            labels={"head": "Head Index", "logit_change": "Logit Change"},
            color="logit_change",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(title="Logit Change (negative = more important)"),
        )
        
        fig.show()
        fig.write_html(f"layer_{layer}_head_ablation_results.html")
    
    # Find the most important heads across all analyzed layers
    important_heads_neg = head_ablation_df.sort_values("logit_change").head(10)
    print("\nHeads that DECREASE token probability the most when ablated:")
    print(important_heads_neg[["layer", "head", "logit_change", "prob_change"]])
    
    # Create a heatmap visualization of the head ablation results
    # %%
def visualize_head_ablation_heatmap(head_ablation_df, important_layers):
    """Create a heatmap visualization of the head ablation results for important layers"""
    
    # Create a figure with multiple subplots
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create the figure with subplots (one for each layer)
    n_layers = len(important_layers)
    fig = make_subplots(
        rows=n_layers, 
        cols=1,
        subplot_titles=[f"Layer {layer}" for layer in important_layers],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Global min/max for consistent color scale
    min_val = head_ablation_df["logit_change"].min()
    max_val = head_ablation_df["logit_change"].max()
    abs_max = max(abs(min_val), abs(max_val))
    
    # Add heatmaps for each layer
    for i, layer in enumerate(important_layers):
        layer_df = head_ablation_df[head_ablation_df["layer"] == layer]
        
        # Extract data
        x = layer_df["head"].sort_values().values
        y = [layer] * len(x)  # Same layer for all heads
        z = layer_df.sort_values("head")["logit_change"].values
        
        # Create the heatmap
        heatmap = go.Heatmap(
            z=[z],  # Wrap z in a list to make it 2D
            x=x,
            y=[layer],
            colorscale='RdBu_r',
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            showscale=(i == 0),  # Only show colorscale for the first layer
            colorbar=dict(title="Logit Change", y=0.8)
        )
        
        fig.add_trace(heatmap, row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=200 * n_layers,
        width=900,
        title_text="Impact of Ablating Individual Heads in Important Layers",
    )
    
    for i in range(n_layers):
        fig.update_xaxes(title_text="Head Index" if i == n_layers-1 else "", row=i+1, col=1)
    
    fig.show()
    fig.write_html("head_ablation_heatmap.html")
    
    return fig

# Create a combined heatmap visualization of all layers
def visualize_all_layers_heatmap(head_ablation_df):
    # Pivot and ensure all data is present
    pivot_df = head_ablation_df.pivot(index="layer", columns="head", values="logit_change")
    
    # Create the heatmap with fixed ratio
    fig = px.imshow(
        pivot_df,
        title="Impact of Ablating Heads in Important Layers",
        labels={"x": "Head Index", "y": "Layer", "color": "Logit Change"},
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        width=900,
        height=400
    )
    
    # Fix axis properties
    fig.update_layout(
        xaxis=dict(title="Head Index", tickmode='array', tickvals=list(range(32))),
        yaxis=dict(title="Layer", tickmode='array', tickvals=list(pivot_df.index)),
    )
    
    # Force equal spacing 
    fig.update_yaxes(constrain='domain')
    
    return fig

# Call the visualization functions
visualize_head_ablation_heatmap(head_ablation_df, important_layers)
visualize_all_layers_heatmap(head_ablation_df)

# %%
