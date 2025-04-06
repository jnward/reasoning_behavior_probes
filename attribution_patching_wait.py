# %%
import json
import torch
import einops
import pandas as pd
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
# Function to prepare context for attribution analysis
def prepare_wait_attribution(sample):
    """
    For a given sample with "Wait" token, prepare it for attribution analysis
    to determine which heads most influence the " Wait" token prediction.
    """
    # Get the full context including problem and reasoning
    problem = sample["problem"]
    
    # Include reasoning up to the position where "Wait" appears
    reasoning = sample["reasoning_chain"]
    wait_pos = reasoning.find(" Wait")
    
    if wait_pos == -1:
        return None
        
    # Include the full context up to the Wait token
    # Format with <|User|> tag for proper model context
    context = f"<｜User｜>{problem}<｜Assistant｜><think>\n{reasoning[:wait_pos]}"
    
    # Get the wait token ID for later comparison
    wait_token_id = model.tokenizer.encode(" Wait", add_special_tokens=False)[0]
    
    return {
        "context": context,
        "wait_token_id": wait_token_id,
        "wait_position": wait_pos
    }

# %%
# Get a sample to test
if len(wait_samples) > 0:
    test_sample = wait_samples[0]
    attribution_data = prepare_wait_attribution(test_sample)
    
    if attribution_data:
        print(f"Context ends with: {attribution_data['context'][-50:]}")
        print(f"Wait token ID: {attribution_data['wait_token_id']}")
    else:
        print("Could not prepare attribution data")

# %%
# Run the attribution analysis
def run_wait_token_attribution(attribution_data):
    """Find which attention heads most influence the " Wait" token prediction"""
    
    context = attribution_data["context"]
    wait_token_id = attribution_data["wait_token_id"]
    
    # The number of layers and heads in the model
    num_layers = len(model.model.layers)
    num_heads = 32  # Typically 32 for this model size
    
    # Storage for attention head outputs and gradients
    attn_outputs = []
    attn_grads = []
    
    # Actually pass the context to the model.trace
    with model.trace(context) as tracer:
        # Run the model on the full context
        # Get attention outputs and track gradients
        for layer in model.model.layers:
            # Get attention output
            attn_out = layer.self_attn.o_proj.input[0]
            # attn_out.requires_grad = True
            attn_outputs.append(attn_out.save())
            attn_grads.append(attn_out.grad.save())
        
        # Get the logits for the next token prediction
        logits = model.lm_head.output[:, -1].save()
        
        # Get the logit for the Wait token
        wait_logit = logits[0, wait_token_id]
        
        # Compute gradients with respect to the Wait token logit
        wait_logit.backward()
    
    # Process results to determine which attention heads most influence the Wait token
    attribution_results = []
    
    for layer_idx, (attn_out, attn_grad) in enumerate(zip(attn_outputs, attn_grads)):
        # Calculate attribution scores for heads in this layer
        # We want to know which heads' outputs most influence the Wait token logit
        # Use the gradient * activation to measure influence
        head_attributions = einops.reduce(
            attn_grad.value[:, -1, :] * attn_out.value[:, -1, :],
            "batch (head dim) -> head",
            "sum",
            head=num_heads,
            dim=128
        )
        
        # Store results for this layer
        for head_idx, score in enumerate(head_attributions.detach().cpu().numpy()):
            attribution_results.append({
                "layer": layer_idx,
                "head": head_idx,
                "attribution_score": float(score)
            })
    
    return attribution_results

# %%
# Run attribution analysis on the test sample
if attribution_data:
    print("Running wait token attribution analysis...")
    attribution_results = run_wait_token_attribution(attribution_data)
    
    # Convert to DataFrame for visualization
    results_df = pd.DataFrame(attribution_results)
    
    # Create a matrix of layer x head for visualization
    attribution_matrix = results_df.pivot(index="layer", columns="head", values="attribution_score")
    
    # Find the top contributing heads
    top_heads = results_df.sort_values("attribution_score", ascending=False).head(10)
    print("\nTop 10 attention heads responsible for boosting the ' Wait' token logit:")
    print(top_heads)
else:
    print("No attribution data available")

# %%
# Visualize the results
if 'attribution_matrix' in locals():
    fig = px.imshow(
        attribution_matrix,
        title="Attribution Scores for ' Wait' Token Prediction",
        labels={"x": "Head", "y": "Layer", "color": "Attribution Score"},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    fig.show()

# %%
# Run analysis on multiple samples for more robust results
def run_multiple_attributions(samples, n_samples=5):
    """Run attribution analysis on multiple samples and aggregate results"""
    all_results = []
    
    for i, sample in enumerate(tqdm(samples[:n_samples])):
        attribution_data = prepare_wait_attribution(sample)
        if attribution_data:
            try:
                results = run_wait_token_attribution(attribution_data)
                all_results.extend(results)
                print(f"Successfully analyzed sample {i+1}")
            except Exception as e:
                print(f"Error analyzing sample {i+1}: {e}")
    
    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        aggregated = results_df.groupby(["layer", "head"])["attribution_score"].mean().reset_index()
        return aggregated
    else:
        return None

# %%
# Run on multiple samples (uncomment to run)
# print("Running attribution analysis on multiple samples...")
# aggregated_results = run_multiple_attributions(wait_samples, n_samples=3)
# 
# if aggregated_results is not None:
#     # Create a matrix for visualization
#     agg_matrix = aggregated_results.pivot(index="layer", columns="head", values="attribution_score")
#     
#     # Visualize
#     fig = px.imshow(
#         agg_matrix,
#         title="Aggregated Attribution Scores for ' Wait' Token Prediction",
#         labels={"x": "Head", "y": "Layer", "color": "Attribution Score"},
#         color_continuous_scale="RdBu",
#         color_continuous_midpoint=0
#     )
#     fig.show()
#     
#     # Find top contributors
#     top_heads = aggregated_results.sort_values("attribution_score", ascending=False).head(10)
#     print("\nTop 10 attention heads responsible for predicting ' Wait':")
#     print(top_heads)
# else:
#     print("No aggregated results available") 