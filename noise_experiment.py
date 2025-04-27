import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio

# Configuration parameters
# noise_stds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Standard deviations of Gaussian noise
# noise_scales = [0, 1, 2, 3]  # norm of noise relative to activation norm
noise_scales = [2.5]
num_prompts = 10
# layers_of_interest = list(range(8, 17, 4))  # Different layers to test
layers_of_interest = [10]
# layers_of_interest = [10]
max_new_tokens = 128
seed = 43

def load_random_prompts(file_path, num_prompts):
    with open(file_path, "r") as f:
        all_prompts = json.load(f)
    
    return random.sample(all_prompts, num_prompts)

def count_patterns(text, patterns=["Hmm", "Wait"]):
    counts = {}
    total_tokens = len(text.split())
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        counts[pattern] = len(matches) / total_tokens * 100  # Convert to percentage
    
    return counts

def apply_noise_intervention(model, text, scale, layer):
    activations = []
    with torch.inference_mode():
        with model.generate(text, max_new_tokens=max_new_tokens) as tracer:
            with model.model.layers.all():
                activation = model.model.layers[layer].output[0].save()

                activations.append(activation)
                # Skip intervention if scale is 0 (baseline)
                if scale == 0:
                    pass  # Do nothing for baseline
                else:
                    # Generate Gaussian noise with the same shape as the activation
                    # noise = torch.randn_like(activation) * std
                    noise = torch.randn_like(activation)
                    act_norm = activation.norm(dim=-1, keepdim=True).save()
                    # noise = noise / noise.norm(dim=-1, keepdim=True) * act_norm * scale
                    noise = noise / noise.norm(dim=-1, keepdim=True) * 20
                    noise.save()
                    
                    # Add noise to the activation
                    activation[:] += noise
                
                out = model.generator.output.save()
    if scale != 0:
        print(layer)
        print(act_norm * scale)
    return model.tokenizer.decode(out[0])

def load_model():
    # Import transformers and related modules
    from nnsight import LanguageModel
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model = LanguageModel(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
    return model

def run_experiment():
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Load random prompts
    prompts = load_random_prompts("reasoning_chains/all_reasoning_chains.json", num_prompts)
    
    # Initialize results dictionary
    results = {layer: {scale: [] for scale in noise_scales} for layer in layers_of_interest}
    
    # Run the experiment
    print("Running experiments...")
    for i, prompt in enumerate(tqdm(prompts)):
        print(f"Processing prompt {i+1}/{num_prompts}")
        prompt_text = prompt["problem"]
        
        # Format with chat template
        formatted_prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate the baseline response once per prompt (scale 0)
        # We'll reuse this for all layers with scale 0
        baseline_text = apply_noise_intervention(
            model,
            formatted_prompt,
            0,
            layers_of_interest[0]  # Doesn't matter which layer we use for baseline
        )
        baseline_counts = count_patterns(baseline_text)
        
        # Apply Gaussian noise at each layer with each scale
        pbar = tqdm(total=len(layers_of_interest) * len(noise_scales), desc="Running experiments")
        for layer in layers_of_interest:
            for scale in noise_scales:
                pbar.update(1)
                if scale == 0:
                    # Use the baseline results for all layers at scale 0
                    results[layer][scale].append(baseline_counts)
                else:
                    generated_text = apply_noise_intervention(
                        model, 
                        formatted_prompt, 
                        scale, 
                        layer
                    )
                    
                    # Count patterns
                    pattern_counts = count_patterns(generated_text)
                    results[layer][scale].append(pattern_counts)
    
    # Average the results
    avg_results = {}
    patterns = ["Hmm", "Wait"]
    
    for layer in layers_of_interest:
        avg_results[layer] = {}
        
        for pattern in patterns:
            avg_results[layer][pattern] = []
            
            for scale in noise_scales:
                # Calculate average percentage for this pattern at this scale
                pattern_percentages = [result[pattern] for result in results[layer][scale]]
                avg_percentage = sum(pattern_percentages) / len(pattern_percentages)
                avg_results[layer][pattern].append(avg_percentage)
    
    # Create a DataFrame from the results
    data = []
    for layer in layers_of_interest:
        for i, scale in enumerate(noise_scales):
            hmm_pct = avg_results[layer]["Hmm"][i]
            wait_pct = avg_results[layer]["Wait"][i]
            combined_pct = hmm_pct + wait_pct
            
            data.append({
                "layer": layer,
                "scale": scale,
                "hmm_percentage": hmm_pct,
                "wait_percentage": wait_pct,
                "combined_percentage": combined_pct
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    results_df.to_csv("noise_intervention_results.csv", index=False)
    print("Results saved to noise_intervention_results.csv")
    
    # Plot the results
    plot_heatmap(results_df, patterns)
    
    return avg_results, results_df

def plot_heatmap(results_df, patterns):
    # Create a heatmap for combined percentage
    pivot_combined = results_df.pivot(index="layer", columns="scale", values="combined_percentage")
    
    # Create a figure with subplots - let Plotly use the DataFrame indices directly
    fig_combined = px.imshow(
        pivot_combined,
        color_continuous_scale="viridis",
        text_auto=".2f",
        aspect="auto"
    )
    
    # Update layout
    fig_combined.update_layout(
        title="Effect of Gaussian Noise on Combined Hesitation Markers (Hmm + Wait)",
        xaxis_title="Noise Scale",
        yaxis_title="Layer",
        width=800,
        height=600
    )
    
    # Save figure
    fig_combined.write_image("plots/noise_combined_heatmap.png")
    
    # Create separate heatmaps for each pattern
    for pattern in patterns:
        pivot_pattern = results_df.pivot(index="layer", columns="scale", values=f"{pattern.lower()}_percentage")
        
        # Create a figure with subplots - let Plotly use the DataFrame indices directly
        fig_pattern = px.imshow(
            pivot_pattern,
            color_continuous_scale="viridis",
            text_auto=".2f",
            aspect="auto"
        )
        
        # Update layout
        fig_pattern.update_layout(
            title=f"Effect of Gaussian Noise on {pattern} Tokens",
            xaxis_title="Noise Scale",
            yaxis_title="Layer",
            width=800,
            height=600
        )
        
        # Save figure
        fig_pattern.write_image(f"plots/noise_{pattern.lower()}_heatmap.png")

if __name__ == "__main__":
    avg_results, results_df = run_experiment() 
    # df = pd.read_csv("noise_intervention_results.csv")
    # plot_heatmap(df, ["Hmm", "Wait"])