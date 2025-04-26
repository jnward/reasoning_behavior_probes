import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pandas as pd
from tqdm import tqdm
from generate_steering_vectors import generate_steering_vectors

# Configuration parameters
intervention_magnitudes = [0, 2, 4, 6, 8, 10, 12, 15]
num_prompts = 10
layer_of_interest = 10
max_new_tokens = 128
seed = 42
intervention_type = "backtracking"  # Choose the behavior to analyze
offsets = [0, -4, -8, -12, -16, -20]  # Different offsets to compare
window_size = 4

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

def apply_intervention(model, text, steering_vector, magnitude, layer_of_interest):
    with model.generate(text, max_new_tokens=max_new_tokens) as tracer:
        with model.model.layers.all():
            activation = model.model.layers[layer_of_interest].output[0]
            
            # Skip intervention if magnitude is 0 (baseline)
            if magnitude == 0:
                pass  # Do nothing for baseline
            else:
                # Apply the steering vector
                activation[:] += magnitude * steering_vector.to(activation.device)
            
            out = model.generator.output.save()
    
    return model.tokenizer.decode(out[0])

def run_experiment():
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate steering vectors for each offset
    print("Generating steering vectors for different offsets...")
    offset_steering_vectors = {}
    model = None
    
    for offset in offsets:
        print(f"Generating vectors with offset {offset}...")
        steering_vectors, overall_mean, model = generate_steering_vectors(
            "annotated_chains/all_annotated_chains.json", 
            layer_of_interest=layer_of_interest,
            offset=offset,
            window_size=window_size
        )
        offset_steering_vectors[offset] = steering_vectors[intervention_type]
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Load random prompts
    prompts = load_random_prompts("reasoning_chains/all_reasoning_chains.json", num_prompts)
    
    # Initialize results dictionary
    results = {offset: {mag: [] for mag in intervention_magnitudes} for offset in offsets}
    
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
        
        # Generate the baseline response once per prompt (magnitude 0)
        # We'll reuse this for all offsets with magnitude 0
        baseline_text = apply_intervention(
            model,
            formatted_prompt,
            offset_steering_vectors[offsets[0]],  # Doesn't matter which vector we use for baseline
            0,
            layer_of_interest
        )
        baseline_counts = count_patterns(baseline_text)
        
        # Apply steering vector from each offset at each magnitude
        for offset in offsets:
            for magnitude in intervention_magnitudes:
                if magnitude == 0:
                    # Use the baseline results for all offsets at magnitude 0
                    results[offset][magnitude].append(baseline_counts)
                else:
                    generated_text = apply_intervention(
                        model, 
                        formatted_prompt, 
                        offset_steering_vectors[offset], 
                        magnitude, 
                        layer_of_interest
                    )
                    
                    # Count patterns
                    pattern_counts = count_patterns(generated_text)
                    results[offset][magnitude].append(pattern_counts)
    
    # Average the results
    avg_results = {}
    patterns = ["Hmm", "Wait"]
    
    for offset in offsets:
        avg_results[offset] = {}
        
        for pattern in patterns:
            avg_results[offset][pattern] = []
            
            for magnitude in intervention_magnitudes:
                # Calculate average percentage for this pattern at this magnitude
                pattern_percentages = [result[pattern] for result in results[offset][magnitude]]
                avg_percentage = sum(pattern_percentages) / len(pattern_percentages)
                avg_results[offset][pattern].append(avg_percentage)
    
    # Create a DataFrame from the results
    data = []
    for offset in offsets:
        for i, magnitude in enumerate(intervention_magnitudes):
            hmm_pct = avg_results[offset]["Hmm"][i]
            wait_pct = avg_results[offset]["Wait"][i]
            combined_pct = hmm_pct + wait_pct
            
            data.append({
                "offset": offset,
                "magnitude": magnitude,
                "hmm_percentage": hmm_pct,
                "wait_percentage": wait_pct,
                "combined_percentage": combined_pct
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    results_df.to_csv(f"offset_intervention_results_{intervention_type}.csv", index=False)
    print(f"Results saved to offset_intervention_results_{intervention_type}.csv")
    
    # Plot the results
    plot_results(avg_results, patterns, offsets, intervention_magnitudes, intervention_type)
    
    return avg_results, results_df

def plot_results(avg_results, patterns, offsets, intervention_magnitudes, intervention_type):
    # Create separate plots for each pattern
    for pattern in patterns:
        plt.figure(figsize=(12, 8))
        
        for offset in offsets:
            plt.plot(
                intervention_magnitudes, 
                avg_results[offset][pattern], 
                marker='o', 
                linewidth=2, 
                label=f"Offset {offset}"
            )
        
        plt.xlabel('Intervention Magnitude')
        plt.ylabel(f'Average {pattern} Token %')
        plt.title(f'Effect of {intervention_type} Steering Vector (Different Offsets) on {pattern} Token Percentage')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/{pattern}_offset_intervention_results_{intervention_type}.png')
        plt.show()
    
    # Create a combined plot with the sum of both patterns
    plt.figure(figsize=(15, 10))
    
    for offset in offsets:
        # Calculate the sum of both pattern percentages for each magnitude
        combined_percentages = []
        for i in range(len(intervention_magnitudes)):
            combined = avg_results[offset]["Hmm"][i] + avg_results[offset]["Wait"][i]
            combined_percentages.append(combined)
        
        # Plot the combined line
        plt.plot(
            intervention_magnitudes, 
            combined_percentages, 
            marker='o', 
            linewidth=2, 
            label=f"Offset {offset}"
        )
    
    plt.xlabel('Intervention Magnitude')
    plt.ylabel('Average Combined "Hmm" + "Wait" Token %')
    plt.title(f'Effect of {intervention_type} Steering Vector (Different Offsets) on Combined Hesitation Markers')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/combined_offset_intervention_results_{intervention_type}.png')
    plt.show()

if __name__ == "__main__":
    avg_results, results_df = run_experiment() 