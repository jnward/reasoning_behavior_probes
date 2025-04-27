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
intervention_magnitudes = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25]  # Added 0 for baseline
# intervention_magnitudes = [0, 12]
num_prompts = 10
layer_of_interest = 10
max_new_tokens = 128
seed = 43
# intervention_types = ["backtracking", "deduction", "initializing", "noise", "overall_mean", "self"]
intervention_types = ["noise"]
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

def apply_intervention(model, text, intervention_type, intervention_vector, magnitude, layer_of_interest):
    with model.generate(text, max_new_tokens=max_new_tokens) as tracer:
        with model.model.layers.all():
            activation = model.model.layers[layer_of_interest].output[0]
            
            # Skip intervention if magnitude is 0 (baseline)
            if magnitude == 0:
                pass  # Do nothing for baseline
            elif intervention_type == "backtracking":
                intervention = magnitude * intervention_vector["backtracking"]
                activation[:] += intervention.to(activation.device)
            elif intervention_type == "deduction":
                intervention = magnitude * intervention_vector["deduction"]
                activation[:] += intervention.to(activation.device)
            elif intervention_type == "initializing":
                intervention = magnitude * intervention_vector["initializing"]
                activation[:] += intervention.to(activation.device)
            elif intervention_type == "noise":
                noise = torch.randn_like(activation)
                noise = noise / noise.norm(dim=-1, keepdim=True) * magnitude
                activation[:] += noise
            elif intervention_type == "overall_mean":
                mean_vec = intervention_vector["overall_mean"] / intervention_vector["overall_mean"].norm()
                activation[:] += magnitude * mean_vec.to(activation.device)
            elif intervention_type == "self":
                # Normalize the activation itself and add it back
                norm_activation = activation / activation.norm(dim=-1, keepdim=True)
                activation[:] += magnitude * norm_activation
            
            out = model.generator.output.save()
    
    return model.tokenizer.decode(out[0])

def run_experiment():
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate steering vectors
    print("Generating steering vectors...")
    steering_vectors, overall_mean, model = generate_steering_vectors("annotated_chains/all_annotated_chains.json", layer_of_interest)
    
    # Prepare intervention vectors
    intervention_vectors = steering_vectors.copy()
    intervention_vectors["overall_mean"] = overall_mean
    
    # Load random prompts
    prompts = load_random_prompts("reasoning_chains/all_reasoning_chains.json", num_prompts)
    
    # Initialize results dictionary
    results = {intervention_type: {mag: [] for mag in intervention_magnitudes} for intervention_type in intervention_types}
    
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
        # We'll reuse this for all intervention types with magnitude 0
        if intervention_magnitudes[0] == 0:  # Make sure the first magnitude is 0
            baseline_text = apply_intervention(
                model,
                formatted_prompt,
                "baseline",  # Doesn't matter which type we use for baseline
                intervention_vectors,
                0,
                layer_of_interest
            )
            baseline_counts = count_patterns(baseline_text)
        
        # Apply each intervention type at each magnitude
        for intervention_type in intervention_types:
            for magnitude in intervention_magnitudes:
                if magnitude == 0:
                    # Use the baseline results for all intervention types at magnitude 0
                    results[intervention_type][magnitude].append(baseline_counts)
                else:
                    generated_text = apply_intervention(
                        model, 
                        formatted_prompt, 
                        intervention_type, 
                        intervention_vectors, 
                        magnitude, 
                        layer_of_interest
                    )
                    
                    # Count patterns
                    pattern_counts = count_patterns(generated_text)
                    results[intervention_type][magnitude].append(pattern_counts)
    
    # Average the results
    avg_results = {}
    patterns = ["Hmm", "Wait"]
    
    for intervention_type in intervention_types:
        avg_results[intervention_type] = {}
        
        for pattern in patterns:
            avg_results[intervention_type][pattern] = []
            
            for magnitude in intervention_magnitudes:
                # Calculate average percentage for this pattern at this magnitude
                pattern_percentages = [result[pattern] for result in results[intervention_type][magnitude]]
                avg_percentage = sum(pattern_percentages) / len(pattern_percentages)
                avg_results[intervention_type][pattern].append(avg_percentage)
    
    # Create a DataFrame from the results
    data = []
    for intervention_type in intervention_types:
        for i, magnitude in enumerate(intervention_magnitudes):
            hmm_pct = avg_results[intervention_type]["Hmm"][i]
            wait_pct = avg_results[intervention_type]["Wait"][i]
            combined_pct = hmm_pct + wait_pct
            
            data.append({
                "intervention_type": intervention_type,
                "magnitude": magnitude,
                "hmm_percentage": hmm_pct,
                "wait_percentage": wait_pct,
                "combined_percentage": combined_pct
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    results_df.to_csv("intervention_results.csv", index=False)
    print(f"Results saved to intervention_results.csv")
    
    # Plot the results
    plot_results(avg_results, patterns, intervention_types, intervention_magnitudes)
    
    return avg_results, results_df

def plot_results(avg_results, patterns, intervention_types, intervention_magnitudes):
    # Create separate plots for each pattern
    for pattern in patterns:
        plt.figure(figsize=(12, 8))
        
        for intervention_type in intervention_types:
            plt.plot(
                intervention_magnitudes, 
                avg_results[intervention_type][pattern], 
                marker='o', 
                linewidth=2, 
                label=intervention_type
            )
        
        plt.xlabel('Intervention Magnitude')
        plt.ylabel(f'Average {pattern} Token %')
        plt.title(f'Effect of Interventions on {pattern} Token Percentage')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{pattern}_intervention_results.png')
        plt.show()
    
    # Create a combined plot with the sum of both patterns
    plt.figure(figsize=(15, 10))
    
    for intervention_type in intervention_types:
        # Calculate the sum of both pattern percentages for each magnitude
        combined_percentages = []
        for i in range(len(intervention_magnitudes)):
            combined = avg_results[intervention_type]["Hmm"][i] + avg_results[intervention_type]["Wait"][i]
            combined_percentages.append(combined)
        
        # Plot the combined line
        plt.plot(
            intervention_magnitudes, 
            combined_percentages, 
            marker='o', 
            linewidth=2, 
            label=intervention_type
        )
    
    plt.xlabel('Intervention Magnitude')
    plt.ylabel('Average Combined "Hmm" + "Wait" Token %')
    plt.title('Effect of Interventions on Combined Hesitation Markers')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_intervention_results.png')
    plt.show()

if __name__ == "__main__":
    avg_results, results_df = run_experiment() 