# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the intervention results
df = pd.read_csv('intervention_results.csv')

# Get unique intervention types and magnitudes
intervention_types = df['intervention_type'].unique()
intervention_magnitudes = sorted(df['magnitude'].unique())

# Organize data in the same structure as run_experiment.py
avg_results = {}
for intervention_type in intervention_types:
    avg_results[intervention_type] = {
        "Hmm": [],
        "Wait": []
    }
    for magnitude in intervention_magnitudes:
        row = df[(df['intervention_type'] == intervention_type) & (df['magnitude'] == magnitude)]
        if not row.empty:
            avg_results[intervention_type]["Hmm"].append(row['hmm_percentage'].values[0])
            avg_results[intervention_type]["Wait"].append(row['wait_percentage'].values[0])

# Define patterns (same as in run_experiment.py)
patterns = ["Hmm", "Wait"]

def plot_results(avg_results, patterns, intervention_types, intervention_magnitudes):
    # Create separate plots for each pattern
    for pattern in patterns:
        plt.figure(dpi=200)
        
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
        # plt.savefig(f'{pattern}_intervention_results.png')
        plt.savefig('fig4_baseline.pdf', format='pdf', bbox_inches='tight')
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

# Plot the results
plot_results(avg_results, patterns, intervention_types, intervention_magnitudes)
# %%
