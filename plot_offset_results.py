import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_offset_results(csv_file, intervention_type="backtracking"):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Get unique offsets and magnitudes
    offsets = sorted(df['offset'].unique())
    magnitudes = sorted(df['magnitude'].unique())
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # List of patterns to plot
    patterns = ['hmm_percentage', 'wait_percentage', 'combined_percentage']
    pattern_names = {'hmm_percentage': 'Hmm', 'wait_percentage': 'Wait', 'combined_percentage': 'Combined Hmm+Wait'}
    
    # Create a plot for each pattern
    for pattern in patterns:
        plt.figure(figsize=(12, 8))
        
        # Plot each magnitude as a separate line
        for magnitude in magnitudes:
            magnitude_data = df[df['magnitude'] == magnitude]
            # Sort by offset to ensure correct line plotting
            magnitude_data = magnitude_data.sort_values('offset')
            
            plt.plot(
                magnitude_data['offset'],
                magnitude_data[pattern],
                marker='o',
                linewidth=2,
                label=f"Magnitude {magnitude}"
            )
        
        # Set x-ticks to show all offsets
        plt.xticks(offsets)
        
        # Add vertical line at offset=0 for reference
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Offset (tokens)')
        plt.ylabel(f'Average {pattern_names[pattern]} Token %')
        plt.title(f'Effect of Offset on {pattern_names[pattern]} Token % ({intervention_type})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'plots/magnitude_lines_{pattern_names[pattern].lower()}_{intervention_type}.png')
        plt.show()
    
    # Create a 3D surface plot for combined percentage
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(offsets, magnitudes)
    Z = np.zeros_like(X, dtype=float)
    
    # Fill in Z values
    for i, magnitude in enumerate(magnitudes):
        for j, offset in enumerate(offsets):
            row = df[(df['magnitude'] == magnitude) & (df['offset'] == offset)]
            if not row.empty:
                Z[i, j] = row['combined_percentage'].values[0]
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Combined Hmm+Wait %')
    
    # Set labels
    ax.set_xlabel('Offset')
    ax.set_ylabel('Magnitude')
    ax.set_zlabel('Combined Hmm+Wait %')
    ax.set_title(f'3D Surface: Effect of Offset and Magnitude on Hesitation Markers ({intervention_type})')
    
    # Save the 3D plot
    plt.savefig(f'plots/3d_surface_combined_{intervention_type}.png')
    plt.show()
    
    # Create a heatmap for combined percentage
    plt.figure(figsize=(12, 8))
    
    # Pivot the data for heatmap
    heatmap_data = df.pivot(index='magnitude', columns='offset', values='combined_percentage')
    
    # Create the heatmap
    im = plt.imshow(heatmap_data, cmap='viridis')
    
    # Set up axes
    plt.colorbar(im, label='Combined Hmm+Wait %')
    plt.xticks(np.arange(len(offsets)), offsets)
    plt.yticks(np.arange(len(magnitudes)), magnitudes)
    plt.xlabel('Offset')
    plt.ylabel('Magnitude')
    plt.title(f'Heatmap: Effect of Offset and Magnitude on Hesitation Markers ({intervention_type})')
    
    # Add text annotations in the heatmap cells
    for i in range(len(magnitudes)):
        for j in range(len(offsets)):
            text = plt.text(j, i, f"{heatmap_data.iloc[i, j]:.1f}",
                          ha="center", va="center", color="w" if heatmap_data.iloc[i, j] > heatmap_data.values.mean() else "black")
    
    plt.tight_layout()
    plt.savefig(f'plots/heatmap_combined_{intervention_type}.png')
    plt.show()

if __name__ == "__main__":
    # Default CSV file path (assumes the script is run from the same directory)
    csv_file = "offset_intervention_results_backtracking.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        print("Please provide the correct path to the CSV file.")
        csv_file = input("Enter the path to the CSV file: ")
    
    # Extract intervention type from filename
    intervention_type = csv_file.split('_')[-1].split('.')[0]
    
    # Plot the results
    plot_offset_results(csv_file, intervention_type) 