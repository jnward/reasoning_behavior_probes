#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p steering_vector_plots

# Loop through all layers from 0 to 31
for layer in {0..31}; do
    echo "Processing layer $layer..."
    
    # Run the Python script for the current layer
    python steering_vector_analysis.py $layer
    
    # Clear CUDA cache between runs (helps prevent memory issues)
    python -c "import torch; torch.cuda.empty_cache()"
    
    echo "Completed layer $layer"
    echo "-----------------------------------"
done

echo "All layers processed successfully!" 