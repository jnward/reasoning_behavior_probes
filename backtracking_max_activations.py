# %%
import torch
from nnsight import LanguageModel
import os
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
import textwrap

# Disable gradients for inference
torch.set_grad_enabled(False)

# %%
# Configuration
layer_of_interest = 10
num_tokens_to_process = 100000
max_seq_length = 512
batch_size = 4
top_k = 50  # Number of top activating examples to keep

# %%
# Load the Llama R1 model
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer directly
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
# Load steering vectors
print("Loading steering vectors...")
base_steering_vectors = torch.load("base_steering_vectors.pt")

# Get the backtracking vector and move to device with correct dtype
backtracking_vector = base_steering_vectors["backtracking"].to(base_model.device, dtype=torch.bfloat16)
print(f"Backtracking vector shape: {backtracking_vector.shape}")

# %%
# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",  # Use v1 config
    split="train",
    streaming=True,
)

# %%
class TopKTracker:
    """Efficiently track top-k activating examples"""
    def __init__(self, k=50):
        self.k = k
        self.top_values = []
        self.top_indices = []
        self.top_contexts = []
        self.min_value = float('-inf')
    
    def update(self, values, indices, contexts):
        """Update with new values, indices, and contexts"""
        # Combine new values with existing top values
        all_values = self.top_values + list(values)
        all_indices = self.top_indices + list(indices)
        all_contexts = self.top_contexts + list(contexts)
        
        # Sort by value (descending)
        sorted_indices = sorted(range(len(all_values)), 
                               key=lambda i: all_values[i], 
                               reverse=True)
        
        # Keep only top k
        self.top_values = [all_values[i] for i in sorted_indices[:self.k]]
        self.top_indices = [all_indices[i] for i in sorted_indices[:self.k]]
        self.top_contexts = [all_contexts[i] for i in sorted_indices[:self.k]]
        
        # Update minimum value threshold
        if len(self.top_values) >= self.k:
            self.min_value = self.top_values[-1]

# %%
# Initialize trackers for positive and negative activations
positive_tracker = TopKTracker(k=top_k)
negative_tracker = TopKTracker(k=top_k)

# Track activation statistics
activation_stats = {
    'mean': 0,
    'std': 0,
    'count': 0,
    'sum': 0,
    'sum_sq': 0
}

# %%
def process_batch(input_ids, attention_mask):
    """Process a batch and extract activations"""
    batch_size, seq_len = input_ids.shape
    
    # Storage for activations
    activations = None
    
    # Create a hook to capture activations
    def hook_fn(module, input):
        nonlocal activations
        activations = input[0].detach().clone()
    
    # Register hook
    hook_handle = base_model.model.layers[layer_of_interest].register_forward_pre_hook(hook_fn)
    
    # Run forward pass
    with torch.no_grad():
        _ = base_model.model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Remove hook
    hook_handle.remove()
    
    # Get the saved activations
    act_values = activations
    
    # Compute mean activation across all positions (for centering)
    # Shape: [batch_size, seq_len, hidden_dim]
    global_mean = act_values.mean(dim=(0, 1), keepdim=True)  # [1, 1, hidden_dim]
    
    # Center the activations
    centered_acts = act_values - global_mean
    
    # Project onto backtracking vector
    # backtracking_vector shape: [hidden_dim]
    # centered_acts shape: [batch_size, seq_len, hidden_dim]
    projections = torch.matmul(centered_acts, backtracking_vector)  # [batch_size, seq_len]
    
    # Extract top activating positions
    for b in range(batch_size):
        for pos in range(seq_len):
            if attention_mask[b, pos] == 0:
                continue
                
            value = projections[b, pos].item()
            
            # Update statistics
            activation_stats['count'] += 1
            activation_stats['sum'] += value
            activation_stats['sum_sq'] += value ** 2
            
            # Get context window around this position
            context_start = max(0, pos - 20)
            context_end = min(seq_len, pos + 20)
            
            # Decode the context
            context_tokens = input_ids[b, context_start:context_end]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
            
            # Get the specific token
            token = tokenizer.decode(input_ids[b, pos], skip_special_tokens=False)
            
            # Create context info
            context_info = {
                'text': context_text,
                'token': token,
                'position': pos - context_start,  # Position within context
                'value': value,
                'full_sequence': tokenizer.decode(input_ids[b], skip_special_tokens=False)
            }
            
            # Update trackers
            if value > 0 and value > positive_tracker.min_value:
                positive_tracker.update([value], [(b, pos)], [context_info])
            elif value < 0 and abs(value) > abs(negative_tracker.min_value):
                negative_tracker.update([abs(value)], [(b, pos)], [context_info])

# %%
# Process dataset
tokens_processed = 0
batch_texts = []

print(f"Processing {num_tokens_to_process} tokens...")
pbar = tqdm(total=num_tokens_to_process)

for example in dataset:
    # Get the messages and apply chat template
    messages = example.get('messages', [])
    if not messages:
        continue
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    batch_texts.append(text)
    
    # Process batch when full
    if len(batch_texts) >= batch_size:
        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(base_model.device)
        attention_mask = encoding['attention_mask'].to(base_model.device)
        
        # Process batch
        process_batch(input_ids, attention_mask)
        
        # Update token count
        batch_tokens = attention_mask.sum().item()
        tokens_processed += batch_tokens
        pbar.update(batch_tokens)
        
        # Clear batch
        batch_texts = []
        
        # Check if we've processed enough tokens
        if tokens_processed >= num_tokens_to_process:
            break

# Process remaining batch
if batch_texts:
    encoding = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(base_model.device)
    attention_mask = encoding['attention_mask'].to(base_model.device)
    
    process_batch(input_ids, attention_mask)

pbar.close()

# %%
# Compute final statistics
activation_stats['mean'] = activation_stats['sum'] / activation_stats['count']
activation_stats['std'] = np.sqrt(
    activation_stats['sum_sq'] / activation_stats['count'] - activation_stats['mean'] ** 2
)

print(f"\nActivation Statistics:")
print(f"Mean: {activation_stats['mean']:.4f}")
print(f"Std: {activation_stats['std']:.4f}")
print(f"Total tokens processed: {tokens_processed}")

# %%
# Create visualization of top activating examples
def create_html_visualization(tracker, title, is_positive=True):
    """Create HTML visualization of top activating examples"""
    html_parts = [f"<h2>{title}</h2>"]
    
    for i, (value, idx, context) in enumerate(zip(tracker.top_values[:20], 
                                                  tracker.top_indices[:20], 
                                                  tracker.top_contexts[:20])):
        actual_value = value if is_positive else -value
        html_parts.append(f"<div style='margin: 20px 0; padding: 10px; border: 1px solid #ccc;'>")
        html_parts.append(f"<h3>Rank {i+1}: Activation = {actual_value:.4f}</h3>")
        
        # Show the specific token
        html_parts.append(f"<p><strong>Token:</strong> <code>{context['token']}</code></p>")
        
        # Show context with highlighting
        text = context['text']
        position = context['position']
        
        # Split text into tokens for highlighting
        tokens = tokenizer.tokenize(text)
        decoded_tokens = [tokenizer.convert_tokens_to_string([t]) for t in tokens]
        
        # Create highlighted version
        highlighted_text = ""
        current_pos = 0
        for j, token_text in enumerate(decoded_tokens):
            if j == position:
                highlighted_text += f"<span style='background-color: yellow; font-weight: bold;'>{token_text}</span>"
            else:
                highlighted_text += token_text
        
        html_parts.append(f"<p><strong>Context:</strong></p>")
        html_parts.append(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{highlighted_text}</pre>")
        html_parts.append("</div>")
    
    return "\n".join(html_parts)

# %%
# Display results
positive_html = create_html_visualization(positive_tracker, "Top Positive Activations", is_positive=True)
negative_html = create_html_visualization(negative_tracker, "Top Negative Activations", is_positive=False)

display(HTML(positive_html))
display(HTML(negative_html))

# %%
# Plot activation distribution
plt.figure(figsize=(10, 6))

# Collect all activation values for histogram
all_positive_values = positive_tracker.top_values
all_negative_values = [-v for v in negative_tracker.top_values]

plt.hist(all_positive_values, bins=30, alpha=0.5, label='Positive', color='blue')
plt.hist(all_negative_values, bins=30, alpha=0.5, label='Negative', color='red')

plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Activation Value')
plt.ylabel('Count')
plt.title(f'Distribution of Top {top_k} Activations for Backtracking Vector')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Save results
results = {
    'positive_activations': {
        'values': positive_tracker.top_values,
        'contexts': positive_tracker.top_contexts
    },
    'negative_activations': {
        'values': negative_tracker.top_values,
        'contexts': negative_tracker.top_contexts
    },
    'statistics': activation_stats,
    'config': {
        'layer': layer_of_interest,
        'tokens_processed': tokens_processed,
        'top_k': top_k
    }
}

torch.save(results, 'backtracking_max_activations.pt')
print("\nResults saved to backtracking_max_activations.pt")

# %%