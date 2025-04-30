# %%
import torch
import plotly.express as px
from sparseautoencoder import SparseAutoencoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import plotly.express as px
import pandas as pd
import numpy as np
from tqdm import tqdm
device = "cuda"

torch.manual_seed(42)
torch.set_grad_enabled(False)

import os
os.environ["HF_TOKEN"] = "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"

# %%
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map=device, torch_dtype=torch.bfloat16)

# %%
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)
dataset = dataset.shuffle(seed=42)

n_features = 65536
k = 100
layer_num = 4

sae = SparseAutoencoder(
    d_in=4096,
    n_features=n_features,
    k=k,
    device=device,
)

# %%
state_dict = torch.load("saes/r1-sae-layer5_65536_100-33000.pt")
sae.load_state_dict(state_dict)

# %%
sae

# %%
def get_token_iter(dataset, tokenizer, min_ctx_len=512):
    for example in dataset:
        prompt = example["messages"][0]
        prompt_formatted = tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
        response = example["messages"][1]["content"]
        prompt_formatted += response
        tokens = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)
        if tokens.input_ids.shape[1] < min_ctx_len:
            continue
        yield tokens.input_ids

my_data_generator = get_token_iter(dataset, tokenizer, min_ctx_len=512)

example = next(iter(my_data_generator))
print(tokenizer.decode(example[0]))

# %%
example


# %%
def track_feature_activations(
    model,
    sae: SparseAutoencoder,
    dataset,
    layer_num,
    batch_size=8,
    ctx_len=512,
    max_examples=100000,
    top_k=10
):
    print(f"Tracking activations for {max_examples} examples with batch size {batch_size}")
    print(f"Using {sae.n_features} features, keeping top {top_k} examples per feature")
    
    # Initialize feature data structures
    activation_counts = np.zeros(sae.n_features, dtype=int)

    # Initialize top examples storage
    top_examples = [[] for _ in range(sae.n_features)]
    
    # Process batches
    my_data_generator = get_token_iter(
        dataset=dataset,
        tokenizer=tokenizer,
        min_ctx_len=ctx_len,
    )
    
    pbar = tqdm(total=max_examples, desc="Processing examples")
    examples_processed = 0
    
    for batch in my_data_generator:
        batch = batch.to(device)
        
        # Get hidden states from both models
        out = model.forward(batch, output_hidden_states=True)
        
        hidden = out.hidden_states[layer_num][:, 1:]  # remove BOS
        flat_hidden = hidden.view(-1, hidden.shape[-1])

        flat_features = sae.encode(flat_hidden)
        
        # Reshape to match the batch shape
        features = flat_features.view(batch.shape[0], -1, flat_features.shape[-1])

        # Process each token's activations
        for i in range(batch.shape[0]):
            for j in range(1, batch.shape[1]):  # Start from 1 to skip BOS token
                examples_processed += 1
                pbar.update(1)

                # Get token and context
                token = tokenizer.decode(batch[i, j].item())
                context_start = max(0, j-7)
                context_end = min(batch.shape[1], j+5)
                context = tokenizer.decode(batch[i, context_start:context_end].tolist())
                
                # Get activated features for this token
                token_features = features[i, j-1].nonzero()
                
                # Handle case of no features
                if token_features.size(0) == 0:
                    continue
                
                # Extract the feature indices
                token_features = token_features.squeeze(-1)
                
                # Handle case where squeeze removes dimension for single feature
                if token_features.dim() == 0:
                    token_features = token_features.unsqueeze(0)
                
                # Update activation counts
                feat_indices = token_features.cpu().numpy()
                activation_counts[feat_indices] += 1
                
                # Update examples for each activated feature
                for feat_id in token_features.cpu():
                    feat_id_int = feat_id.item()
                    
                    activation_value = features[i, j-1, feat_id_int].item()
                    
                    example = {
                        'token': token,
                        'context': context,
                        'activation': activation_value
                    }
                    
                    # Just append the example (no sorting by activation strength)
                    top_examples[feat_id_int].append(example)
                    # Keep only top k examples
                    if len(top_examples[feat_id_int]) > top_k:
                        # Sort by activation value and keep top k
                        top_examples[feat_id_int] = sorted(top_examples[feat_id_int], 
                                                         key=lambda x: x['activation'], 
                                                         reverse=True)[:top_k]
                
                if examples_processed >= max_examples:
                    break
            
            if examples_processed >= max_examples:
                break
        
        if examples_processed >= max_examples:
            break
    
    pbar.close()
    
    # Organize examples for each feature
    print("Preparing feature dataframe...")
    all_top_examples = []
    
    for feat_id in tqdm(range(sae.n_features), desc="Processing features"):
        # Make sure they're sorted by activation
        sorted_examples = sorted(top_examples[feat_id], 
                               key=lambda x: x['activation'], 
                               reverse=True)
        all_top_examples.append(sorted_examples)
    
    # Create the dataframe
    feature_df = pd.DataFrame({
        'feature_id': range(sae.n_features),
        'activation_count': activation_counts,
        'top_examples': all_top_examples
    })
    
    return feature_df


# %%
# Run feature tracking
feature_df = track_feature_activations(
    model,
    sae,
    dataset,
    layer_num=4,
    batch_size=8,
    max_examples=100000
)

# %%
# Basic statistics
print(f"Total features: {len(feature_df)}")
print(f"Features with activations: {(feature_df['activation_count'] > 0).sum()}")
print(f"Total activations recorded: {feature_df['activation_count'].sum()}")

# %%
# Save dataframe
feature_df.to_pickle("feature_activations.pkl")
print("\nDataframe saved to feature_activations.pkl")

# %%
# Additional feature analyses
def analyze_feature_distribution(feature_df, feature_idx, min_activations=10, total_tokens=10000):
    # Filter features that were activated at least min_activations times
    print(f"Total tokens processed: {total_tokens}")

    row = feature_df[feature_df['feature_id'] == feature_idx].iloc[0]
    pct_active = (row['activation_count'] / total_tokens) * 100
    print(f"Activated in {pct_active:.2f}% of tokens")
    
    # Create tabular display for examples
    examples_data = []
    for example in row['top_examples'][:10]:  # Show top 10 examples
        examples_data.append({
            'Token': example['token'],
            'Activation': f"{example['activation']:.4f}",
            'Context': example['context']
        })
    examples_df = pd.DataFrame(examples_data)
    display(examples_df)
    return row

# %%
feature_idx = 100
# %%
# Run additional analyses
feature_idx += 1
row = analyze_feature_distribution(feature_df, feature_idx=feature_idx, min_activations=10, total_tokens=10000)
row

# %%