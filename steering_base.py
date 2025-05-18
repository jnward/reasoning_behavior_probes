# %%

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pandas as pd
from tqdm import tqdm
from generate_steering_vectors import generate_steering_vectors
import os

torch.set_grad_enabled(False)

os.environ["HF_TOKEN"] = "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"


# Configuration parameters
intervention_magnitudes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Added 0 for baseline
# intervention_magnitudes = [0, 8]
num_prompts = 3
layer_of_interest = 10
max_new_tokens = 128
seed = 43

def load_random_prompts(file_path, num_prompts):
    with open(file_path, "r") as f:
        all_prompts = json.load(f)
    
    return random.sample(all_prompts, num_prompts)

def count_patterns(text, patterns=["Wait"]):
    counts = {}
    total_tokens = len(text.split())
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        counts[pattern] = len(matches) / total_tokens * 100  # Convert to percentage
    
    return counts

def apply_intervention(model, text, intervention_vector, magnitude, layer_of_interest):
    with model.generate(text, max_new_tokens=max_new_tokens) as tracer:
        with model.model.layers.all():
            activation = model.model.layers[layer_of_interest].output[0]
            intervention = magnitude * intervention_vector
            activation[:] += intervention.to(activation.device)
            out = model.generator.output.save()
    
    return model.tokenizer.decode(out[0])

# %%
from nnsight import LanguageModel

# Load random prompts
prompts = load_random_prompts("reasoning_chains/all_reasoning_chains.json", num_prompts)

base_model = LanguageModel(
    "meta-llama/Meta-Llama-3.1-8B",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    )
ft_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    )

def convert_to_base_tokens(tokens: torch.Tensor):
    """
    Convert r1 tokens to base tokens. Only works for Llama tokenizers.
    """
    # patch_token = 77627 # ` ############`
    patch_token = 27370 # ` ####`
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens

tokenizer = ft_model.tokenizer

base_steering_vectors = torch.load("base_steering_vectors.pt")
ft_steering_vectors = torch.load("ft_steering_vectors.pt")

base_steering_vectors = {k: v / v.norm(dim=-1, keepdim=True) for k, v in base_steering_vectors.items()}
ft_steering_vectors = {k: v / v.norm(dim=-1, keepdim=True) for k, v in ft_steering_vectors.items()}

# Run the experiment
print("Running experiments...")
bb_results = [[] for _ in intervention_magnitudes]
bf_results = [[] for _ in intervention_magnitudes]
fb_results = [[] for _ in intervention_magnitudes]
ff_results = [[] for _ in intervention_magnitudes]

# %%
prompts = load_random_prompts("reasoning_chains/all_reasoning_chains.json", 100)[:3]

for i, prompt in enumerate(tqdm(prompts)):
    print(f"Processing prompt {i+1}/{num_prompts}")
    prompt_text = prompt["problem"]
    
    # Format with chat template
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ft_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt")[0]

    # Convert to base tokens
    base_tokens = convert_to_base_tokens(ft_tokens)

    for j, magnitude in enumerate(intervention_magnitudes):
        # model_steer
        base_base = apply_intervention(base_model, formatted_prompt, base_steering_vectors["backtracking"], magnitude, layer_of_interest)
        base_ft = apply_intervention(base_model, formatted_prompt, ft_steering_vectors["backtracking"], magnitude, layer_of_interest)
        ft_base = apply_intervention(ft_model, formatted_prompt, base_steering_vectors["backtracking"], magnitude, layer_of_interest)
        ft_ft = apply_intervention(ft_model, formatted_prompt, ft_steering_vectors["backtracking"], magnitude, layer_of_interest)

        # count patterns
        base_base_counts = count_patterns(base_base)['Wait']
        base_ft_counts = count_patterns(base_ft)['Wait']
        ft_base_counts = count_patterns(ft_base)['Wait']
        ft_ft_counts = count_patterns(ft_ft)['Wait']

        bb_results[j].append(base_base_counts)
        bf_results[j].append(base_ft_counts)
        fb_results[j].append(ft_base_counts)
        ff_results[j].append(ft_ft_counts)

# %%
# get means and stds
bb_results_np = np.array(bb_results)
bf_results_np = np.array(bf_results)
fb_results_np = np.array(fb_results)
ff_results_np = np.array(ff_results)

# %%

bb_results_mean = np.mean(bb_results_np, axis=1)
bb_results_std = np.std(bb_results_np, axis=1)
bf_results_mean = np.mean(bf_results_np, axis=1)
bf_results_std = np.std(bf_results_np, axis=1)
fb_results_mean = np.mean(fb_results_np, axis=1)
fb_results_std = np.std(fb_results_np, axis=1)
ff_results_mean = np.mean(ff_results_np, axis=1)
ff_results_std = np.std(ff_results_np, axis=1)

# %%
magnitudes = intervention_magnitudes
x = np.arange(len(magnitudes))
width = 0.2
fig, ax = plt.subplots(dpi=200)
ax.bar(x - 1.5*width, bb_results_mean, width, yerr=bb_results_std, label='Base model + base steering vector (always zero)')
ax.bar(x - 0.5*width, bf_results_mean, width, yerr=bf_results_std, label='Base model + reasoning steering vector (always zero)')
ax.bar(x + 0.5*width, fb_results_mean, width, yerr=fb_results_std, label='Reasoning model + base steering vector')
ax.bar(x + 1.5*width, ff_results_mean, width, yerr=ff_results_std, label='Reasoning model + reasoning steering vector')
ax.set_xticks(x)
ax.set_xticklabels(magnitudes)
ax.set_xlabel('Steering Magnitude')
ax.set_ylabel('"Wait" token proportion')
ax.set_title('Steering Effect Comparison')
ax.set_ylim(0, 75)
ax.legend()
plt.show()

# %%
