# %%
import torch
from nnsight import LanguageModel
from tqdm import tqdm
import os

torch.set_grad_enabled(False)

os.environ["HF_TOKEN"] = "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"

model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)

with model.trace("Hello!") as tracer:
    out = model.output.save()

# %%

unembed = model.lm_head.weight
unembed.shape
# %%
import json

def load_chain(file_path):
    """Load annotated chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)
    
chains = load_chain("reasoning_chains/all_reasoning_chains.json")
chains

def format_chain(chain):
    prompt = chain["problem"]
    response = chain["reasoning_chain"]
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return model.tokenizer.apply_chat_template(messages, tokenize=False)

formatted_chains = [format_chain(chain) for chain in chains]
formatted_chains[0]
    
# %%
activations = []
logits = []
for chain in tqdm(formatted_chains[:100]):
    with model.trace(chain) as tracer:
        # acts = model.lm_head.input.save()
        acts = model.model.layers[10].output[0].save()
        out = model.output[0].save()
    activations.append(acts.cpu())
    logits.append(out.cpu())

activations = torch.cat(activations, dim=1)
logits = torch.cat(logits, dim=1)

print(activations.shape)
print(logits.shape)

# %%
probs = torch.nn.functional.softmax(logits, dim=-1)
entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
print(entropy.shape)
# %%
entropy = entropy.squeeze(0)
activations = activations.squeeze(0)
del logits, probs

# %%

X = activations.cuda().float()
y = entropy.cuda().float()

# X: [N, d_model], y: [N]
X_mean = X.mean(dim=0, keepdim=True)
y_mean = y.mean()

X_centered = X - X_mean
y_centered = y - y_mean

# Compute the least squares solution
# direction: [d_model]
direction = torch.linalg.lstsq(X_centered, y_centered.unsqueeze(1)).solution.squeeze()


# %%
import einops

random_dir = torch.randn(direction.shape).cuda().float()

logit_shift = einops.einsum(direction, unembed.float(), "d_model, d_vocab d_model -> d_vocab")
random_shift = einops.einsum(random_dir, unembed.float(), "d_model, d_vocab d_model -> d_vocab")

print(logit_shift.abs().mean())
print(random_shift.abs().mean())

# %%
top_logits = logit_shift.topk(100, dim=-1).indices
logit_shift[top_logits]
# %%
for token in top_logits:
    print(model.tokenizer.decode(token))
# %%
torch.save(direction.cpu().float(), "l10_entropy_direction.pt")
# %%
