# %%
# from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"
dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    torch_dtype=dtype,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
from sparseautoencoder import SparseAutoencoder

sae = SparseAutoencoder(
    d_in=4096,
    n_features=65536,
    k=100,
    device=device
)

state_dict = torch.load("saes/r1-sae-layer5_65536_100-33000.pt", map_location=device)
sae.load_state_dict(state_dict)

# %%
from datasets import load_dataset
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)
example = dataset[20000]
print(example) 

# %%
# prompt = "What is the third tallest building in NYC?"
# messages = [
#     {"role": "user", "content": prompt}
# ]
messages = [example["messages"][0]]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
formatted_response = tokenizer.encode(example["messages"][1]["content"], add_special_tokens=False)
formatted_prompt += tokenizer.decode(formatted_response)
print(formatted_prompt)

input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
print(input_ids.shape)

# %%
# with model.generate(formatted_prompt, max_new_tokens=256, do_sample=False, temperature=None, top_p=None) as gen:
#     out = model.generator.output.save()

# %%
# generated = model.tokenizer.decode(out[0])
# print(generated)


# %%
input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

out = model(input_ids, output_hidden_states=True)
resid = out.hidden_states[5]

print(resid.shape)

# %%
features = sae.encode(resid[0, 1:]).unsqueeze(0)
# reconstructions[:, 0] = resid[:, 0]
print(features.shape)
# %%
features = features.detach()
reconstructions = sae.decode(features)
# reconstructions = torch.cat([resid[:, :15], reconstructions], dim=1)
reconstructions.shape

# (reconstructions - resid).norm(dim=-1) / resid.norm(dim=-1)

target = resid[0, 1:]
reconstructions = reconstructions[0]
e = reconstructions - target
total_variance = (target - target.mean(0)).pow(2).sum()
squared_error = e.pow(2)
fvu = squared_error.sum() / total_variance
fvu

# %%
torch.nn.functional.mse_loss(reconstructions, target)





# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
del model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=dtype, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
# 1. run full model to get all hidden states
with torch.no_grad():
    outs = model(input_ids, output_hidden_states=True)

# 2. extract layer 5 activations
h4 = outs.hidden_states[5]               # shape [batch, seq_len, hidden]

# 3. compute position IDs / cache positions
seq_len = h4.shape[1]
cache_pos = torch.arange(seq_len, device=h4.device)
pos_ids   = cache_pos.unsqueeze(0)       # shape [1, seq_len]

# 4. get rotary embeddings for layer 5 inputs
rotary    = model.model.rotary_emb
cos_sin   = rotary(h4, pos_ids)          # tuple of (cos, sin)

# 5. call layer 6 properly
layer5    = model.model.layers[5]
out5 = layer5(
    h4,
    attention_mask=None,
    position_ids=pos_ids,
    cache_position=cache_pos,
    position_embeddings=cos_sin,
)[0]
# %%
true_out5 = model.forward(input_ids, output_hidden_states=True).hidden_states[6]
((out5 - true_out5).norm(dim=-1) / true_out5.norm(dim=-1)).mean()



# %%
steering_vectors = torch.load("ft_steering_vectors.pt")
steering_vector = steering_vectors["backtracking"].to(device).to(dtype)


# %%
def forward_partial(acts, start_layer, end_layer):
    for i in range(start_layer, end_layer + 1):
        layer = model.model.layers[i]
        seq_len = acts.shape[1]
        cache_pos = torch.arange(seq_len, device=acts.device)
        pos_ids = cache_pos.unsqueeze(0)
        cos_sin = model.model.rotary_emb(acts, pos_ids)
        acts = layer(
            acts,
            attention_mask=None,
            position_ids=pos_ids,
            cache_position=cache_pos,
            position_embeddings=cos_sin,
            use_cache=False
        )[0]
    return acts

# forward pass
out = model(input_ids, output_hidden_states=True)

# prepare features
input_4 = out.hidden_states[4]
features = sae.encode(input_4[:, 1:, :])
features.requires_grad_()
features.retain_grad()

# build inputs to layer 5
layer_4_acts = out.hidden_states[5]
reconstructions = sae.decode(features)
input_activations = torch.cat([layer_4_acts[:, :1], reconstructions.to(dtype)], dim=1)

# run through layers 5→10
layer_10_output = forward_partial(input_activations, 5, 10)

# compute metric and backprop
metric = layer_10_output[0, -1] @ steering_vector
print(metric)
print(metric.shape)
metric.backward()

# inspect gradients
print(features.grad)

# %%
attribution = features.grad * features
((attribution * features) > 0).sum()
# %%
attribution.shape
# %%
for i in range(attribution.shape[1]):
    pos_feats = attribution[0, i] > 0
    pos_feats = torch.where(pos_feats)[0].tolist()
    if pos_feats:
        print(i, pos_feats)
# %%
fftzf = sae.decoder.W_dec[55205]

torch.cosine_similarity(fftzf, steering_vector, dim=0)
# %%
