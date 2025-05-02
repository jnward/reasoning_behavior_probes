# %%
# from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

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

sae_layer_input = 21

sae = SparseAutoencoder(
    d_in=4096,
    n_features=65536,
    k=100,
    device=device
)

# state_dict = torch.load("saes/r1-sae-layer5_65536_100_multi-21000.pt", map_location=device)
# state_dict = torch.load("saes/r1-sae-layer7_65536_100_multi-39000.pt", map_location=device)
# state_dict = torch.load("saes/r1-sae-layer11_65536_100_multi-48827.pt", map_location=device)
state_dict = torch.load("saes/r1-sae-layer21_65536_100_multi-24000.pt", map_location=device)
sae.load_state_dict(state_dict)

# %%
from datasets import load_dataset
import random

dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)

test_example_idx = 0
test_output = ""

while "Wait" not in test_output:
    test_example_idx = random.randint(0, len(dataset) - 1)
    example = dataset[test_example_idx]
    print(example["messages"][0]["content"])

    # prompt = "What is the third tallest building in NYC?"
    prompt = example["messages"][0]["content"]
    messages = [
        {"role": "user", "content": prompt}
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    print(tokenizer.decode(tokens[0]))
    output = model.generate(
        tokens,
        max_new_tokens=256,
        # do_sample=False,
        # temperature=None,
        # top_p=None
    )
    print(output.shape)
    print(tokenizer.decode(output[0]))
    test_output = tokenizer.decode(output[0])

print(f"Found Wait in example {test_example_idx}")
input_ids = output
# %%

# formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# formatted_response = tokenizer.encode(example["messages"][1]["content"], add_special_tokens=False)
# formatted_prompt += tokenizer.decode(formatted_response)
# print(formatted_prompt)

# input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
# print(input_ids.shape)

# with model.generate(formatted_prompt, max_new_tokens=256, do_sample=False, temperature=None, top_p=None) as gen:
#     out = model.generator.output.save()

# generated = model.tokenizer.decode(out[0])
# print(generated)

# input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)

out = model(input_ids, output_hidden_states=True)
resid = out.hidden_states[sae_layer_input]

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
# find Wait index
wait_token_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
wait_pos = (input_ids[0] == wait_token_id).nonzero()[0][0].item()

for i, token in enumerate(input_ids[0]):
    print(f"{i}: {repr(tokenizer.decode(token))}")

print(wait_pos)
# %%
torch.nn.functional.mse_loss(reconstructions, target)


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
output_sae_layer_input = out.hidden_states[sae_layer_input].float()
features = sae.encode(output_sae_layer_input[:, 1:, :])
features.requires_grad_()
features.retain_grad()
reconstructions = sae.decode(features)
input_activations = torch.cat([output_sae_layer_input[:, :1], reconstructions.to(dtype)], dim=1)

error = output_sae_layer_input - input_activations
error_coefficient = torch.ones(error.shape[:2], device=device).unsqueeze(-1)
# error_coefficient = torch.zeros(error.shape[:2], device=device).unsqueeze(-1)
error_coefficient.requires_grad_()
error_coefficient.retain_grad()
perturbed_acts = input_activations + error_coefficient * error.detach()

perturbed_acts = perturbed_acts.to(dtype)
# layer_10_output = forward_partial(perturbed_acts, sae_layer_input, 10)  # input to layer 5, get output of 10
# assert torch.allclose(layer_10_output, out.hidden_states[11])


layer_31_output = forward_partial(perturbed_acts, sae_layer_input, 31)
post_norm = model.model.norm(layer_31_output)
assert torch.allclose(post_norm, out.hidden_states[32])

logits = model.lm_head(post_norm)
logits.shape

# %%
# import plotly.express as px
# projection = layer_10_output[0] @ steering_vector
# px.line(projection.cpu().detach().float().numpy())

# %%
# metric_type = "steering_vec"
metric_type = "Wait-I"
# metric_type = "KL"
if metric_type == "steering_vec":
    # compute metric and backprop
    offset = 12
    offset = 24
    # window_size = 4
    window_start = wait_pos - offset - 1
    # window_end = window_start + window_size + 1
    window_end = wait_pos


    metric = layer_10_output[0, window_start:window_end] @ steering_vector
    metric = metric.sum()
elif metric_type == "Wait-I":
    before_wait_logits = logits[:, wait_pos-1]
    topk_tokens = torch.topk(before_wait_logits, k=10).indices
    print(topk_tokens)
    for token in topk_tokens[0]:
        print(repr(tokenizer.decode(token)))
    print(before_wait_logits.shape)
    wait_logit = before_wait_logits[0, tokenizer.encode("Wait", add_special_tokens=False)[0]]
    # other_token = "I"
    other_token = "So"
    i_logit = before_wait_logits[0, tokenizer.encode(other_token, add_special_tokens=False)[0]]
    metric = wait_logit - i_logit
elif metric_type == "KL":
    before_wait_logits = logits[:, wait_pos-1]
    probs = torch.nn.functional.log_softmax(before_wait_logits, dim=-1)
    print(probs.shape)
    probs_detached = probs.clone().detach()
    kl_div = torch.nn.functional.kl_div(probs, probs_detached, reduction="batchmean", log_target=True)
    metric = kl_div
else:
    raise ValueError(f"Invalid metric type: {metric_type}")


print(metric)
print(metric.shape)


# %%
metric.backward()

# inspect gradients
print(features.grad)

# %%
import plotly.express as px
import plotly.graph_objects as go

attribution = features.grad * features
# prepend zero to attribution (has shape [1, 317, 65536])
attribution = torch.cat([torch.zeros(1, 1, attribution.shape[2], device=attribution.device), attribution], dim=1)

print(attribution.shape)
print(error_coefficient.grad.shape)
print(f"Wait position: {wait_pos}")

# %%

# Create the first plot
if metric_type == "steering_vec":
    title = "Attribution to Steering Vector"
elif metric_type == "Wait-I":
    title = f"Attribution to Logit Difference between \"Wait\" and \"{other_token}\""
elif metric_type == "KL":
    title = "Attribution to KL at token before Wait"
else:
    raise ValueError(f"Invalid metric type: {metric_type}")



fig = px.line(error_coefficient.grad.cpu().detach().float().numpy()[0], 
             title=title)
fig.data[0].line.color = 'red'
fig.data[0].name = 'Error Coefficient Gradient'

# Add the second line
positive_attribution = attribution.clamp(min=0).sum(dim=-1)


fig.add_scatter(y=positive_attribution.cpu().detach().float().numpy()[0], 
               mode='lines', 
               name='Positive Feature Attribution Sum',
               line=dict(color='blue'))

if metric_type == "steering_vec":
    # Highlight the window region
    fig.add_shape(
        type="rect",
        x0=window_start,
        x1=window_end,
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="green",
        opacity=0.4,
        layer="below",
        line_width=0,
    )

    # Add a label for the metric window
    fig.add_annotation(
        x=(window_start + window_end) / 2,
        y=1,
        yref="paper",
        text="Metric Window",
        showarrow=False,
        font=dict(color="black", size=12),
        opacity=0.8,
        textangle=15,
    )

# Add a vertical line at the wait position
fig.add_vline(
    x=wait_pos,
    line_dash="dash",
    line_color="green",
    opacity=0.7,
    annotation_text="Wait token",
    annotation_position="top right",
    annotation=dict(textangle=15),
)

fig.show()

# %%
print(wait_pos)
for i in range(attribution.shape[1]):
    pos_feats = attribution[0, i] > 0
    pos_feats = torch.where(pos_feats)[0].tolist()
    token_id = input_ids[0, i].item()
    print(f"{i}: {repr(tokenizer.decode(token_id))}\t {pos_feats}")
# %%

# %%
# Visualization of per-token error term attribution
from IPython.core.display import HTML
from html import escape
import numpy as np

# compute positive gradients for each token
error_grad = error_coefficient.grad.detach().cpu().numpy().squeeze(-1)[0]
pos_error_grad = np.clip(error_grad, a_min=0, a_max=None)
feature_grad = positive_attribution.detach().cpu().numpy()[0]
pos_feature_grad = np.clip(feature_grad, a_min=0, a_max=None)
max_error = pos_error_grad.max() if pos_error_grad.max() > 0 else 1.0
max_feature = pos_feature_grad.max() if pos_feature_grad.max() > 0 else 1.0
# decode tokens
tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
html_str = ''
for i, tok in enumerate(tokens):
    if i > 0:
        red = int(127 * pos_error_grad[i] / max_error)
        blue = int(127 * pos_feature_grad[i] / max_feature)
        style = f'background-color:rgb({255-blue},{255-(red+blue)},{255-red});'
    else:
        style = 'background-color:white;'
    if i == wait_pos:
        style += 'border:2px solid green;'
    escaped_tok = escape(tok)
    html_frag = re.sub(r'\n+', lambda m: '\\n' * len(m.group(0)) + '<br>' * len(m.group(0)), escaped_tok)
    html_str += f'<span style="{style}">{html_frag}</span>'
# display result in notebook
display(HTML(html_str))







# %%

decoded_output = tokenizer.decode(output[0])

new_out = model.generate(output, max_new_tokens=128)

print(tokenizer.decode(new_out[0]))
# %%
