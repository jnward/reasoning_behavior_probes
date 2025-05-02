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

# state_dict = torch.load("saes/r1-sae-layer5_65536_100_multi-21000.pt", map_location=device)
state_dict = torch.load("saes/r1-sae-layer11_65536_100_multi-48827.pt", map_location=device)
sae.load_state_dict(state_dict)

# %%
steering_vectors = torch.load("ft_steering_vectors.pt")
steering_vector = steering_vectors["backtracking"].to(device).to(dtype)

decoder_weights = sae.decoder.W_dec.data
decoder_weights.shape

# %%
cosine_sims = torch.cosine_similarity(decoder_weights, steering_vector, dim=1)

# get top 100
top_features = torch.topk(cosine_sims, k=100).indices

for idx in top_features:
    print(f"{idx}: {cosine_sims[idx]:.4f}")






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
output_4 = out.hidden_states[5].float()
features = sae.encode(output_4[:, 1:, :])
features.requires_grad_()
features.retain_grad()
reconstructions = sae.decode(features)
input_activations = torch.cat([output_4[:, :1], reconstructions.to(dtype)], dim=1)

error = output_4 - input_activations
error_coefficient = torch.ones(error.shape[:2], device=device).unsqueeze(-1)
error_coefficient.requires_grad_()
error_coefficient.retain_grad()
perturbed_acts = input_activations + error_coefficient * error.detach()

perturbed_acts = perturbed_acts.to(dtype)
layer_10_output = forward_partial(perturbed_acts, 5, 10)  # input to layer 5, get output of 10
assert torch.allclose(layer_10_output, out.hidden_states[11])


layer_31_output = forward_partial(perturbed_acts, 5, 31)
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
# metric_type = "Wait-I"
metric_type = "KL"
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
    i_logit = before_wait_logits[0, tokenizer.encode("I", add_special_tokens=False)[0]]
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
print(f"Wait position: {wait_pos}")

# Create the first plot
if metric_type == "steering_vec":
    title = "Attribution to Steering Vector"
elif metric_type == "Wait-I":
    title = "Attribution to Logit Difference between \"Wait\" and \"I\""
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
    token_id = input_ids[0, i+1].item()
    print(f"{i+1}: {repr(tokenizer.decode(token_id))}\t {pos_feats}")
# %%












# %%
fftzf = sae.decoder.W_dec[55205]

torch.cosine_similarity(fftzf, steering_vector, dim=0)
# %%



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

# %%
