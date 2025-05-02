# %%
from nnsight import LanguageModel
import torch

device = "cuda"

model = LanguageModel(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map=device,
    torch_dtype=torch.bfloat16
)

# %%
steering_vectors = torch.load("ft_steering_vectors.pt")
for key, vector in steering_vectors.items():
    steering_vectors[key] = vector / vector.norm()
    print(steering_vectors[key].shape)
    print(steering_vectors[key].norm())
# steering_vector = steering_vectors["backtracking"].to(device).to(torch.bfloat16)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

prompt = "What is 32^2?"
strengths = [0, 5, 10, 15, 20, 25, 30]

def test_confidence(prompt, my_steering_vector, use_noise=False):
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    # print(formatted_messages)
    input_ids = tokenizer.encode(formatted_messages, add_special_tokens=False, return_tensors="pt").to(device)

    with model.generate(input_ids, max_new_tokens=512) as gen:
        out = model.generator.output.save()


    output_text = tokenizer.decode(out[0])
    # print(output_text)

    if "</think>" not in output_text:
        print("No thinking text found")
        return None

    thinking_text = output_text.split("</think>")[0]
    thinking_text += "\nI should estimate my confidence in this answer. My confidence level is"
    # print(thinking_text)

    logit_diffs = []

    for strength in strengths:
        with model.trace(thinking_text) as tracer:
            if use_noise:
                with model.model.layers.all():
                    acts = model.model.layers[10].output[0][:]
                    noise = torch.randn_like(acts)
                    noise_normed = noise / noise.norm(dim=-1, keepdim=True)
                    model.model.layers[10].output[0][:] += strength * noise_normed
                logits = model.output[0].save()
            else:
                with model.model.layers.all():
                    model.model.layers[10].output[0][:] += strength * my_steering_vector
                logits = model.output[0].save()

        logits[0, -1, :].shape
        high_logit = logits[0, -1, [tokenizer.encode(' high', add_special_tokens=False)[0]]].item()
        low_logit = logits[0, -1, [tokenizer.encode(' low', add_special_tokens=False)[0]]].item()
        logit_diff = high_logit - low_logit
        logit_diffs.append(logit_diff)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        # print(f"Strength: {strength}")
        # print(f"Logit diff: {high_logit - low_logit}")
        # for i in range(5):
        #     print(f"{repr(tokenizer.decode(top_indices[i]))}: {top_probs[i]:.2%}")
        # print()
    return logit_diffs

# test_confidence(prompt)
# %%
prompts = [
    "Is 111 a prime number?",
    "What is 32^2?",
    "What is the tallest building in NYC?",
    "Jack is two years older than Jill's twin sister, Sarah. Sarah is 21 years old. How old is Jack?",
    "What is the square root of 144?",
    "If a triangle has angles of 30, 60, and 90 degrees, what is the sum of its angles?",
    "What is 15% of 80?",
    "If x + y = 10 and x - y = 4, what is the value of x?",
    "How many sides does a hexagon have?",
    "What is the next number in the sequence: 2, 4, 8, 16, ...?",
    "If a car travels at 60 miles per hour, how far will it travel in 2.5 hours?"
]

category_logit_diffs = []
for steering_category in steering_vectors.keys():
    all_logit_diffs = []
    for prompt in prompts:
        logit_diffs = test_confidence(prompt, steering_vectors[steering_category])
        all_logit_diffs.append(logit_diffs)
    category_logit_diffs.append(all_logit_diffs)
# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

categories = list(steering_vectors.keys())
fig = make_subplots(rows=2, cols=3, subplot_titles=categories)

for idx, (steering_category, all_logit_diffs) in enumerate(zip(steering_vectors.keys(), category_logit_diffs)):
    row = idx // 3 + 1
    col = idx % 3 + 1
    for i, logit_diff in enumerate(all_logit_diffs):
        if logit_diff is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=strengths,
                y=logit_diff,
                mode='lines',
                name=prompts[i],
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

fig.update_xaxes(title_text='Steering Strength')
# apply y-axis title only to left column subplots
for row in [1, 2]:
    fig.update_yaxes(title_text='Logit Difference (high - low)', range=[0, 12], row=row, col=1)
# apply y-axis range to other subplots
for row in [1, 2]:
    for col in [1, 2, 3]:
        fig.update_yaxes(range=[-2, 14], row=row, col=col)
# move legend below plots
fig.update_layout(
    height=800,
    width=1000,
    title_text='Effect of Steering Strength on Confidence Estimate',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.6,
        xanchor='center',
        x=0.3
    )
)
fig.show()

# %%
noise_logit_diffs = []
for prompt in prompts:
    logit_diffs = test_confidence(prompt, None, use_noise=True)
    noise_logit_diffs.append(logit_diffs)

# %%
import plotly.express as px
import pandas as pd

# Prepare noise baseline DataFrame
df_noise = pd.DataFrame()
for i, logit_diff in enumerate(noise_logit_diffs):
    if logit_diff is None:
        continue
    temp = pd.DataFrame({
        'Steering Strength': strengths,
        'Logit Diff': logit_diff,
        'Prompt': [prompts[i]] * len(strengths)
    })
    df_noise = pd.concat([df_noise, temp], ignore_index=True)

# Plot noise baseline
fig_noise = px.line(
    df_noise,
    x='Steering Strength',
    y='Logit Diff',
    color='Prompt',
    title='Noise Baseline: Effect of Noise on Confidence Estimate'
)
fig_noise.update_layout(
    xaxis_title='Steering Strength',
    yaxis_title='Logit Difference (high - low)',
    # legend=dict(orientation='h', yanchor='bottom', y=-0.6, xanchor='center', x=0.5)
)
fig_noise.update_yaxes(range=[-2, 14])
fig_noise.show()

# %%
