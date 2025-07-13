# %%
import torch
from nnsight import LanguageModel
import os
import textwrap

# Disable gradients for inference
torch.set_grad_enabled(False)

# %%
# Configuration
layer_of_interest = 10
intervention_magnitude = 6
max_new_tokens = 256

# %%
# Load the Llama R1 model
print("Loading model...")
model = LanguageModel(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# %%
# Load steering vectors
print("Loading steering vectors...")
base_steering_vectors = torch.load("base_steering_vectors.pt")

# Normalize steering vectors
base_steering_vectors = {k: v / v.norm(dim=-1, keepdim=True) for k, v in base_steering_vectors.items()}

# %%
# Define the intervention function
def apply_intervention(model, text, intervention_vector, magnitude, layer_of_interest):
    """Apply steering intervention during generation"""
    with model.generate(text, max_new_tokens=max_new_tokens) as tracer:
        with model.model.layers.all():
            activation = model.model.layers[layer_of_interest].output[0]
            intervention = magnitude * intervention_vector
            activation[:] += intervention.to(activation.device)
            out = model.generator.output.save()
    
    return model.tokenizer.decode(out[0])

# %%
# Hardcoded problem/prompt
problem = "How many edges does a 4-simplex have?"

# Format prompt with chat template
formatted_prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": problem}],
    tokenize=True,
    add_generation_prompt=True,
)

print(f"Prompt: {problem}")
print("-" * 50)

# %%
# Create animated visualization
from IPython.display import HTML, display

# Generate steered output with specific magnitude for visualization
vis_magnitude = 5
print("Generating output for visualization...")
vis_output = apply_intervention(
    model, 
    formatted_prompt, 
    base_steering_vectors["backtracking"], 
    vis_magnitude, 
    layer_of_interest
)

# Tokenize the output for animation
# Remove the prompt part to focus on generated text
prompt_text = model.tokenizer.decode(formatted_prompt)
generated_text = vis_output[len(prompt_text):]

# Split the text into characters/tokens while preserving structure
# This approach will handle both newlines and special characters properly
import re

# Strip leading whitespace from the entire generated text first
generated_text = generated_text.lstrip()

# First, let's use a simple approach: split by word boundaries but keep everything
tokens = []
# Split on word boundaries while keeping all characters including spaces
parts = re.findall(r'\S+|\s+', generated_text)
for part in parts:
    if '\n' in part:
        # Split on newlines but keep them
        subparts = part.split('\n')
        for i, subpart in enumerate(subparts):
            if i > 0:
                tokens.append('\n')
            if subpart:
                tokens.append(subpart)
    else:
        tokens.append(part)

# Convert tokens to list for JavaScript
import json
tokens_json = json.dumps(tokens)

# %%
# Create HTML visualization
import time
unique_id = str(int(time.time() * 1000))  # Use timestamp for unique ID

html_content = f"""
<div style="width: 600px; background: white; padding: 20px; font-family: Arial, sans-serif;">
    <!-- Top section with emojis and arrow -->
    <div style="height: 150px; position: relative;">
        <!-- Base Model with Swirl emoji (visible from start) -->
        <div style="position: absolute; left: 40px; top: 10px; text-align: center;">
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">Base Model</div>
            <div style="font-size: 96px;">🌀</div>
        </div>
        
        <!-- Arrow and label (fade in) -->
        <div id="arrow-container-{unique_id}" style="position: absolute; left: 220px; top: 60px; opacity: 0; transition: opacity 1s ease-in;">
            <div style="font-size: 48px; display: inline-block; font-weight: bold;">→</div>
            <div style="font-size: 12px; position: absolute; top: 55px; left: -15px; white-space: nowrap; text-align: center;">
                steering vector
            </div>
        </div>
        
        <!-- Reasoning Model with Robot emoji (fade in) -->
        <div id="robot-{unique_id}" style="position: absolute; right: 40px; top: 10px; text-align: center; opacity: 0; transition: opacity 1s ease-in;">
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">Reasoning Model</div>
            <div style="font-size: 96px;">🤖</div>
        </div>
    </div>
    
    <!-- Text generation area (no border, no scroll) -->
    <div style="padding-top: 15px;">
        <div id="generated-text-{unique_id}" style="font-size: 14px; line-height: 1.5; font-family: monospace; white-space: pre-wrap;">
        </div>
    </div>
</div>

<style>
    @keyframes fadeIn-{unique_id} {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    #generated-text-{unique_id} .token {{
        display: inline;
        opacity: 0;
    }}
    
    #generated-text-{unique_id} .token.visible {{
        animation: fadeIn-{unique_id} 0.3s forwards;
    }}
    
    #generated-text-{unique_id} .token.highlight {{
        background-color: #ffcccc;
        color: #cc0000;
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 3px;
    }}
</style>

<script>
    // Wrap everything in a function to ensure proper scoping
    (function() {{
        const tokens = {tokens_json};
        const uniqueId = '{unique_id}';
        
        // Wait for DOM to be ready
        setTimeout(() => {{
            const textContainer = document.getElementById('generated-text-' + uniqueId);
            const robotEmoji = document.getElementById('robot-' + uniqueId);
            const arrowContainer = document.getElementById('arrow-container-' + uniqueId);
            
            if (!textContainer || !robotEmoji || !arrowContainer) {{
                console.error('Elements not found for ID:', uniqueId);
                return;
            }}
            
            // Function to check if token should be highlighted
            function shouldHighlight(token, index) {{
                const lowerToken = token.toLowerCase();
                
                // Always highlight "wait"
                if (lowerToken.includes('wait')) {{
                    return true;
                }}
                
                // Highlight "confus" or "overcomplicat"
                if (lowerToken.includes('confus') || lowerToken.includes('overcomplicat')) {{
                    return true;
                }}
                
                // Check for "wait, but" or "wait, no" pattern
                // Look back through recent tokens for "wait"
                // Check if current token is or starts with "but" or "no"
                if (lowerToken === 'but' || lowerToken === 'no' || 
                    lowerToken.startsWith('no') || lowerToken.startsWith('but')) {{
                    for (let i = Math.max(0, index - 6); i < index; i++) {{
                        if (tokens[i] && tokens[i].toLowerCase().includes('wait')) {{
                            return true;
                        }}
                    }}
                }}
                
                return false;
            }}
            
            // Create token elements with merged highlights
            let currentHighlightSpan = null;
            let inWaitPattern = false; // Track if we're in a wait pattern
            
            tokens.forEach((token, index) => {{
                const isHighlighted = shouldHighlight(token, index);
                
                // Check if this is part of a wait pattern
                if (token.toLowerCase().includes('wait')) {{
                    inWaitPattern = true;
                }} else if (inWaitPattern && !token.trim() && index < tokens.length - 1) {{
                    // This is whitespace in a wait pattern - check if next non-space token should be highlighted
                    let nextNonSpaceIndex = index + 1;
                    while (nextNonSpaceIndex < tokens.length && !tokens[nextNonSpaceIndex].trim()) {{
                        nextNonSpaceIndex++;
                    }}
                    if (nextNonSpaceIndex < tokens.length && 
                        (tokens[nextNonSpaceIndex].toLowerCase().startsWith('no') || 
                         tokens[nextNonSpaceIndex].toLowerCase().startsWith('but'))) {{
                        // Continue the highlight through this space
                        if (currentHighlightSpan) {{
                            currentHighlightSpan.textContent += token;
                        }} else {{
                            currentHighlightSpan = document.createElement('span');
                            currentHighlightSpan.className = 'token highlight';
                            currentHighlightSpan.textContent = token;
                            textContainer.appendChild(currentHighlightSpan);
                        }}
                        return;
                    }}
                }}
                
                // Reset wait pattern if we hit something that's not part of it
                if (token.trim() && !isHighlighted && !token.includes(',')) {{
                    inWaitPattern = false;
                }}
                
                // Handle newlines
                if (token === '\\n' || token.includes('\\n')) {{
                    // End current highlight span if exists
                    currentHighlightSpan = null;
                    inWaitPattern = false;
                    
                    // Split on actual newlines and create elements
                    const parts = token.split('\\n');
                    parts.forEach((part, i) => {{
                        if (i > 0) {{
                            const br = document.createElement('br');
                            textContainer.appendChild(br);
                        }}
                        if (part) {{
                            const span = document.createElement('span');
                            span.className = 'token';
                            span.textContent = part;
                            if (shouldHighlight(part, index)) {{
                                span.classList.add('highlight');
                            }}
                            textContainer.appendChild(span);
                        }}
                    }});
                }} else {{
                    if (isHighlighted || (inWaitPattern && token === ',')) {{
                        // If we're already in a highlight span, just append to it
                        if (currentHighlightSpan) {{
                            currentHighlightSpan.textContent += token;
                        }} else {{
                            // Start a new highlight span
                            currentHighlightSpan = document.createElement('span');
                            currentHighlightSpan.className = 'token highlight';
                            currentHighlightSpan.textContent = token;
                            textContainer.appendChild(currentHighlightSpan);
                        }}
                    }} else {{
                        // Not highlighted - end current highlight span if exists
                        currentHighlightSpan = null;
                        
                        const span = document.createElement('span');
                        span.className = 'token';
                        span.textContent = token;
                        textContainer.appendChild(span);
                    }}
                }}
            }});
            
            // Animation sequence
            setTimeout(() => {{
                // Fade in robot and arrow using opacity transition
                robotEmoji.style.opacity = '1';
                arrowContainer.style.opacity = '1';
                
                // After robot appears, show tokens
                setTimeout(() => {{
                    const allElements = textContainer.querySelectorAll('.token, br');
                    let delay = 0;
                    allElements.forEach((element) => {{
                        if (element.tagName === 'BR') {{
                            // Don't add delay for line breaks
                            return;
                        }}
                        setTimeout(() => {{
                            element.classList.add('visible');
                        }}, delay);
                        // Adjust delay based on text length for merged highlights
                        const textLength = element.textContent.length;
                        delay += Math.min(50 * Math.max(1, textLength / 10), 150);
                    }});
                }}, 1500);
            }}, 500);
        }}, 100); // Small delay to ensure DOM is ready
    }})();
</script>
"""

# Display the visualization
display(HTML(html_content))

# %%