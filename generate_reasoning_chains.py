import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# Path to the directory containing task files
TASKS_DIR = "new_tasks"
OUTPUT_DIR = "new_reasoning_chains"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model ID for DeepSeek-R1-Distill-Llama-8B
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# use Pythia 1B
# MODEL_ID = "EleutherAI/pythia-160m"
def load_model_and_tokenizer():
    """Load the DeepSeek-R1-Distill-Llama-8B model and tokenizer"""
    print(f"Loading model {MODEL_ID}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load model with fp16 precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

def generate_reasoning_chain(model, tokenizer, task_text, max_tokens=500):
    """Generate a reasoning chain for a given task"""
    # Create prompt with correct user/assistant tokens
    # prompt = f"Task: {task_text}
    # Reasoning Chain:"
    messages = [
        {"role": "user", "content": task_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)
    print("*"*len(prompt))
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Generate with greedy decoding
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the generated text (without the prompt)
    full_output = tokenizer.decode(output[0], skip_special_tokens=False)
    generated_text = full_output[len(prompt):]
    return generated_text.strip()

def process_category(category_file, model, tokenizer):
    """Process all tasks in a category file"""
    category_name = os.path.basename(category_file).replace("_tasks.json", "")
    
    # Load tasks
    with open(os.path.join(TASKS_DIR, category_file), 'r') as f:
        tasks = json.load(f)
    
    # Prepare output data structure
    outputs = []
    
    # Process each task
    for task in tqdm(tasks, desc=f"Processing {category_name}"):
        task_id = task["id"]
        problem = task["problem"]
        
        # Generate reasoning chain
        reasoning_chain = generate_reasoning_chain(model, tokenizer, problem)
        
        # Display the task and reasoning chain
        print("\n" + "="*80)
        print(f"Task ID: {task_id}")
        print(f"Problem: {problem}")
        print("-"*40)
        print("Reasoning Chain:")
        print(reasoning_chain)
        print("="*80)
        
        # Save output
        output_data = {
            "task_id": task_id,
            "problem": problem,
            "reasoning_chain": reasoning_chain
        }
        
        outputs.append(output_data)
            
    
    # Save outputs for this category
    output_file = os.path.join(OUTPUT_DIR, f"{category_name}_reasoning_chains.json")
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print(f"Saved {len(outputs)} reasoning chains to {output_file}")
    
    return outputs

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Get all task files
    category_files = [f for f in os.listdir(TASKS_DIR) if f.endswith("_tasks.json") and not f.startswith("all_")]
    
    # Process each category
    all_outputs = []
    for category_file in category_files:
        category_outputs = process_category(category_file, model, tokenizer)
        all_outputs.extend(category_outputs)
    
    # Save all outputs to a combined file
    combined_output_file = os.path.join(OUTPUT_DIR, "all_reasoning_chains.json")
    with open(combined_output_file, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    
    print(f"Completed processing. Total reasoning chains generated: {len(all_outputs)}")

if __name__ == "__main__":
    main()