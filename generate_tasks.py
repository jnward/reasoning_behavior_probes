import json
import time
import os
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Anthropic API
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Create output directory if it doesn't exist
os.makedirs("tasks", exist_ok=True)

# Categories from the paper
categories = [
    {
        "name": "Mathematical Logic",
        "description": "Problems requiring formal logical operations, mathematical proofs, and numerical reasoning",
        "id_prefix": "math_logic"
    },
    {
        "name": "Spatial Reasoning",
        "description": "Tasks involving visualization, geometric manipulation, and understanding spatial relationships",
        "id_prefix": "spatial"
    },
    {
        "name": "Verbal Logic",
        "description": "Problems focused on language-based reasoning, syllogisms, and verbal analogies",
        "id_prefix": "verbal"
    },
    {
        "name": "Pattern Recognition",
        "description": "Questions requiring identification and continuation of sequences or abstract patterns",
        "id_prefix": "pattern"
    },
    {
        "name": "Lateral Thinking",
        "description": "Problems that require creative, non-linear approaches to reach unconventional solutions",
        "id_prefix": "lateral"
    },
    {
        "name": "Causal Reasoning",
        "description": "Tasks involving understanding cause-and-effect relationships and making causal inferences",
        "id_prefix": "causal"
    },
    {
        "name": "Probabilistic Thinking",
        "description": "Problems requiring reasoning about uncertainty, probability, and statistical concepts",
        "id_prefix": "probability"
    },
    {
        "name": "Systems Thinking",
        "description": "Questions about complex systems, interconnected components, and emergent behaviors",
        "id_prefix": "systems"
    },
    {
        "name": "Creative Problem Solving",
        "description": "Open-ended problems requiring novel approaches and innovative solutions",
        "id_prefix": "creative"
    },
    {
        "name": "Scientific Reasoning",
        "description": "Tasks involving hypothesis formation, experimental design, and evidence evaluation",
        "id_prefix": "scientific"
    }
]

# categories = categories[:1]

# Template for task generation prompt
prompt_template = """
Generate exactly 30 challenging problems for the category of {category_name}.

Category Description: {category_description}

Return the problems as a JSON array with the following structure for each problem:
[
  {{
    "id": "{id_prefix}_0",
    "category": "{category_name}",
    "problem": "The full text of the problem goes here",
  }},
  ...
]

Make sure each problem is unique and requires reasoning to solve.
Each problem should be self-contained and not require additional information or resources.
Ensure all IDs follow the pattern {id_prefix}_0 through {id_prefix}_29.

Only return valid JSON with no additional text, explanations, or markdown formatting.
"""

def generate_tasks_for_category(category):
    """Generate tasks for a specific category using Claude with streaming"""
    print(f"Generating tasks for {category['name']}...")
    
    formatted_prompt = prompt_template.format(
        category_name=category["name"],
        category_description=category["description"],
        id_prefix=category["id_prefix"]
    )
    
    try:
        print("-" * 50)
        
        # Stream the response
        full_content = ""
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            # model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            messages=[{"role": "user", "content": formatted_prompt}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_content += text
        
        print("\n" + "-" * 50)
        
        # Clean the response if necessary (remove markdown code blocks if present)
        if full_content.startswith("```json"):
            full_content = full_content.strip("```json").strip("```").strip()
        
        # Parse JSON
        tasks = json.loads(full_content)
        
        return tasks
    
    except Exception as e:
        print(f"\nError generating tasks for {category['name']}: {str(e)}")
        return None

# Main execution
all_tasks = []

for category in categories:
    # Generate tasks for this category
    tasks = generate_tasks_for_category(category)
    
    if tasks:
        # Save to category-specific file
        filename = f"tasks/{category['id_prefix']}_tasks.json"
        with open(filename, "w") as f:
            json.dump(tasks, f, indent=2)
        
        print(f"Successfully saved {len(tasks)} tasks for {category['name']} to {filename}")
        
        # Add to all tasks
        all_tasks.extend(tasks)
    
    # Rate limiting - be nice to the API
    time.sleep(2)

# Save all tasks to a combined file
with open("tasks/all_reasoning_tasks.json", "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"Task generation complete. Total tasks generated: {len(all_tasks)}")