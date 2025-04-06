import json
import os
import time
import argparse
from openai import OpenAI
from tqdm import tqdm

# Constants
REASONING_CHAINS_DIR = "reasoning_chains"
ANNOTATED_CHAINS_DIR = "annotated_chains"

# Create output directory
os.makedirs(ANNOTATED_CHAINS_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI()

def load_reasoning_chains(file_path):
    """Load reasoning chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def annotate_chain(thinking_process):
    """Annotate a reasoning chain using GPT-4o"""
    # Format prompt according to paper
    prompt = f"""Please split the following reasoning chain of an LLM into
annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple
parts if it incorporates multiple behaviours indicated by the labels.

Available labels:
0. initializing -> The model is rephrasing the given task and states initial thoughts.
1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
2. adding-knowledge -> The model is enriching the current approach with recalled facts.
3. example-testing -> The model generates examples to test its current approach.
4. uncertainty-estimation -> The model is stating its own uncertainty.
5. backtracking -> The model decides to change its approach.

The reasoning chain to analyze:
{thinking_process}

Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during annotation: {str(e)}")
        return None

def debug_annotation(annotated_chain):
    """Debug function to analyze annotation format issues"""
    print("\n=== ANNOTATION DEBUG INFO ===")
    print(f"Total length: {len(annotated_chain)} characters")
    
    # Count opening and closing tags
    opening_tags = annotated_chain.count('["')
    closing_tags = annotated_chain.count('["end-section"]')
    
    print(f"Found {opening_tags} opening tags and {closing_tags} closing tags")
    
    # Count category instances
    categories = ["initializing", "deduction", "adding-knowledge", 
                  "example-testing", "uncertainty-estimation", "backtracking"]
    
    for cat in categories:
        count = annotated_chain.count(f'["{cat}"]')
        print(f"Category '{cat}' appears {count} times")
    
    print("=== END DEBUG INFO ===\n")

def process_category_file(file_path):
    """Process a category file and annotate all reasoning chains"""
    # Extract category name from file path
    category_name = os.path.basename(file_path).replace("_reasoning_chains.json", "")
    print(f"Processing category: {category_name}")
    
    # Load reasoning chains
    chains = load_reasoning_chains(file_path)
    
    # Prepare output data structure
    annotated_chains = []
    
    # Process each reasoning chain
    for chain_data in tqdm(chains, desc=f"Annotating {category_name}"):
        task_id = chain_data["task_id"]
        problem = chain_data["problem"]
        reasoning_chain = chain_data["reasoning_chain"]
        
        # Clean the reasoning chain - remove <think> tag if present
        if reasoning_chain.startswith("<think>"):
            reasoning_chain = reasoning_chain[7:]
        if reasoning_chain.endswith("</think>"):
            reasoning_chain = reasoning_chain[:-8]
        
        # Annotate the reasoning chain
        annotated_chain = annotate_chain(reasoning_chain)
        
        if annotated_chain:
            # Create annotated data
            annotated_data = {
                "task_id": task_id,
                "problem": problem,
                "reasoning_chain": reasoning_chain,
                "annotated_chain": annotated_chain
            }
            
            annotated_chains.append(annotated_data)
            
            # Print a sample and debug information for the first chain
            if len(annotated_chains) == 1:
                print("\nSample Annotation:")
                print(f"Problem: {problem}")
                print(f"Annotated Chain (excerpt): {annotated_chain[:200]}...\n")
                
                # Add debugging for first chain
                debug_annotation(annotated_chain)
        
        # Rate limit to avoid API throttling
        time.sleep(0.5)
        # break
    
    # Save annotated chains
    output_file = os.path.join(ANNOTATED_CHAINS_DIR, f"{category_name}_annotated_chains.json")
    with open(output_file, 'w') as f:
        json.dump(annotated_chains, f, indent=2)
    
    print(f"Saved {len(annotated_chains)} annotated chains to {output_file}")
    
    return annotated_chains

def analyze_annotations(annotated_chains):
    """Analyze annotated chains to get token distribution for each behavioral category"""
    # Define the categories we're looking for
    categories = ["initializing", "deduction", "adding-knowledge", 
                  "example-testing", "uncertainty-estimation", "backtracking"]
    
    # Initialize counters
    category_counts = {cat: 0 for cat in categories}
    total_tokens = 0
    
    # Process each chain
    for chain_data in annotated_chains:
        annotated_chain = chain_data["annotated_chain"]
        
        # Process each category in this chain
        for category in categories:
            # Split the annotated chain on the category tag
            parts = annotated_chain.split(f'["{category}"]')
            
            # Skip the first part (comes before the first occurrence of this category)
            for i in range(1, len(parts)):
                part = parts[i]
                # Extract the text up to the end-section tag
                if '["end-section"]' in part:
                    text = part.split('["end-section"]')[0].strip()
                    # Count tokens (simple whitespace splitting)
                    tokens = len(text.split())
                    category_counts[category] += tokens
                    total_tokens += tokens
    
    # Calculate percentages
    category_percentages = {cat: (count / total_tokens * 100 if total_tokens > 0 else 0) 
                           for cat, count in category_counts.items()}
    
    return category_percentages, category_counts

def print_category_distribution(category_percentages, category_counts=None):
    """Print a detailed summary of category distribution"""
    print("\n=== Behavioral Category Distribution ===")
    
    # Sort categories by percentage for better visualization
    sorted_categories = sorted(
        category_percentages.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Calculate total tokens if counts are provided
    total_tokens = sum(category_counts.values()) if category_counts else None
    
    print(f"{'Category':<25} {'Percentage':<10} {'Token Count':<12}")
    print("-" * 47)
    
    for cat, percentage in sorted_categories:
        count = category_counts[cat] if category_counts else "N/A"
        print(f"{cat:<25} {percentage:>8.2f}%  {count:>10}")
    
    if total_tokens:
        print("-" * 47)
        print(f"{'TOTAL':<25} {100:>8.2f}%  {total_tokens:>10}")
    
    print("=" * 47)

def main():
    parser = argparse.ArgumentParser(description='Annotate reasoning chains with GPT-4o')
    parser.add_argument('--category', type=str, help='Specific category to process (e.g., "lateral")', default=None)
    parser.add_argument('--all', action='store_true', help='Process all categories')
    parser.add_argument('--test', action='store_true', help='Run in test mode to verify parsing')
    args = parser.parse_args()
    
    if args.test:
        # Run in test mode - analyze existing annotations to verify parsing
        annotated_dirs = [ANNOTATED_CHAINS_DIR]
        
        # Try to load existing files
        found_files = False
        for directory in annotated_dirs:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if f.endswith("_annotated_chains.json")]
                if files:
                    found_files = True
                    for file_name in files:
                        file_path = os.path.join(directory, file_name)
                        print(f"\nAnalyzing: {file_path}")
                        
                        # Load annotated chains
                        with open(file_path, 'r') as f:
                            annotated_chains = json.load(f)
                        
                        # Analyze distributions
                        category_percentages, category_counts = analyze_annotations(annotated_chains)
                        
                        # Print summary
                        print(f"Successfully analyzed {len(annotated_chains)} chains")
                        print_category_distribution(category_percentages, category_counts)
                        
        if not found_files:
            print("No existing annotated files found for analysis.")
            print("Please run the script first to generate annotations.")
    
    elif args.all:
        # Process all categories
        all_files = [f for f in os.listdir(REASONING_CHAINS_DIR) 
                     if f.endswith("_reasoning_chains.json") and not f.startswith("all_")]
        
        all_annotated_chains = []
        for file_name in all_files:
            file_path = os.path.join(REASONING_CHAINS_DIR, file_name)
            annotated_chains = process_category_file(file_path)
            all_annotated_chains.extend(annotated_chains)
            
        # Save combined annotated chains
        combined_output_file = os.path.join(ANNOTATED_CHAINS_DIR, "all_annotated_chains.json")
        with open(combined_output_file, 'w') as f:
            json.dump(all_annotated_chains, f, indent=2)
        
        # Analyze annotations
        category_percentages, category_counts = analyze_annotations(all_annotated_chains)
        
        # Print analysis
        print_category_distribution(category_percentages, category_counts)
            
    elif args.category:
        # Process specific category
        file_path = os.path.join(REASONING_CHAINS_DIR, f"{args.category}_reasoning_chains.json")
        if os.path.exists(file_path):
            annotated_chains = process_category_file(file_path)
            
            # Analyze annotations
            category_percentages, category_counts = analyze_annotations(annotated_chains)
            
            # Print analysis
            print_category_distribution(category_percentages, category_counts)
        else:
            print(f"Error: File not found - {file_path}")
    else:
        print("Please specify either --category, --all, or --test")

if __name__ == "__main__":
    main() 