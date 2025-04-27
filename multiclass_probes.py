# %%
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nnsight import LanguageModel
from tqdm import tqdm
from collections import defaultdict
from IPython.display import display, HTML

def load_annotated_chain(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_annotations(annotated_text):
    annotations = []
    
    # Find all annotation segments
    current_pos = 0
    while True:
        # Find the next category tag
        start_tag_pos = annotated_text.find('[\"', current_pos)
        if start_tag_pos == -1:
            break
            
        end_tag_pos = annotated_text.find('\"]', start_tag_pos)
        if end_tag_pos == -1:
            break
            
        # Extract the category
        category = annotated_text[start_tag_pos+2:end_tag_pos]
        
        # Skip if this is an end-section tag
        if category == "end-section":
            current_pos = end_tag_pos + 2
            continue
            
        # Find the corresponding end-section tag
        start_text_pos = end_tag_pos + 2
        end_section_tag = annotated_text.find('[\"end-section\"]', start_text_pos)
        
        # Extract the text between category tag and end-section tag
        if end_section_tag != -1:
            text = annotated_text[start_text_pos:end_section_tag]
            
            # Move past the end-section tag
            after_end_tag_pos = end_section_tag + 15  # Length of "[\"end-section\"]"
            
            # Check if there's another category tag following
            next_tag_pos = annotated_text.find('[\"', after_end_tag_pos)
            
            # If there is text between this end-section and the next category tag, include it
            if next_tag_pos != -1 and next_tag_pos > after_end_tag_pos:
                # Include the text between tags (which may contain newlines)
                between_tags_text = annotated_text[after_end_tag_pos:next_tag_pos]
                text += between_tags_text
            
            annotations.append((category, text))
            current_pos = after_end_tag_pos if next_tag_pos == -1 else next_tag_pos
        else:
            # If no end-section tag, move to the next position
            current_pos = end_tag_pos + 2
    
    return annotations

def process_chain(model, chain):
    # Format problem with user/assistant tags
    problem = chain["problem"]
    messages = [
        {"role": "user", "content": problem},
    ]
    formatted_problem = model.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    # Tokenize the formatted problem
    tokenized_problem = model.tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]
    
    # Extract all annotations from the annotated chain
    annotations = extract_annotations(chain["annotated_chain"])
    
    # Track token indices for each category
    annotation_indices = {}
    
    # Current token position
    current_token_pos = len(tokenized_problem)
    
    # Full tokenized text (starting with the tokenized problem)
    full_tokens = tokenized_problem.tolist()
    
    # Process each annotation
    for i, (category, text) in enumerate(annotations):
        # Tokenize this text segment
        segment_tokens = model.tokenizer.encode(text, add_special_tokens=False)
        
        # Record start and end token indices
        start_idx = current_token_pos
        end_idx = start_idx + len(segment_tokens) - 1
        
        # Add to annotation indices for this category
        if category not in annotation_indices:
            annotation_indices[category] = []
        annotation_indices[category].append((start_idx, end_idx))
        
        # Add segment tokens to full tokens
        full_tokens.extend(segment_tokens)
        
        # Update current token position
        current_token_pos += len(segment_tokens)
    
    # Convert full tokens back to tensor
    tokenized_full_text = torch.tensor(full_tokens)
    
    return tokenized_full_text, annotation_indices

def process_chains_iterator(model, chains):
    for i, chain in enumerate(chains):
        # Process this chain
        tokenized_text, annotation_indices = process_chain(model, chain)
        
        # Yield the results
        yield tokenized_text, annotation_indices

def prepare_multiclass_probe_data(model, chains, layer_of_interest, 
                             categories=["backtracking", "uncertainty-estimation", "deduction", 
                                        "example-testing", "adding-knowledge", "initializing"],
                             n_tokens=5, buffer_tokens=5, max_chains=None, random_seed=42):
    """
    Prepare data for training multi-class linear probes to predict the next category.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries
        layer_of_interest: Layer to extract activations from
        categories: List of categories to predict
        n_tokens: Number of token positions to use for the window
        buffer_tokens: Number of tokens to skip between window and annotation
        max_chains: Maximum number of chains to process (None for all)
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Features (activations)
        y: Labels (category index)
        label_map: Mapping from class index to category name
    """
    # Create a mapping from category name to class index
    category_to_index = {cat: i for i, cat in enumerate(categories)}
    label_map = {i: cat for i, cat in enumerate(categories)}
    
    # Lists to collect examples
    examples = []
    
    processed_chains = list(process_chains_iterator(model, chains[:max_chains] if max_chains else chains))
    
    for tokens, indices in tqdm(processed_chains):
        # Need to decode tokens back to text for model.trace
        text = model.tokenizer.decode(tokens)
        
        with torch.inference_mode():
            if layer_of_interest == -1:
                with model.trace(text) as tracer:
                    layer_activations = model.model.layers[0].input[0].unsqueeze(0).save()
            else:   
                with model.trace(text) as tracer:
                    layer_activations = model.model.layers[layer_of_interest].output[0].save()
            
        # Create sequence of annotations in order of appearance
        all_annotations = []
        for category, index_tuples in indices.items():
            for start, end in index_tuples:
                all_annotations.append((category, start, end))
        
        # Sort by start index
        all_annotations.sort(key=lambda x: x[1])
        
        # For each annotation, collect data for predicting the next category
        for i in range(len(all_annotations) - 1):
            current_cat, current_start, current_end = all_annotations[i]
            next_cat, next_start, next_end = all_annotations[i + 1]
            
            # Check if we have enough tokens for window + buffer
            window_start = current_end - n_tokens - buffer_tokens
            window_end = current_end - buffer_tokens
            
            # Skip if the next category is not in our target categories
            if next_cat not in category_to_index:
                continue
                
            # Skip if we don't have enough tokens
            if window_start < 0:
                continue
                
            # Get the activations for the n tokens, leaving a buffer
            token_activations = layer_activations[0, window_start:window_end]
            
            # Take the mean across the token dimension
            mean_activations = token_activations.mean(dim=0).to(torch.float32).detach().cpu().numpy()
            
            # Skip if we have NaN values or empty activations
            if np.isnan(mean_activations).any() or mean_activations.shape[0] == 0:
                continue
            
            # Get the label index for the next category
            label = category_to_index[next_cat]
            
            # Add example
            examples.append((mean_activations, label))
    
    # Count examples per class
    class_counts = defaultdict(int)
    for _, label in examples:
        class_counts[label] += 1
    
    print(f"Collected examples per class:")
    for idx, count in class_counts.items():
        print(f"  {label_map[idx]}: {count} examples")
    
    # Shuffle the examples
    np.random.seed(random_seed)
    np.random.shuffle(examples)
    
    # Split into features and labels
    X = np.array([example[0] for example in examples])
    y = np.array([example[1] for example in examples])
    
    return X, y, label_map

def train_and_evaluate_multiclass_probe(X, y, label_map, test_size=0.2, random_state=42):
    """
    Train and evaluate a multi-class linear probe.
    
    Args:
        X: Features (activations)
        y: Labels (category indices)
        label_map: Mapping from class index to category name
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model and evaluation metrics
    """
    # Split data into train and test sets
    n_samples = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Print class distribution
    class_names = [label_map[i] for i in range(len(label_map))]
    
    print(f"Train set: {len(y_train)} examples")
    train_counts = np.bincount(y_train, minlength=len(label_map))
    for i, count in enumerate(train_counts):
        print(f"  {label_map[i]}: {count} examples ({count/len(y_train):.2%})")
    
    print(f"Test set: {len(y_test)} examples")
    test_counts = np.bincount(y_test, minlength=len(label_map))
    for i, count in enumerate(test_counts):
        print(f"  {label_map[i]}: {count} examples ({count/len(y_test):.2%})")
    
    # Train the logistic regression model (multi-class)
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state,
        solver='liblinear',
        C=1e-2,
        multi_class='ovr'  # One-vs-rest strategy
    )
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Format confusion matrix for display
    cm_str = "Predicted\n"
    cm_str += " " * 20 + " ".join([f"{label_map[i]:<15}" for i in range(len(label_map))]) + "\n"
    for i in range(len(label_map)):
        cm_str += f"{label_map[i]:<20}"
        for j in range(len(label_map)):
            cm_str += f"{cm[i,j]:<15}"
        cm_str += f" | {test_counts[i]}\n"
    print(cm_str)
    
    return model, accuracy, report, cm

def experiment_with_token_windows(model, chains, layer_of_interest,
                                 categories=["backtracking", "uncertainty-estimation", "deduction", 
                                            "example-testing", "adding-knowledge", "initialization"],
                                 token_windows=[1, 3, 5, 10, 20], buffer_tokens=5, max_chains=None):
    """
    Experiment with different token window sizes for multi-class prediction.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries
        layer_of_interest: Layer to extract activations from
        categories: List of categories to predict
        token_windows: List of token window sizes to try
        buffer_tokens: Number of tokens to skip between window and annotation
        max_chains: Maximum number of chains to process
        
    Returns:
        Dictionary mapping window sizes to trained models and metrics
    """
    results = {}
    
    # Convert buffer_tokens to a list if it's a single value
    if not isinstance(buffer_tokens, list):
        buffer_tokens = [buffer_tokens]
    
    for buffer in buffer_tokens:
        if buffer not in results:
            results[buffer] = {}
        
        for n_tokens in token_windows:
            print(f"\nTraining multi-class probe with {n_tokens} token window, {buffer} buffer:")
            X, y, label_map = prepare_multiclass_probe_data(
                model, chains, layer_of_interest, categories, 
                n_tokens, buffer, max_chains
            )
            
            if len(X) == 0:
                print(f"No data for window size {n_tokens} with buffer {buffer}")
                continue
                
            print(f"Data shape: {X.shape}, Labels: {y.shape}")
            probe, accuracy, report, cm = train_and_evaluate_multiclass_probe(X, y, label_map)
            
            results[buffer][n_tokens] = {
                'probe': probe,
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'label_map': label_map,
                'data_shape': X.shape
            }
            
            # Save the trained probe
            import pickle
            with open(f"multiclass_probe_{n_tokens}tokens_{buffer}buffer_layer{layer_of_interest}.pkl", "wb") as f:
                pickle.dump({
                    'probe': probe, 
                    'accuracy': accuracy,
                    'report': report,
                    'label_map': label_map
                }, f)
    
    return results

def visualize_multiclass_probe_predictions(model, probe, label_map, chain, layer_of_interest=10, 
                                          n_tokens=5, buffer_tokens=5, color_by_score=True):
    """
    Create HTML visualization of a multi-class probe's predictions on a reasoning chain.
    
    Args:
        model: Language model with tokenizer
        probe: Trained multi-class linear probe model
        label_map: Mapping from class index to category name
        chain: Dictionary containing problem and annotated_chain
        layer_of_interest: Layer to extract activations from
        n_tokens: Number of token positions used for the window
        buffer_tokens: Number of tokens skipped between window and target
        color_by_score: If True, color intensity varies by prediction probability
        
    Returns:
        HTML string with highlighted tokens
    """
    # Define colors for each category (using distinct colors)
    category_colors = {
        "backtracking": "255,0,0",          # Red
        "uncertainty-estimation": "255,165,0", # Orange
        "deduction": "0,128,0",             # Green
        "example-testing": "0,0,255",       # Blue
        "adding-knowledge": "128,0,128",    # Purple
        "initializing": "0,128,128"       # Teal
    }
    
    # Process the chain
    tokens, indices = process_chain(model, chain)
    text = model.tokenizer.decode(tokens)
    token_texts = [model.tokenizer.decode(t) for t in tokens]
    
    # Get activations for the entire text
    with model.trace(text) as tracer:
        layer_activations = model.model.layers[layer_of_interest].output[0].save()
    
    # Get predictions for each position where we have enough context
    predictions = []
    probabilities = []
    
    # For each possible token position
    for pos in range(n_tokens + buffer_tokens, len(tokens)):
        # Check if we have enough tokens for window + buffer
        window_start = pos - n_tokens - buffer_tokens
        window_end = pos - buffer_tokens
        
        if window_start >= 0:
            # Get the activations for the n tokens, leaving a buffer
            token_activations = layer_activations[0, window_start:window_end]
            
            # Take the mean across the token dimension
            mean_activations = token_activations.mean(dim=0).to(torch.float32).detach().cpu().numpy()
            
            # Check for NaN values and skip if found
            if np.isnan(mean_activations).any() or mean_activations.shape[0] == 0:
                predictions.append(None)
                probabilities.append(None)
                continue
                
            # Get prediction and probabilities
            pred = probe.predict([mean_activations])[0]
            probs = probe.predict_proba([mean_activations])[0]
            
            predictions.append(pred)
            probabilities.append(probs)
        else:
            predictions.append(None)
            probabilities.append(None)
    
    # Add padding for tokens at the beginning that didn't get predictions
    padding = [None] * (n_tokens + buffer_tokens)
    all_predictions = padding + predictions
    all_probabilities = padding + probabilities
    
    # Create HTML with token highlighting
    html = "<div style='font-family:monospace; line-height:1.5; background-color:white; color:black; padding:10px;'>"
    html += f"<p><strong>Multi-class probe visualization:</strong> {n_tokens} token window, {buffer_tokens} buffer</p>"
    
    # Display color legend
    html += "<div style='margin-bottom:10px;'>"
    for category, color in category_colors.items():
        html += f"<span style='background-color:rgba({color},0.7); padding:2px 5px; margin-right:10px;'>{category}</span>"
    html += "</div>"
    
    for i, token in enumerate(token_texts):
        if i < len(all_probabilities) and all_probabilities[i] is not None:
            probs = all_probabilities[i]
            pred = all_predictions[i]
            pred_category = label_map[pred]
            color = category_colors.get(pred_category, "0,0,0")
            
            if color_by_score:
                # Use confidence as intensity
                intensity = 0.3 + probs[pred] * 0.7
                html += f"<span style='background-color:rgba({color},{intensity:.2f})'>{token}</span>"
            else:
                # Simple coloring with fixed intensity
                html += f"<span style='background-color:rgba({color},0.5)'>{token}</span>"
        else:
            # Token with no prediction (beginning of text)
            html += f"<span style='opacity:0.5'>{token}</span>"
    
    html += "</div>"
    
    return html

def display_multiclass_probe_predictions(model, probe, label_map, chain, layer_of_interest=10, 
                                        n_tokens=5, buffer_tokens=5, color_by_score=True):
    """Display the visualization of multi-class probe predictions in a notebook"""
    html_viz = visualize_multiclass_probe_predictions(
        model, probe, label_map, chain, layer_of_interest, n_tokens, buffer_tokens, color_by_score
    )
    display(HTML(html_viz))
    
    # Also save to file
    with open(f"multiclass_probe_predictions_{n_tokens}tokens_{buffer_tokens}buffer.html", "w") as f:
        f.write(html_viz)
    
    return html_viz

def load_multiclass_probe(file_path):
    """Load a trained multi-class probe from a pickle file"""
    import pickle
    with open(file_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    if isinstance(probe_data, dict):
        return probe_data.get('probe'), probe_data.get('label_map')
    return probe_data, None

# Set up the model with CUDA and bfloat16
print("Loading model with CUDA and bfloat16...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

model = LanguageModel(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
    device_map=device,
    torch_dtype=torch.bfloat16
)
print("Model loaded successfully")

# Load chains
chains = load_annotated_chain("new_annotated_chains/all_annotated_chains.json")
print(f"Loaded {len(chains)} chains")

# Set parameters for multi-class probe training
categories = ["backtracking", "uncertainty-estimation", "deduction", 
             "example-testing", "adding-knowledge", "initializing"]
layer_of_interest = 16
max_chains = None  # Limit to avoid memory issues, set to None for all chains

# Experiment with different token windows and buffer sizes
token_windows = [4]
buffer_tokens = [14]
print(f"Running multi-class experiments for predicting {len(categories)} categories...")

results = experiment_with_token_windows(
    model,
    chains,
    layer_of_interest, 
    categories=categories,
    token_windows=token_windows,
    buffer_tokens=buffer_tokens,
    max_chains=max_chains
)

# Print summary in a formatted table
print("\nResults summary:")
print(f"{'Buffer':<8} {'Window':<8} {'Accuracy':<10}")
print("-" * 30)

for buffer in sorted(results.keys()):
    for n_tokens in sorted(results[buffer].keys()):
        acc = results[buffer][n_tokens]['accuracy']
        print(f"{buffer:<8} {n_tokens:<8} {acc:.4f}")

# Visualize trained probe predictions on a chain
if len(results) > 0:
    # Choose a well-performing probe from the results
    # Find the buffer and window with the highest accuracy
    best_acc = 0
    best_buffer = None
    best_window = None
    
    for buffer in results:
        for window in results[buffer]:
            acc = results[buffer][window]['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_buffer = buffer
                best_window = window
    
    if best_buffer is not None and best_window is not None:
        print(f"Visualizing predictions from best probe: window={best_window}, buffer={best_buffer}, accuracy={best_acc:.4f}")
        probe = results[best_buffer][best_window]['probe']
        label_map = results[best_buffer][best_window]['label_map']
        
        # Choose a chain with multiple categories if possible
        for i in range(min(5, len(chains))):
            categories_in_chain = set()
            for category in categories:
                if "annotated_chain" in chains[i] and category in chains[i]["annotated_chain"]:
                    categories_in_chain.add(category)
            
            if len(categories_in_chain) >= 3:  # Look for chains with at least 3 categories
                print(f"Visualizing predictions on chain {i} with {len(categories_in_chain)} categories")
                html = display_multiclass_probe_predictions(
                    model, probe, label_map, chains[i], 
                    layer_of_interest=layer_of_interest,
                    n_tokens=best_window, 
                    buffer_tokens=best_buffer,
                    color_by_score=True
                )
                print(f"Visualization saved to multiclass_probe_predictions_{best_window}tokens_{best_buffer}buffer.html")
                break
else:
    print("No trained probes available yet. Run the experiment first.") 
# %%
