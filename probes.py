# %%
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nnsight import LanguageModel
from tqdm import tqdm
from collections import defaultdict
from IPython.display import display, HTML

# %%
def load_annotated_chain(file_path):
    """Load annotated chains from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

# %% 
# Test load_annotated_chain
try:
    chains = load_annotated_chain("annotated_chains/all_annotated_chains.json")
    print(f"Loaded {len(chains)} chains")
    print(f"First chain keys: {chains[0].keys()}")
except FileNotFoundError:
    print("File not found. Will continue when the file is available.")

# %%
def extract_annotations(annotated_text):
    """Extract all annotations and their text from an annotated chain"""
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
            text = annotated_text[start_text_pos:end_section_tag].strip()
            annotations.append((category, text))
            current_pos = end_section_tag + 15  # Length of "[\"end-section\"]"
        else:
            # If no end-section tag, move to the next position
            current_pos = end_tag_pos + 2
    
    return annotations

# %%
# Test extract_annotations with a simple example
sample_text = 'First I need to [\"initializing\"]understand the problem[\"end-section\"] then I will [\"backtracking\"]check my understanding[\"end-section\"]'
annotations = extract_annotations(sample_text)
for category, text in annotations:
    print(f"Category: {category}")
    print(f"Text: {text}")
    print("-" * 30)

# %%
def process_chain(model, chain):
    """
    Process a chain to format it with user/assistant tags and track annotation indices.
    
    Args:
        model: Language model with tokenizer
        chain: Dictionary containing problem and annotated_chain
        
    Returns:
        tuple: (tokenized_full_text, annotation_indices)
            where annotation_indices is a dict mapping categories to lists of (start, end) token pairs
    """
    # Format problem with user/assistant tags
    problem = chain["problem"]
    formatted_problem = f"<|User|>{problem}<|Assistant|><think>\n"
    
    # Tokenize the formatted problem
    tokenized_problem = model.tokenizer.encode(formatted_problem, return_tensors="pt")[0]
    
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
        if i > 0:
            text = " " + text
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

# %%
def process_chains_iterator(model, chains):
    """
    Process multiple chains and yield tokenized text with annotation indices for each.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries containing problem and annotated_chain
        
    Yields:
        tuple: (tokenized_full_text, annotation_indices)
            where annotation_indices is a dict mapping categories to lists of (start, end) token pairs
    """
    for i, chain in enumerate(chains):
        # Process this chain
        tokenized_text, annotation_indices = process_chain(model, chain)
        
        # Yield the results
        yield tokenized_text, annotation_indices

# %%
def prepare_probe_data(model, chains, layer_of_interest, target_category="backtracking", 
                      n_tokens=5, buffer_tokens=5, max_chains=None, balance_ratio=1.0, random_seed=42):
    """
    Prepare data for training linear probes to predict if the next sentence is a target category.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries
        layer_of_interest: Layer to extract activations from
        target_category: Category to predict (e.g., "backtracking")
        n_tokens: Number of token positions to use for the window
        buffer_tokens: Number of tokens to skip between window and annotation
        max_chains: Maximum number of chains to process (None for all)
        balance_ratio: Ratio of negative to positive examples (1.0 = equal classes)
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Features (activations)
        y: Labels (1 if followed by target_category, 0 otherwise)
    """
    # Lists to collect examples by class
    positive_examples = []
    negative_examples = []
    
    processed_chains = list(process_chains_iterator(model, chains[:max_chains] if max_chains else chains))
    
    for tokens, indices in tqdm(processed_chains):
        # Need to decode tokens back to text for model.trace
        text = model.tokenizer.decode(tokens)
        
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
        
        # For each annotation, check if the next one is target_category
        for i in range(len(all_annotations) - 1):
            current_cat, current_start, current_end = all_annotations[i]
            next_cat, next_start, next_end = all_annotations[i + 1]
            
            # Check if we have enough tokens for window + buffer
            window_start = current_end - n_tokens - buffer_tokens
            window_end = current_end - buffer_tokens
            
            if window_start >= 0:
                # Get the activations for the n tokens, leaving a buffer
                token_activations = layer_activations[0, window_start:window_end]
                
                # Take the mean across the token dimension instead of flattening
                # This ensures the feature vector has the same dimension regardless of n_tokens
                mean_activations = token_activations.mean(dim=0).to(torch.float32).detach().cpu().numpy()
                
                # Skip if we have NaN values or empty activations
                if np.isnan(mean_activations).any() or mean_activations.shape[0] == 0:
                    continue
                
                # Add to appropriate list based on label
                if next_cat == target_category:
                    positive_examples.append((mean_activations, 1))
                else:
                    negative_examples.append((mean_activations, 0))
    
    print(f"Collected {len(positive_examples)} positive examples and {len(negative_examples)} negative examples")
    
    # Balance the dataset using random undersampling
    np.random.seed(random_seed)
    if balance_ratio > 0:
        # Determine number of negative examples to keep
        n_neg_to_keep = min(len(negative_examples), int(len(positive_examples) * balance_ratio))
        
        # Randomly select negative examples
        if n_neg_to_keep < len(negative_examples):
            selected_neg_indices = np.random.choice(
                len(negative_examples), n_neg_to_keep, replace=False
            )
            negative_examples = [negative_examples[i] for i in selected_neg_indices]
    
    # Combine positive and negative examples
    examples = positive_examples + negative_examples
    np.random.shuffle(examples)
    
    # Split into features and labels
    X = np.array([example[0] for example in examples])
    y = np.array([example[1] for example in examples])
    
    print(f"After balancing: {np.sum(y)} positive examples and {len(y) - np.sum(y)} negative examples")
    print(f"Class distribution: {np.mean(y):.2%} positive")
    
    return X, y

# %%
def train_and_evaluate_probe(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate a linear probe.
    
    Args:
        X: Features (activations)
        y: Labels
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
    train_pos = np.sum(y_train) / len(y_train)
    test_pos = np.sum(y_test) / len(y_test)
    print(f"Train set: {len(y_train)} examples, {train_pos:.2%} positive")
    print(f"Test set: {len(y_test)} examples, {test_pos:.2%} positive")
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, y_train_pred, average='binary')
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f}")
    print(f"Train F1-Score: {train_f1:.4f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    return model, (accuracy, precision, recall, f1)

# %%
def experiment_with_token_windows(model, chains, layer_of_interest, target_category="backtracking", 
                                 token_windows=[1, 3, 5, 10, 20], buffer_tokens=5, max_chains=None,
                                 balance_ratio=1.0):
    """
    Experiment with different token window sizes.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries
        layer_of_interest: Layer to extract activations from
        target_category: Category to predict
        token_windows: List of token window sizes to try
        buffer_tokens: Number of tokens to skip between window and annotation, can be a single value or a list
        max_chains: Maximum number of chains to process
        balance_ratio: Ratio of negative to positive examples (1.0 = equal classes)
        
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
            print(f"\nTraining probe with {n_tokens} token window, {buffer} buffer:")
            X, y = prepare_probe_data(
                model, chains, layer_of_interest, target_category, 
                n_tokens, buffer, max_chains, balance_ratio
            )
            
            if len(X) == 0:
                print(f"No data for window size {n_tokens} with buffer {buffer}")
                continue
                
            print(f"Data shape: {X.shape}, Labels: {y.shape}, Positive: {np.sum(y)}/{len(y)} ({np.mean(y):.2%})")
            probe, metrics = train_and_evaluate_probe(X, y)
            
            results[buffer][n_tokens] = {
                'probe': probe,
                'metrics': metrics,
                'data_shape': X.shape,
                'positive_ratio': np.mean(y)
            }
            
            # Save the trained probe
            import pickle
            balanced_str = f"_balanced{balance_ratio}" if balance_ratio != 0 else ""
            with open(f"probe_{target_category}_{n_tokens}tokens_{buffer}buffer{balanced_str}_layer{layer_of_interest}.pkl", "wb") as f:
                pickle.dump({'probe': probe, 'metrics': metrics}, f)
    
    return results

# %%
def visualize_results_heatmap(results, metric_index=0, title=None):
    """
    Visualize probe results as a heatmap.
    
    Args:
        results: Results dictionary from experiment_with_token_windows
        metric_index: Index of the metric to plot (0=accuracy, 1=precision, 2=recall, 3=f1)
        title: Optional title for the plot
    
    Returns:
        Plotly figure
    """
    import plotly.express as px
    import pandas as pd
    
    # Extract metrics from results dictionary
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_name = metric_names[metric_index]
    
    # Create lists for buffer_tokens, token_windows, and corresponding values
    buffers = []
    windows = []
    values = []
    
    # Extract all token windows and buffer values
    all_windows = set()
    for buffer in results:
        for window in results[buffer]:
            all_windows.add(window)
    
    # Sort buffers and windows
    buffer_tokens = sorted(list(results.keys()))
    token_windows = sorted(list(all_windows))
    
    # Create a 2D array to hold the values
    value_matrix = np.zeros((len(buffer_tokens), len(token_windows)))
    value_matrix[:] = np.nan  # Fill with NaN initially
    
    # Fill in the matrix with metric values
    for i, buffer in enumerate(buffer_tokens):
        for j, window in enumerate(token_windows):
            if window in results[buffer]:
                metrics = results[buffer][window]['metrics']
                value_matrix[i, j] = metrics[metric_index]
    
    # Create a DataFrame for easier plotting with plotly
    df = pd.DataFrame(value_matrix, 
                     index=[f"Buffer: {b}" for b in buffer_tokens], 
                     columns=[f"Window: {w}" for w in token_windows])
    
    # Create heatmap using plotly
    if title is None:
        title = f"{metric_name} by Token Window Size and Buffer Distance"
        
    fig = px.imshow(df, 
                   labels=dict(x="Token Window Size", y="Buffer Size", color=metric_name),
                   text_auto='.3f',
                   color_continuous_scale="Viridis",
                   title=title)
    
    fig.update_layout(xaxis_side="top")
    
    return fig

# %%
# Visualization functions for training examples
def visualize_training_examples(model, chain, target_category="backtracking", 
                               n_tokens=5, buffer_tokens=5,
                               positive_color="red", negative_color="blue"):
    """
    Create HTML visualization of the training examples in a reasoning chain.
    
    Args:
        model: Language model with tokenizer
        chain: Dictionary containing problem and annotated_chain
        target_category: Category to predict (e.g., "backtracking")
        n_tokens: Number of token positions to use for the window
        buffer_tokens: Number of tokens to skip between window and annotation
        positive_color: Color for positive examples (followed by target_category)
        negative_color: Color for negative examples
        
    Returns:
        HTML string with highlighted tokens
    """
    # Define color RGB values
    color_map = {
        "blue": "0,0,255",
        "orange": "255,165,0",
        "red": "255,0,0",
        "green": "0,128,0",
        "purple": "128,0,128"
    }
    
    pos_rgb = color_map.get(positive_color, "255,0,0")
    neg_rgb = color_map.get(negative_color, "0,0,255")
    
    # Process the chain
    tokens, indices = process_chain(model, chain)
    token_texts = [model.tokenizer.decode(t) for t in tokens]
    
    # Create sequence of annotations in order of appearance
    all_annotations = []
    for category, index_tuples in indices.items():
        for start, end in index_tuples:
            all_annotations.append((category, start, end))
    
    # Sort by start index
    all_annotations.sort(key=lambda x: x[1])
    
    # Mark which tokens are part of positive/negative examples
    token_classes = [None] * len(tokens)
    
    # For each annotation, check if the next one is the target_category
    for i in range(len(all_annotations) - 1):
        current_cat, current_start, current_end = all_annotations[i]
        next_cat, next_start, next_end = all_annotations[i + 1]
        
        # Check if we have enough tokens for window + buffer
        window_start = current_end - n_tokens - buffer_tokens
        window_end = current_end - buffer_tokens
        
        if window_start >= 0:
            # Get the indices for the n tokens, leaving a buffer
            example_indices = list(range(window_start, window_end))
            
            # Mark these tokens as positive or negative examples
            is_positive = (next_cat == target_category)
            
            for idx in example_indices:
                token_classes[idx] = is_positive
    
    # Create HTML with token highlighting
    html = "<div style='font-family:monospace; line-height:1.5; background-color:white; color:black; padding:10px;'>"
    
    for i, token in enumerate(token_texts):
        if token_classes[i] is None:
            # Not part of a training example
            html += f"<span>{token}</span>"
        elif token_classes[i]:
            # Positive example (followed by backtracking)
            html += f"<span style='background-color:rgba({pos_rgb},0.3)'>{token}</span>"
        else:
            # Negative example (not followed by backtracking)
            html += f"<span style='background-color:rgba({neg_rgb},0.3)'>{token}</span>"
    
    html += "</div>"
    
    return html

def display_training_examples(model, chain, target_category="backtracking", 
                             n_tokens=5, buffer_tokens=5):
    """Display the visualization of training examples in a notebook"""
    html_viz = visualize_training_examples(
        model, chain, target_category, n_tokens, buffer_tokens,
        positive_color="red", negative_color="blue"
    )
    display(HTML(html_viz))
    
    # Also save to file
    with open(f"training_examples_{target_category}_{n_tokens}tokens_{buffer_tokens}buffer.html", "w") as f:
        f.write(html_viz)
    
    return html_viz

# %%
# Visualization functions for probe predictions on new text
def visualize_probe_predictions(model, probe, chain, layer_of_interest=10, n_tokens=5, buffer_tokens=-1,
                               threshold=0.5, high_color="red", low_color="blue", 
                               color_by_score=True):
    """
    Create HTML visualization of a probe's predictions on a reasoning chain.
    
    Args:
        model: Language model with tokenizer
        probe: Trained linear probe model
        chain: Dictionary containing problem and annotated_chain
        layer_of_interest: Layer to extract activations from
        n_tokens: Number of token positions used for the window (must match the probe's training)
        buffer_tokens: Number of tokens skipped between window and target (must match the probe's training)
        threshold: Threshold for binary prediction (default: 0.5)
        high_color: Color for high probability predictions (likely backtracking)
        low_color: Color for low probability predictions (unlikely backtracking)
        color_by_score: If True, color intensity varies by prediction probability; 
                        if False, only prediction above threshold are colored
        
    Returns:
        HTML string with highlighted tokens
    """
    # Define color RGB values
    color_map = {
        "blue": "0,0,255",
        "orange": "255,165,0",
        "red": "255,0,0",
        "green": "0,128,0",
        "purple": "128,0,128"
    }
    
    high_rgb = color_map.get(high_color, "255,0,0")
    low_rgb = color_map.get(low_color, "0,0,255")
    
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
            
            # Take the mean across the token dimension instead of flattening
            mean_activations = token_activations.mean(dim=0).to(torch.float32).detach().cpu().numpy()
            
            # Check for NaN values and skip if found
            if np.isnan(mean_activations).any() or mean_activations.shape[0] == 0:
                predictions.append(None)
                probabilities.append(None)
                continue
                
            # Get binary prediction and probability
            pred = probe.predict([mean_activations])[0]
            prob = probe.predict_proba([mean_activations])[0][1]  # Probability of class 1
            
            predictions.append(pred)
            probabilities.append(prob)
        else:
            predictions.append(None)
            probabilities.append(None)
    
    # Add padding for tokens at the beginning that didn't get predictions
    padding = [None] * (n_tokens + buffer_tokens)
    all_predictions = padding + predictions
    all_probabilities = padding + probabilities
    
    # Create HTML with token highlighting
    html = "<div style='font-family:monospace; line-height:1.5; background-color:white; color:black; padding:10px;'>"
    html += f"<p><strong>Probe visualization:</strong> {n_tokens} token window, {buffer_tokens} buffer, threshold={threshold}</p>"
    
    # Display color legend
    html += "<div style='margin-bottom:10px;'>"
    html += f"<span style='background-color:rgba({high_rgb},0.7); padding:2px 5px; margin-right:10px;'>High probability (likely backtracking)</span>"
    html += f"<span style='background-color:rgba({low_rgb},0.7); padding:2px 5px;'>Low probability (unlikely backtracking)</span>"
    html += "</div>"
    
    for i, token in enumerate(token_texts):
        if i < len(all_probabilities) and all_probabilities[i] is not None:
            prob = all_probabilities[i]
            pred = all_predictions[i]
            
            if color_by_score:
                # Color by probability score (intensity proportional to confidence)
                if prob >= threshold:
                    # Use high color with intensity proportional to probability
                    intensity = 0.3 + (prob - threshold) * (0.7 / (1 - threshold))
                    html += f"<span style='background-color:rgba({high_rgb},{intensity:.2f})'>{token}</span>"
                else:
                    # Use low color with intensity proportional to inverse probability
                    intensity = 0.3 + (threshold - prob) * (0.7 / threshold)
                    html += f"<span style='background-color:rgba({low_rgb},{intensity:.2f})'>{token}</span>"
            else:
                # Simple binary coloring based on threshold
                if pred == 1:
                    html += f"<span style='background-color:rgba({high_rgb},0.3)'>{token}</span>"
                else:
                    html += f"<span style='background-color:rgba({low_rgb},0.3)'>{token}</span>"
        else:
            # Token with no prediction (beginning of text)
            html += f"<span style='opacity:0.5'>{token}</span>"
    
    html += "</div>"
    
    return html

def display_probe_predictions(model, probe, chain, layer_of_interest=10, n_tokens=5, buffer_tokens=-1, 
                             threshold=0.5, high_color="red", low_color="blue", color_by_score=True):
    """Display the visualization of probe predictions in a notebook"""
    html_viz = visualize_probe_predictions(
        model, probe, chain, layer_of_interest, n_tokens, buffer_tokens,
        threshold, high_color, low_color, color_by_score
    )
    display(HTML(html_viz))
    
    # Also save to file
    with open(f"probe_predictions_{n_tokens}tokens_{buffer_tokens}buffer.html", "w") as f:
        f.write(html_viz)
    
    return html_viz

# %%
# Function to load a trained probe from a pickle file
def load_probe(file_path):
    """Load a trained probe from a pickle file"""
    import pickle
    with open(file_path, 'rb') as f:
        probe_data = pickle.load(f)
    return probe_data.get('probe') if isinstance(probe_data, dict) else probe_data

# %% 
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

# %%
# Load chains
chains = load_annotated_chain("annotated_chains/all_annotated_chains.json")
print(f"Loaded {len(chains)} chains")

# %%
# Visualize training examples for the first few chains
if len(chains) > 0:
    print("Visualizing training examples for backtracking...")
    
    # Visualize with 5 token window and 5 token buffer
    n_tokens = 1
    buffer_tokens = -3
    print(f"\nWindow size: {n_tokens} tokens, Buffer: {buffer_tokens} tokens")
    
    # Choose a chain with backtracking annotations if possible
    for i in range(min(5, len(chains))):
        # Check if this chain has backtracking annotations
        if "annotated_chain" in chains[i] and "backtracking" in chains[i]["annotated_chain"]:
            print(f"Visualizing chain {i}")
            html = display_training_examples(model, chains[i], "backtracking", n_tokens, buffer_tokens)
            print(f"HTML visualization saved to training_examples_backtracking_{n_tokens}tokens_{buffer_tokens}buffer.html")
            break

# %%
# Set parameters
layer_of_interest = -1
target_category = "backtracking"
max_chains = None  # Limit to avoid memory issues, set to None for all chains
balance_ratio = 1.0  # Equal number of positive and negative examples

# %%
# Experiment with different token windows and buffer sizes
token_windows = [1]
buffer_tokens = [-3, -2, -1, 0, 1, 2, 3, 4, 5]  # -2 is " Wait", -1 is "."
print(f"Running experiments for predicting '{target_category}' annotations with balanced classes (ratio: {balance_ratio})...")

results = experiment_with_token_windows(
    model, chains, layer_of_interest, 
    target_category=target_category,
    token_windows=token_windows,
    buffer_tokens=buffer_tokens,
    max_chains=max_chains,
    balance_ratio=balance_ratio
)

# %%
# Print summary in a formatted table
print("\nResults summary:")
print(f"{'Buffer':<8} {'Window':<8} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 60)

for buffer in sorted(results.keys()):
    for n_tokens in sorted(results[buffer].keys()):
        acc, prec, rec, f1 = results[buffer][n_tokens]['metrics']
        print(f"{buffer:<8} {n_tokens:<8} {acc:.4f}      {f1:.4f}      {prec:.4f}      {rec:.4f}")

# %%
# Visualize the results as heatmaps for each metric
import plotly.io as pio
# Set default renderer to notebook
pio.renderers.default = "notebook"

# Create and display accuracy heatmap
accuracy_fig = visualize_results_heatmap(results, metric_index=0, 
                                        title=f"Accuracy for '{target_category}' Probes (Layer {layer_of_interest})")
accuracy_fig.show()

# Create and display F1 score heatmap
f1_fig = visualize_results_heatmap(results, metric_index=3, 
                                 title=f"F1 Score for '{target_category}' Probes (Layer {layer_of_interest})")
f1_fig.show()

# Save the figures
accuracy_fig.write_html(f"probe_accuracy_heatmap_{target_category}_layer{layer_of_interest}.html")
f1_fig.write_html(f"probe_f1_heatmap_{target_category}_layer{layer_of_interest}.html")

# %%
# Save the full results dictionary for later analysis
import pickle
with open(f"probe_results_matrix_{target_category}_layer{layer_of_interest}.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
# Visualize trained probe predictions on a chain
if len(results) > 0:
    # Choose a well-performing probe from the results
    # Find the buffer and window with the highest accuracy
    best_acc = 0
    best_buffer = 3
    best_window = 2
    
    # for buffer in results:
    #     for window in results[buffer]:
    #         acc = results[buffer][window]['metrics'][0]  # accuracy is first metric
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_buffer = buffer
    #             best_window = window
    
    if best_buffer is not None and best_window is not None:
        print(f"Visualizing predictions from best probe: window={best_window}, buffer={best_buffer}, accuracy={best_acc:.4f}")
        probe = results[best_buffer][best_window]['probe']
        
        # Choose a chain with backtracking annotations if possible
        for i in range(min(5, len(chains))):
            if "annotated_chain" in chains[i] and "backtracking" in chains[i]["annotated_chain"]:
                print(f"Visualizing predictions on chain {i}")
                html = display_probe_predictions(
                    model, probe, chains[i], 
                    layer_of_interest=layer_of_interest,
                    n_tokens=best_window, 
                    buffer_tokens=best_buffer,
                    threshold=0.5,
                    color_by_score=True
                )
                print(f"Visualization saved to probe_predictions_{best_window}tokens_{best_buffer}buffer.html")
                break
else:
    print("No trained probes available yet. Run the experiment first.")

# %%
def print_only_positive_examples(model, chain, target_category="backtracking", buffer_tokens=5):
    """
    Print only the positive examples (tokens followed by target_category) for a window size of 1.
    
    Args:
        model: Language model with tokenizer
        chain: Dictionary containing problem and annotated_chain
        target_category: Category to predict (e.g., "backtracking")
        buffer_tokens: Number of tokens to skip between window and annotation
    """
    # Process the chain
    tokens, indices = process_chain(model, chain)
    token_texts = [model.tokenizer.decode(t) for t in tokens]
    
    # Create sequence of annotations in order of appearance
    all_annotations = []
    for category, index_tuples in indices.items():
        for start, end in index_tuples:
            all_annotations.append((category, start, end))
    
    # Sort by start index
    all_annotations.sort(key=lambda x: x[1])
    
    # Fixed window size of 1
    n_tokens = 1
    
    # List to collect positive examples
    positive_examples = []
    
    # For each annotation, check if the next one is target_category
    for i in range(len(all_annotations) - 1):
        current_cat, current_start, current_end = all_annotations[i]
        next_cat, next_start, next_end = all_annotations[i + 1]
        
        # Check if we have enough tokens for window + buffer
        window_start = current_end - n_tokens - buffer_tokens
        window_end = current_end - buffer_tokens
        
        if window_start >= 0 and next_cat == target_category:
            # Get the window token
            window_token = token_texts[window_start:window_end]
            window_text = "".join(window_token)
            
            # Add to list
            positive_examples.append((window_text, current_cat, next_cat))
    
    # Print only the positive examples
    print(f"All positive examples (token followed by {target_category}) for buffer={buffer_tokens}:")
    for window_text, current_cat, next_cat in positive_examples:
        print(f"\"{window_text}\"")

# %%
# Add a cell to run the new function
if len(chains) > 0:
    # Choose a chain with backtracking annotations if possible
    for i in range(min(5, len(chains))):
        if "annotated_chain" in chains[i] and "backtracking" in chains[i]["annotated_chain"]:
            print(f"\nChain {i} - Positive examples only:")
            buffer = -2  # Adjust this value to see different buffer sizes
            print_only_positive_examples(
                model, chains[i], "backtracking", buffer_tokens=buffer
            )
            break

# %%
def print_all_chains_positive_examples(model, chains, target_category="backtracking", buffer_tokens=5):
    """
    Analyze token frequencies from all chains in the dataset for a window size of 1.
    Print frequency statistics sorted by frequency.
    
    Args:
        model: Language model with tokenizer
        chains: List of chain dictionaries
        target_category: Category to predict (e.g., "backtracking")
        buffer_tokens: Number of tokens to skip between window and annotation
    
    Returns:
        tuple: (token_counts, all_token_examples)
    """
    # Fixed window size of 1
    n_tokens = 1
    
    # Total count of positive examples
    total_examples = 0
    
    # Dictionary to count token frequencies
    token_counts = {}
    
    # List to store all token examples
    all_token_examples = []
    
    # Process each chain
    for chain_idx, chain in enumerate(chains):
        if "annotated_chain" not in chain or target_category not in chain["annotated_chain"]:
            continue
            
        # Process the chain
        tokens, indices = process_chain(model, chain)
        token_texts = [model.tokenizer.decode(t) for t in tokens]
        
        # Create sequence of annotations in order of appearance
        all_annotations = []
        for category, index_tuples in indices.items():
            for start, end in index_tuples:
                all_annotations.append((category, start, end))
        
        # Sort by start index
        all_annotations.sort(key=lambda x: x[1])
        
        # For each annotation, check if the next one is target_category
        for i in range(len(all_annotations) - 1):
            current_cat, current_start, current_end = all_annotations[i]
            next_cat, next_start, next_end = all_annotations[i + 1]
            
            # Check if we have enough tokens for window + buffer
            window_start = current_end - n_tokens - buffer_tokens
            window_end = current_end - buffer_tokens
            
            if window_start >= 0 and next_cat == target_category:
                # Get the window token
                window_token = token_texts[window_start:window_end]
                window_text = "".join(window_token)
                
                # Add to frequency count
                if window_text in token_counts:
                    token_counts[window_text] += 1
                else:
                    token_counts[window_text] = 1
                
                # Save token example
                all_token_examples.append(window_text)
                
                total_examples += 1
    
    # Print header with totals
    print(f"Token frequency analysis for buffer={buffer_tokens} (tokens followed by {target_category}):")
    print("=" * 80)
    print(f"Total positive examples: {total_examples}")
    print(f"Unique tokens: {len(token_counts)}")
    
    # Print token frequency statistics
    print("\nToken frequencies (sorted by frequency):")
    print("-" * 60)
    print(f"{'Token':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    # Sort tokens by frequency (most frequent first)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    for token, count in sorted_tokens:
        percentage = (count / total_examples) * 100
        # Use repr to show exact token representation (e.g., show whitespace)
        token_repr = repr(token)
        # Truncate if token is too long
        if len(token_repr) > 18:
            token_repr = token_repr[:15] + "..."
        print(f"{token_repr:<20} {count:<10} {percentage:.1f}%")
    
    print("=" * 80)
    
    return token_counts, all_token_examples

# %%
# Add a cell to compare buffers -2 and -3
print("BUFFER -2 (Wait token):")
buffer_neg2_counts, _ = print_all_chains_positive_examples(model, chains, "backtracking", buffer_tokens=-2)

print("\n\nBUFFER -3 (After Wait):")
buffer_neg3_counts, _ = print_all_chains_positive_examples(model, chains, "backtracking", buffer_tokens=-3)

# %%
