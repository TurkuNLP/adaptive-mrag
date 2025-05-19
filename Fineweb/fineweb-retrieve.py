import numpy as np
import json
import re
import pickle

def clean_label(label):
    # 1. Replace any comma between number and text with a dot
    label = re.sub(r'^(\d+)[,.]', r'\1.', label.strip())

    # 2. Remove everything after the presence of a second number
    #    e.g., "1. Technology 2. Gaming" -> "1. Technology"
    match = re.match(r'^(\d+\.\s*[^0-9]*)', label)
    if match:
        label = match.group(1).strip()

    # 3. Remove any trailing punctuation like comma or period
    label = re.sub(r'[,.]\s*$', '', label)

    # 4. Make sure only the first dot remains (fix cases like 1.. Technology)
    label = re.sub(r'^(\d+)\.+', r'\1.', label)

    # 5. Remove extra whitespace around the dot and category
    label = re.sub(r'\s*\.\s*', '.', label)
    label = re.sub(r'\s+', ' ', label)

    return label.strip()

# Load Dataset
with open("data/classified_topics_narrowed.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

dataset = [doc for doc in dataset if doc["topic"] == "14.Politics, Conflict, International"]

# Process Documents
num_heads = 32  # Define the number of attention heads
total_docs = 960  # Total documents

cosine_scores_raw = []
cosine_scores = []
output_path_base = "data/FineWeb-embedd/"

file_counts = 0
for doc in dataset[:total_docs]:  # Assuming dataset is loaded from JSONL

    title = doc["topic"]  # Using text as the title since JSONL does not have a separate title field
    file_counts += 1

    id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base + id + ".pkl"

    all_embd = []

    with open(output_path, "rb") as f:
        while True:
            try:
                # Load one batch at a time
                padded_embeddings = pickle.load(f)
                
                # Apply mean pooling to the batch
                all_embd.append(padded_embeddings)
            
            except EOFError:
                # End of file reached
                break

    all_embd = np.concatenate(all_embd, axis=0)

    n_samples = all_embd.shape[0]
    head_size = all_embd.shape[1] // num_heads
    split_embeddings = all_embd.reshape(n_samples, num_heads, head_size)

    #flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)

    # Step 1: Compute mean embedding per head across all samples
    head_embeddings = split_embeddings.mean(axis=0)  # Shape: (num_heads, head_size)

    # Step 2: Compute L2 magnitude (norm) of each head embedding
    head_magnitudes = np.linalg.norm(head_embeddings, axis=1)  # Shape: (num_heads,)

    # Step 3: Normalize magnitudes to use as weights (optional but common)
    # This keeps scale consistent. If you don't want normalization, skip this.
    weights = head_magnitudes / (np.linalg.norm(head_magnitudes) + 1e-10)

    # Step 4: Apply weights to head embeddings to compute weighted document embedding
    weighted_embedding = np.sum(head_embeddings * weights[:, np.newaxis], axis=0)  # Shape: (head_size,)
