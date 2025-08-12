"""
@mtebad

This file prints the most similar and the most dissimilar docs per head-topic.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import re
import pickle
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

# Model and Tokenizer Setup
model_name = "Salesforce/SFR-Embedding-Mistral"  # Change if needed
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

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

for doc in dataset:
    doc["topic"] = clean_label(doc["topic"])

# Count topic occurrences
topic_counts = Counter(doc["topic"] for doc in dataset)

# Sort dataset by most repeated topics
dataset.sort(key=lambda x: topic_counts[x["topic"]], reverse=True)

# Keep only up to 120 docs per topic
max_per_topic = 120
topic_seen = defaultdict(int)
dataset = [doc for doc in dataset if topic_seen[doc["topic"]] <
           max_per_topic and not topic_seen.__setitem__(doc["topic"], topic_seen[doc["topic"]] + 1)]

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Function to compute embeddings
def get_embeddings(text):
    """Extracts embeddings from model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state, inputs.attention_mask  # (batch_size, seq_length, hidden_size)

# Process Documents
num_heads = 32  # Define the number of attention heads
total_docs = 960  # Total documents

cosine_scores = []
output_path_base = "data/FineWeb-embedd/"

file_counts = 0
for doc in dataset[:total_docs]:  # Assuming dataset is loaded from JSONL

    title = doc["topic"]  # Using text as the title since JSONL does not have a separate title field
    file_counts += 1

    # Get Title Embeddings (Mean Pooling)
    title_hidden, title_mask = get_embeddings(title)
    title_embedding = last_token_pool(title_hidden, title_mask ).cpu().numpy()

    n_samples = title_embedding.shape[0]
    head_size = title_embedding.shape[1] // num_heads

    title_embedding = title_embedding.reshape(num_heads, head_size).mean(axis=0, keepdims=True)  # Shape: (1, 128)


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

    flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)
    doc_embedding = flat_embeddings.mean(axis=0, keepdims=True)

    flat_embeddings = np.concatenate([doc_embedding, flat_embeddings], axis=0)
    # Compute Cosine Similarity (Title vs. Each Head)
    similarities = cosine_similarity(title_embedding, flat_embeddings)[0]  # Shape: (n_samples * num_heads,)
    
    cosine_scores.append(similarities)

# Convert to NumPy Array for Visualization
cosine_scores = np.array(cosine_scores).T  # Shape: (num_heads, num_docs)

# Target topic
target_topic = "14.Politics, Conflict, International"

# Find indices of documents belonging to the target topic
target_indices = [i for i, doc in enumerate(dataset[:total_docs]) if doc["topic"] == target_topic]

# Helper to get top/bottom ids for a given head
def print_extremes_for_head(head_index):
    print(f"\nHead {head_index} — Top & Bottom 5 cosine similarities for topic: '{target_topic}'")

    # Get cosine scores for that head and only for the target topic indices
    scores = cosine_scores[head_index, target_indices]

    # Sort by similarity
    sorted_indices = np.argsort(scores)

    # Get original dataset indices from sorted positions
    top_5 = [target_indices[i] for i in sorted_indices[-5:][::-1]]  # Highest
    bottom_5 = [target_indices[i] for i in sorted_indices[:5]]      # Lowest

    print("Top 5 (highest similarity):")
    for idx in top_5:
        print(f"Doc ID: {dataset[idx]['id']}, Similarity: {cosine_scores[head_index, idx]:.4f}")

    print("\nBottom 5 (lowest similarity):")
    for idx in bottom_5:
        print(f"Doc ID: {dataset[idx]['id']}, Similarity: {cosine_scores[head_index, idx]:.4f}")

# Print for head 10 and 19
print_extremes_for_head(18)