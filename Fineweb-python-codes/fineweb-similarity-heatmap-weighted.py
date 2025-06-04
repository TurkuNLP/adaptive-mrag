"""
@mtebad

This file tries to prune some heads and create a weighted embedding for doc representation. Embeddings from stage 3.
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

def sort_docs_by_topic_similarity(cosine_scores_raw, min_docs_per_topic=10):

    topic_docs = defaultdict(list)

    for doc_similarity, head_similarities, topic in cosine_scores_raw:
        topic_docs[topic].append((doc_similarity, head_similarities))

    sorted_cosine_scores = []
    topic_order = []
    topic_boundaries = []
    current_index = 0

    for topic, docs in topic_docs.items():
        if len(docs) < min_docs_per_topic:
            continue
        # Sort by doc-level similarity
        sorted_docs = sorted(docs, key=lambda x: x[0], reverse=True)

        for sim, head_sim in sorted_docs:
            sorted_cosine_scores.append(np.concatenate([[sim], head_sim]))
            topic_order.append(topic)

        current_index += len(sorted_docs)
        topic_boundaries.append(current_index)

    cosine_scores_sorted = np.array(sorted_cosine_scores).T  # shape: (num_heads+1, total_docs)

    return cosine_scores_sorted, topic_order, topic_boundaries

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


    # Reshape into (num_heads, n_samples, head_size)
    transposed = np.transpose(split_embeddings, (1, 0, 2))  # shape: (num_heads, n_samples, head_size)
    head_embeddings = transposed.mean(axis=1)  # shape: (num_heads, head_size)
    head_magnitudes = np.linalg.norm(head_embeddings, axis=1)  # shape: (num_heads,)
    weighted_doc_embedding = np.sum(head_embeddings * head_magnitudes[:, np.newaxis], axis=0, keepdims=True)  # shape: (1, head_size)

    final_embeddings = np.concatenate([doc_embedding, weighted_doc_embedding], axis=0)

    exclude_heads = {0, 7, 8, 17, 18, 21}
    split_embeddings[:, list(exclude_heads), :] = 0

    # Reshape into (num_heads, n_samples, head_size)
    transposed = np.transpose(split_embeddings, (1, 0, 2))  # shape: (num_heads, n_samples, head_size)

    # Mean per head across tokens
    head_embeddings = transposed.mean(axis=1)  # shape: (num_heads, head_size)

    # Compute L2 norm for each head
    head_magnitudes = np.linalg.norm(head_embeddings, axis=1)  # shape: (num_heads,)

    # Optional: Normalize weights to sum to 1
    #weights = head_magnitudes / (np.sum(head_magnitudes) + 1e-10)  # shape: (num_heads,)

    # Create weighted document embedding
    weighted_doc_embedding = np.sum(head_embeddings * head_magnitudes[:, np.newaxis], axis=0, keepdims=True)  # shape: (1, head_size)

    final_embeddings = np.concatenate([final_embeddings, weighted_doc_embedding], axis=0)

    # Compute Cosine Similarity (Title vs. Each Head)
    similarities = cosine_similarity(title_embedding, final_embeddings)[0]  # Shape: (n_samples * num_heads,)

    # Inside the loop instead of appending full similarities directly:
    doc_similarity = similarities[0]
    head_similarities = similarities[1:]
    cosine_scores_raw.append((doc_similarity, head_similarities, doc["topic"]))

    #cosine_scores.append(similarities)

# Convert to NumPy Array for Visualization
#cosine_scores = np.array(cosine_scores).T  # Shape: (num_heads, num_docs)

# Count how many docs per topic (in order of appearance)
topic_order = [doc["topic"] for doc in dataset[:total_docs]]
topic_counts = Counter(topic_order)
topic_boundaries = []
cosine_scores, topic_order, topic_boundaries = sort_docs_by_topic_similarity(cosine_scores_raw)

# Preserve order and build x_labels
x_ticks = []
tick_position = 0

for topic, count in topic_counts.items():
    if count < 10:
        break
    x_ticks.append((tick_position + count // 2, f"{topic} ({count})"))
    tick_position += count
    topic_boundaries.append(tick_position)

yticklabels = ["full embd"] + ["weighted"] + ["weighted with 0"]

# Plot heatmap
plt.figure(figsize=(20, 4)) # cmap="Reds"
sns.heatmap(cosine_scores, cmap="coolwarm", yticklabels=yticklabels, vmin=-0.35)

# Custom x-ticks in the middle of each topic group
positions, labels = zip(*x_ticks)
plt.xticks(positions, labels, rotation=90, fontsize=8)

# Draw vertical lines at topic boundaries
for boundary in topic_boundaries[:-1]:  # skip last one to avoid line at far right edge
    plt.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5)

plt.xlabel("Documents (Grouped by Category)")
plt.ylabel("Vector Embedding Heads")
plt.title("Heatmap of Cosine Similarity Between Title and Text Attention Heads")

# Show plot
plt.savefig("fineweb_heatmap_topics_full_weighted.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()