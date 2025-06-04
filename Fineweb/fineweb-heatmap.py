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
from collections import Counter
from fineweb_topic_groups import topic_cluster_map

# Model and Tokenizer Setup
model_name = "Salesforce/SFR-Embedding-Mistral"  # Change if needed
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

# Load Dataset
with open("data/classified_topics.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]
dataset = dataset[:2000]
for doc in dataset:
    doc["topic"] = doc["topic"].strip().rstrip(',')

# Count topic occurrences
topic_counts = Counter(doc["topic"] for doc in dataset)

# Sort dataset by most repeated topics
dataset.sort(key=lambda x: topic_counts[x["topic"]], reverse=True)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Function for Mean Pooling
def mean_pooling(hidden_states, attention_mask):
    """Applies mean pooling across valid tokens (ignoring padding)."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

# Function to compute embeddings
def get_embeddings(text):
    """Extracts embeddings from model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state, inputs.attention_mask  # (batch_size, seq_length, hidden_size)

# Process Documents
num_heads = 32  # Define the number of attention heads
num_categories = 4  # Number of categories
docs_per_category = 250  # Documents per category
total_docs = num_categories * docs_per_category  # Total documents

cosine_scores = []
output_path_base = "data/FineWeb-embedd/"

file_counts = 0
for doc in dataset[:total_docs]:  # Assuming dataset is loaded from JSONL

    title = doc["topic"]  # Using text as the title since JSONL does not have a separate title field
    file_counts += 1

    # Get Title Embeddings (Mean Pooling)
    title_hidden, title_mask = get_embeddings(title)
    title_embedding = last_token_pool(title_hidden, title_mask ).cpu().numpy()

    num_heads = 32
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
                #pooled_embeddings = mean_pooling(padded_embeddings)
                all_embd.append(padded_embeddings)
            
            except EOFError:
                # End of file reached
                break

    all_embd = np.concatenate(all_embd, axis=0)

    n_samples = all_embd.shape[0]
    head_size = all_embd.shape[1] // num_heads
    split_embeddings = all_embd.reshape(n_samples, num_heads, head_size)
    flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)

    #print("title_embedding", len(title_embedding), len(title_embedding[0]))
    #print("all_embd, 2: ", len(flat_embeddings), len(flat_embeddings[0]))

    # Compute Cosine Similarity (Title vs. Each Head)
    #similarities = [cosine_similarity(title_embedding, flat_embeddings[i].reshape(1, -1))[0, 0] for i in range(num_heads)]
    similarities = cosine_similarity(title_embedding, flat_embeddings)[0]  # Shape: (n_samples * num_heads,)
    
    cosine_scores.append(similarities)

# Convert to NumPy Array for Visualization
cosine_scores = np.array(cosine_scores).T  # Shape: (num_heads, num_docs)

# Normalize each head's similarities to [0, 1]
normalized_scores = []

for head_scores in cosine_scores:
    min_val = np.min(head_scores)
    max_val = np.max(head_scores)
    norm = (head_scores - min_val) / (max_val - min_val + 1e-9)  # Avoid division by zero
    normalized_scores.append(norm)

# Convert back to NumPy array
#cosine_scores = np.array(normalized_scores)  # Still (num_heads, num_docs)

#cosine_scores_normalized = (cosine_scores - cosine_scores.min(axis=1, keepdims=True)) / \
#                           (cosine_scores.max(axis=1, keepdims=True) - cosine_scores.min(axis=1, keepdims=True) + 1e-9)

# Count how many docs per topic (in order of appearance)
#topic_order = [topic_cluster_map.get(doc["topic"], "Other") for doc in dataset[:total_docs]]
topic_order = [doc["topic"] for doc in dataset[:total_docs]]
topic_counts = Counter(topic_order)
topic_boundaries = []

# Preserve order and build x_labels
x_ticks = []
tick_position = 0

for topic, count in topic_counts.items():
    if count < 10:
        break
    x_ticks.append((tick_position + count // 2, f"{topic} ({count})"))
    tick_position += count
    topic_boundaries.append(tick_position)

# Plot heatmap
plt.figure(figsize=(20, 10)) # cmap="Reds"
sns.heatmap(cosine_scores, cmap="coolwarm", xticklabels=docs_per_category, yticklabels=[f"Head {i+1}" for i in range(num_heads)])

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
plt.savefig("fineweb_heatmap_topics.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()