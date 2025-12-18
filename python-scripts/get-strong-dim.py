import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import re
import pickle
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict


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
    # use outputs.last_hidden_state for stage 3
    return outputs.last_hidden_state, inputs.attention_mask #model.layers[-1].self_attn.o_proj.cached_heads, inputs.attention_mask  # (batch_size, seq_length, hidden_size) 

# Model and Tokenizer Setup
model_name = "Salesforce/SFR-Embedding-Mistral"  # Change if needed
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, dtype=torch.float16).cuda()

# Load Dataset
with open("/scratch/project_2000539/maryam/adaptive-mrag/data/classified_topics_narrowed.jsonl", "r", encoding="utf-8") as f:
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

# Process Documents
num_heads = 32  # Define the number of attention heads
total_docs = 960  # Total documents

cosine_scores_raw = []
cosine_scores = []
output_path_base = "data/Fineweb/FineWeb-embedd-stage3_10000/"

# Choose histogram bins over cosine range [-1, 1]
bins = np.linspace(-0.5, 0.5, 101)  # 100 bins
counts_similarities = np.zeros(len(bins) - 1, dtype=np.int64)
counts_doc_similarity = np.zeros(len(bins) - 1, dtype=np.int64)

topic_docs = defaultdict(list)
for doc in dataset[:total_docs]:  # Assuming dataset is loaded from JSONL

    title = doc["topic"]  # Using text as the title since JSONL does not have a separate title field

    # Get Title Embeddings (Mean Pooling)
    title_hidden, title_mask = get_embeddings(title)
    title_full_embedding = last_token_pool(title_hidden, title_mask ).cpu().numpy()

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

    all_embd_norm = all_embd[0] / np.linalg.norm(all_embd[0])

    x = title_full_embedding.astype(np.float32, copy=False)  # promote to fp32 for safe math
    norm = np.linalg.norm(x, axis=1, keepdims=True)          # (1, 1)
    norm = np.maximum(norm, 1e-12)                           # avoid divide-by-zero

    title_embd_norm = x / norm                               # (1, 4096) float32

    topic_docs[title].append(all_embd_norm * title_embd_norm)

for topic, dot_products in topic_docs.items():

    print(topic)

    dot_products = np.stack(dot_products, axis=0)       # shape (N, 4096)
    dot_products = np.mean(dot_products, axis=0, keepdims=True)[0]  # shape (1, 4096)

    dot_products = dot_products.flatten()
    print("dot_products",np.max(dot_products), np.min(dot_products))

    sorted_indices = np.argsort(dot_products)

    top_5 = sorted_indices[-5:][::-1]
    min_5 = sorted_indices[:5]

    print("Top 5 values:")
    for i in top_5:
        print(f"Index: {i}, Value: {dot_products[i]:.6f}")

    print("\nBottom 5 values:")
    for i in min_5:
        print(f"Index: {i}, Value: {dot_products[i]:.6f}")
