"""
@mtebad

This file checks how much the embedding model and the annotator model correlate.
"""

import numpy as np
import torch
import json
import pickle
import re
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# === Model Setup ===
model_name = "Salesforce/SFR-Embedding-Mistral"
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()
#model.layers[-1].self_attn.o_proj = torch.nn.Identity()

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = last_token_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy()
    head_size = pooled.shape[1] // 32  # 4096 / 32 = 128
    split = pooled.reshape(32, head_size)  # shape: (32, 128)
    mean_emb = split.mean(axis=0, keepdims=True)  # shape: (1, 128)

    return mean_emb

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

# === Load Dataset ===
dataset = []
with open("data/classified_topics_narrowed.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

for doc in dataset:
    doc["topic"] = clean_label(doc["topic"])

# === Precompute Topic Embeddings (pooled vector) ===
unique_topics = sorted(set(doc["topic"] for doc in dataset))

# Keep only up to 120 docs per topic
max_per_topic = 120
topic_seen = defaultdict(int)
dataset = [doc for doc in dataset if topic_seen[doc["topic"]] <
           max_per_topic and not topic_seen.__setitem__(doc["topic"], topic_seen[doc["topic"]] + 1)]

topic_embeddings = {}
for topic in unique_topics:
    topic_embeddings[topic] = get_embeddings(topic)  # shape: (1, hidden_size)

# === Compute Document Embeddings and Compare to Topics ===
output_path_base = "data/FineWeb-embedd/"
num_heads = 32
most_similar_topics = []
max_similarities = []

valid_indices = []

for idx, doc in enumerate(dataset):
    doc_id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base + doc_id + ".pkl"

    try:
        with open(output_path, "rb") as f:
            all_embd = []
            while True:
                try:
                    all_embd.append(pickle.load(f))
                except EOFError:
                    break
        all_embd = np.concatenate(all_embd, axis=0)
    except FileNotFoundError:
        print(f"Missing file: {output_path}")
        most_similar_topics.append(None)
        max_similarities.append(None)
        continue

    n_samples = all_embd.shape[0]
    head_size = all_embd.shape[1] // num_heads
    split_embeddings = all_embd.reshape(n_samples, num_heads, head_size)

    flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)
    doc_embedding = flat_embeddings.mean(axis=0, keepdims=True)

    exclude_heads = {0, 7, 8, 17, 18, 21}
    #split_embeddings[:, list(exclude_heads), :] = 0

    # Reshape into (num_heads, n_samples, head_size)
    transposed = np.transpose(split_embeddings, (1, 0, 2))  # shape: (num_heads, n_samples, head_size)

    # Mean per head across tokens
    head_embeddings = transposed.mean(axis=1)  # shape: (num_heads, head_size)

    # Compute L2 norm for each head
    head_magnitudes = np.linalg.norm(head_embeddings, axis=1)  # shape: (num_heads,)

    # Create weighted document embedding
    weighted_doc_embedding = np.sum(head_embeddings * head_magnitudes[:, np.newaxis], axis=0, keepdims=True)  # shape: (1, head_size)

    #doc_embedding = all_embd.mean(axis=0, keepdims=True)  # shape: (1, hidden_size)

    sims = {
        topic: cosine_similarity(topic_emb, doc_embedding)[0][0]
        for topic, topic_emb in topic_embeddings.items()
    }

    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    top1_topic, top1_score = sorted_sims[0]

    most_similar_topics.append(top1_topic)
    max_similarities.append(top1_score)
    valid_indices.append(idx)

# === Top-1 Accuracy ===
match_count = sum(
    1 for i in valid_indices
    if dataset[i]["topic"] == most_similar_topics[i]
)
match_ratio = match_count / len(valid_indices)

# === Top-5 and 10 Accuracy ===
top5_match_count = 0
top10_match_count = 0

top5_but_not_top1_examples = []
top10_but_not_top5_examples = []
not_in_top10_examples = []

for idx in valid_indices:
    doc = dataset[idx]
    doc_topic = doc["topic"]
    doc_id = doc["id"]
    embedding_path = output_path_base + ''.join(re.findall(r'[\d-]+', doc_id)) + ".pkl"

    try:
        with open(embedding_path, "rb") as f:
            all_embd = []
            while True:
                try:
                    all_embd.append(pickle.load(f))
                except EOFError:
                    break
        all_embd = np.concatenate(all_embd, axis=0)
        n_samples = all_embd.shape[0]
        head_size = all_embd.shape[1] // num_heads
        split_embeddings = all_embd.reshape(n_samples, num_heads, head_size)

        flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)
        doc_embedding = flat_embeddings.mean(axis=0, keepdims=True)

        exclude_heads = {0, 7, 8, 17, 18, 21}
        #split_embeddings[:, list(exclude_heads), :] = 0

        transposed = np.transpose(split_embeddings, (1, 0, 2))  # shape: (num_heads, n_samples, head_size)
        head_embeddings = transposed.mean(axis=1)  # shape: (num_heads, head_size)
        head_magnitudes = np.linalg.norm(head_embeddings, axis=1)  # shape: (num_heads,)
        weighted_doc_embedding = np.sum(head_embeddings * head_magnitudes[:, np.newaxis], axis=0, keepdims=True)  # shape: (1, head_size)

    except FileNotFoundError:
        continue

    sims = {
        topic: cosine_similarity(topic_emb, doc_embedding)[0][0]
        for topic, topic_emb in topic_embeddings.items()
    }

    sorted_topics = sorted(sims.items(), key=lambda x: -x[1])
    top5_topics = [topic for topic, _ in sorted_topics[:5]]
    top10_topics = [topic for topic, _ in sorted_topics[:10]]

    if doc_topic in top10_topics:
        top10_match_count += 1

        if doc_topic in top5_topics:
            top5_match_count += 1
            if doc_topic != top5_topics[0] and len(top5_but_not_top1_examples) < 5:
                top5_but_not_top1_examples.append({
                    "doc_id": doc_id,
                    "true_topic": doc_topic,
                    "top5": top5_topics
                })
        elif len(top10_but_not_top5_examples) < 5:
            top10_but_not_top5_examples.append({
                "doc_id": doc_id,
                "true_topic": doc_topic,
                "top10": top10_topics
            })
    elif len(not_in_top10_examples) < 5:
        not_in_top10_examples.append({
            "doc_id": doc_id,
            "true_topic": doc_topic,
            "top10": top10_topics
        })

# === Accuracy Reporting
top5_ratio = top5_match_count / len(valid_indices)
top10_ratio = top10_match_count / len(valid_indices)

print(f"\n✅ Top-1 match accuracy: {match_ratio:.2%} of documents")
print(f"✅ Top-5 match accuracy: {top5_ratio:.2%} of documents")
print(f"✅ Top-10 match accuracy: {top10_ratio:.2%} of documents")

# === Examples Reporting
print("\n📌 Examples where the true topic was in top-5 but not top-1:")
for ex in top5_but_not_top1_examples:
    print(f"Doc ID: {ex['doc_id']}")
    print(f"True Topic: {ex['true_topic']}")
    print(f"Top-5 Predicted Topics: {ex['top5']}")
    print("—" * 40)

print("\n📌 Examples where the true topic was in top-10 but not top-5:")
for ex in top10_but_not_top5_examples:
    print(f"Doc ID: {ex['doc_id']}")
    print(f"True Topic: {ex['true_topic']}")
    print(f"Top-10 Predicted Topics: {ex['top10']}")
    print("—" * 40)

print("\n📌 Examples where the true topic was NOT in top-10 at all:")
for ex in not_in_top10_examples:
    print(f"Doc ID: {ex['doc_id']}")
    print(f"True Topic: {ex['true_topic']}")
    print(f"Top-10 Predicted Topics: {ex['top10']}")
    print("—" * 40)