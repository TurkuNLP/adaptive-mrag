import numpy as np
import torch
import json
import pickle
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# === Model Setup ===
model_name = "Salesforce/SFR-Embedding-Mistral"
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

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
    return last_token_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy()


# === Load Dataset ===
dataset = []
with open("data/classified_topics.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

dataset = dataset[:1000]  # Adjust if needed
for doc in dataset:
    parts = doc["topic"].strip().rstrip('.').split(',')
    doc["topic"] = ','.join(parts[:3]).strip()

# === Precompute Topic Embeddings (pooled vector) ===
unique_topics = sorted(set(doc["topic"] for doc in dataset))
topic_embeddings = {}

for topic in unique_topics:
    topic_embeddings[topic] = get_embeddings(topic)  # shape: (1, hidden_size)

# === Compute Document Embeddings and Compare to Topics ===
output_path_base = "data/FineWeb-embedd/"
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

    doc_embedding = all_embd.mean(axis=0, keepdims=True)  # shape: (1, hidden_size)

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
        doc_embedding = all_embd.mean(axis=0, keepdims=True)
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

