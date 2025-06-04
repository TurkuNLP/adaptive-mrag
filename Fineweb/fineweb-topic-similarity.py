import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

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
"""
# === Load and Clean Topics ===
with open("data/classified_topics.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]"""

# Extract cleaned unique topics
cleaned_topics = set()

with open("images-txts/FineWeb/categories-narrowed.txt", "r", encoding="utf-8") as f:
    for line in f:
        cleaned_topics.add(line.strip())
"""
for doc in dataset:
    parts = doc["topic"].rstrip('.').strip().split(',')
    cleaned = ','.join(parts[:3]).strip()
    cleaned_topics.add(cleaned)"""
    
topic_names = sorted(cleaned_topics)

# === Embed Topics ===
print(f"Embedding {len(topic_names)} unique topics...")
topic_embeddings = {}
for topic in topic_names:
    topic_embeddings[topic] = get_embeddings(topic)

# === Create Similarity Matrix ===
topic_matrix = np.vstack([topic_embeddings[t] for t in topic_names])
similarity_matrix = cosine_similarity(topic_matrix)

# === Plot Heatmap ===
plt.figure(figsize=(20,16))
sns.heatmap(similarity_matrix, xticklabels=topic_names, yticklabels=topic_names,
            cmap="coolwarm", square=True, annot=False, fmt=".2f", cbar=True)

plt.title("Cosine Similarity Between Topic Embeddings", fontsize=16)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("topic_similarity_heatmap.png", dpi=300)
plt.show()
