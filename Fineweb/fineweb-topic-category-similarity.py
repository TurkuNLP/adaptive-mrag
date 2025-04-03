import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# === MODEL SETUP ===
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
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return last_token_pool(outputs.last_hidden_state, inputs["attention_mask"]).cpu().numpy()

def clean_topic_string(text):
    text = re.sub(r'^\d+\.\s*', '', text)
    text = text.strip().rstrip('.,')
    parts = text.split(',')
    return ','.join(parts[:3]).strip()

# === LOAD CATEGORY STRUCTURE ===
def load_super_and_sub(filepath):
    super_to_sub = {}
    current_super = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+\.', line):
                current_super = re.sub(r'^\d+\.\s*', '', line)
                super_to_sub[current_super] = []
            else:
                sub = clean_topic_string(line)
                super_to_sub[current_super].append(sub)
    return super_to_sub

category_file = "categories-general.txt"
super_to_sub = load_super_and_sub(category_file)

# === EMBED SUPER-TOPICS & SUB-TOPICS ===
print("Embedding categories and topics...")
super_embeddings = {sup: get_embeddings(sup) for sup in super_to_sub}
all_sub_topics = sorted({sub for subs in super_to_sub.values() for sub in subs})
sub_embeddings = {sub: get_embeddings(sub) for sub in all_sub_topics}

# === COMPUTE SIMILARITY MATRIX ===
super_names = list(super_to_sub.keys())
sub_names = list(sub_embeddings.keys())

super_matrix = np.vstack([super_embeddings[s] for s in super_names])      # (num_super, dim)
sub_matrix = np.vstack([sub_embeddings[s] for s in sub_names])            # (num_sub, dim)

similarity_matrix = cosine_similarity(super_matrix, sub_matrix)           # (num_super, num_sub)

# === PLOT HEATMAP ===
plt.figure(figsize=(len(sub_names) * 0.35, len(super_names) * 0.8))
sns.heatmap(similarity_matrix, cmap="coolwarm",
            xticklabels=sub_names,
            yticklabels=super_names,
            square=False, annot=False)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=9)
plt.title("Cosine Similarity Between Categories and Topics")
plt.tight_layout()
plt.savefig("category_topic_similarity_heatmap.png", dpi=300)
plt.show()
