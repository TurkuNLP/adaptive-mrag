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
all_doc_embeddings = []  # Will store (embedding, topic)

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
                all_embd.append(padded_embeddings)
            
            except EOFError:
                # End of file reached
                break

    all_embd = np.concatenate(all_embd, axis=0)

    n_samples = all_embd.shape[0]
    head_size = all_embd.shape[1] // num_heads
    split_embeddings = all_embd.reshape(n_samples, num_heads, head_size)
    flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)

    doc_embedding = flat_embeddings.mean(axis=0)  # or use title_embedding.squeeze()
    topic = doc["topic"]
    all_doc_embeddings.append((doc_embedding, topic))

# Convert list to arrays
embeddings = np.array([e for e, _ in all_doc_embeddings])
topics = [t for _, t in all_doc_embeddings]

same_topic_sims = []
diff_topic_sims = []

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
        if topics[i] == topics[j]:
            same_topic_sims.append(sim)
        else:
            diff_topic_sims.append(sim)

print(f"Average similarity (same topic): {np.mean(same_topic_sims):.4f}")
print(f"Average similarity (different topic): {np.mean(diff_topic_sims):.4f}")

plt.hist(same_topic_sims, bins=30, alpha=0.6, label="Same Topic")
plt.hist(diff_topic_sims, bins=30, alpha=0.6, label="Different Topic")
plt.title("Cosine Similarity Distribution")
plt.xlabel("Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("fineweb_heatmap_topics.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
