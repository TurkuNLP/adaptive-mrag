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

def capture_heads(module, input, output):
    out = output[0]
    # this is only needed if "output is used"
    if out.dim() == 2:
        out = out.unsqueeze(0)  # convert [seq_len, hidden] → [1, seq_len, hidden] for output only
    elif out.dim() != 3:
        raise ValueError(f"Unexpected shape: {out.shape}")
    module.cached_heads = out 

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
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()
# delete this for stage3
#hook = model.layers[-1].self_attn.o_proj.register_forward_hook(capture_heads)

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

file_counts = 0
for doc in dataset[:total_docs]:  # Assuming dataset is loaded from JSONL

    title = doc["topic"]  # Using text as the title since JSONL does not have a separate title field
    file_counts += 1

    # Get Title Embeddings (Mean Pooling)
    title_hidden, title_mask = get_embeddings(title)
    title_full_embedding = last_token_pool(title_hidden, title_mask ).cpu().numpy()

    n_samples = title_full_embedding.shape[0]
    head_size = title_full_embedding.shape[1] // num_heads

    title_embedding = title_full_embedding.reshape(num_heads, head_size).mean(axis=0, keepdims=True)  # Shape: (1, 128)


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

    similarities = cosine_similarity(title_embedding, flat_embeddings)[0]  # Shape: (n_samples * num_heads,)
    doc_similarity = cosine_similarity(title_full_embedding, all_embd)[0]

    counts_similarities += np.histogram(similarities, bins=bins)[0]
    counts_doc_similarity += np.histogram(doc_similarity, bins=bins)[0]

# Midpoints for plotting bars/curves
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Plot frequency (counts). If you prefer density, divide by counts.sum() and bin width.
plt.figure()
plt.bar(bin_centers, counts_similarities, width=(bins[1]-bins[0]))
plt.title("Frequency distribution: title vs. per-head similarities")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("freq_per_head_similarities.png", dpi=150)

plt.figure()
plt.bar(bin_centers, counts_doc_similarity, width=(bins[1]-bins[0]))
plt.title("Frequency distribution: title vs. per-row (all_embd) similarities")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("freq_doc_similarity.png", dpi=150)

# Also save raw counts to disk for later analysis
np.savez(
    "similarity_histograms.npz",
    bins=bins,
    counts_similarities=counts_similarities,
    counts_doc_similarity=counts_doc_similarity
)