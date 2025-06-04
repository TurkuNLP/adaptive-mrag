import json
import re
import numpy as np
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

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

def load_super_topics(filepath):
    """Parses categories-general.txt into sub-topic → super-topic mapping"""
    topic_to_super = {}
    with open(filepath, "r", encoding="utf-8") as f:
        current_super = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+\.', line):
                current_super = line.split(".", 1)[1].strip()
            else:
                cleaned_topic = clean_topic_string(line)
                topic_to_super[cleaned_topic] = current_super
    return topic_to_super

def get_super(topic, mapping):
    return mapping.get(topic, "Unknown")

def clean_topic_string(text):
    # Remove leading numbers and dots (e.g. "35. ")
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove trailing commas or periods
    text = text.strip().rstrip('.,')
    # Keep only first 3 comma-separated parts
    parts = text.split(',')
    return ','.join(parts[:3]).strip()


# === Load Data ===
with open("data/classified_topics.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

dataset = dataset[:1000]  # Adjust as needed

# Clean topics: keep first 3 comma-separated elements
for doc in dataset:
    doc["topic"] = clean_topic_string(doc["topic"])

# === Load Super-topic Mappings ===
super_topic_file = "categories-general.txt"
sub_to_super = load_super_topics(super_topic_file)


used_supers = set(get_super(doc["topic"], sub_to_super) for doc in dataset)
print(f"Unique super-topics in dataset: {len(used_supers)}")

# === Get All Unique Topics and Embed ===
unique_topics = sorted(set(doc["topic"] for doc in dataset))
topic_embeddings = {}

print("Embedding unique topics...")
for topic in unique_topics:
    topic_embeddings[topic] = get_embeddings(topic)

# === Evaluate Documents ===
output_path_base = "data/FineWeb-embedd/"
valid_indices = []
top1_super_matches = 0
top5_super_matches = 0
examples_top5_but_not_top1 = []
examples_not_in_top5 = []
total_valid = 0

print("Evaluating document-topic similarities...")

for idx, doc in enumerate(dataset):
    true_topic = doc["topic"]
    true_super = get_super(true_topic, sub_to_super)
    doc_id = ''.join(re.findall(r'[\d-]+', doc["id"]))
    embedding_path = output_path_base + doc_id + ".pkl"

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
        print(f"Missing: {embedding_path}")
        continue

    # Compute cosine similarities
    sims = {
        topic: cosine_similarity(topic_emb, doc_embedding)[0][0]
        for topic, topic_emb in topic_embeddings.items()
        if topic != true_topic
    }

    sorted_topics = sorted(sims.items(), key=lambda x: -x[1])
    top5 = [t for t, _ in sorted_topics[:5]]
    top1 = top5[0]

    top5_supers = [get_super(t, sub_to_super) for t in top5]
    top1_super = get_super(top1, sub_to_super)

    total_valid += 1
    if true_super == top1_super:
        top1_super_matches += 1
    if true_super in top5_supers:
        top5_super_matches += 1
        if true_super != top1_super and len(examples_top5_but_not_top1) < 5:
            examples_top5_but_not_top1.append({
                "doc_id": doc_id,
                "true_super": true_super,
                "top5_super_topics": top5_supers
            })
    elif len(examples_not_in_top5) < 5:
        examples_not_in_top5.append({
            "doc_id": doc_id,
            "true_super": true_super,
            "top5_super_topics": top5_supers
        })

# === Results ===
print(f"\n✅ Super-topic Top-1 accuracy: {top1_super_matches / total_valid:.2%}")
print(f"✅ Super-topic Top-5 accuracy: {top5_super_matches / total_valid:.2%}")

print("\n📌 Examples where true super-topic was in top-5 but not top-1:")
for ex in examples_top5_but_not_top1:
    print(f"Doc ID: {ex['doc_id']}")
    print(f"True Super-topic: {ex['true_super']}")
    print(f"Top-5 Super-topics: {ex['top5_super_topics']}")
    print("—" * 40)

print("\n📌 Examples where true super-topic was NOT in top-5:")
for ex in examples_not_in_top5:
    print(f"Doc ID: {ex['doc_id']}")
    print(f"True Super-topic: {ex['true_super']}")
    print(f"Top-5 Super-topics: {ex['top5_super_topics']}")
    print("—" * 40)

print("\n📌 Raw topic predictions for manual sanity check:")
for i, (topic, sim) in enumerate(sorted_topics[:5]):
    print(f"  {i+1}. {topic} ({sim:.4f}) → Super-topic: {get_super(topic, sub_to_super)}")
