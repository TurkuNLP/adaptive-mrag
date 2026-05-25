import json
import os
from collections import Counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CACHE_DIR = "/scratch/project_2000539/maryam/embed/.cache"
BATCH_SIZE = 32
TOP_KS = (1, 3, 5)
OUTPUT_DIR = "results/eval_simple_datasets_combined_knn"

DATASETS = [
    {
        "input_path": "data/simple_datasets/finnish_cities_dataset.jsonl",
        "dataset_name": "finnish_cities_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/rare_names_dataset.jsonl",
        "dataset_name": "rare_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/informative_names_dataset.jsonl",
        "dataset_name": "informative_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/person_actions_dataset.jsonl",
        "dataset_name": "person_actions_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def embed_texts(texts, tokenizer, model, batch_size=32, device="cpu"):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)
            pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy())

    return np.vstack(all_vecs).astype(np.float32, copy=False)


def load_combined_rows():
    texts = []
    labels = []
    dataset_names = []

    for config in DATASETS:
        rows = load_jsonl(config["input_path"])
        texts.extend(row[config["text_field"]] for row in rows)
        labels.extend(str(row[config["label_field"]]) for row in rows)
        dataset_names.extend([config["dataset_name"]] * len(rows))

    return texts, np.array(labels), np.array(dataset_names)


def leave_one_out_nearest_centroid_accuracy(embeddings, labels):
    unique_labels = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    sums = np.zeros((len(unique_labels), embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_labels), dtype=np.int32)

    for emb, label in zip(embeddings, labels):
        idx = label_to_idx[label]
        sums[idx] += emb
        counts[idx] += 1

    correct = 0
    for emb, label in zip(embeddings, labels):
        best_label = None
        best_score = -np.inf
        for candidate in unique_labels:
            idx = label_to_idx[candidate]
            count = counts[idx] - (1 if candidate == label else 0)
            if count <= 0:
                continue
            centroid = (sums[idx] - (emb if candidate == label else 0.0)) / count
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            score = float(np.dot(emb, centroid))
            if score > best_score:
                best_score = score
                best_label = candidate
        correct += int(best_label == label)
    return correct / len(labels)


def leave_one_out_knn_hit_rates(embeddings, labels, top_ks):
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    sorted_idx = np.argsort(sims, axis=1)[:, ::-1]

    results = {}
    for k in top_ks:
        hits = 0
        for i in range(len(labels)):
            neighbor_idx = sorted_idx[i, :k]
            hits += int(np.any(labels[neighbor_idx] == labels[i]))
        results[k] = hits / len(labels)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    texts, labels, dataset_names = load_combined_rows()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emb_cache = os.path.join(OUTPUT_DIR, "embeddings_cache.npz")

    print(f"Combined rows={len(texts)}")
    print("Dataset distribution:", Counter(dataset_names))
    print("Label distribution:", Counter(labels))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    ).to(device)
    model.eval()

    if os.path.exists(emb_cache):
        cached = np.load(emb_cache)
        embeddings = cached["embeddings"].astype(np.float32, copy=False)
        print(f"Loaded cached embeddings: {emb_cache} | shape={embeddings.shape}")
    else:
        embeddings = embed_texts(
            texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=BATCH_SIZE,
            device=device,
        )
        np.savez_compressed(emb_cache, embeddings=embeddings)
        print(f"Saved embedding cache: {emb_cache}")

    centroid_acc = leave_one_out_nearest_centroid_accuracy(embeddings, labels)
    knn_hits = leave_one_out_knn_hit_rates(embeddings, labels, TOP_KS)

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Combined simple-dataset evaluation\n")
        f.write(f"Rows: {len(texts)}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Embedding dim: {embeddings.shape[1]}\n")
        f.write(f"Datasets: {', '.join(cfg['dataset_name'] for cfg in DATASETS)}\n")
        f.write(
            "Dataset distribution: "
            + ", ".join(
                f"{name}={count}" for name, count in Counter(dataset_names).items()
            )
            + "\n"
        )
        f.write(f"Leave-one-out nearest-centroid accuracy: {centroid_acc:.4f}\n")
        for k in TOP_KS:
            f.write(
                f"Leave-one-out top-{k} neighbor hit rate: {knn_hits[k]:.4f}\n"
            )

    print(f"[OK] Leave-one-out nearest-centroid accuracy: {centroid_acc:.4f}")
    for k in TOP_KS:
        print(f"[OK] Leave-one-out top-{k} neighbor hit rate: {knn_hits[k]:.4f}")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
