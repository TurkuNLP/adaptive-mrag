import json
import os
from collections import Counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CACHE_DIR = "/scratch/project_2000539/maryam/embed/.cache"
BATCH_SIZE = 32
N_COMPONENTS = 30

DATASETS = [
    {
        "input_path": "data/simple_datasets/finnish_cities_dataset.jsonl",
        "output_dir": "results/eval_finnish_cities_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/rare_names_dataset.jsonl",
        "output_dir": "results/eval_rare_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/informative_names_dataset.jsonl",
        "output_dir": "results/eval_informative_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/person_actions_dataset.jsonl",
        "output_dir": "results/eval_person_actions_dataset",
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


def pca_components(embeddings, n_components):
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    use_components = min(n_components, vh.shape[0])
    components = vh[:use_components].T
    projected = centered @ components
    return projected, components, mean[0], use_components


def component_signal_scores(projected, labels):
    unique_labels = np.unique(labels)
    overall_mean = projected.mean(axis=0)
    scores = []
    for dim in range(projected.shape[1]):
        between = 0.0
        within = 0.0
        for label in unique_labels:
            values = projected[labels == label, dim]
            label_mean = values.mean()
            between += len(values) * (label_mean - overall_mean[dim]) ** 2
            within += np.sum((values - label_mean) ** 2)
        scores.append(between / max(within, 1e-12))
    return np.array(scores)


def class_centroids(embeddings, labels):
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        centroids[label] = embeddings[labels == label].mean(axis=0)
    return centroids


def leave_one_out_nearest_centroid_cosine_accuracy(embeddings, labels):
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


def leave_one_out_sentence_knn1_accuracy(embeddings, labels):
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    nearest_idx = np.argmax(sims, axis=1)
    preds = labels[nearest_idx]
    return float(np.mean(preds == labels))


def evaluate_pca_components_strict(embeddings, labels):
    max_components = min(N_COMPONENTS, embeddings.shape[0] - 1, embeddings.shape[1])
    signal_sums = np.zeros(max_components, dtype=np.float64)
    centroid_correct = np.zeros(max_components, dtype=np.float64)
    knn1_correct = np.zeros(max_components, dtype=np.float64)
    eval_counts = np.zeros(max_components, dtype=np.int32)

    for i in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[i] = False
        train_embeddings = embeddings[train_mask]
        train_labels = labels[train_mask]
        test_embedding = embeddings[i]
        test_label = labels[i]

        projected_train, components, train_mean, used_components = pca_components(
            train_embeddings, N_COMPONENTS
        )
        signal_scores = component_signal_scores(projected_train[:, :used_components], train_labels)
        centroids = class_centroids(train_embeddings, train_labels)
        projected_centroids = {
            label: (centroid - train_mean) @ components[:, :used_components]
            for label, centroid in centroids.items()
        }
        projected_test = (test_embedding - train_mean) @ components[:, :used_components]

        for dim in range(used_components):
            test_value = float(projected_test[dim])

            best_label = None
            best_dist = np.inf
            for label, centroid_proj in projected_centroids.items():
                dist = abs(test_value - float(centroid_proj[dim]))
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            centroid_correct[dim] += float(best_label == test_label)

            train_values = projected_train[:, dim]
            nearest = int(np.argmin(np.abs(train_values - test_value)))
            knn1_correct[dim] += float(train_labels[nearest] == test_label)

            signal_sums[dim] += signal_scores[dim]
            eval_counts[dim] += 1

    component_rows = []
    for dim in range(max_components):
        if eval_counts[dim] == 0:
            continue
        component_rows.append(
            (
                dim + 1,
                float(signal_sums[dim] / eval_counts[dim]),
                float(centroid_correct[dim] / eval_counts[dim]),
                float(knn1_correct[dim] / eval_counts[dim]),
            )
        )
    return component_rows


def format_top_components(rows, score_idx, top_k=10):
    top_rows = sorted(rows, key=lambda row: row[score_idx], reverse=True)[:top_k]
    return ", ".join(f"c{row[0]}={row[score_idx]:.4f}" for row in top_rows)


def evaluate_dataset(config, tokenizer, model, device):
    rows = load_jsonl(config["input_path"])
    texts = [row[config["text_field"]] for row in rows]
    labels = np.array([str(row[config["label_field"]]) for row in rows])

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    emb_cache = os.path.join(output_dir, "embeddings_cache.npz")

    print(f"Dataset: {config['input_path']} | rows={len(rows)}")
    print("Label distribution:", Counter(labels))

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

    full_centroid_acc = leave_one_out_nearest_centroid_cosine_accuracy(
        embeddings, labels
    )
    full_knn1_acc = leave_one_out_sentence_knn1_accuracy(embeddings, labels)
    component_rows = evaluate_pca_components_strict(embeddings, labels)
    used_components = len(component_rows)

    best_signal = max(component_rows, key=lambda row: row[1])
    best_centroid = max(component_rows, key=lambda row: row[2])
    best_knn1 = max(component_rows, key=lambda row: row[3])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {config['input_path']}\n")
        f.write(f"Rows: {len(rows)}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Embedding dim: {embeddings.shape[1]}\n")
        f.write(f"Full-embedding centroid accuracy: {full_centroid_acc:.4f}\n")
        f.write(f"Full-embedding 1-NN accuracy: {full_knn1_acc:.4f}\n")
        f.write(f"PCA components requested: {N_COMPONENTS}\n")
        f.write(f"PCA components used: {used_components}\n")
        f.write(f"Top 10 signal components: {format_top_components(component_rows, 1)}\n")
        f.write(
            f"Top 10 centroid components: {format_top_components(component_rows, 2)}\n"
        )
        f.write(f"Top 10 1-NN components: {format_top_components(component_rows, 3)}\n")
        f.write(
            "Evaluation: strict leave-one-out PCA, full-space centroids projected to each PC\n"
        )
        f.write("Labels=" + ", ".join(np.unique(labels)) + "\n")
        f.write("Per-component results:\n")
        for component_id, signal, centroid_acc, knn1_acc in component_rows:
            f.write(
                f"c{component_id}: signal={signal:.4f} "
                f"centroid_acc={centroid_acc:.4f} knn1_acc={knn1_acc:.4f}\n"
            )

    print(
        f"[OK] Full-embedding centroid accuracy: {full_centroid_acc:.4f}"
    )
    print(f"[OK] Full-embedding 1-NN accuracy: {full_knn1_acc:.4f}")
    print(
        f"[OK] Top 10 signal components: {format_top_components(component_rows, 1)}"
    )
    print(
        f"[OK] Top 10 centroid components: {format_top_components(component_rows, 2)}"
    )
    print(
        f"[OK] Top 10 1-NN components: {format_top_components(component_rows, 3)}"
    )
    print(f"[OK] Summary saved to: {summary_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    for config in DATASETS:
        evaluate_dataset(config, tokenizer=tokenizer, model=model, device=device)


if __name__ == "__main__":
    main()
