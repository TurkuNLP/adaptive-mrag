import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CACHE_DIR = "/scratch/project_2000539/maryam/embed/.cache"
BATCH_SIZE = 32
N_COMPONENTS = 30
NOISE_SEED = 42
NOISE_AXIS = 0
NOISE_STD_FRACTION = 1

DATASETS = [
    {
        "input_path": "data/simple_datasets/finnish_cities_dataset.jsonl",
        "output_dir": "results/pca_finnish_cities_dataset",
        "title": "Centroid projection: Finnish cities dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/rare_names_dataset.jsonl",
        "output_dir": "results/pca_rare_names_dataset",
        "title": "Centroid projection: rare names dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/informative_names_dataset.jsonl",
        "output_dir": "results/pca_informative_names_dataset",
        "title": "Centroid projection: informative names dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/person_actions_dataset.jsonl",
        "output_dir": "results/pca_person_actions_dataset",
        "title": "Centroid projection: person actions dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
]
"""
DATASETS = [
    {
        "input_path": "data/simple_datasets/generic_names_50x50_dataset.jsonl",
        "output_dir": "results/pca_generic_names_50x50_dataset",
        "title": "Centroid projection: generic names",
        "label_field": "name",
        "text_field": "sentence",
    },
]
"""
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def generate_distinct_styles(n_labels):
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    palette_blocks = [
        list(plt.get_cmap("tab20").colors),
        list(plt.get_cmap("tab20b").colors),
        list(plt.get_cmap("tab20c").colors),
    ]
    base_colors = []
    max_block = max(len(block) for block in palette_blocks)
    for idx in range(max_block):
        for block in palette_blocks:
            if idx < len(block):
                base_colors.append(block[idx])

    styles = []
    for idx in range(n_labels):
        styles.append((base_colors[idx % len(base_colors)], marker_cycle[idx % len(marker_cycle)]))
    return styles


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

    return np.vstack(all_vecs)


def centroid_components(embeddings, labels, n_components):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        centroids.append(embeddings[labels == label].mean(axis=0))
    centroid_matrix = np.vstack(centroids)
    centroid_matrix = centroid_matrix - centroid_matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centroid_matrix, full_matrices=False)
    use_components = min(n_components, vh.shape[0])
    components = vh[:use_components].T
    return embeddings @ components, unique_labels, use_components


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


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def leave_one_out_centroid_accuracy(embeddings, labels):
    embeddings = normalize_rows(np.asarray(embeddings, dtype=np.float32))
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
        scores = []
        for candidate in unique_labels:
            idx = label_to_idx[candidate]
            count = counts[idx] - (1 if candidate == label else 0)
            if count <= 0:
                scores.append(-1.0)
                continue
            centroid = (sums[idx] - (emb if candidate == label else 0.0)) / count
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            scores.append(float(np.dot(emb, centroid)))
        pred = unique_labels[int(np.argmax(scores))]
        correct += int(pred == label)
    return correct / len(labels)


def write_sample_centroid_similarities(path, texts, embeddings, labels, max_queries=10, top_n=10):
    embeddings = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    unique_labels = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    sums = np.zeros((len(unique_labels), embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_labels), dtype=np.int32)

    for emb, label in zip(embeddings, labels):
        idx = label_to_idx[label]
        sums[idx] += emb
        counts[idx] += 1

    query_indices = []
    for label in unique_labels[:max_queries]:
        query_indices.append(int(np.flatnonzero(labels == label)[0]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("Sample centroid similarities\n")
        f.write(f"Queries shown: {len(query_indices)}\n")
        f.write(f"Closest centroids shown per query: {top_n}\n\n")

        for query_idx in query_indices:
            emb = embeddings[query_idx]
            label = labels[query_idx]
            centroid_scores = []
            for candidate in unique_labels:
                idx = label_to_idx[candidate]
                count = counts[idx] - (1 if candidate == label else 0)
                if count <= 0:
                    continue
                centroid = (sums[idx] - (emb if candidate == label else 0.0)) / count
                centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
                score = float(np.dot(emb, centroid))
                centroid_scores.append((candidate, score))

            centroid_scores.sort(key=lambda item: item[1], reverse=True)
            f.write(f"Query index: {query_idx}\n")
            f.write(f"Query label: {label}\n")
            f.write(f"Query text: {texts[query_idx]}\n")
            f.write("Closest centroids:\n")
            for rank, (candidate, score) in enumerate(centroid_scores[:top_n], start=1):
                hit = "yes" if candidate == label else "no"
                f.write(
                    f"  {rank}. centroid={candidate} score={score:.4f} hit={hit}\n"
                )
            f.write("\n")


def leave_one_out_knn_accuracy(embeddings, labels, k=3):
    embeddings = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    correct = 0
    for i in range(len(labels)):
        neighbor_idx = np.argsort(sims[i])[::-1][:k]
        neighbor_labels = labels[neighbor_idx]
        vote_counts = Counter(neighbor_labels)
        pred = sorted(vote_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        correct += int(pred == labels[i])
    return correct / len(labels)


def save_projection_plot(path, projected, labels, unique_labels, plot_styles, dims, title):
    plt.figure(figsize=(10, 8))
    for class_id, label in enumerate(unique_labels):
        idx = labels == label
        color, marker = plot_styles[class_id]
        plt.scatter(
            projected[idx, 0],
            projected[idx, 1],
            s=6,
            alpha=0.8,
            color=color,
            marker=marker,
            label=label,
        )

    x_dim = int(dims[0]) + 1
    y_dim = int(dims[1]) + 1
    plt.xlabel(f"Centroid component {x_dim}")
    plt.ylabel(f"Centroid component {y_dim}")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        frameon=True,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def add_gaussian_noise_to_axis(projected, axis, std_fraction, seed):
    noisy = projected.copy()
    axis_std = float(np.std(noisy[:, axis]))
    noise_std = axis_std * std_fraction
    rng = np.random.default_rng(seed)
    noisy[:, axis] += rng.normal(loc=0.0, scale=noise_std, size=noisy.shape[0])
    return noisy, noise_std


def noise_tag(axis, std_fraction, seed):
    frac_str = str(std_fraction).replace(".", "_")
    return f"gaussian_axis{axis + 1}_frac{frac_str}_seed{seed}"


def plot_dataset(config, tokenizer, model, device):
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
        embeddings = cached["embeddings"]
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

    embeddings = embeddings.astype(np.float32, copy=False)
    X_proj_all, unique_labels, used_components = centroid_components(
        embeddings, labels, N_COMPONENTS
    )
    signal_scores = component_signal_scores(X_proj_all, labels)
    best_dims = np.argsort(signal_scores)[::-1][: min(2, len(signal_scores))]
    X_proj = X_proj_all[:, best_dims]
    original_dims = np.array([0, 1]) if used_components >= 2 else best_dims
    X_proj_original = X_proj_all[:, original_dims]
    X_proj_original_noisy, noise_std = add_gaussian_noise_to_axis(
        X_proj_original,
        axis=NOISE_AXIS,
        std_fraction=NOISE_STD_FRACTION,
        seed=NOISE_SEED,
    )
    centroid_acc = leave_one_out_centroid_accuracy(embeddings, labels)
    knn_acc = leave_one_out_knn_accuracy(embeddings, labels, k=1)
    original_pc12_centroid_acc = leave_one_out_centroid_accuracy(X_proj_original, labels)
    original_pc12_knn_acc = leave_one_out_knn_accuracy(X_proj_original, labels, k=1)
    noisy_pc12_centroid_acc = leave_one_out_centroid_accuracy(X_proj_original_noisy, labels)
    noisy_pc12_knn_acc = leave_one_out_knn_accuracy(X_proj_original_noisy, labels, k=1)
    plot_styles = generate_distinct_styles(len(unique_labels))

    plot_path = os.path.join(output_dir, "centroid_projection_sentences.png")
    plot_path_original = os.path.join(output_dir, "centroid_projection_sentences_original_pc12.png")
    noisy_tag = noise_tag(NOISE_AXIS, NOISE_STD_FRACTION, NOISE_SEED)
    plot_path_original_noisy = os.path.join(
        output_dir, f"centroid_projection_sentences_original_pc12_{noisy_tag}.png"
    )
    save_projection_plot(
        plot_path,
        X_proj,
        labels,
        unique_labels,
        plot_styles,
        best_dims,
        config["title"],
    )
    save_projection_plot(
        plot_path_original,
        X_proj_original,
        labels,
        unique_labels,
        plot_styles,
        original_dims,
        f"{config['title']} (original components 1 and 2)",
    )
    save_projection_plot(
        plot_path_original_noisy,
        X_proj_original_noisy,
        labels,
        unique_labels,
        plot_styles,
        original_dims,
        (
            f"{config['title']} (original PC1/PC2 + Gaussian noise on "
            f"component {int(original_dims[NOISE_AXIS]) + 1})"
        ),
    )

    summary_path = os.path.join(output_dir, "summary.txt")
    centroid_samples_path = os.path.join(output_dir, "sample_centroid_similarities.txt")
    write_sample_centroid_similarities(
        centroid_samples_path,
        texts,
        embeddings,
        labels,
        max_queries=10,
        top_n=10,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {config['input_path']}\n")
        f.write(f"Rows: {len(rows)}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Embedding dim: {embeddings.shape[1]}\n")
        f.write(f"Centroid components requested: {N_COMPONENTS}\n")
        f.write(f"Centroid components used: {used_components}\n")
        f.write("Top signal components: " + ", ".join(str(int(dim) + 1) for dim in best_dims) + "\n")
        f.write(f"Leave-one-out centroid accuracy: {centroid_acc:.4f}\n")
        f.write(f"Leave-one-out kNN accuracy (k=1): {knn_acc:.4f}\n")
        f.write(
            f"Original PC1/PC2 leave-one-out centroid accuracy: "
            f"{original_pc12_centroid_acc:.4f}\n"
        )
        f.write(
            f"Original PC1/PC2 leave-one-out kNN accuracy (k=1): "
            f"{original_pc12_knn_acc:.4f}\n"
        )
        f.write(
            f"Noisy PC1/PC2 leave-one-out centroid accuracy: "
            f"{noisy_pc12_centroid_acc:.4f}\n"
        )
        f.write(
            f"Noisy PC1/PC2 leave-one-out kNN accuracy (k=1): "
            f"{noisy_pc12_knn_acc:.4f}\n"
        )
        f.write(f"Plot: {plot_path}\n")
        f.write(f"Original PC1/PC2 plot: {plot_path_original}\n")
        f.write(f"Original PC1/PC2 noisy plot: {plot_path_original_noisy}\n")
        f.write(
            f"Noise: Gaussian on component {int(original_dims[NOISE_AXIS]) + 1}, "
            f"std_fraction={NOISE_STD_FRACTION:.2f}, std={noise_std:.6f}, "
            f"seed={NOISE_SEED}\n"
        )
        f.write(f"Sample centroid similarities: {centroid_samples_path}\n")
        f.write("Labels=" + ", ".join(unique_labels) + "\n")
        f.write(
            "Signal scores="
            + ", ".join(f"c{idx + 1}:{score:.4f}" for idx, score in enumerate(signal_scores))
            + "\n"
        )

    print(f"[OK] Plot saved to: {plot_path}")
    print(
        "[OK] Top signal components:",
        ", ".join(str(int(dim) + 1) for dim in best_dims),
    )
    print(f"[OK] Leave-one-out centroid accuracy: {centroid_acc:.4f}")
    print(f"[OK] Leave-one-out kNN accuracy (k=1): {knn_acc:.4f}")
    print(
        f"[OK] Original PC1/PC2 leave-one-out centroid accuracy: "
        f"{original_pc12_centroid_acc:.4f}"
    )
    print(
        f"[OK] Original PC1/PC2 leave-one-out kNN accuracy (k=1): "
        f"{original_pc12_knn_acc:.4f}"
    )
    print(
        f"[OK] Noisy PC1/PC2 leave-one-out centroid accuracy: "
        f"{noisy_pc12_centroid_acc:.4f}"
    )
    print(
        f"[OK] Noisy PC1/PC2 leave-one-out kNN accuracy (k=1): "
        f"{noisy_pc12_knn_acc:.4f}"
    )
    print(f"[OK] Original PC1/PC2 plot saved to: {plot_path_original}")
    print(f"[OK] Original PC1/PC2 noisy plot saved to: {plot_path_original_noisy}")
    print(f"[OK] Sample centroid similarities saved to: {centroid_samples_path}")
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
        plot_dataset(config, tokenizer=tokenizer, model=model, device=device)


if __name__ == "__main__":
    main()
