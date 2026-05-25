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
TOP_KS = (1, 5)
CUMULATIVE_COMPONENTS = (1, 2, 3, 5, 10, 30)
SAMPLES_PER_CLASS = 2
RETRIEVALS_TO_SHOW = 10

DATASETS = [
    {
        "input_path": "data/simple_datasets/finnish_cities_dataset.jsonl",
        "output_dir": "results/retrieve_finnish_cities_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/rare_names_dataset.jsonl",
        "output_dir": "results/retrieve_rare_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/informative_names_dataset.jsonl",
        "output_dir": "results/retrieve_informative_names_dataset",
        "label_field": "name",
        "text_field": "sentence",
    },
    {
        "input_path": "data/simple_datasets/person_actions_dataset.jsonl",
        "output_dir": "results/retrieve_person_actions_dataset",
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
    return projected, use_components


def cosine_similarity_matrix(embeddings):
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    return sims


def negative_l2_similarity_matrix(values):
    if values.ndim == 1:
        values = values[:, None]
    diffs = values[:, None, :] - values[None, :, :]
    sims = -np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(sims, -np.inf)
    return sims


def retrieval_metrics_from_similarity(sims, labels, top_ks=TOP_KS):
    labels = np.asarray(labels)
    recalls = {k: 0 for k in top_ks}
    reciprocal_ranks = []
    first_ranks = []

    for i in range(len(labels)):
        ranked_idx = np.argsort(sims[i])[::-1]
        ranked_labels = labels[ranked_idx]
        hits = ranked_labels == labels[i]
        hit_positions = np.flatnonzero(hits)
        if len(hit_positions) == 0:
            first_ranks.append(len(labels))
            reciprocal_ranks.append(0.0)
        else:
            first_hit = int(hit_positions[0]) + 1
            first_ranks.append(first_hit)
            reciprocal_ranks.append(1.0 / first_hit)
        for k in top_ks:
            recalls[k] += int(np.any(hits[:k]))

    n = len(labels)
    metrics = {f"Recall@{k}": recalls[k] / n for k in top_ks}
    metrics["MRR"] = float(np.mean(reciprocal_ranks))
    metrics["MeanFirstHitRank"] = float(np.mean(first_ranks))
    return metrics


def format_metrics(metrics):
    return " ".join(
        [
            f"R@1={metrics['Recall@1']:.4f}",
            f"R@5={metrics['Recall@5']:.4f}",
            f"MRR={metrics['MRR']:.4f}",
            f"MeanFirstHitRank={metrics['MeanFirstHitRank']:.2f}",
        ]
    )


def write_sample_retrievals(path, texts, labels, sims, samples_per_class, top_n):
    unique_labels = np.unique(labels)
    ranked_idx = np.argsort(sims, axis=1)[:, ::-1]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Sample retrievals from full-embedding cosine similarity\n")
        f.write(f"Queries per class: {samples_per_class}\n")
        f.write(f"Retrieved items shown per query: {top_n}\n\n")

        for label in unique_labels:
            class_idx = np.flatnonzero(labels == label)[:samples_per_class]
            f.write(f"=== Label: {label} ===\n")
            for query_idx in class_idx:
                f.write(f"Query index: {query_idx}\n")
                f.write(f"Query text: {texts[query_idx]}\n")
                f.write("Top retrievals:\n")
                for rank, cand_idx in enumerate(ranked_idx[query_idx, :top_n], start=1):
                    score = float(sims[query_idx, cand_idx])
                    hit = "yes" if labels[cand_idx] == labels[query_idx] else "no"
                    f.write(
                        f"  {rank}. score={score:.4f} hit={hit} "
                        f"label={labels[cand_idx]} text={texts[cand_idx]}\n"
                    )
                f.write("\n")


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

    full_sims = cosine_similarity_matrix(embeddings)
    full_metrics = retrieval_metrics_from_similarity(full_sims, labels)

    projected, used_components = pca_components(embeddings, N_COMPONENTS)
    projected = projected[:, :used_components]

    per_component = []
    for dim in range(used_components):
        sims = negative_l2_similarity_matrix(projected[:, dim])
        metrics = retrieval_metrics_from_similarity(sims, labels)
        per_component.append((dim + 1, metrics))

    cumulative = []
    for k in CUMULATIVE_COMPONENTS:
        if k > used_components:
            continue
        sims = negative_l2_similarity_matrix(projected[:, :k])
        metrics = retrieval_metrics_from_similarity(sims, labels)
        cumulative.append((k, metrics))

    top_r1 = sorted(per_component, key=lambda item: item[1]["Recall@1"], reverse=True)[:10]
    top_mrr = sorted(per_component, key=lambda item: item[1]["MRR"], reverse=True)[:10]

    summary_path = os.path.join(output_dir, "summary.txt")
    samples_path = os.path.join(output_dir, "sample_retrievals.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {config['input_path']}\n")
        f.write(f"Rows: {len(rows)}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Embedding dim: {embeddings.shape[1]}\n")
        f.write(f"PCA components requested: {N_COMPONENTS}\n")
        f.write(f"PCA components used: {used_components}\n")
        f.write(f"Full embedding retrieval: {format_metrics(full_metrics)}\n")
        f.write("Top 10 PCs by Recall@1:\n")
        for component_id, metrics in top_r1:
            f.write(f"c{component_id}: {format_metrics(metrics)}\n")
        f.write("Top 10 PCs by MRR:\n")
        for component_id, metrics in top_mrr:
            f.write(f"c{component_id}: {format_metrics(metrics)}\n")
        f.write("Cumulative top-k PCs:\n")
        for k, metrics in cumulative:
            f.write(f"top_{k}: {format_metrics(metrics)}\n")

    write_sample_retrievals(
        samples_path,
        texts,
        labels,
        full_sims,
        samples_per_class=SAMPLES_PER_CLASS,
        top_n=RETRIEVALS_TO_SHOW,
    )

    print(f"[OK] Full embedding retrieval: {format_metrics(full_metrics)}")
    print(
        "[OK] Top 10 PCs by Recall@1: "
        + ", ".join(f"c{cid}={m['Recall@1']:.4f}" for cid, m in top_r1)
    )
    print(
        "[OK] Top 10 PCs by MRR: "
        + ", ".join(f"c{cid}={m['MRR']:.4f}" for cid, m in top_mrr)
    )
    print(f"[OK] Sample retrievals saved to: {samples_path}")
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
