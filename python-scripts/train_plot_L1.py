import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def clean_label(label):
    label = re.sub(r'^(\d+)[,.]', r'\1.', label.strip())
    match = re.match(r'^(\d+\.\s*[^0-9]*)', label)
    if match:
        label = match.group(1).strip()
    label = re.sub(r'[,.]\s*$', '', label)
    label = re.sub(r'^(\d+)\.+', r'\1.', label)
    label = re.sub(r'\s*\.\s*', '.', label)
    label = re.sub(r'\s+', ' ', label)
    return label.strip()


def visualize_embedding_heads(
    embeddings,
    y,
    nonzero_indices,
    method='tsne',
    ax=None,
    boost_factor=1,
    save_path=None
):
    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]

    if method == 'tsne':
        perplexity = min(30, n_samples - 1)
        reducer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("tsne", TSNE(n_components=2,
                          perplexity=perplexity,
                          random_state=42)),
        ])
    else:
        raise ValueError("Method must be 'tsne'")

    if ax is None:
        ax = plt.gca()

    boosted = embeddings_array.copy()

    valid_idx = [i for i in nonzero_indices if 0 <= i < boosted.shape[1]]
    if valid_idx:
        mask = np.ones(boosted.shape[1], dtype=bool)
        mask[valid_idx] = False
        boosted[:, mask] = 0
        boosted[:, valid_idx] *= boost_factor

    print("Silhouette original vs L1:",
          silhouette_score(embeddings_array, y),
          silhouette_score(boosted, y))

    reduced = reducer.fit_transform(boosted)

    # ---- subsample after dim reduction ----
    n_arrows = min(20, reduced.shape[0])   # don’t exceed available points
    idx = np.random.choice(reduced.shape[0], size=n_arrows, replace=False)

    # tails at origin, heads at reduced direction vectors
    """tails_x = np.zeros(n_samples)
    tails_y = np.zeros(n_samples)
    dx = reduced[:, 0]
    dy = reduced[:, 1]"""
    tails_x = np.zeros(n_arrows)
    tails_y = np.zeros(n_arrows)
    dx = reduced[idx, 0]
    dy = reduced[idx, 1]

    # optional: scale arrows a bit so they don't explode
    lengths = np.sqrt(dx**2 + dy**2)
    max_length = lengths.max()
    if max_length > 0:
        dx = dx / max_length
        dy = dy / max_length

    # ---------- NEW: plot EN → FI directions instead of dots ----------  ###
    # assume first half = EN, second half = FI
    """n_pairs = n_samples // 2
    en_2d = reduced[:n_pairs]
    fi_2d = reduced[n_pairs: n_pairs * 2]

    en_2d = en_2d[:20]
    fi_2d = fi_2d[:20]

    dx = fi_2d[:, 0] - en_2d[:, 0]
    dy = fi_2d[:, 1] - en_2d[:, 1]

    # compute relative lengths
    lengths = np.sqrt(dx**2 + dy**2)

    # target: make the longest arrow ~10% of plot width
    max_length = lengths.max()
    if max_length > 0:
        scale_factor = 0.10 * (en_2d[:, 0].max() - en_2d[:, 0].min()) / max_length
        dx *= scale_factor
        dy *= scale_factor"""


    # colors based on EN labels (first half of y)
    y_en = y[:n_pairs]
    colors_en = np.where(
        np.arange(n_pairs) < 3000,
        np.where(y_en == 1, 'crimson', 'deepskyblue'),
        np.where(y_en == 1, 'orange', 'green')
    )

    # draw arrows: tails at EN, heads at FI
    plt.quiver(
        #en_2d[:, 0], en_2d[:, 1],
        tails_x, tails_y,
        dx, dy,
        color=colors_en,
        angles='xy',
        scale_units='xy',
        scale=1,
        alpha=0.7,
        width=0.002
    )
    # -------------------------------------------------------------------  ###

    plt.title("t-SNE EN→FI directions (L1 dims)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Plot saved to: {save_path}")

    plt.close()


def topk_hits(scores, y_true, k=42):
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return idx, int(y_true[idx].sum())

# ----------------- LOAD DATA -----------------

with open("/scratch/project_2000539/maryam/adaptive-mrag/data/classified_topics_narrowed.jsonl",
          "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

for doc in dataset:
    doc["topic"] = clean_label(doc["topic"])

target_topic = "18.Health, Nutrition, Diet, Medicine, Diseases, Biology"
output_path_base_en = "data/Fineweb/FineWeb-embedd-e5-multi-en/"
output_path_base_fi = "data/Fineweb/FineWeb-embedd-e5-multi-fi/"

X, y, y_lang = [], [], []

for doc in dataset:
    emb_path = output_path_base_en + doc['id'] + ".pkl"

    all_embd = []
    with open(emb_path, "rb") as f:
        while True:
            try:
                arr = pickle.load(f)
                all_embd.append(arr)
            except EOFError:
                break

    all_embd = np.concatenate(all_embd, axis=0)
    X.append(all_embd.mean(axis=0))
    y.append(1 if doc["topic"] == target_topic else 0)
    y_lang.append(0)

for doc in dataset:
    emb_path = output_path_base_fi + doc['id'] + ".pkl"

    all_embd = []
    with open(emb_path, "rb") as f:
        while True:
            try:
                arr = pickle.load(f)
                all_embd.append(arr)
            except EOFError:
                break

    all_embd = np.concatenate(all_embd, axis=0)
    X.append(all_embd.mean(axis=0))
    y.append(1 if doc["topic"] == target_topic else 0)    
    y_lang.append(1)

X = np.stack(X)
y = np.array(y)
y_lang = np.array(y_lang)

print("Shape:", X.shape, "| Positives:", y.sum())

# --- build EN–FI direction vectors ---
n_pairs = len(dataset)                 # number of docs
X_en = X[:n_pairs]                     # EN embeddings
X_fi = X[n_pairs:]                     # FI embeddings
X_dir = X_fi - X_en                    # direction EN -> FI
y_dir = y[:n_pairs]                    # use EN labels for pairs

# ----------------- TRAIN/TEST SPLIT (with indices) -----------------

n_docs = X.shape[0]
all_idx = np.arange(n_docs)

idx_train, idx_test, y_train, y_test = train_test_split(
    all_idx,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

X_train, X_test = X[idx_train], X[idx_test]

TOP_K = 84

# ----------------- MAIN LOOP -----------------

for C in [1, 0.5, 0.1, 0.01]:
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)

    scores = clf.decision_function(X_test)
    topk_idx_test, hits = topk_hits(scores, y_test, k=TOP_K)

    w = clf.coef_[0]
    nz_idx = np.nonzero(w)[0]

    print(f"C={C} | top-{TOP_K} hits: {hits}/{TOP_K} | nonzero={len(nz_idx)}")

if C == 0.01:
    # map top-k indices from test space -> global doc indices
    hit_indices_global = idx_test[topk_idx_test]
    save_path = f"tsne_original_dir_merge_docs_C{str(C).replace('.', '_')}_20s.png"

    visualize_embedding_heads(
        #X,          # ALL docs: first EN, then FI
        #y,
        X_dir,         
        y_dir,          
        nonzero_indices=nz_idx,
        save_path=save_path
    )

