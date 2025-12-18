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

def visualize_embedding_heads(embeddings, y, nonzero_indices, method='tsne', ax=None, boost_factor=1):

    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]

    """
    # Step 1: Split each embedding into 32 parts (heads)
    num_heads = 32
    head_size = embeddings_array.shape[1] // num_heads
    split_embeddings = embeddings_array.reshape(n_samples, num_heads, head_size)
    #flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)
    """

    # Step 2: Apply dimensionality reduction
    if method == 'tsne':
        if n_samples < 2:
            raise ValueError("Number of samples is too small for t-SNE")
        
        perplexity = min(30, n_samples - 1)
        reducer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
            ("scaler", StandardScaler()),                # Scale features
            ("tsne", TSNE(n_components=2, perplexity=perplexity, random_state=42))  # Apply t-SNE
        ])
        #reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("Method must be 'tsne'")

    # Step 3: Plot with varying color intensities based on magnitude for each head
    if ax is None:
        ax = plt.gca()
    
    base_colors = ['crimson', 'deepskyblue']

    all_points = []
    indices = np.arange(n_samples)  # FIX: define indices
    # rows to affect: only docs with Y == 1

    boosted = embeddings_array.copy()
    # keep only valid indices
    valid_idx = [i for i in nonzero_indices if 0 <= i < boosted.shape[1]]
    #rng = np.random.default_rng()
    #valid_idx = rng.integers(low=0, high=4095, size=36).tolist()
    if valid_idx:

        # --- STEP 1: Zero out everything except the valid_idx for ALL rows ---
        mask = np.ones(boosted.shape[1], dtype=bool)
        mask[valid_idx] = False
        boosted[:, mask] = 0

        # --- STEP 2: Apply boost ONLY to rows where y == 1 ---
        #row_mask = (y == 1)
        #boosted[np.ix_(row_mask, valid_idx)] *= boost_factor
        boosted[:, valid_idx] *= boost_factor

    sil_orig = silhouette_score(embeddings_array.copy(), y)
    sil_l1   = silhouette_score(boosted, y)
    print(sil_orig, sil_l1)

    actual_nonzero_cols = np.unique(np.nonzero(boosted)[1])
    print("Actual nonzero columns:", actual_nonzero_cols)
    print("Expected valid_idx:", valid_idx)
    print("len valid_idx", len(valid_idx))

    print("Match:", np.array_equal(np.sort(actual_nonzero_cols), np.sort(valid_idx)))

    reduced_boosted = reducer.fit_transform(boosted)

    print("all indices", indices)
    for idx in indices:
        all_points.append({
            'x': reduced_boosted[idx, 0],
            'y': reduced_boosted[idx, 1],
            'color': base_colors[0] if y[idx] == 1 else base_colors[1],
            'alpha': 0.7,
            'size': 1
        })

    # Shuffle the entire collection of points
    np.random.shuffle(all_points)
    xs = [point['x'] for point in all_points]
    ys = [point['y'] for point in all_points]
    colors = [point['color'] for point in all_points]

    # Plot in the new order
    plt.scatter(xs, ys, c=colors, alpha=0.7, s=1)

    ax.set_title(f'Embedding Visualization({method.upper()})') #with Magnitude Shading
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    #ax.legend(title="Heads", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    #plt.savefig("plot_5_fi_e5.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

with open("/scratch/project_2000539/maryam/adaptive-mrag/data/classified_topics_narrowed.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

for doc in dataset:
    doc["topic"] = clean_label(doc["topic"])

target_topic = "18.Health, Nutrition, Diet, Medicine, Diseases, Biology"
output_path_base = "data/Fineweb/FineWeb-embedd-stage3_10000/" #3000 docs


X = []
y = []

for doc in dataset:
    topic = doc["topic"]
    id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base +  id + ".pkl"

    all_embd = []
    with open(output_path, "rb") as f:
        while True:
            try:
                padded_embeddings = pickle.load(f)
                all_embd.append(padded_embeddings)
            except EOFError:
                break

    all_embd = np.concatenate(all_embd, axis=0)
    X.append(all_embd[0])
    y.append(1 if topic == target_topic else 0)

y = np.array(y)

#FI SFR
#nonzero_indices  = [93, 204, 705, 777, 985, 1116, 1195, 1372, 1641, 1661, 2410, 2618, 2679, 2682, 2798, 3071, 3121, 3298, 3307, 3501, 3701, 3855, 4091]

#EN SFR
nonzero_indices= [93, 204, 322, 488, 688, 701, 705, 777, 1030, 1143, 1195, 1372, 1375, 1428, 1661, 2049, 2157, 2284, 2410, 2524, 2554, 2618, 2679, 2682, 2763, 2798, 2816, 2976, 3070, 3071, 3298, 3307, 3355, 3501, 3855, 4091]

#FI e5 multi
#nonzero_indices= [16, 34, 49, 92, 122, 157, 158, 176, 210, 223, 282, 286, 306, 327, 378, 388, 435, 454, 469, 478, 505, 507, 509, 513, 539, 562, 563, 564, 622, 663, 681, 693, 726, 727, 744, 763, 779, 810, 828, 843, 870, 877, 909, 965, 967, 1003, 1019]

visualize_embedding_heads(X, y, nonzero_indices, method='tsne', ax=None)
