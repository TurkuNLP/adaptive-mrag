import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def visualize_embedding_heads(embeddings, method='tsne', ax=None):
    """
    Visualize clusters in the embeddings by splitting each into 32 heads and using t-SNE, PCA, or ICA.

    :param embeddings: Embedding array to cluster and visualize.
    :param method: The method to use for dimensionality reduction ('tsne', 'pca', or 'ica').
    :param ax: The axis to plot on (optional).
    """    
    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]

    # Step 1: Split each embedding into 32 parts (heads)
    num_heads = 32
    head_size = embeddings_array.shape[1] // num_heads
    split_embeddings = embeddings_array.reshape(n_samples, num_heads, head_size)
    flat_embeddings = split_embeddings.reshape(n_samples * num_heads, head_size)

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

    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'ica':
        reducer = FastICA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne', 'pca', or 'ica'")

    reduced_embeddings = reducer.fit_transform(flat_embeddings)

    # Step 3: Plot with consistent colors for each head
    if ax is None:
        ax = plt.gca()

    # Assign 32 unique colors, one for each head
    colors = plt.cm.get_cmap('tab20b', num_heads).colors  # Alternative colormap with 32 options

    for head_idx in range(num_heads):
        indices = np.arange(head_idx, n_samples * num_heads, num_heads)
        ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                   color=colors[head_idx], label=f'Head {head_idx+1}', s=1, alpha=0.7)

    ax.set_title(f'Embedding Visualization by Head ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    #ax.legend(title="Heads", loc='best', fontsize='small', ncol=2)
    ax.legend(title="Heads", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.savefig("stella_plot.png", dpi=300)
    plt.show()

all_embd = []
with open("/scratch/project_2000539/maryam/rag_data/volkswagen_articles.pkl", "rb") as f:
    while True:
        try:
            # Load one batch at a time
            padded_embeddings = pickle.load(f)
            
            # Apply mean pooling to the batch
            #pooled_embeddings = mean_pooling(padded_embeddings)
            all_embd.append(padded_embeddings)
        
        except EOFError:
            # End of file reached
            break

all_embd = np.concatenate(all_embd, axis=0)

visualize_embedding_heads(all_embd, method='tsne', ax=None)