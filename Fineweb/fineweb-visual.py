import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
import gc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datasets import load_from_disk
import os

def visualize_embedding_heads_with_magnitude_shading(embeddings, method='tsne', ax=None):
    """
    Visualize clusters in the embeddings by splitting each into 32 heads with color intensity representing magnitude.
    
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
        #reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'ica':
        reducer = FastICA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne', 'pca', or 'ica'")

    reduced_embeddings = reducer.fit_transform(flat_embeddings)

    # Step 3: Plot with varying color intensities based on magnitude for each head
    if ax is None:
        ax = plt.gca()

    # Assign 32 base colors, one for each head
    base_colors = plt.cm.get_cmap('tab20b', num_heads).colors  # Colormap with 32 colors

    for head_idx in range(num_heads):
        indices = np.arange(head_idx, n_samples * num_heads, num_heads)

        # Compute magnitudes for the current head's embeddings
        head_embeddings = split_embeddings[:, head_idx, :]
        magnitudes = np.linalg.norm(head_embeddings, axis=1)

        # Normalize magnitudes to [0, 1] for color intensity
        norm_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-9)
        
        # Adjust color intensity based on magnitude
        head_colors = [base_colors[head_idx] + (1 - norm) * (1 - np.array(base_colors[head_idx])) for norm in norm_magnitudes]

        ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                   color=head_colors, label=f'Head {head_idx+1}', s=1, alpha=0.7)

    ax.set_title(f'Embedding Visualization by Head with Magnitude Shading ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(title="Heads", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.savefig("fineweb_sfr_plot_shaded.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def visualize_embedding_specific_heads_with_magnitude_shading(embeddings, specific_heads, method='tsne', ax=None):
    """
    Visualize clusters in the embeddings by splitting each into 32 heads with color intensity representing magnitude.
    
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
        #reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'ica':
        reducer = FastICA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne', 'pca', or 'ica'")

    reduced_embeddings = reducer.fit_transform(flat_embeddings)

    # Step 3: Plot with varying color intensities based on magnitude for each head
    if ax is None:
        ax = plt.gca()

    # Assign 32 base colors, one for each head
    base_colors = plt.cm.get_cmap('tab20b', num_heads).colors  # Colormap with 32 colors
    tops = []

    for head_idx in specific_heads:
        indices = np.arange(head_idx, n_samples * num_heads, num_heads)

        # Compute magnitudes for the current head's embeddings
        head_embeddings = split_embeddings[:, head_idx, :]
        magnitudes = np.linalg.norm(head_embeddings, axis=1)

        # Normalize magnitudes to [0, 1] for color intensity
        norm_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-9)
        
        sorted_indices = np.argsort(magnitudes)  # Sort magnitudes in ascending order
        top_5_indices = sorted_indices[-10:][::-1]  # Last 5, reversed for descending order
        low_5_indices = sorted_indices[:5] 

        tops.append([top_5_indices, low_5_indices])
        
        # Adjust color intensity based on magnitude
        head_colors = [base_colors[head_idx] + (1 - norm) * (1 - np.array(base_colors[head_idx])) for norm in norm_magnitudes]

        ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                   color=head_colors, label=f'Head {head_idx+1}', s=1, alpha=0.7)

    ax.set_title(f'Embedding Visualization by Head with Magnitude Shading ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(title="Heads", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.savefig("fineweb_sfr_heads.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return tops

all_embd = []
directory_path = "FineWeb-embedd-modified"
"""
all_txt = []
print("loading from disk")
dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")
"""
for file_index, filename in enumerate(os.listdir(directory_path)):

    #all_txt.append(dataset[file_index]['text'])
    with open(os.path.join(directory_path, filename), "rb") as f:
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

print("everything loaded")
print(len(all_embd))
print(len(all_embd[0]))
print(len(all_embd[0][0]))

all_embd = np.concatenate(all_embd, axis=0) # [0][:partition]

print("everything concTINATED")
print(len(all_embd))
print(len(all_embd[0]))
#print(len(all_embd[0][0]))

#visualize_embedding_heads(all_embd, method='tsne', ax=None)
print("visualize_embedding_specific_heads_with_magnitude_shading")
visualize_embedding_heads_with_magnitude_shading(all_embd, method='tsne', ax=None)

heads = [3,13,26]
print("visualize_embedding_specific_heads_with_magnitude_shading")
tops_and_lows = visualize_embedding_specific_heads_with_magnitude_shading(all_embd,heads, method='tsne', ax=None)
"""all_txt = np.array(all_txt)
for head_idx, (top_5, low_5) in enumerate(tops_and_lows):
    print("head: ", heads[head_idx])
    print("Top 5: ", all_txt[top_5])
    print("Low 5: ", all_txt[low_5])"""