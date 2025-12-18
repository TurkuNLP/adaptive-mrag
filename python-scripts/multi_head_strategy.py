import pickle
import numpy as np
import heapq
import os
from typing import Callable, List, Tuple

class LocalMultiHeadStrategyFromFlat:
    def __init__(self, all_flat_embeddings: List[np.ndarray], weight_fn: Callable[[float, int, float], float]):
        self.num_heads = 32
        print("check: ", all_flat_embeddings[0], all_flat_embeddings[0].shape[0])
        self.head_dim = all_flat_embeddings[0].shape[0] // self.num_heads
        self.weight_fn = weight_fn
        self.documents = [
            {
                "id": i,
                "embedding": emb.reshape(self.num_heads, self.head_dim),
            }
            for i, emb in enumerate(all_flat_embeddings)
        ]
        self.head_scales = self._compute_head_scales()

    def _compute_head_scales(self) -> List[float]:
        # Compute: mean pairwise distance * mean norm for each head
        scales = []
        for head_idx in range(self.num_heads):
            head_embs = np.array([doc["embedding"][head_idx] for doc in self.documents])
            mean_norm = np.mean(np.linalg.norm(head_embs, axis=1))

            dists = []
            for i in range(len(head_embs)):
                for j in range(i+1, len(head_embs)):
                    dists.append(np.linalg.norm(head_embs[i] - head_embs[j]))
            mean_dist = np.mean(dists)
            scales.append(mean_dist * mean_norm)
        return scales

    def _search(self, query: np.ndarray, n: int) -> List[List[Tuple[float, int]]]:

        query_heads = query.reshape(self.num_heads, self.head_dim)
        results = []

        for head_idx in range(self.num_heads):
            q = query_heads[head_idx]
            distances = [
                (np.linalg.norm(q - doc["embedding"][head_idx]), doc["id"])
                for doc in self.documents
            ]
            top_n = sorted(distances, key=lambda x: x[0])[:n]
            results.append(top_n)
        return results

    def _multi_vote(self, query: np.ndarray, n: int) -> List[Tuple[float, int]]:
        ranking = self._search(query, n)
        votes = {}

        for head_idx, head_ranking in enumerate(ranking):
            scale = self.head_scales[head_idx]
            for rank, (dist, doc_id) in enumerate(head_ranking):
                score = self.weight_fn(scale, rank, dist)
                votes[doc_id] = votes.get(doc_id, 0.0) + score

        top_docs = heapq.nlargest(n, votes.items(), key=lambda x: x[1])
        print("top_docs:", top_docs)
        return top_docs

    def get_top_docs(self, query: np.ndarray, n: int = 5) -> List[int]:
        return [doc_id for doc_id, _ in self._multi_vote(query, n)]

def default_weight_fn(scale, rank, dist):
    return scale * 2 ** (-rank)


all_embd = []
directory_path = "data/Fineweb/FineWeb-embedd-stage3_10000"
"""
all_txt = []
print("loading from disk")
dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")
"""
for file_index, filename in enumerate(os.listdir(directory_path)):

    #all_txt.append(dataset[file_index]['text'])

    if(len(all_embd) > 10000):
        break

    with open(os.path.join(directory_path, filename), "rb") as f:
        while True:
            try:
                # Load one batch at a time
                padded_embeddings = pickle.load(f)
                
                # Apply mean pooling to the batch
                # Make sure it's 1D: (4096,) instead of (1, 4096)
                flat = np.array(padded_embeddings).flatten()
                if flat.shape[0] % 32 != 0:
                    print(f"Skipping embedding with unexpected shape: {flat.shape}")
                    continue
                all_embd.append(flat)
            
            except EOFError:
                # End of file reached
                break

strategy = LocalMultiHeadStrategyFromFlat(all_embd, default_weight_fn)

query = all_embd[0]  # for testing, use one of the docs

top_doc_ids = strategy.get_top_docs(query, n=5)
print("Top 5 doc IDs:", top_doc_ids)
print([os.listdir(directory_path)[i] for i in top_doc_ids])
