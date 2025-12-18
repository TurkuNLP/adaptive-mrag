#!/usr/bin/env python3
"""
Partial-head mixing tests with precomputed U.

Loads:
  - A (Stage-1/2) from pickle dir
  - E3 (Stage-3) from pickle dir
  - U (saved npy from earlier regression)

Runs:
  - Per-block effective ranks (SVD)
  - Head-block top-k reconstruction curve
  - Mini-block top-k reconstruction curve
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Paths (adjust if needed)
# --------------------------
stage1_dir = "data/Fineweb/FineWeb-embedd-stage1_10000/"
stage3_dir = "data/Fineweb/FineWeb-embedd-stage3_10000/"
U_path     = "U_stage1_to_stage3.npy"
out_dir    = "results_partial_head"
os.makedirs(out_dir, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def load_dir_concat(dir_path, limit=None):
    files = sorted(os.listdir(dir_path))
    if limit is not None:
        files = files[:limit]
    all_chunks = []
    for fn in files:
        fp = os.path.join(dir_path, fn)
        with open(fp, "rb") as f:
            while True:
                try:
                    arr = pickle.load(f)
                    arr = np.asarray(arr, dtype=np.float32)
                    all_chunks.append(arr)
                except EOFError:
                    break
    if not all_chunks:
        raise RuntimeError(f"No data loaded from {dir_path}")
    return np.concatenate(all_chunks, axis=0)

def rel_err(X, Y, eps=1e-12):
    return np.linalg.norm(X - Y) / (np.linalg.norm(Y) + eps)

def blockify(U, H, dh):
    return U.reshape(H, dh, H, dh).transpose(0, 2, 1, 3)

def debockify(U_blocks):
    H, _, dh, _ = U_blocks.shape
    return U_blocks.transpose(0, 2, 1, 3).reshape(H*dh, H*dh)

def miniblockify(U, H, dh, G):
    assert dh % G == 0
    m = dh // G
    U4 = U.reshape(H, dh, H, dh)
    U6 = U4.reshape(H, G, m, H, G, m)
    return U6, m

def deminiblockify(U6):
    H, Gin, m, Hout, Gout, _ = U6.shape
    dh = Gin * m
    U4 = U6.reshape(H, dh, Hout, dh)
    return U4.reshape(H*dh, Hout*dh)

def svd_effective_rank(block, energy=0.90):
    s = np.linalg.svd(block, compute_uv=False)
    if s.size == 0:
        return 0
    total = (s**2).sum()
    if total <= 1e-20:
        return 0
    cum = np.cumsum(s**2) / total
    return int(np.searchsorted(cum, energy) + 1)

def plot_heatmap(M, title, path_png):
    plt.figure(figsize=(7,6))
    im = plt.imshow(M, aspect="auto")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def save_curve(xs, ys, path_csv, path_png, title, xlabel, ylabel):
    with open(path_csv, "w") as f:
        f.write("k,rel_err\n")
        for k,y in zip(xs,ys):
            f.write(f"{k},{y:.8f}\n")
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

# --------------------------
# Load data
# --------------------------
print("Loading paired Stage-1/2 (A) and Stage-3 (E3) embeddings...")
A  = load_dir_concat(stage1_dir)
E3 = load_dir_concat(stage3_dir)
U  = np.load(U_path)
print("Shapes:", A.shape, E3.shape, U.shape)

# --------------------------
# Setup
# --------------------------
H = 32
d = A.shape[1]
dh = d // H
print(f"Hidden size d={d}, heads={H}, dh={dh}")

E3_hat_full = A @ U
print("RelErr full U:", rel_err(E3_hat_full, E3))

# --------------------------
# A) Block effective ranks
# --------------------------
U_blocks = blockify(U, H, dh)
R = np.zeros((H,H), dtype=np.int32)
for h in range(H):
    for s in range(H):
        R[h,s] = svd_effective_rank(U_blocks[h,s], energy=0.90)
np.save(os.path.join(out_dir, "block_effective_rank.npy"), R)
plot_heatmap(R, "Effective rank per head→slice block (90% energy)",
             os.path.join(out_dir, "block_effective_rank.png"))
print("Saved block effective rank.")

# --------------------------
# B) Mini-block top-k curve
# --------------------------
def topk_curve_mini(A, E3, U, H, dh, G, ks, out_dir):
    U6, m = miniblockify(U, H, dh, G)
    norms = np.linalg.norm(U6, ord='fro', axis=(2,5))
    rels = []
    for k in ks:
        M6 = np.zeros_like(U6)
        for s in range(H):
            slab = norms[:,:,s,:]
            flat = slab.reshape(-1)
            idx_keep = np.argpartition(flat, -k)[-k:]
            Hin, Gin, Gout = np.unravel_index(idx_keep, slab.shape)
            for hi, gi, go in zip(Hin, Gin, Gout):
                M6[hi,gi,:,s,go,:] = U6[hi,gi,:,s,go,:]
        U_k = deminiblockify(M6)
        rel = rel_err(A @ U_k, E3)
        rels.append(rel)
        print(f"[mini-block] k={k:3d} RelErr={rel:.6f}")
    save_curve(ks, rels,
        os.path.join(out_dir,"topk_miniblock_curve.csv"),
        os.path.join(out_dir,"topk_miniblock_curve.png"),
        f"Mini-block top-k reconstruction (G={G})",
        xlabel="k mini-blocks per slice", ylabel="Relative error")
    return rels

G = 8 if dh % 8 == 0 else 4
ks_mini = [1,2,4,8,16,32,64,128]
topk_curve_mini(A,E3,U,H,dh,G,ks_mini,out_dir)

# --------------------------
# C) Head-block top-k curve
# --------------------------
def topk_curve_head(A, E3, U, H, dh, ks, out_dir):
    U_blocks = blockify(U,H,dh)
    norms = np.linalg.norm(U_blocks, ord='fro', axis=(2,3))
    rels = []
    for k in ks:
        M = np.zeros_like(U_blocks)
        for s in range(H):
            top_rows = np.argsort(norms[:,s])[::-1][:k]
            M[top_rows,s,:,:] = U_blocks[top_rows,s,:,:]
        U_k = debockify(M)
        rel = rel_err(A @ U_k, E3)
        rels.append(rel)
        print(f"[head-block] k={k:2d} RelErr={rel:.6f}")
    save_curve(ks, rels,
        os.path.join(out_dir,"topk_head_curve.csv"),
        os.path.join(out_dir,"topk_head_curve.png"),
        "Head-block top-k reconstruction",
        xlabel="k heads per slice", ylabel="Relative error")
    return rels

ks_head = [1,2,4,8,16,32]
topk_curve_head(A,E3,U,H,dh,ks_head,out_dir)

print("Done. Results saved in", out_dir)
