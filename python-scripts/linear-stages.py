"""
Linear unmixing: learn U so that A @ U ≈ E3, then analyze head mixing.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment

# --------------------------
# Config
# --------------------------
num_heads = 32
d = 4096
dh = d // num_heads   # head slice size
assert d % num_heads == 0, "d must be divisible by num_heads"

# Adjust these to your paths
stage1_dir = "data/Fineweb/FineWeb-embedd-stage1_10000/"  # <- your Stage-1 (or Stage-2) dumps
stage3_dir = "data/Fineweb/FineWeb-embedd-stage3_10000/"  # <- your Stage-3 dumps
save_U_path = "U_stage1_to_stage3.npy"

# --------------------------
# Helpers
# --------------------------
def load_dir_concat(dir_path, limit=None):
    """
    Loads all pickled arrays from dir_path and concatenates along axis 0.
    Each file may contain multiple pickled arrays (batches).
    Returns a single np.array [N, d].
    """
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
                    # arr should be [batch, d], or similar; convert to float32
                    arr = np.asarray(arr, dtype=np.float32)
                    all_chunks.append(arr)
                except EOFError:
                    break
    if not all_chunks:
        raise RuntimeError(f"No data loaded from {dir_path}")
    return np.concatenate(all_chunks, axis=0)

def block_norm_matrix(U, H, dh):
    """
    Reshape U [d,d] into HxH blocks of size dh x dh and compute Frobenius norm per block.
    Returns [H,H] matrix of block norms.
    """
    U4 = U.reshape(H, dh, H, dh).transpose(0, 2, 1, 3)  # [H,H,dh,dh]
    norms = np.linalg.norm(U4, ord='fro', axis=(2,3))    # [H,H]
    return norms

# --------------------------
# 1) Load paired data
# --------------------------
print("Loading paired Stage-1/2 (A) and Stage-3 (E3) embeddings...")
A = load_dir_concat(stage1_dir, limit=None)   # [N, d]
E3 = load_dir_concat(stage3_dir,  limit=None) # [N, d]

# Defensive checks
if A.shape[0] != E3.shape[0]:
    raise ValueError(f"Mismatched sample counts: A={A.shape[0]} vs E3={E3.shape[0]}")
if A.shape[1] != d or E3.shape[1] != d:
    raise ValueError(f"Expected dim {d}, got A={A.shape[1]}, E3={E3.shape[1]}")

N = A.shape[0]
print(f"Loaded N={N} samples, d={d}")

# Optionally subsample for speed while testing
# idx = np.random.RandomState(0).choice(N, size=min(N, 200000), replace=False)
# A = A[idx]; E3 = E3[idx]

# --------------------------
# 2) Fit linear map U (no intercept)
# --------------------------
print("Fitting linear regression A @ U ≈ E3 ...")
reg = LinearRegression(fit_intercept=False, n_jobs=-1)
reg.fit(A, E3)
U = reg.coef_.astype(np.float32)  # shape [d, d]; sklearn stores as [out_dim, in_dim]
U = U.T                           # convert to [in_dim, out_dim] so A @ U ≈ E3

np.save(save_U_path, U)
print(f"Saved U to {save_U_path}")

# Reconstruction quality
E3_hat = A @ U
mse = np.mean((E3_hat - E3)**2)
rel_err = np.linalg.norm(E3_hat - E3) / (np.linalg.norm(E3) + 1e-12)
print(f"Reconstruction MSE: {mse:.6e}")
print(f"Relative error  : {rel_err:.6e}")

# --------------------------
# 3) Head mixing analysis
# --------------------------
print("Analyzing block mixing (32x32 blocks)...")
bn = block_norm_matrix(U, num_heads, dh)  # [H,H] block Frobenius norms

# Normalize rows to sum to 1 for interpretability
row_sums = bn.sum(axis=1, keepdims=True) + 1e-12
bn_rownorm = bn / row_sums

# Off-diagonal energy (lower is closer to permutation)
off_diag = (bn_rownorm.sum() - np.trace(bn_rownorm)) / num_heads
print(f"Avg off-diagonal mass (row-normalized): {off_diag:.4f} (0=ideal permutation, 1=fully mixed)")

# Hungarian matching to find best head->slice assignment
# We want to maximize block norm on diagonal after permutation ⇒ minimize negative norms
cost = -bn
row_ind, col_ind = linear_sum_assignment(cost)
perm_mass = bn[row_ind, col_ind].sum() / (bn.sum() + 1e-12)
print(f"Hungarian matched diagonal mass (fraction of total): {perm_mass:.4f}")

# Also report how peaky each row is (top-1 minus top-2)
top1 = bn_rownorm.max(axis=1)
top2 = np.partition(bn_rownorm, -2, axis=1)[:, -2]
avg_margin = float(np.mean(top1 - top2))
print(f"Avg row peak margin (top1-top2): {avg_margin:.4f} (higher=cleaner 1-1 mapping)")

# Optional: simple textual heat summary (which slice each head maps to)
mapping = [(h, int(bn_rownorm[h].argmax()), float(bn_rownorm[h].max())) for h in range(num_heads)]
print("Head -> Best slice (row-normalized mass):")
for h, s, mass in mapping[:10]:  # print first 10 for brevity
    print(f"  head {h:2d} -> slice {s:2d} (mass={mass:.3f})")
