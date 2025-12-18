import numpy as np
import os
import pickle

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

print("Loading paired Stage-1/2 (A) and Stage-3 (E3) embeddings...")
A = load_dir_concat(stage1_dir, limit=None)   # [N, d]
E3 = load_dir_concat(stage3_dir,  limit=None) # [N, d]

H, d = 32, 4096
dh = d // H
U = np.load(save_U_path)  # [d,d]

# U is [d, d]
U_blocks = U.reshape(H, dh, H, dh)          # [H, dh, H, dh]

# 32x32 mask for keeping only diagonal head-to-slice blocks
mask = np.eye(H, dtype=U.dtype)             # [H, H]
mask4 = mask[:, None, :, None]              # [H, 1, H, 1] -> broadcasts over dh, dh

# Zero off-diagonal blocks by broadcast-multiplication
U_bd_blocks = U_blocks * mask4              # keep diag blocks, zero others

# Back to [d, d]
U_bd = U_bd_blocks.reshape(d, d)

E3_hat_full = A @ U
E3_hat_bd   = A @ U_bd

def rel_err(X, Y): 
    return np.linalg.norm(X-Y) / (np.linalg.norm(Y)+1e-12)

print("RelErr full:", rel_err(E3_hat_full, E3))
print("RelErr blkdiag:", rel_err(E3_hat_bd, E3))


def block_norm_matrix(U, H, dh):
    """
    Reshape U [d,d] into HxH blocks of size dh x dh and compute Frobenius norm per block.
    Returns [H,H] matrix of block norms.
    """
    U4 = U.reshape(H, dh, H, dh).transpose(0, 2, 1, 3)  # [H,H,dh,dh]
    norms = np.linalg.norm(U4, ord='fro', axis=(2,3))    # [H,H]
    return norms

bn = block_norm_matrix(U, H, dh) 

from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression

# block norms from before: bn = block_norm_matrix(U, H, dh)  # [H,H]
row_ind, col_ind = linear_sum_assignment(-bn)  # maximize norms

# Split A and E3 into head/slice blocks
A_blocks  = A.reshape(-1, H, dh)    # [N, H, dh]
E_blocks  = E3.reshape(-1, H, dh)   # [N, H, dh]

R_list = []
for h, s in zip(row_ind, col_ind):
    Xh = A_blocks[:, h, :]  # [N, dh]
    YS = E_blocks[:, s, :]  # [N, dh]
    reg = LinearRegression(fit_intercept=False).fit(Xh, YS)
    R_list.append(reg.coef_.T)  # [dh, dh]

# Build block-diagonal R in permuted order
P = np.zeros((H,H)); P[row_ind, col_ind] = 1
R = np.zeros((d,d))
for h in range(H):
    s = col_ind[h]  # matched slice
    R[h*dh:(h+1)*dh, s*dh:(s+1)*dh] = R_list[h]

E3_hat_perm = A @ R
print("RelErr perm+perhead:", rel_err(E3_hat_perm, E3))

import numpy as np

def procrustes_orth(X, Y):
    # X @ R ≈ Y, R orthogonal
    U_, S_, Vt_ = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U_ @ Vt_

R_list_ortho = []
for h, s in zip(row_ind, col_ind):
    Xh = A_blocks[:, h, :]  # [N, dh]
    YS = E_blocks[:, s, :]  # [N, dh]
    R_h = procrustes_orth(Xh, YS)
    R_list_ortho.append(R_h)

R_ortho = np.zeros((d,d))
for h in range(H):
    s = col_ind[h]
    R_ortho[h*dh:(h+1)*dh, s*dh:(s+1)*dh] = R_list_ortho[h]

E3_hat_ortho = A @ R_ortho
print("RelErr perm+orthoprocrustes:", rel_err(E3_hat_ortho, E3))

"""results:
RelErr full: 0.10007241856866812
RelErr blkdiag: 1.7256154794603904
RelErr perm+perhead: 0.31216960433469243
RelErr perm+orthoprocrustes: 0.9472323017042406
"""