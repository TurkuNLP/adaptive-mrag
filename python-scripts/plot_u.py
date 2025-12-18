import numpy as np
import matplotlib.pyplot as plt

# Load the .npy matrix file
matrix = np.load("U_stage1_to_stage3.npy")

print("matrix.shape", matrix.shape)
"""
# Plot full heatmap (4096x4096) - might be heavy but possible
plt.figure(figsize=(8, 8))
plt.imshow(matrix, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("Full Matrix Heatmap (4096x4096)")
plt.savefig("s1.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
"""
# Downsampled version (512x512)
step = matrix.shape[0] // 512
matrix_downsampled = matrix[::step, ::step]

plt.figure(figsize=(8, 8))
plt.imshow(matrix_downsampled, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("Downsampled Matrix Heatmap (512x512)")
plt.savefig("s2.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

n = matrix.shape[0]

def block_average_downsample(arr, target_n):
    """Downsample arr to (target_n, target_n) by block averaging (no aliasing by simple striding)."""
    assert arr.shape[0] == arr.shape[1]
    f = arr.shape[0] // target_n  # integer factor (assumes divisible)
    # reshape to (target_n, f, target_n, f) and average over the small blocks
    return arr.reshape(target_n, f, target_n, f).mean(3).mean(1)

# Create several downsampled versions
sizes = [1024, 512, 256, 128]
downs = {}

for s in sizes:
    downs[s] = block_average_downsample(matrix, s)

# Show a clear overview with a 256x256 heatmap
plt.figure(figsize=(7, 7))
plt.imshow(downs[256], aspect="auto")
plt.colorbar()
plt.title("Downsampled (block-averaged) Heatmap — 256×256")
plt.savefig("s3.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()