"""
@mtebad

This file visualizes the projection matrix of mistral model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel

# Model and Tokenizer Setup
model_name = "Salesforce/SFR-Embedding-Mistral"  # Change if needed
cache_dir = "/scratch/project_2000539/maryam/embed/.cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()
matrix = model.layers[-1].self_attn.o_proj

# Assume matrix is a torch.nn.Linear layer, e.g. model.layers[-1].self_attn.o_proj
# Extract weights
weight_matrix = matrix.weight.detach().cpu().numpy()[:400,:400]  # shape: (out_features, in_features)

print("Min value:", weight_matrix.min())
print("Max value:", weight_matrix.max())
print("Mean value:", weight_matrix.mean())
print("Standard deviation:", weight_matrix.std())

# Reversed
"""
weight_matrix = matrix.weight  
reversed_matrix = torch.flip(weight_matrix, dims=[0])  # flip rows (dim 0)
reversed_matrix_np = reversed_matrix.detach().cpu().numpy()
"""

# Normalized
"""
normalized = (weight_matrix - weight_matrix.mean()) / weight_matrix.std()
sns.heatmap(normalized, cmap='coolwarm', center=0)
plt.title("Z-score Normalized o_proj Weights")
"""


sns.heatmap(weight_matrix, cmap='viridis', cbar=True, center=0, vmin=-0.01, vmax=0.01)
plt.title("Heatmap of o_proj Weight Matrix")
plt.xlabel("Input Features")
plt.ylabel("Output Features")
plt.tight_layout()
plt.savefig("fineweb_projection_matrix_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()