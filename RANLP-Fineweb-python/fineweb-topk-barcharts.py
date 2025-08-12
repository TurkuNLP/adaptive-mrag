import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

methods = ['Full Embd', 'Weighted Heads Pruned', 'Weighted heads']
categories = ['Top-1', 'Top-5', 'Top-10', 'Other']

# Simulated normalized similarity contribution for each category (must sum to 1 per method)
scores = [
    [0.19, 0.33, 0.11, 0.37],  # Average Pooling
    [0.16, 0.32, 0.22, 0.30],     # Weighted Pooling - with 0s
    [0.03, 0.39, 0.17, 0.42]    # Weighted Pooling
]

# Create a horizontal stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Position for each bar
y_pos = np.arange(len(methods))

# Accumulator for stacking
left = np.zeros(len(methods))

# Colors for segments (use different colors for each topic)
colors = plt.cm.tab20c(np.linspace(0, 1, 4))

for i in range(len(categories)):
    values = [scores[j][i] for j in range(len(methods))]
    labels = [categories[i]] * len(methods)
    ax.barh(y_pos, values, left=left, color=colors[i], edgecolor='white', label=categories[i])
    
    # Annotate with category names
    for j in range(len(methods)):
        ax.text(left[j] + values[j]/2, y_pos[j], f"{categories[i]} ({values[j]*100:.0f}%)",
                ha='center', va='center', fontsize=9, color='black')
    
    # Update stacking
    left += values

# Y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('Normalized Similarity Contribution')
ax.set_title('Top-K Topic Similarity Contribution by Embedding Method')
ax.tick_params(axis='y', pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

plt.tight_layout()
plt.savefig("fineweb_topk.png", dpi=300, bbox_inches="tight")
plt.show()