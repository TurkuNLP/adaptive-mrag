import numpy as np

# Prepare data for scatter plot
topics = [
    "Entertainment", "Business", "Sports", "Gaming/Tech",
    "Personal", "Health", "Politics", "Places"
]


# Prepare reduced data: only first top positive and first two bottom negative per topic
reduced_topic_data = {
    "Entertainment": [(835, 0.019063), (53, -0.064584), (2232, -0.041303)],
    "Business": [(3855, 0.012488), (53, -0.068330), (2232, -0.040116)],
    "Sports": [(835, 0.019761), (53, -0.063408), (2232, -0.042412)],
    "Gaming/Tech": [(3855, 0.013420), (53, -0.078358), (2232, -0.035867)],
    "Personal": [(835, 0.016032), (53, -0.071323), (2232, -0.039737)],
    "Health": [(1641, 0.006414), (53, -0.071392), (2232, -0.041982)],
    "Politics": [(835, 0.017735), (53, -0.090621), (2232, -0.053162)],
    "Places": [(835, 0.012029), (53, -0.071553), (2232, -0.043484)]
}

# Assign colors to topics
colors = plt.cm.tab10(np.linspace(0, 1, len(reduced_topic_data)))

# Plot scatter with jittering for clarity
plt.figure(figsize=(12, 6))
np.random.seed(42)

for topic, color in zip(reduced_topic_data.keys(), colors):
    indices = np.array([i for i, _ in reduced_topic_data[topic]]) + np.random.uniform(-5, 5, 3)
    values = [v for _, v in reduced_topic_data[topic]]
    plt.scatter(indices, values, label=topic, s=80, alpha=0.8, color=color, edgecolor='k')

plt.axhline(0, color='black', linewidth=1)
plt.title("Bipolar Scatter Plot (Top + 2 Bottom Contributions per Topic)", fontsize=14)
plt.xlabel("Embedding Index (with jitter)")
plt.ylabel("Contribution Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
