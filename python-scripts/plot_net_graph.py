import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Data: top positive + two strongest negatives per topic
topic_data = {
    "Entertainment": [(835, 0.019063), (53, -0.064584), (2232, -0.041303)],
    "Business": [(3855, 0.012488), (53, -0.068330), (2232, -0.040116)],
    "Sports": [(835, 0.019761), (53, -0.063408), (2232, -0.042412)],
    "Gaming/Tech": [(3855, 0.013420), (53, -0.078358), (2232, -0.035867)],
    "Personal": [(835, 0.016032), (53, -0.071323), (2232, -0.039737)],
    "Health": [(1641, 0.006414), (53, -0.071392), (2232, -0.041982)],
    "Politics": [(835, 0.017735), (53, -0.090621), (2232, -0.053162)],
    "Places": [(835, 0.012029), (53, -0.071553), (2232, -0.043484)]
}

# Build a bipartite graph
G = nx.Graph()
topic_nodes = []
index_nodes = set()

for topic, pairs in topic_data.items():
    G.add_node(topic, type='topic')
    topic_nodes.append(topic)
    for idx, val in pairs:
        G.add_node(idx, type='index')
        index_nodes.add(idx)
        G.add_edge(topic, idx, weight=abs(val), value=val)

index_nodes = list(index_nodes)

# Layout: topics left, indices right (bipartite look)
pos = {}
pos.update({t: (-1, i) for i, t in enumerate(sorted(topic_nodes))})
pos.update({idx: (1, i) for i, idx in enumerate(sorted(index_nodes))})

# Split edges by sign for styling
pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['value'] > 0]
neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['value'] < 0]

# Scale widths by absolute value
all_abs = [abs(d['value']) for _, _, d in G.edges(data=True)]
max_abs = max(all_abs) if all_abs else 1.0
widths = { (u, v): 2.5 * abs(G[u][v]['value']) / max_abs for u, v in G.edges() }

# Draw
plt.figure(figsize=(12, 7))

# Draw nodes with different shapes (topics: circles, indices: squares)
topic_positions = {n: pos[n] for n in topic_nodes}
index_positions = {n: pos[n] for n in index_nodes}

nx.draw_networkx_nodes(G, topic_positions, nodelist=topic_nodes, node_shape='o', node_size=1400)
nx.draw_networkx_nodes(G, index_positions, nodelist=index_nodes, node_shape='s', node_size=1400)

# Draw edges: positive (solid), negative (dashed). No explicit colors.
nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
                       width=[widths[e] for e in pos_edges], style='solid')
nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                       width=[widths[e] for e in neg_edges], style='dashed')

# Labels
labels = {n: (n if isinstance(n, str) else f"Idx {n}") for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

plt.title("Topic–Index Network Graph (solid = positive, dashed = negative)")
plt.axis('off')
plt.tight_layout()
plt.show()
