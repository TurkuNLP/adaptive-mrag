import os

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["A", "B", "C", "D", "E"]
X_OFFSETS = {"A": -4.0, "B": -2.0, "C": 0.0, "D": 2.0, "E": 4.0}
CONTENT_POINTS = [
    (10.0, 0.0),
    (6.0, 5.0),
    (0.0, 8.0),
    (-6.0, 5.0),
    (-10.0, 0.0),
    (-6.0, -5.0),
    (0.0, -8.0),
    (6.0, -5.0),
]
OUTPUT_DIR = "results/centroid_vs_knn_example"


def build_points():
    points = []
    point_labels = []
    content_ids = []

    for content_id, (y, z) in enumerate(CONTENT_POINTS):
        for label in LABELS:
            x = X_OFFSETS[label]
            points.append([x, y, z])
            point_labels.append(label)
            content_ids.append(content_id)

    return (
        np.array(points, dtype=np.float32),
        np.array(point_labels),
        np.array(content_ids),
    )


def leave_one_out_centroid_accuracy(points, labels):
    unique_labels = np.unique(labels)
    sums = {label: points[labels == label].sum(axis=0) for label in unique_labels}
    counts = {label: int(np.sum(labels == label)) for label in unique_labels}

    correct = 0
    for point, label in zip(points, labels):
        best_label = None
        best_dist = np.inf
        for candidate in unique_labels:
            count = counts[candidate] - (1 if candidate == label else 0)
            if count <= 0:
                continue
            centroid = (sums[candidate] - (point if candidate == label else 0.0)) / count
            dist = np.linalg.norm(point - centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = candidate
        correct += int(best_label == label)
    return correct / len(labels)


def leave_one_out_knn1_accuracy(points, labels):
    correct = 0
    for i, point in enumerate(points):
        dists = np.linalg.norm(points - point, axis=1)
        dists[i] = np.inf
        nearest = int(np.argmin(dists))
        correct += int(labels[nearest] == labels[i])
    return correct / len(labels)


def plot_example(points, labels, content_ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = {
        "A": "tab:blue",
        "B": "tab:orange",
        "C": "tab:green",
        "D": "tab:red",
        "E": "tab:purple",
    }
    centroids = np.vstack([points[labels == label].mean(axis=0) for label in LABELS])

    fig, ax = plt.subplots(figsize=(9, 7))
    for label in LABELS:
        idx = labels == label
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            s=60,
            marker="o",
            color=colors[label],
            alpha=0.85,
            label=label,
        )

    for label, centroid in zip(LABELS, centroids):
        ax.scatter(
            centroid[0],
            centroid[1],
            s=180,
            marker="X",
            color=colors[label],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(centroid[0] + 0.12, centroid[1] + 0.12, f"{label} centroid", fontsize=8)

    ax.set_title("2D example: class colors and centroid markers")
    ax.set_xlabel("class offset (x)")
    ax.set_ylabel("content axis y")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "centroid_vs_knn_example.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_summary(points, labels, content_ids, centroid_acc, knn1_acc, plot_path):
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Synthetic centroid-vs-knn example\n")
        f.write(f"Classes: {', '.join(LABELS)}\n")
        f.write(f"Points: {len(points)}\n")
        f.write(f"Content templates: {len(np.unique(content_ids))}\n")
        f.write(f"Leave-one-out centroid accuracy: {centroid_acc:.4f}\n")
        f.write(f"Leave-one-out kNN accuracy (k=1): {knn1_acc:.4f}\n")
        f.write("Interpretation: x stores class offset, y/z store content structure.\n")
        f.write(f"Plot: {plot_path}\n")
    return summary_path


def main():
    points, labels, content_ids = build_points()
    centroid_acc = leave_one_out_centroid_accuracy(points, labels)
    knn1_acc = leave_one_out_knn1_accuracy(points, labels)
    plot_path = plot_example(points, labels, content_ids)
    summary_path = write_summary(
        points,
        labels,
        content_ids,
        centroid_acc,
        knn1_acc,
        plot_path,
    )

    print(f"[OK] Leave-one-out centroid accuracy: {centroid_acc:.4f}")
    print(f"[OK] Leave-one-out kNN accuracy (k=1): {knn1_acc:.4f}")
    print(f"[OK] Plot saved to: {plot_path}")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
