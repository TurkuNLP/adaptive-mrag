import os
import re
import json
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def clean_label(label):
    # 1. Replace any comma between number and text with a dot
    label = re.sub(r'^(\d+)[,.]', r'\1.', label.strip())

    # 2. Remove everything after the presence of a second number
    #    e.g., "1. Technology 2. Gaming" -> "1. Technology"
    match = re.match(r'^(\d+\.\s*[^0-9]*)', label)
    if match:
        label = match.group(1).strip()

    # 3. Remove any trailing punctuation like comma or period
    label = re.sub(r'[,.]\s*$', '', label)

    # 4. Make sure only the first dot remains (fix cases like 1.. Technology)
    label = re.sub(r'^(\d+)\.+', r'\1.', label)

    # 5. Remove extra whitespace around the dot and category
    label = re.sub(r'\s*\.\s*', '.', label)
    label = re.sub(r'\s+', ' ', label)

    return label.strip()

with open("/scratch/project_2000539/maryam/adaptive-mrag/data/classified_topics_narrowed.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

for doc in dataset:
    doc["topic"] = clean_label(doc["topic"])

target_topic = "18.Health, Nutrition, Diet, Medicine, Diseases, Biology"
output_path_base = "data/Fineweb/FineWeb-embedd-e5-multi-fi/"


X = []
y = []

for doc in dataset:
    topic = doc["topic"]
    #id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base +  doc['id'] + ".pkl"

    all_embd = []
    with open(output_path, "rb") as f:
        while True:
            try:
                padded_embeddings = pickle.load(f)
                all_embd.append(padded_embeddings)
            except EOFError:
                break

    all_embd = np.concatenate(all_embd, axis=0)
    # Mean over all samples to get one vector per document
    doc_embedding = all_embd.mean(axis=0)
    X.append(doc_embedding)
    y.append(1 if topic == target_topic else 0)

X = np.stack(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Positive examples:", sum(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

results = []  # store tuples (C, nonzero, total, report)

def topk_hits(scores, y_true, k=42):
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return int(y_true[idx].sum())

for C in [1, 0.5, 0.1, 0.01]: 
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)

    scores = clf.decision_function(X_test)  # w·x + b
    hits = topk_hits(scores, y_test, k=42)
    print(f"C={C:<5}  top-42 health hits: {hits}/42")

    weights = clf.coef_[0]
    nonzero = np.count_nonzero(weights)
    total = len(weights)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    results.append((C, nonzero, total, report))
    print(f"C={C}: {nonzero}/{total} nonzero weights")
    if C == 0.1:
        print("nonzero_indices, C:", C, np.nonzero(weights)[0])

    print("-----------------------------------------")

# ---- save results ----

output_path = "l1_experiment_results_FI_e5.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for C, nonzero, total, report in results:
        f.write(f"C={C}\n")
        f.write(f"Nonzero weights: {nonzero}/{total}\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")

print(f"[OK] Saved results to {output_path}")
