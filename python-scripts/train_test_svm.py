import json
import os
import re
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

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
output_path_base = "data/Fineweb/FineWeb-embedd-stage3_10000/"

# Process Documents
total_docs = 960  # Total documents

X = []
y = []

for doc in dataset[:total_docs]:
    topic = doc["topic"]
    id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base + id + ".pkl"

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

# ---- Train Linear SVM ----
# Standardization + LinearSVC with class weights to balance positives
clf = make_pipeline(StandardScaler(with_mean=False), 
                    LinearSVC(class_weight="balanced", random_state=42))
clf.fit(X, y)

X_test = []
y_test = []

for doc in dataset[total_docs:]:
    topic = doc["topic"]
    id = ''.join(re.findall(r'[\d-]+', doc['id']))
    output_path = output_path_base + id + ".pkl"

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
    X_test.append(doc_embedding)
    y_test.append(1 if topic == target_topic else 0)

X_test = np.stack(X_test)
y_test = np.array(y_test)

print("Feature matrix shape:", X_test.shape)
print("Positive examples:", sum(y_test))

# Option 1: evaluate normally
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
