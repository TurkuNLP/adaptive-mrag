"""
@mtebad

This file prints the text of given documents.
"""

from datasets import load_from_disk

# Load local dataset
dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")

# UUIDs from your results (cleaned to remove angle brackets)
target_ids = {
    "<urn:uuid:c970d9a2-a5ce-4050-9ea3-58d7bbd609a8>",
    "<urn:uuid:8ba5fa5a-1f92-4f5a-95e3-85dbb7befbfe>",
    "<urn:uuid:4ad1d60f-e986-44b4-8b58-e833a6dddba8>",
    "<urn:uuid:9a56e7c7-5d04-41c9-8c2c-bd8518657d95>",
    "<urn:uuid:095340b9-8a67-45ac-afb3-250bec8e3efd>",
}

# Search and print matching texts
print("Iterating the dataset...")
for record in dataset:
    if record["id"] in target_ids:
        print(f"\n--- Document ID: {record['id']} ---")
        print(record["text"])
        print("\n" + "-"*80)
