from datasets import load_from_disk

# Load local dataset
dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")

# UUIDs from your results (cleaned to remove angle brackets)
target_ids = {
    "<urn:uuid:b661bc06-9425-48a3-990b-ed8cdf93dcb9>",
    "<urn:uuid:dce2b195-b978-468f-a8f1-59486a7b8340>",
    "<urn:uuid:de805fbe-a7c9-4436-bdbc-60980db04159>",
    "<urn:uuid:cda9e1fb-722e-4a86-b774-a117bdfe2171>",
    "<urn:uuid:d10aaf0a-9479-4803-9974-0316a807836a>",
    "<urn:uuid:6fee35ad-ae85-43ec-b57b-b3367be6f892>",
    "<urn:uuid:5d19ca73-bc1b-44f8-8316-84154f0d17d8>",
    "<urn:uuid:a6f89f97-77ec-4701-b616-b8cc74c8e0ed>",
    "<urn:uuid:8280daa2-29c5-4b63-8795-63a2ef93bba9>",
    "<urn:uuid:9e375e75-dd91-4f9b-9c6b-35521022cfd6>",
}

# Search and print matching texts
print("Iterating the dataset...")
for record in dataset:
    if record["id"] in target_ids:
        print(f"\n--- Document ID: {record['id']} ---")
        print(record["text"])
        print("\n" + "-"*80)
