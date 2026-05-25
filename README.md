# adaptive-mrag

This repository contains experiments around embedding-based analysis, retrieval, and small controlled dataset evaluations.

## Structure

- `python-scripts/`
  - experiment and analysis scripts, mostly belong to research after RANLP2025.
  - `python-scripts/simple_datasets/` contains the scripts for the simple-dataset experiments (latest ongoing research)
- `data/simple_datasets/`
  - JSONL files for the simple-dataset experiments

## Simple datasets

The simple-dataset code is grouped under `python-scripts/simple_datasets/`.

The corresponding dataset files are in:

- `data/simple_datasets/finnish_cities_dataset.jsonl`
- `data/simple_datasets/rare_names_dataset.jsonl`
- `data/simple_datasets/informative_names_dataset.jsonl`
- `data/simple_datasets/person_actions_dataset.jsonl`
- `data/simple_datasets/generic_names_50x50_dataset.jsonl`

## Running

Run scripts from the repository root, for example:

```bash
python python-scripts/simple_datasets/plot_pca_simple_datasets.py
python python-scripts/simple_datasets/eval_simple_datasets_centroid_knn.py
python python-scripts/simple_datasets/retrieve_simple_datasets_pca.py
```

## Outputs

Generated outputs are written under `results/`.
