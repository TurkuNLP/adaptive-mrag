"""
@mtebad

(Part of) a dataset can be downloaded with this file.
"""

import os

cachedir = '/scratch/project_2000539/maryam/embed/.cashe/'
os.environ["HF_HOME"] = "/scratch/project_2000539/maryam/embed/.cashe/"

from datasets import load_dataset

# Load 10% of the dataset (not streaming mode)
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train[:5%]", cache_dir=cachedir
                       , keep_in_memory=False, streaming=False, num_proc=4)

save_path = "/scratch/project_2000539/maryam/fineweb_dataset"
dataset.save_to_disk(save_path)

