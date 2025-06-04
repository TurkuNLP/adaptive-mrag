from datasets import load_dataset

cachedir = '/scratch/project_2000539/maryam/embed/.cashe/'

# Load 20% of the dataset (not streaming mode)
dataset_en = load_dataset("mteb/mtop_domain", name="en", split="train", cache_dir=cachedir
                       , keep_in_memory=False, streaming=False, num_proc=4)

dataset_de = load_dataset("mteb/mtop_domain", name="de", split="train", cache_dir=cachedir
                       , keep_in_memory=False, streaming=False, num_proc=4)

save_path_en = "/scratch/project_2000539/maryam/mtop_domain/en"
dataset_en.save_to_disk(save_path_en)

save_path_de = "/scratch/project_2000539/maryam/mtop_domain/de"
dataset_de.save_to_disk(save_path_de)
