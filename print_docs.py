from datasets import load_from_disk
from tqdm import tqdm
import re
import os

idx = [0, 7673, 4150, 7141, 4306]
directory_path = "data/Fineweb/FineWeb-embedd-stage3_10000"
files = os.listdir(directory_path)
selected_files = [files[i] for i in idx]

dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")
number_of_doc = 0

# Process files
for file in tqdm(dataset, desc="Processing documents", total=len(dataset)):

    if number_of_doc > 10000:
        break

    id = ''.join(re.findall(r'[\d-]+', file['id']))
    embd_file =  id + ".pkl"

    if embd_file not in selected_files:
        continue

    index = selected_files.index(embd_file)
    text = file['text']
    number_of_doc += 1

    print(idx[index], text)

