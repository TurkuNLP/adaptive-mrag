# https://huggingface.co/datasets/mteb/sts12-sts

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from mteb import MTEB, get_tasks

import functools
import tqdm.auto as tqdma
tqdma.tqdm = functools.partial(tqdma.tqdm, disable=True)

bnb = BitsAndBytesConfig(load_in_8bit=True)

# Load ST model without custom hooks
model = SentenceTransformer(
    "Salesforce/SFR-Embedding-Mistral",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_folder=os.environ["HF_HOME"],
    model_kwargs={
        "quantization_config": bnb,
        "trust_remote_code": True,   # SFR provides its own encode pipeline
    }
)

# match your truncation length
model.max_seq_length = 32768

# MTEB run
tasks = get_tasks(tasks=["STS12"])
evaluation = MTEB(tasks=tasks, eval_splits=["test"])

# give it a stable name + cosine (MTEB detects cosine for ST models automatically,
# but we can be explicit via model.similarity if you wrap; not required here)
results = evaluation.run(
    model,
    output_folder="./sts-12-test",
    encode_kwargs={"batch_size": 4, "show_progress": True},  # bump batch size!
)

for tr in results:
    print(f"{tr.task_name}: {tr.get_score():.4f}  (main: {tr.task.metadata.main_score})")

    print(tr.only_main_score().to_dict())
