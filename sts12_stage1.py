# pip install mteb faiss-cpu ir_measures pytrec_eval transformers datasets
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch, numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from mteb import MTEB, get_tasks
import random
import os
from tqdm import tqdm
from datasets import Dataset


# sanity check env
for k in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
    print(k, os.environ.get(k))
# GPU (cuda) optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- your hook path (unchanged) ----
def capture_heads(module, input, output):
    out = input[0]
    if out.dim() == 2: out = out.unsqueeze(0)
    elif out.dim() != 3: raise ValueError(f"Unexpected shape: {out.shape}")
    module.cached_heads = out

def mean_pooling(embeddings, mask):
    input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask

# ---- minimal model wrapper MTEB needs ----
class HookEncoder:
    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral", cache_dir=None, max_length=32768, fp16=True):
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # GPU (cuda) optimization
        #bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
        #                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        bnb = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir,
            quantization_config=bnb, # torch_dtype=(torch.float16 if fp16 else torch.float32)
            device_map="auto"
        )
        self.model.config.use_cache = False
        self.model.eval()

        # verify:
        print(any("Linear4bit" in m.__class__.__name__ for m in self.model.modules()))

        self.model.layers[-1].self_attn.o_proj.register_forward_hook(capture_heads)
        self.max_length = max_length
        self.similarity_fn_name = "cosine"

    @torch.no_grad()
    def encode(self, texts, batch_size=1, show_progress=True, **kwargs):
        outs = []
        all_heads, all_masks = [], []

        total = len(texts)
        pbar = tqdm(total=total, disable=not show_progress, desc="Encoding")  # <-- progress bar

        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i+batch_size], return_tensors="pt", padding="max_length",
                           truncation=True, max_length=self.max_length)
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = self.model(**enc)  # fills .cached_heads
            heads = self.model.layers[-1].self_attn.o_proj.cached_heads  # [B, L, H]

            #heads = heads.to(dtype=torch.float32)

            all_heads.append(heads.detach().cpu())
            all_masks.append(enc["attention_mask"].detach().cpu())

            # GPU (cuda) optimization
            self.model.layers[-1].self_attn.o_proj.cached_heads = None
            del heads, enc
            torch.cuda.empty_cache()

            pbar.update(min(batch_size, total - i))  # <-- update

        pbar.close()

        H = torch.cat(all_heads, dim=0)   # [N_total, L, H]
        M = torch.cat(all_masks, dim=0)   # [N_total, L]
        pooled = mean_pooling(H, M).cpu().numpy()
        return pooled

# ---- one line to run MLDR ----
tasks = get_tasks(tasks=["STS12"])

# downsampling
# 1) Get the task and load its data (this populates task.dataset)
task = get_tasks(tasks=["STS12"])[0]
task.load_data(split="test")  # returns None; fills task.dataset

# 2) Helper to take a fraction of a Hugging Face Dataset
def take_frac(ds: Dataset, frac=0.01, seed=42):
    n = max(1, int(len(ds) * frac))
    return ds.shuffle(seed=seed).select(range(n))

# 3) Grab the test split from task.dataset and subset it
test_split = task.dataset.get("test")

# STS12 may be a Dataset or a dict of language -> Dataset. Handle both.
if isinstance(test_split, Dataset):
    small = take_frac(test_split, frac=0.01, seed=42)
elif isinstance(test_split, dict):
    # e.g., {"en": Dataset, ...}
    small = {lang: take_frac(ds, frac=0.01, seed=42) for lang, ds in test_split.items()}
else:
    raise RuntimeError(f"Unexpected test split type: {type(test_split)}")

# 4) Replace the task's test split with the subset
task.dataset["test"] = small

evaluation = MTEB(tasks=[task], eval_splits=["test"])

model = HookEncoder(cache_dir="/scratch/project_2000539/maryam/embed/.cache", max_length=32768)
model.name = "local/SFR-Embedding-Mistral-o-proj-hook"

meta = {
    "name": "local/SFR-Embedding-Mistral-o-proj-hook-64",   # must be org/name-ish
}

results = evaluation.run(
    model, 
    output_folder="./sts-12-stage1",
    encode_kwargs={"batch_size": 1, "show_progress": True},
    model_meta=meta,                # <-- this bypasses the auto-introspection
)
for tr in results:
    print(f"{tr.task_name}: {tr.get_score():.4f}  (main: {tr.task.metadata.main_score})")

    print(tr.only_main_score().to_dict())