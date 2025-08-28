# pip install mteb faiss-cpu ir_measures pytrec_eval transformers datasets
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch, numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from mteb import MTEB, get_tasks
import os

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
    def __init__(self, model_name="Salesforce/SFR-Embedding-Mistral", cache_dir=None, max_length=64, fp16=True):
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # GPU (cuda) optimization
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir,
            quantization_config=bnb, # torch_dtype=(torch.float16 if fp16 else torch.float32)
            device_map="cuda"
        )
        self.model.config.use_cache = False
        self.model.eval()

        self.model.layers[-1].self_attn.o_proj.register_forward_hook(capture_heads)
        self.max_length = max_length
        self.similarity_fn_name = "cosine"

    @torch.no_grad()
    def encode(self, texts, batch_size=1, **kwargs):
        outs = []
        all_heads, all_masks = [], []

        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i+batch_size], return_tensors="pt", padding="max_length",
                           truncation=True, max_length=self.max_length)
            enc = {k: v.cuda() for k, v in enc.items()}
            _ = self.model(**enc)  # fills .cached_heads
            heads = self.model.layers[-1].self_attn.o_proj.cached_heads  # [B, L, H]

            heads = heads.to(dtype=torch.float32)

            all_heads.append(heads)
            all_masks.append(enc["attention_mask"])

            # GPU (cuda) optimization
            self.model.layers[-1].self_attn.o_proj.cached_heads = None
            del heads, enc
            torch.cuda.empty_cache()

        H = torch.cat(all_heads, dim=0)   # [N_total, L, H]
        M = torch.cat(all_masks, dim=0)   # [N_total, L]
        pooled = mean_pooling(H, M).cpu().numpy()
        return pooled

# ---- one line to run MLDR ----
model = HookEncoder(cache_dir="/scratch/project_2000539/maryam/embed/.cache", max_length=64)
tasks = get_tasks(tasks=["MultiLongDocRetrieval"])   # <- not "MLDR"
evaluation = MTEB(tasks=tasks, eval_splits=["test"])

model.name = "local/SFR-Embedding-Mistral-o-proj-hook"

try:
    from sentence_transformers import SentenceTransformerModelCardData
    model.model_card_data = SentenceTransformerModelCardData(
        model_name=model.name,
        language=["eng-Latn"]
    )
except Exception:
    class _Card:
        def __init__(self, model_name, languages):
            self.model_name = model_name; self.languages = languages
    model.model_card_data = _Card(model.name, ["eng-Latn"])

meta = {
    "name": "local/SFR-Embedding-Mistral-o-proj-hook",   # must be org/name-ish
    "languages": ["eng-Latn"],
}

results = evaluation.run(
    model, 
    output_folder="./mldr-results",
    encode_kwargs={"batch_size": 1},
    model_meta=meta,                # <-- this bypasses the auto-introspection
    task_langs=["en"],
)
for tr in results:
    print(f"{tr.task_name}: {tr.get_score():.4f}  (main: {tr.task.metadata.main_score})")

    print(tr.only_main_score().to_dict())