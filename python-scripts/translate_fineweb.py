import os
import json
from datasets import load_from_disk, concatenate_datasets
from datasets import DatasetInfo
from google.oauth2 import service_account
from google.cloud import translate_v3 as translate  # pip install google-cloud-translate

keep_ids = set()
with open("data/classified_topics_narrowed.jsonl") as f:
    for line in f:
        if line.strip():
            keep_ids.add(json.loads(line)["id"])

PROJECT_ID = "educa-tapio-nojonen"
LOCATION = "global"  # or your region
TARGET_LANG = "fi"
SERVICE_ACCOUNT_FILE = "service_account.json"

_client = None
_parent = None
def _get_client_and_parent():
    global _client, _parent
    if _client is None:
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        _client = translate.TranslationServiceClient(credentials=creds)
        _parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    return _client, _parent

ds = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")

# pick your source text column name here
SOURCE_COL = "text"  # change if your dataset uses another field (e.g., "content")

MAX_CHARS_PER_REQ = 30000  # GCP hard-ish limit; stay well under it
BATCH_SIZE = 64            # tune for throughput

def translate_batch(batch, target_lang):
    client, parent = _get_client_and_parent()
    contents = batch[SOURCE_COL]

    # 1) Clip long items into smaller pieces (same as before, but keep it simple)
    PER_ITEM = MAX_CHARS_PER_REQ // 2  # ~15k to be safe per slice
    clipped = []
    idx_map = []  # maps back to original row index
    for i, s in enumerate(contents):
        s = s or ""
        for start in range(0, len(s), PER_ITEM):
            clipped.append(s[start:start + PER_ITEM])
            idx_map.append(i)

    if not clipped:
        return {"text_fi": [""] * len(contents)}

    # 2) Send multiple requests so each stays under total-char budget
    MAX_TOTAL = 28000                 # keep well under 30720
    MAX_CONTENTS_PER_REQ = 64         # API also has a per-request count cap
    merged = [""] * len(contents)

    start = 0
    n = len(clipped)
    while start < n:
        total = 0
        count = 0
        # greedily pack as many slices as we can without exceeding limits
        while (start + count < n and
               count < MAX_CONTENTS_PER_REQ and
               total + len(clipped[start + count]) <= MAX_TOTAL):
            total += len(clipped[start + count])
            count += 1

        block = clipped[start:start + count]
        resp = client.translate_text(
            request={
                "parent": parent,
                "contents": block,
                "mime_type": "text/plain",
                "target_language_code": target_lang,
            }
        )

        for j, tr in enumerate(resp.translations):
            merged_idx = idx_map[start + j]
            merged[merged_idx] += tr.translated_text

        start += count

    return {"text_fi": merged}

OUT_DIR = "/scratch/project_2000539/maryam/fineweb_dataset_fi"

print("len(ds) =", len(ds))
ds_filtered = ds.filter(lambda row: row["id"] in keep_ids)
print("len(ds_filtered) =", len(ds_filtered))

for start in range(1800, len(ds_filtered), 10):
    end = min(start + 10, len(ds_filtered))
    print(f"Translating and saving docs {start}-{end} ...")
    
    chunk = ds_filtered.select(range(start, end))
    ds_fi_chunk = chunk.map(
        translate_batch,
        fn_kwargs={"target_lang": TARGET_LANG},
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=1,                 # tune to your CPU quota; start with 2–4
        writer_batch_size=1000,     # bigger writes -> fewer disk syncs
    )

    info = ds_fi_chunk.info.copy()
    info.description = (info.description or "") + "\n\nTranslated EN→FI using Google Cloud Translation v3 (translate_text)."
    info.citation = (info.citation or "") + "\n@software{gcp_translate_v3, title={Google Cloud Translation v3}, year={2025}}"
    info.license = info.license or "other"  # set appropriately
    ds_fi_chunk._info = info

    chunk_dir = os.path.join(OUT_DIR, f"chunk_{start}_{end}")
    ds_fi_chunk.save_to_disk(chunk_dir)
    print(f"✅ Saved {chunk_dir}")

print("Done. Each 10-doc chunk is saved under", OUT_DIR)

paths = [os.path.join(OUT_DIR, d) for d in sorted(os.listdir(OUT_DIR)) if d.startswith("chunk_")]
merged = concatenate_datasets([load_from_disk(p) for p in paths])
merged.save_to_disk(os.path.join(OUT_DIR, "merged"))