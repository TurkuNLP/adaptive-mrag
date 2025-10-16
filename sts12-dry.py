from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from mteb import MTEB, get_tasks
import numpy as np
from collections import Counter

class ProbeEncoder:
    def __init__(self, dim=128):
        self.name = "probe/fast-counter"
        self.similarity_fn_name = "cosine"
        self.dim = dim
        self.total = 0
        self.by_role = Counter()   # e.g., {"corpus": N, "query": M}
        self.calls = []            # sizes of each encode() call

    def encode(self, texts, batch_size=1024, show_progress=False, **kwargs):
        n = len(texts)
        self.total += n
        self.calls.append(n)
        role = "query" if kwargs.get("is_query") else "corpus"
        self.by_role[role] += n
        # return dummy embeddings
        return np.zeros((n, self.dim), dtype="float32")

# run the probe
probe = ProbeEncoder(dim=128)
tasks = get_tasks(tasks=["STS12"])
evaluation = MTEB(tasks=tasks, eval_splits=["test"])

_ = evaluation.run(
    probe,
    output_folder=None,  # don't write results
    encode_kwargs={"batch_size": 4096, "show_progress": False},
)

print("TOTAL items to encode:", probe.total)
print("Breakdown:", probe.by_role)
print("# of encode() calls:", len(probe.calls), "first few:", probe.calls[:10])
