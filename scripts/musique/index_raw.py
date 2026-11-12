"""Official HippoRAG2 indexing of the MuSiQue corpus with HY4 + NV-Embed-v2.

The working index lives on local disk (the project mount is Ceph and far too
slow for parquet stores of ~10^5 embeddings); ``--sync`` copies the finished
state back to ``outputs/musique/graphs`` for persistence.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HY4_RATE_FILE", "musique")
os.environ.setdefault("HY4_GLOBAL_QPS", "4")

from src.musique.data import OUT, load_corpus, passage_text  # noqa: E402
from src.musique.hippo import WORK, open_musique_hippo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="raw")
ap.add_argument("--limit", type=int, default=0, help="index only the first N passages (smoke test)")
ap.add_argument("--sync", action="store_true")
args = ap.parse_args()

corpus = load_corpus()
docs = [passage_text(p) for p in corpus]
if args.limit:
    docs = docs[: args.limit]
save_dir = WORK / "graphs" / args.name
t0 = time.time()
with open_musique_hippo(save_dir) as hippo:
    hippo.index(docs)
    info = hippo.get_graph_info()
info["seconds"] = round(time.time() - t0, 1)
info["n_docs"] = len(docs)
print(json.dumps(info, indent=1))
(save_dir / "build_info.json").write_text(json.dumps(info, indent=1))
if args.sync:
    dst = OUT / "graphs" / args.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(save_dir, dst, symlinks=True)
