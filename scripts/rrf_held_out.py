"""
Generate RRF held-out submission combining:
  A: final_best_wd030_wbc065_we5c035.json  (cite pipeline, 0.69 codabench)
  B: BM25 + e5-large-v2 dense (replicating retrieve_advanced, 0.70 codabench)
     using cached held-out embeddings

Expected: fusion >= 0.70 on held-out
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_E5      = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

# ── Corpus ────────────────────────────────────────────────────────────────────
print("Loading corpus + e5 embeddings...")
e5_c = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)
e5_cid2i = {c: i for i, c in enumerate(e5_cids)}

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

# ── Held-out query embeddings (already cached) ────────────────────────────────
print("Loading held-out query embeddings...")
e5_hq = np.load(EMB_DIR_E5 / "held_out/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "held_out/query_ids.json") as f: e5_hqids = json.load(f)
e5_hqid2i = {q: i for i, q in enumerate(e5_hqids)}

held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(c, "") for c in e5_cids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

# ── BM25 on title+abstract ────────────────────────────────────────────────────
print("Building BM25 on title+abstract...")
corpus_ta  = (corpus["title"].fillna("") + " " + corpus["abstract"].fillna("")).tolist()
bm25       = BM25Okapi([t.lower().split() for t in corpus_ta], k1=1.8, b=0.75)
cids_list  = corpus["doc_id"].tolist()
c_reindex  = {c: i for i, c in enumerate(cids_list)}
e5_to_bm25 = np.array([c_reindex.get(c, 0) for c in e5_cids])

held_ta = dict(zip(held_df["doc_id"],
    (held_df["title"].fillna("") + " " + held_df["abstract"].fillna("")).tolist()))

def mm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

# ── Generate BM25+e5 held-out (matching retrieve_advanced weights) ─────────────
print("Generating BM25+e5 held-out submission...")
WB, WD, DB = 0.6, 0.3, 0.2   # retrieve_advanced defaults

sub_bm25 = {}
for qid in e5_hqids:
    qi = e5_hqid2i[qid]
    s_e5   = e5_c @ e5_hq[qi]

    bm25_raw = bm25.get_scores(held_ta.get(qid, "").lower().split())
    s_bm25   = bm25_raw[e5_to_bm25]

    score = WB * np.power(mm(s_bm25), 1.2) + WD * np.power(mm(s_e5), 0.9)

    q_dom = held_doms.get(qid, "")
    if q_dom in domain_masks:
        score += DB * domain_masks[q_dom].astype(np.float32)

    top = np.argsort(-score)[:100]
    sub_bm25[qid] = [e5_cids[i] for i in top]

# ── Load cite held-out ────────────────────────────────────────────────────────
with open(SUBMISSIONS_DIR / "final_best_wd030_wbc065_we5c035.json") as f:
    sub_cite = json.load(f)

print(f"BM25+e5 queries: {len(sub_bm25)}, cite queries: {len(sub_cite)}")

# ── RRF ───────────────────────────────────────────────────────────────────────
def rrf(lists, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

# Try the configs that worked best on train
configs = [
    ("RRF k=10",            lambda qid: rrf([sub_cite.get(qid,[]), sub_bm25.get(qid,[])], k=10)[:100]),
    ("RRF k=20",            lambda qid: rrf([sub_cite.get(qid,[]), sub_bm25.get(qid,[])], k=20)[:100]),
    ("RRF cite×2 k=20",     lambda qid: rrf([sub_cite.get(qid,[])]*2 + [sub_bm25.get(qid,[])], k=20)[:100]),
    ("RRF cite×3 k=20",     lambda qid: rrf([sub_cite.get(qid,[])]*3 + [sub_bm25.get(qid,[])], k=20)[:100]),
    ("bm25 only",           lambda qid: sub_bm25.get(qid, [])[:100]),
    ("cite only",           lambda qid: sub_cite.get(qid, [])[:100]),
]

qids = list(sub_cite.keys())
print(f"\nGenerating {len(configs)} held-out submissions...\n")

for name, fn in configs:
    sub = {qid: fn(qid) for qid in qids}
    jp = SUBMISSIONS_DIR / f"rrf_held_{name.replace(' ','_').replace('×','x')}.json"
    zp = SUBMISSIONS_DIR / f"rrf_held_{name.replace(' ','_').replace('×','x')}.zip"
    with open(jp, "w") as f: json.dump(sub, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"  Saved: {zp.name}")

print("\nDone. Submit these to codabench:")
print("  - rrf_held_bm25_only: should match 0.70 baseline")
print("  - rrf_held_cite_only: should match 0.69 baseline")
print("  - rrf_held_RRF_k=20:  target > 0.70")
print("  - rrf_held_RRF_cite×2_k=20: target > 0.70")
print("  - rrf_held_RRF_cite×3_k=20: target > 0.70")
