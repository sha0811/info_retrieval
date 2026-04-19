import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

DATA_DIR         = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR  = Path(__file__).parent.parent / "submissions"
EMB_DIR_SPECTER  = DATA_DIR / "embeddings" / "specter2"
CACHE_DIR        = DATA_DIR / "bm25_cache"

# ── Load held-out data ─────────────────────────────────────────────────────────
print("Loading held-out embeddings...")
spec_c = np.load(EMB_DIR_SPECTER / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SPECTER / "corpus_ids.json") as f: spec_cids = json.load(f)

spec_hq = np.load(EMB_DIR_SPECTER / "held_out" / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SPECTER / "held_out" / "query_ids.json") as f: spec_hqids = json.load(f)
spec_hqid2i = {q: i for i, q in enumerate(spec_hqids)}

bm25_held = np.load(CACHE_DIR / "bm25_held_scores.npy").astype(np.float32)
with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
with open(CACHE_DIR / "bm25_held_query_ids.json") as f: bm25_hqids = json.load(f)
bm25_hqi    = {q: i for i, q in enumerate(bm25_hqids)}
bm25_cid2i  = {c: i for i, c in enumerate(bm25_cids)}
bm25_to_spec = np.array([bm25_cid2i.get(c, 0) for c in spec_cids])

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_domain_map  = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(c, "") for c in spec_cids])
domain_masks       = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))

def mm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

print("Building BM25(full)+specter2 held-out scores...")
sub_bm25spec = {}
for qid in spec_hqids:
    s_spec = spec_c @ spec_hq[spec_hqid2i[qid]]
    s_bm25 = bm25_held[bm25_hqi[qid]][bm25_to_spec]
    score  = 0.6 * np.power(mm(s_bm25), 1.2) + 0.3 * np.power(mm(s_spec), 0.9)
    q_dom  = held_doms.get(qid, "")
    if q_dom in domain_masks:
        score += 0.2 * domain_masks[q_dom].astype(np.float32)
    top = np.argsort(-score)[:100]
    sub_bm25spec[qid] = [spec_cids[i] for i in top]

print("Loading cite held-out...")
with open(SUBMISSIONS_DIR / "final_best_wd030_wbc065_we5c035.json") as f:
    sub_cite = json.load(f)

qids = list(sub_cite.keys())

def rrf(lists, k=60):
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

configs = [
    ("cite2_bm25spec_k3",  0.7332, lambda qid: rrf([sub_cite.get(qid,[])] * 2 + [sub_bm25spec.get(qid,[])], k=3)[:100]),
    ("cite3_bm25spec_k3",  0.7321, lambda qid: rrf([sub_cite.get(qid,[])] * 3 + [sub_bm25spec.get(qid,[])], k=3)[:100]),
    ("cite1_bm25spec_k3",  0.7264, lambda qid: rrf([sub_cite.get(qid,[])] + [sub_bm25spec.get(qid,[])], k=3)[:100]),
    ("cite2_bm25spec_k5",  0.7303, lambda qid: rrf([sub_cite.get(qid,[])] * 2 + [sub_bm25spec.get(qid,[])], k=5)[:100]),
    # Triple: rrf(cite+bm25spec, k=3) then rrf that with cite again k=3
    ("triple_cite_bm25spec_k3", 0.7336, None),  # build manually below
]

sub_inner = {qid: rrf([sub_cite.get(qid,[])] + [sub_bm25spec.get(qid,[])], k=3)[:100] for qid in qids}
sub_triple = {qid: rrf([sub_inner.get(qid,[])] + [sub_cite.get(qid,[])], k=3)[:100] for qid in qids}

print("\nGenerating held-out submissions...")
for name, train_ndcg, fn in configs:
    if fn is None:
        sub = sub_triple
    else:
        sub = {qid: fn(qid) for qid in qids}
    jp = SUBMISSIONS_DIR / f"best_rrf_{name}.json"
    zp = SUBMISSIONS_DIR / f"best_rrf_{name}.zip"
    with open(jp, "w") as f: json.dump(sub, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"  Saved: {zp.name}  (train NDCG@10={train_ndcg:.4f})")
