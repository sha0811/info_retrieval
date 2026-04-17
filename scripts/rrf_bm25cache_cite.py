"""
RRF fusion using:
  A: BM25(full_text, cached) + e5-large-v2 → replicates the 0.70 held-out pipeline
  B: final_best_wd030_wbc065_we5c035.json  → ft+bge+cite, 0.69 held-out

This should combine the complementary signals and exceed 0.70.
All heavy computation is precomputed; this runs in seconds.
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_E5      = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
CACHE_DIR       = DATA_DIR / "bm25_cache"

# ── Load all embeddings ────────────────────────────────────────────────────────
print("Loading e5-large-v2 embeddings...")
e5_c = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)

# Train queries
e5_tq = np.load(EMB_DIR_E5 / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "query_ids.json") as f: e5_tqids = json.load(f)

# Held-out queries
e5_hq = np.load(EMB_DIR_E5 / "held_out" / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "held_out" / "query_ids.json") as f: e5_hqids = json.load(f)

# ── Load cached BM25 scores ────────────────────────────────────────────────────
print("Loading cached BM25 scores...")
bm25_train = np.load(CACHE_DIR / "bm25_train_scores.npy").astype(np.float32)
bm25_held  = np.load(CACHE_DIR / "bm25_held_scores.npy").astype(np.float32)
with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tqids = json.load(f)
with open(CACHE_DIR / "bm25_held_query_ids.json") as f: bm25_hqids = json.load(f)

# Map BM25 corpus order to e5 corpus order
bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
bm25_to_e5  = np.array([bm25_cid2i.get(c, 0) for c in e5_cids])  # e5 index -> bm25 index

# Maps for query lookup
bm25_tqi = {q: i for i, q in enumerate(bm25_tqids)}
bm25_hqi = {q: i for i, q in enumerate(bm25_hqids)}

# ── Domain info ────────────────────────────────────────────────────────────────
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(c, "") for c in e5_cids])

queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels      = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))

all_domains = np.unique(corpus_domains_arr)
domain_masks = {d: corpus_domains_arr == d for d in all_domains if d}

def mm(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-9)

def rrf(lists, k=60):
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

# ── Build BM25+e5 score fusion (replicating retrieve_advanced defaults) ────────
WB, WD, DB = 0.6, 0.3, 0.2  # same as retrieve_advanced defaults

def make_bm25_e5_sub(qids, e5_qembs, e5_qid2i, bm25_scores, bm25_qid2i, doms):
    sub = {}
    for qid in qids:
        qi_e5   = e5_qid2i[qid]
        qi_bm25 = bm25_qid2i[qid]

        s_e5   = e5_c @ e5_qembs[qi_e5]                     # (20K,) raw cosine
        s_bm25 = bm25_scores[qi_bm25][bm25_to_e5]           # (20K,) aligned to e5 corpus order

        score = WB * np.power(mm(s_bm25), 1.2) + WD * np.power(mm(s_e5), 0.9)

        q_dom = doms.get(qid, "")
        if q_dom in domain_masks:
            score += DB * domain_masks[q_dom].astype(np.float32)

        top = np.argsort(-score)[:100]
        sub[qid] = [e5_cids[i] for i in top]
    return sub

# ── Train evaluation of BM25+e5 ───────────────────────────────────────────────
print("\nEvaluating BM25(full_text)+e5 on train...")
e5_tqid2i = {q: i for i, q in enumerate(e5_tqids)}
sub_bm25e5_train = make_bm25_e5_sub(
    list(e5_tqid2i.keys()), e5_tq, e5_tqid2i, bm25_train, bm25_tqi, query_domains
)
res = evaluate(sub_bm25e5_train, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
print(f"  BM25(full)+e5  NDCG@10={res['overall']['NDCG@10']:.4f}  R@100={res['overall']['Recall@100']:.4f}")

# ── Load cite submissions ──────────────────────────────────────────────────────
print("\nLoading cite submissions...")
with open(SUBMISSIONS_DIR / "dual_cite_v2_train.json") as f:
    sub_cite_train = json.load(f)
with open(SUBMISSIONS_DIR / "final_best_wd030_wbc065_we5c035.json") as f:
    sub_cite_held  = json.load(f)

# Eval cite on train
res_cite = evaluate(sub_cite_train, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
print(f"  Cite pipeline  NDCG@10={res_cite['overall']['NDCG@10']:.4f}  R@100={res_cite['overall']['Recall@100']:.4f}")

# ── RRF sweep on train to find best k ─────────────────────────────────────────
print("\nRRF sweep on train (BM25(full)+e5  ×  cite):")
best_ndcg, best_k, best_n = 0.0, 60, 1
qids_train = list(sub_cite_train.keys())

for k in [5, 10, 20, 30, 40, 60]:
    for n_cite in [1, 2, 3]:
        sub = {qid: rrf([sub_cite_train.get(qid,[])] * n_cite + [sub_bm25e5_train.get(qid,[])], k=k)[:100]
               for qid in qids_train}
        res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        rec  = res["overall"]["Recall@100"]
        mark = " <-- BEST" if ndcg > best_ndcg else ""
        print(f"  k={k:<3} cite×{n_cite}  NDCG@10={ndcg:.4f}  R@100={rec:.4f}{mark}")
        if ndcg > best_ndcg:
            best_ndcg, best_k, best_n = ndcg, k, n_cite

print(f"\nBest on train: k={best_k}, cite×{best_n} -> {best_ndcg:.4f}  (cite baseline: {res_cite['overall']['NDCG@10']:.4f})")

# ── Generate held-out submissions ─────────────────────────────────────────────
print("\nGenerating held-out submissions...")
e5_hqid2i = {q: i for i, q in enumerate(e5_hqids)}
sub_bm25e5_held = make_bm25_e5_sub(
    list(e5_hqid2i.keys()), e5_hq, e5_hqid2i, bm25_held, bm25_hqi, held_doms
)
print(f"  BM25+e5 held-out: {len(sub_bm25e5_held)} queries")

qids_held = list(sub_cite_held.keys())

configs = [
    ("bm25full_e5_only",     1, 999,  lambda qid: sub_bm25e5_held.get(qid, [])[:100]),
    ("cite_only",            1, 999,  lambda qid: sub_cite_held.get(qid, [])[:100]),
]

# Add best k from sweep
for k in [best_k] + ([20, 40] if best_k not in [20, 40] else []):
    for n_cite in [best_n] + ([1, 2] if best_n not in [1, 2] else []):
        name = f"rrf_k{k}_citeX{n_cite}"
        def make_fn(k=k, n=n_cite):
            return lambda qid: rrf([sub_cite_held.get(qid,[])] * n + [sub_bm25e5_held.get(qid,[])], k=k)[:100]
        configs.append((name, n_cite, k, make_fn()))

for name, _, _, fn in configs:
    sub = {qid: fn(qid) for qid in qids_held}
    jp = SUBMISSIONS_DIR / f"rrf_bm25cache_{name}.json"
    zp = SUBMISSIONS_DIR / f"rrf_bm25cache_{name}.zip"
    with open(jp, "w") as f: json.dump(sub, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"  Saved: {zp.name}")

print("\nDone.")
print(f"Best train RRF config: k={best_k}, cite×{best_n} -> {best_ndcg:.4f}")
print("Submit rrf_bm25cache_bm25full_e5_only.zip  (should match ~0.70 baseline)")
print("Submit rrf_bm25cache_rrf_k*_citeX*.zip     (target: >0.70)")
