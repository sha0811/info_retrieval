"""
Targeted test: add BM25 on top of the existing best dual-dense+cite pipeline.
Fast script — only 8 configs, should run in ~5 minutes.

Current best train: 0.7170 (ft-small + bge-large + cite, domain=0.30)
retrieve_advanced:  0.668 train / 0.70 codabench  (BM25 helps on held-out)

Hypothesis: BM25 on top of dual-dense+cite will improve both.
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_FT  = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5  = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

# ── Load embeddings ───────────────────────────────────────────────────────────
print("Loading embeddings...")
ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)

e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)

bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx  = {cid: i for i, cid in enumerate(e5_corpus_ids)}
bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]

ft_q_embs = np.load(EMB_DIR_FT / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_q_ids = json.load(f)

bge_q_embs = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_q_ids = json.load(f)

bge_cite_embs = np.load(EMB_DIR_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)

e5_cite_embs = np.load(EMB_DIR_E5 / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_q_ids)}
qid_to_bge_cite = defaultdict(list)
for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
qid_to_e5_cite = defaultdict(list)
for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

# ── BM25 on title+abstract (fast) ─────────────────────────────────────────────
print("Building BM25 on title+abstract...")
corpus_ta = (corpus["title"].fillna("") + " " + corpus["abstract"].fillna("")).tolist()
bm25 = BM25Okapi([t.lower().split() for t in corpus_ta], k1=1.8, b=0.75)
corpus_ids_list = corpus["doc_id"].tolist()
corpus_reindex = {cid: i for i, cid in enumerate(corpus_ids_list)}
ft_to_bm25_idx = np.array([corpus_reindex.get(cid, 0) for cid in ft_corpus_ids])

query_ta = dict(zip(queries_df["doc_id"],
    (queries_df["title"].fillna("") + " " + queries_df["abstract"].fillna("")).tolist()))

print(f"BM25 ready. Starting eval...\n")


def minmax(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-9)


def retrieve(w_bm25=0.0, w_domain=0.30, w_bge_cite=0.65, w_e5_cite=0.35, cite_top_k=1000):
    sub = {}
    for qidx, qid in enumerate(ft_q_ids):
        q_ft = ft_q_embs[qidx]
        q_domain = query_domains.get(qid, "")
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        e5_cite_rows  = qid_to_e5_cite.get(qid, [])

        ft_scores  = ft_corpus_embs @ q_ft
        bge_idx = bge_qid_to_idx.get(qid)
        bge_scores = bge_corpus_aligned @ bge_q_embs[bge_idx] if bge_idx is not None else ft_scores

        # Dual-dense score (existing best pipeline)
        scores = 0.5 * ft_scores + 0.5 * bge_scores

        # Domain boost
        if w_domain > 0 and q_domain in domain_masks:
            scores = scores + w_domain * domain_masks[q_domain].astype(np.float32)

        # BM25 additive (nonlinear, normalized)
        if w_bm25 > 0:
            bm25_raw = bm25.get_scores(query_ta.get(qid, "").lower().split())
            bm25_aligned = bm25_raw[ft_to_bm25_idx]
            scores = scores + w_bm25 * np.power(minmax(bm25_aligned), 1.2)

        # Cite context
        if w_bge_cite > 0 and bge_cite_rows:
            q_cite = bge_cite_embs[bge_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bi = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bi is not None:
                    scores[cidx] += w_bge_cite * float((q_cite @ bge_corpus_embs[bi]).max())

        if w_e5_cite > 0 and e5_cite_rows:
            q_cite_e5 = e5_cite_embs[e5_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                ei = e5_cid_to_idx.get(ft_corpus_ids[cidx])
                if ei is not None:
                    scores[cidx] += w_e5_cite * float((q_cite_e5 @ e5_corpus_embs[ei]).max())

        top_idx = np.argsort(-scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


PREV_BEST = 0.7170
best_ndcg = 0.0
best_cfg = None
best_sub = None

configs = [
    # name,                        w_bm25, w_domain, w_bge_cite, w_e5_cite
    ("baseline (no bm25)",          0.00,   0.30,     0.65,       0.35),
    ("bm25=0.10",                   0.10,   0.30,     0.65,       0.35),
    ("bm25=0.20",                   0.20,   0.30,     0.65,       0.35),
    ("bm25=0.30",                   0.30,   0.30,     0.65,       0.35),
    ("bm25=0.40",                   0.40,   0.30,     0.65,       0.35),
    ("bm25=0.50",                   0.50,   0.30,     0.65,       0.35),
    ("bm25=0.60",                   0.60,   0.30,     0.65,       0.35),
    ("bm25=0.30 no_cite",           0.30,   0.30,     0.00,       0.00),
    ("bm25=0.40 no_cite",           0.40,   0.30,     0.00,       0.00),
    ("bm25=0.50 no_cite",           0.50,   0.30,     0.00,       0.00),
    ("bm25=0.40 domain=0.20",       0.40,   0.20,     0.65,       0.35),
    ("bm25=0.40 domain=0.00",       0.40,   0.00,     0.65,       0.35),
    ("bm25=0.50 domain=0.20",       0.50,   0.20,     0.65,       0.35),
    ("bm25=0.50 domain=0.00",       0.50,   0.00,     0.65,       0.35),
]

print(f"{'Config':<40}  NDCG@10   R@100")
print("-"*65)

for name, w_bm25, w_domain, w_bge_cite, w_e5_cite in configs:
    sub = retrieve(w_bm25, w_domain, w_bge_cite, w_e5_cite)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    marker = " <-- NEW BEST" if ndcg > best_ndcg else ""
    print(f"  {name:<38}  {ndcg:.4f}   {rec:.4f}{marker}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg = (w_bm25, w_domain, w_bge_cite, w_e5_cite, name)
        best_sub = sub

print(f"\nBest: {best_cfg[-1]} -> NDCG@10={best_ndcg:.4f}")
if best_ndcg > PREV_BEST:
    print(f"IMPROVEMENT: +{best_ndcg - PREV_BEST:.4f} over previous best {PREV_BEST}")
    out = SUBMISSIONS_DIR / "bm25_on_best_train.json"
    with open(out, "w") as f: json.dump(best_sub, f)
    print(f"Saved -> {out}")
else:
    print(f"No improvement over {PREV_BEST}.")
