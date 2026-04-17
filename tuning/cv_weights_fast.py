"""
5-fold CV weight search for the full pipeline.

Precomputes all signals (dense, BM25, cite, chunk), then sweeps a weight grid
cheaply over those precomputed scores. Picks the best weights per fold and averages them
to get params that generalize better than tuning on the full training set.

Note: pipeline logic is duplicated inline here (it's basically a tuning script for models/pipeline.py).
"""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate, save_results
import pandas as pd

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR   = DATA_DIR / "bm25_cache"
EMB_FT    = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE   = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_E5    = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

ft_c  = np.load(EMB_FT  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "corpus_ids.json") as f: ft_cids = json.load(f)
bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)
e5_c  = np.load(EMB_E5  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "corpus_ids.json") as f: e5_cids = json.load(f)

cc_embs = np.load(EMB_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_chunk_doc_ids.json") as f: cc_doc_ids = json.load(f)
doc_to_chunk_rows = defaultdict(list)
for row_idx, doc_id in enumerate(cc_doc_ids):
    doc_to_chunk_rows[doc_id].append(row_idx)

bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
e5_cid2i  = {c: i for i, c in enumerate(e5_cids)}
bge_c_al  = bge_c[[bge_cid2i[c] for c in ft_cids]]

with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids_raw = json.load(f)
bm25_cid2i_raw = {c: i for i, c in enumerate(bm25_cids_raw)}
bm25_to_ft = np.array([bm25_cid2i_raw.get(c, 0) for c in ft_cids])

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
cdmap = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
cdom_arr = np.array([cdmap.get(c, "") for c in ft_cids])
dmasks = {d: cdom_arr == d for d in np.unique(cdom_arr) if d}

ft_q  = np.load(EMB_FT  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "train/query_ids.json") as f: ft_qids = json.load(f)
bge_q = np.load(EMB_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/query_ids.json") as f: bge_qids = json.load(f)
bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
bge_q_al = np.array([bge_q[bge_qid2i[q]] if q in bge_qid2i else np.zeros(bge_q.shape[1], dtype=np.float32) for q in ft_qids])

bge_cite = np.load(EMB_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
e5_cite  = np.load(EMB_E5  / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)
bge_pool = np.load(EMB_BGE / "train/cite_context_pooled_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/cite_context_pooled_query_ids.json") as f: bge_pool_qids = json.load(f)

bm25_tr = np.load(CACHE_DIR / "bm25_train_scores.npy")
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tr_qids = json.load(f)
bm25_tr_qi = {q: i for i, q in enumerate(bm25_tr_qids)}

queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
qdmap = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

q2bc_all = defaultdict(list)
for i, q in enumerate(bge_cite_qids): q2bc_all[q].append(i)
q2ec_all = defaultdict(list)
for i, q in enumerate(e5_cite_qids): q2ec_all[q].append(i)
pool_bge_qi_all = {q: i for i, q in enumerate(bge_pool_qids)}

def mm(v): 
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


# Pre-compute base scores (dense + BM25) without tunable signals
print("Pre-computing base scores...")
base_sc = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
for qi, qid in enumerate(ft_qids):
    base_sc[qi] = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * (bge_c_al @ bge_q_al[qi])
    if qid in bm25_tr_qi:
        base_sc[qi] += 0.35 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
print("  Done.")

# Pre-compute domain boost vectors
dom_boost = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
for qi, qid in enumerate(ft_qids):
    qd = qdmap.get(qid, "")
    if qd in dmasks: dom_boost[qi] = dmasks[qd].astype(np.float32)

# Pre-compute pooled BCE sim
print("Pre-computing pool sims...")
pool_sim = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
for qi, qid in enumerate(ft_qids):
    if qid in pool_bge_qi_all:
        pool_sim[qi] = bge_pool[pool_bge_qi_all[qid]] @ bge_c_al.T
print("  Done.")

# Pre-compute per-candidate cite scores for top-2000 per query
print("Pre-computing candidate cite scores (this takes a while)...")
CTK = 500  # reduced for speed
bc_sc  = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
ec_sc  = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
cc_sc  = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)

for qi, qid in enumerate(ft_qids):
    # We get an approximate ranking
    sc = base_sc[qi] + 0.38 * dom_boost[qi] + 0.15 * pool_sim[qi]
    top_cands = np.argsort(-sc)[:CTK]
    rows_bc = q2bc_all.get(qid, [])
    rows_ec = q2ec_all.get(qid, [])
    if rows_bc:
        qc = bge_cite[rows_bc]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            bi = bge_cid2i.get(doc_id)
            if bi is not None:
                bc_sc[qi, cidx] = float((qc @ bge_c[bi]).max())
                chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                if chunk_rows:
                    cc_sc[qi, cidx] = float((qc @ cc_embs[chunk_rows].T).max())
    if rows_ec:
        qc = e5_cite[rows_ec]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            ei = e5_cid2i.get(doc_id)
            if ei is not None:
                ec_sc[qi, cidx] = float((qc @ e5_c[ei]).max())
    if qi % 10 == 0: print(f"  {qi+1}/{len(ft_qids)}", flush=True)

print("  Done.")


def build_sub_fast(qid_list, w_dom, w_pool, w_bc, w_ec, w_cc):
    sub = {}
    for qi, qid in enumerate(ft_qids):
        if qid not in qid_list: continue
        sc = base_sc[qi] + w_dom * dom_boost[qi] + w_pool * pool_sim[qi]
        sc = sc + w_bc * bc_sc[qi] + w_ec * ec_sc[qi] + w_cc * cc_sc[qi]
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub


def eval_sub(sub, qids):
    sub_f = {q: sub[q] for q in qids if q in sub}
    r = evaluate(sub_f, qrels, ks=[10], query_domains=qdmap, verbose=False)
    return r["overall"]["NDCG@10"]


# Small grid for CV - focus on cite weights since they're most important
GRID = [
    (dom, pool, bc, ec, cc)
    for dom  in [0.3, 0.38, 0.5]
    for pool in [0.1, 0.15, 0.2]
    for bc   in [0.7, 0.95, 1.2]
    for ec   in [0.25, 0.35, 0.45]
    for cc   in [0.0, 0.07, 0.15]
]
print(f"\nGrid size: {len(GRID)}")

# 5-fold CV
np.random.seed(42)
qids_arr = np.array(ft_qids)
fold_idx = np.random.permutation(len(qids_arr))
folds = np.array_split(fold_idx, 5)

print("Running 5-fold CV...")
fold_best_params = []
fold_best_cv_scores = []

for fold_i, val_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(5) if j != fold_i])
    train_qids = set(qids_arr[train_idx])
    val_qids   = set(qids_arr[val_idx])

    best_s, best_p = 0, None
    for params in GRID:
        sub = build_sub_fast(train_qids, *params)
        s = eval_sub(sub, train_qids)
        if s > best_s:
            best_s = s
            best_p = params

    sub_val = build_sub_fast(val_qids, *best_p)
    val_s = eval_sub(sub_val, val_qids)
    fold_best_params.append(best_p)
    fold_best_cv_scores.append(val_s)
    print(f"  Fold {fold_i+1}: train={best_s:.4f} val={val_s:.4f} params=dom{best_p[0]} pool{best_p[1]} bc{best_p[2]} ec{best_p[3]} cc{best_p[4]}")

print(f"\nCV val scores: {[f'{s:.4f}' for s in fold_best_cv_scores]}")
print(f"Mean CV val:   {np.mean(fold_best_cv_scores):.4f}")

avg_params = tuple(np.mean([p[i] for p in fold_best_params]) for i in range(5))
print(f"Avg params: dom={avg_params[0]:.3f} pool={avg_params[1]:.3f} bc={avg_params[2]:.3f} ec={avg_params[3]:.3f} cc={avg_params[4]:.3f}")

# Full train eval
sub_full = build_sub_fast(set(ft_qids), *avg_params)
full_s = eval_sub(sub_full, set(ft_qids))
print(f"Full train with avg params: {full_s:.4f}")

# Also eval default best params
sub_best = build_sub_fast(set(ft_qids), 0.38, 0.15, 0.95, 0.35, 0.07)
best_s = eval_sub(sub_best, set(ft_qids))
print(f"Full train with best params (0.38,0.15,0.95,0.35,0.07): {best_s:.4f}")

# Save results
results = evaluate(sub_full, qrels, ks=[10], query_domains=qdmap, verbose=False)
save_results(results, RESULTS_DIR / "pipeline.csv",
             hyperparameters={"w_domain": avg_params[0], "w_pool": avg_params[1],
                              "w_bc": avg_params[2], "w_ec": avg_params[3],
                              "w_cc": avg_params[4], "source": "cv5fast"})
print(f"\nResults saved to results/pipeline.csv")
print(f"To generate a submission with these weights, update the constants in models/pipeline.py and run:")
print(f"  python models/pipeline.py --split held_out --zip")
