"""
Deep error analysis: understand why specific queries fail.
Focus on queries where relevant docs are not even in top 100.
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate

DATA_DIR  = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "bm25_cache"
EMB_FT    = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE   = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_E5    = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

ft_c  = np.load(EMB_FT  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "corpus_ids.json") as f: ft_cids = json.load(f)
bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)

bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
bge_c_al  = bge_c[[bge_cid2i[c] for c in ft_cids]]
ft_cid2i = {c: i for i, c in enumerate(ft_cids)}

bge_cite = np.load(EMB_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)

ft_q  = np.load(EMB_FT  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "train/query_ids.json") as f: ft_qids = json.load(f)
bge_q = np.load(EMB_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/query_ids.json") as f: bge_qids = json.load(f)
bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
bge_q_al = np.array([bge_q[bge_qid2i[q]] if q in bge_qid2i else np.zeros(bge_q.shape[1], dtype=np.float32) for q in ft_qids])

bm25_tr = np.load(CACHE_DIR / "bm25_train_scores.npy")
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tr_qids = json.load(f)
bm25_tr_qi = {q: i for i, q in enumerate(bm25_tr_qids)}
with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids_raw = json.load(f)
bm25_cid2i_raw = {c: i for i, c in enumerate(bm25_cids_raw)}
bm25_to_ft = np.array([bm25_cid2i_raw.get(c, 0) for c in ft_cids])

queries_df = load_queries(DATA_DIR / "queries.parquet")
corpus_df  = pd.read_parquet(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
qdmap = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

q2bc = defaultdict(list)
for i, q in enumerate(bge_cite_qids): q2bc[q].append(i)

corpus_ta   = dict(zip(corpus_df["doc_id"], corpus_df["ta"].fillna("")))
query_ta    = dict(zip(queries_df["doc_id"], queries_df["ta"].fillna("")))
ft_qid2i    = {q: i for i, q in enumerate(ft_qids)}

def mm(v): 
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

print("=== ANALYZING FAILURES ===\n")
print("For each query, checking rank of each relevant doc in different retrieval systems:\n")

# Pick 5 worst-scoring queries from the full pipeline
# (those where relevant docs are not in top 100)
with open(DATA_DIR.parent / "submissions/BEST_v4_final_bm0.35_bc0.95_ec0.35_dom0.38_pbc0.15_cc0.07_train0.7401.json") as f:
    best_sub = json.load(f)

# We get train qids from ft_qids, not best_sub (which has held-out)
# Rerun a quick version of the pipeline for train
def quick_score(qid, qi):
    sc_ft   = ft_c @ ft_q[qi]   # FT
    sc_bge  = bge_c_al @ bge_q_al[qi]  # BGE
    sc_dense = 0.5 * sc_ft + 0.5 * sc_bge
    sc_bm25 = bm25_tr[bm25_tr_qi[qid]][bm25_to_ft] if qid in bm25_tr_qi else None
    if sc_bm25 is not None:
        sc_full = sc_dense + 0.35 * np.power(mm(sc_bm25), 1.2)
    else:
        sc_full = sc_dense.copy()
    # cite
    rows_bc = q2bc.get(qid, [])
    if rows_bc:
        qc = bge_cite[rows_bc]
        # Quick: just compute for all corpus
        cite_sc = (qc @ bge_c_al.T).max(axis=0)
        sc_full += 0.95 * cite_sc
    return sc_dense, sc_bm25, sc_full

# Find challenging queries
hard_queries = []
for qi, qid in enumerate(ft_qids):
    rel_set = set(qrels.get(qid, []))
    if not rel_set:
        continue
    _, _, sc_full = quick_score(qid, qi)
    sorted_idx = np.argsort(-sc_full)
    ranks = {did: np.where(sorted_idx == ft_cid2i.get(did, -1))[0] for did in rel_set}
    min_rank = min((r[0] if len(r) > 0 else 99999) for r in ranks.values())
    hard_queries.append((qid, qi, min_rank))

hard_queries.sort(key=lambda x: x[2], reverse=True)

print("Top 10 hardest queries (highest rank for best relevant doc):\n")
for qid, qi, min_rank in hard_queries[:10]:
    rel_set = set(qrels.get(qid, []))
    sc_dense, sc_bm25, sc_full = quick_score(qid, qi)
    sorted_dense = np.argsort(-sc_dense)
    sorted_full  = np.argsort(-sc_full)
    sorted_bm25  = np.argsort(-sc_bm25) if sc_bm25 is not None else None

    print(f"Query: {qid[:20]} | domain={qdmap.get(qid,'')} | n_rel={len(rel_set)}")
    print(f"  Query TA: {query_ta.get(qid, '')[:100]}")
    for did in list(rel_set)[:5]:
        rank_dense = np.where(sorted_dense == ft_cid2i.get(did, -1))[0]
        rank_full  = np.where(sorted_full == ft_cid2i.get(did, -1))[0]
        rank_bm25  = np.where(sorted_bm25 == ft_cid2i.get(did, -1))[0] if sorted_bm25 is not None else [-1]
        r_d = rank_dense[0] if len(rank_dense) > 0 else 99999
        r_f = rank_full[0]  if len(rank_full) > 0  else 99999
        r_b = rank_bm25[0]  if len(rank_bm25) > 0  else 99999
        print(f"  Rel doc {did[:20]} | rank_dense={r_d:6d} rank_bm25={r_b:6d} rank_full={r_f:6d}")
        print(f"    Corpus TA: {corpus_ta.get(did,'')[:100]}")
    print()

# Now check: are the relevant docs actually in the corpus?
print("=== CHECKING IF RELEVANT DOCS ARE IN CORPUS ===")
all_rel_docs = set(d for v in qrels.values() for d in v)
ft_cids_set = set(ft_cids)
missing = all_rel_docs - ft_cids_set
print(f"Total unique relevant docs: {len(all_rel_docs)}")
print(f"Relevant docs NOT in ft_cids corpus: {len(missing)}")
if missing:
    for did in list(missing)[:5]:
        print(f"  Missing: {did}")
