"""Quick script to generate best-pipeline train submission for use as hard negatives for finetune_biencoder.py"""


import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries

DATA_DIR  = Path(__file__).parent.parent / "data"
SUBS_DIR  = Path(__file__).parent.parent / "submissions"
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

bm25_tr = np.load(CACHE_DIR / "bm25_train_scores.npy")
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tr_qids = json.load(f)
bm25_tr_qi = {q: i for i, q in enumerate(bm25_tr_qids)}

queries_df = load_queries(DATA_DIR / "queries.parquet")
qdmap = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

def mm(v): 
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

# Simple dense + BM25 + domain for hard negatives (no cite — that would leak)
sub = {}
for qi, qid in enumerate(ft_qids):
    sc = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * (bge_c_al @ bge_q_al[qi])
    if qid in bm25_tr_qi:
        sc += 0.35 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
    qd = qdmap.get(qid, "")
    if qd in dmasks: sc += 0.38 * dmasks[qd].astype(np.float32)
    sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]

out = SUBS_DIR / "best_pipeline_train_for_finetune.json"
with open(out, "w") as f: json.dump(sub, f)
print(f"Saved {len(sub)} queries to {out}")
