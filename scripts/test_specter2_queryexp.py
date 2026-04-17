"""
Test:
1. Specter2_base as dense retrieval signal (allenai/specter2_base)
2. Query expansion: average(query_emb, cite_sentence_embs) as retrieval signal
3. BGE cite sentences used as expanded queries (treat each cite sentence as a query)
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate

DATA_DIR  = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "bm25_cache"
EMB_FT    = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE   = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_E5    = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
EMB_SP2B  = DATA_DIR / "embeddings" / "allenai_specter2_base"

ft_c  = np.load(EMB_FT  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "corpus_ids.json") as f: ft_cids = json.load(f)
bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)
e5_c  = np.load(EMB_E5  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "corpus_ids.json") as f: e5_cids = json.load(f)
sp2b_c = np.load(EMB_SP2B / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_SP2B / "corpus_ids.json") as f: sp2b_cids = json.load(f)

cc_embs = np.load(EMB_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_chunk_doc_ids.json") as f: cc_doc_ids = json.load(f)
doc_to_chunk_rows = defaultdict(list)
for row_idx, doc_id in enumerate(cc_doc_ids):
    doc_to_chunk_rows[doc_id].append(row_idx)

bge_cid2i  = {c: i for i, c in enumerate(bge_cids)}
e5_cid2i   = {c: i for i, c in enumerate(e5_cids)}
sp2b_cid2i = {c: i for i, c in enumerate(sp2b_cids)}
bge_c_al   = bge_c[[bge_cid2i[c] for c in ft_cids]]
sp2b_c_al  = sp2b_c[[sp2b_cid2i.get(c, 0) for c in ft_cids]]

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

sp2b_q  = np.load(EMB_SP2B / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_SP2B / "train/query_ids.json") as f: sp2b_qids = json.load(f)
sp2b_qid2i = {q: i for i, q in enumerate(sp2b_qids)}
sp2b_q_al = np.array([sp2b_q[sp2b_qid2i[q]] if q in sp2b_qid2i else np.zeros(sp2b_q.shape[1], dtype=np.float32) for q in ft_qids])

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

pool_bge_qi = {q: i for i, q in enumerate(bge_pool_qids)}
pool_bge_sim = bge_pool @ bge_c_al.T

q2bc = defaultdict(list)
for i, q in enumerate(bge_cite_qids): q2bc[q].append(i)
q2ec = defaultdict(list)
for i, q in enumerate(e5_cite_qids): q2ec[q].append(i)

def mm(v): return (v - v.min()) / (v.max() - v.min() + 1e-9)
def ndcg10(sub):
    r = evaluate(sub, qrels, ks=[10], query_domains=qdmap, verbose=False)
    return r["overall"]["NDCG@10"]

W_BM25, W_BC, W_EC, W_DOM, W_POOL_BC, W_CHUNK_CITE = 0.35, 0.95, 0.35, 0.38, 0.15, 0.07

# Pre-compute Specter2 dense similarities
print("Pre-computing Specter2_base dense sims...")
sp2b_sims = sp2b_q_al @ sp2b_c_al.T  # (100, 20000)
print(f"  Shape: {sp2b_sims.shape}")

# Pre-compute BGE cite-expanded query sims
# Idea: for each query, average the query embedding with its cite sentence embeddings
# Then use that as an expanded query for dense retrieval
print("Pre-computing BGE cite-expanded query sims...")
bge_cite_expanded_sims = np.zeros((len(ft_qids), len(ft_cids)), dtype=np.float32)
for qi, qid in enumerate(ft_qids):
    rows_bc = q2bc.get(qid, [])
    if rows_bc:
        # Average of BGE query + all cite sentences
        q_emb = bge_q_al[qi]  # (1024,)
        cite_embs = bge_cite[rows_bc]  # (n, 1024)
        expanded = np.concatenate([q_emb[None], cite_embs], axis=0).mean(axis=0)
        expanded = expanded / (np.linalg.norm(expanded) + 1e-9)
        bge_cite_expanded_sims[qi] = expanded @ bge_c_al.T
    else:
        bge_cite_expanded_sims[qi] = bge_q_al[qi] @ bge_c_al.T
print(f"  Shape: {bge_cite_expanded_sims.shape}")

def run_base():
    sub = {}
    for qi, qid in enumerate(ft_qids):
        sc = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * (bge_c_al @ bge_q_al[qi])
        if qid in bm25_tr_qi:
            sc += W_BM25 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
        qd = qdmap.get(qid, "")
        if qd in dmasks: sc += W_DOM * dmasks[qd].astype(np.float32)
        if qid in pool_bge_qi: sc += W_POOL_BC * pool_bge_sim[pool_bge_qi[qid]]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        top_cands = np.argsort(-sc)[:1000]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            bi = bge_cid2i.get(doc_id)
            if bi is not None and rows_bc:
                qc = bge_cite[rows_bc]
                sc[cidx] += W_BC * float((qc @ bge_c[bi]).max())
                if W_CHUNK_CITE > 0:
                    chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                    if chunk_rows:
                        sc[cidx] += W_CHUNK_CITE * float((qc @ cc_embs[chunk_rows].T).max())
            ei = e5_cid2i.get(doc_id)
            if ei is not None and rows_ec:
                sc[cidx] += W_EC * float((e5_cite[rows_ec] @ e5_c[ei]).max())
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub

def run_sp2b(w_sp2b=0.3):
    sub = {}
    for qi, qid in enumerate(ft_qids):
        sc = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * (bge_c_al @ bge_q_al[qi])
        sc += w_sp2b * sp2b_sims[qi]
        if qid in bm25_tr_qi:
            sc += W_BM25 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
        qd = qdmap.get(qid, "")
        if qd in dmasks: sc += W_DOM * dmasks[qd].astype(np.float32)
        if qid in pool_bge_qi: sc += W_POOL_BC * pool_bge_sim[pool_bge_qi[qid]]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        top_cands = np.argsort(-sc)[:1000]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            bi = bge_cid2i.get(doc_id)
            if bi is not None and rows_bc:
                qc = bge_cite[rows_bc]
                sc[cidx] += W_BC * float((qc @ bge_c[bi]).max())
                if W_CHUNK_CITE > 0:
                    chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                    if chunk_rows:
                        sc[cidx] += W_CHUNK_CITE * float((qc @ cc_embs[chunk_rows].T).max())
            ei = e5_cid2i.get(doc_id)
            if ei is not None and rows_ec:
                sc[cidx] += W_EC * float((e5_cite[rows_ec] @ e5_c[ei]).max())
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub

def run_cite_expanded(w_ce=0.3):
    """Replace BGE query with cite-expanded BGE query."""
    sub = {}
    for qi, qid in enumerate(ft_qids):
        sc = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * (bge_c_al @ bge_q_al[qi])
        sc += w_ce * bge_cite_expanded_sims[qi]  # additional expanded query signal
        if qid in bm25_tr_qi:
            sc += W_BM25 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
        qd = qdmap.get(qid, "")
        if qd in dmasks: sc += W_DOM * dmasks[qd].astype(np.float32)
        if qid in pool_bge_qi: sc += W_POOL_BC * pool_bge_sim[pool_bge_qi[qid]]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        top_cands = np.argsort(-sc)[:1000]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            bi = bge_cid2i.get(doc_id)
            if bi is not None and rows_bc:
                qc = bge_cite[rows_bc]
                sc[cidx] += W_BC * float((qc @ bge_c[bi]).max())
                if W_CHUNK_CITE > 0:
                    chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                    if chunk_rows:
                        sc[cidx] += W_CHUNK_CITE * float((qc @ cc_embs[chunk_rows].T).max())
            ei = e5_cid2i.get(doc_id)
            if ei is not None and rows_ec:
                sc[cidx] += W_EC * float((e5_cite[rows_ec] @ e5_c[ei]).max())
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub

def run_replace_bge_with_expanded(alpha=0.5):
    """Replace BGE query with alpha*bge_query + (1-alpha)*mean_cite."""
    sub = {}
    for qi, qid in enumerate(ft_qids):
        # Use expanded query instead of plain bge query for base dense
        expanded_sc = bge_cite_expanded_sims[qi]
        bge_sc = bge_c_al @ bge_q_al[qi]
        sc = 0.5 * (ft_c @ ft_q[qi]) + 0.5 * ((1-alpha)*bge_sc + alpha*expanded_sc)
        if qid in bm25_tr_qi:
            sc += W_BM25 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2)
        qd = qdmap.get(qid, "")
        if qd in dmasks: sc += W_DOM * dmasks[qd].astype(np.float32)
        if qid in pool_bge_qi: sc += W_POOL_BC * pool_bge_sim[pool_bge_qi[qid]]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        top_cands = np.argsort(-sc)[:1000]
        for cidx in top_cands:
            doc_id = ft_cids[cidx]
            bi = bge_cid2i.get(doc_id)
            if bi is not None and rows_bc:
                qc = bge_cite[rows_bc]
                sc[cidx] += W_BC * float((qc @ bge_c[bi]).max())
                if W_CHUNK_CITE > 0:
                    chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                    if chunk_rows:
                        sc[cidx] += W_CHUNK_CITE * float((qc @ cc_embs[chunk_rows].T).max())
            ei = e5_cid2i.get(doc_id)
            if ei is not None and rows_ec:
                sc[cidx] += W_EC * float((e5_cite[rows_ec] @ e5_c[ei]).max())
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub

print("Testing signals...")
base = ndcg10(run_base())
print(f"Baseline: {base:.4f}")
best = base

print("\n=== Specter2_base dense ===")
for w in [0.1, 0.2, 0.3, 0.5]:
    s = ndcg10(run_sp2b(w))
    mk = " ***" if s > best else ""
    print(f"  w_sp2b={w}: {s:.4f}{mk}")
    if s > best: best = s

print("\n=== BGE Cite-Expanded Query (additional dense signal) ===")
for w in [0.1, 0.2, 0.3, 0.5]:
    s = ndcg10(run_cite_expanded(w))
    mk = " ***" if s > best else ""
    print(f"  w_ce={w}: {s:.4f}{mk}")
    if s > best: best = s

print("\n=== Replace BGE with Cite-Expanded Query ===")
for alpha in [0.3, 0.5, 0.7, 1.0]:
    s = ndcg10(run_replace_bge_with_expanded(alpha))
    mk = " ***" if s > best else ""
    print(f"  alpha={alpha}: {s:.4f}{mk}")
    if s > best: best = s

print(f"\nBest: {best:.4f}")
