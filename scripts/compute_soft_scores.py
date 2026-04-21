"""
Compute & cache the soft-pipeline score matrix (n_docs x n_queries) for a split.

Used as a feature by the v7 reranker. Also exposes `get_candidates`, the
hard-domain candidate selector (kept here to avoid a circular import with
`models/hard_pipeline_with_cite.py`).

Cache path: data/soft_scores/soft_scores_{split}.npz

Usage:
    python scripts/compute_soft_scores.py --split train
    python scripts/compute_soft_scores.py --split held_out
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import load_corpus, load_queries
from models.hard_pipeline_with_cite import mm

DATA_DIR  = ROOT / "data"
EMB_DIR   = DATA_DIR / "embeddings"
CACHE_DIR = DATA_DIR / "bm25_cache"
SOFT_DIR  = DATA_DIR / "soft_scores"
SOFT_DIR.mkdir(exist_ok=True)

EMB_FT  = EMB_DIR / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE = EMB_DIR / "BAAI_bge-large-en-v1.5"
EMB_E5  = EMB_DIR / "intfloat_e5-large-v2"

PROGRESS = Path(__file__).parent / "compute_soft_scores_progress.txt"


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_soft_scores(split="train"):
    """Compute the soft pipeline's full score matrix (n_docs × n_queries).

    Returns aligned to the soft pipeline's corpus_ids order (the finetuned-bge
    ordering). Cached to disk.
    """
    cache = SOFT_DIR / f"soft_scores_{split}.npz"
    if cache.exists():
        log(f"Loading soft scores from cache {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["scores"], list(d["query_ids"]), list(d["corpus_ids"])
    log(f"Computing soft pipeline scores for split={split}...")
    t0 = time.time()

    ft_c = np.load(EMB_FT / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_FT / "corpus_ids.json") as f: ft_cids = json.load(f)
    bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)
    bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
    e5_c = np.load(EMB_E5 / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)
    e5_cid2i = {c: i for i, c in enumerate(e5_cids)}

    bge_c_al = bge_c[[bge_cid2i[c] for c in ft_cids]]

    ft_q = np.load(EMB_FT / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_FT / f"{split}/query_ids.json") as f: ft_qids = json.load(f)
    bge_q = np.load(EMB_BGE / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/query_ids.json") as f: bge_qids = json.load(f)
    bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
    bge_q_al = np.array([bge_q[bge_qid2i[q]] if q in bge_qid2i else np.zeros(bge_q.shape[1], np.float32)
                         for q in ft_qids])

    bge_cite = np.load(EMB_BGE / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
    e5_cite = np.load(EMB_E5 / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_E5 / f"{split}/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)
    bge_pool = np.load(EMB_BGE / f"{split}/cite_context_pooled_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/cite_context_pooled_query_ids.json") as f: bge_pool_qids = json.load(f)
    q2bc = defaultdict(list)
    for i, q in enumerate(bge_cite_qids): q2bc[q].append(i)
    q2ec = defaultdict(list)
    for i, q in enumerate(e5_cite_qids): q2ec[q].append(i)
    pool_qi = {q: i for i, q in enumerate(bge_pool_qids)}

    bm25_key = "train" if split == "train" else "held"
    bm25_sc = np.load(CACHE_DIR / f"bm25_{bm25_key}_scores.npy")
    with open(CACHE_DIR / f"bm25_{bm25_key}_query_ids.json") as f: bm25_qids = json.load(f)
    bm25_qi = {q: i for i, q in enumerate(bm25_qids)}
    with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
    bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
    bm25_to_ft = np.array([bm25_cid2i.get(c, 0) for c in ft_cids])

    cc_embs = np.load(EMB_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_chunk_doc_ids.json") as f: cc_doc_ids = json.load(f)
    doc_to_chunk_rows = defaultdict(list)
    for row_idx, doc_id in enumerate(cc_doc_ids):
        doc_to_chunk_rows[doc_id].append(row_idx)

    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    cdmap = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
    cdom_arr = np.array([cdmap.get(c, "") for c in ft_cids])
    dmasks = {d: cdom_arr == d for d in np.unique(cdom_arr) if d}
    query_file = DATA_DIR / ("queries.parquet" if split == "train" else "held_out_queries.parquet")
    queries_df = load_queries(query_file)
    qdmap = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

    # Soft pipeline weights (same as models/pipeline.py)
    W_FT, W_BGE, W_BM25 = 0.5, 0.5, 0.35
    W_DOMAIN, W_POOL = 0.38, 0.15
    W_BC, W_EC, W_CC = 0.95, 0.35, 0.07
    BM25_EXP = 1.2
    CITE_TOP_K = 500

    n_docs = len(ft_cids); n_q = len(ft_qids)
    scores = np.zeros((n_docs, n_q), np.float32)
    for qi, qid in enumerate(ft_qids):
        sc = W_FT * (ft_c @ ft_q[qi]) + W_BGE * (bge_c_al @ bge_q_al[qi])
        if qid in bm25_qi:
            sc += W_BM25 * np.power(mm(bm25_sc[bm25_qi[qid]][bm25_to_ft]), BM25_EXP)
        qd = qdmap.get(qid, "")
        if qd in dmasks:
            sc += W_DOMAIN * dmasks[qd].astype(np.float32)
        if qid in pool_qi:
            sc += W_POOL * (bge_pool[pool_qi[qid]] @ bge_c_al.T)
        top_cands = np.argsort(-sc)[:CITE_TOP_K]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        if rows_bc:
            qc_bge = bge_cite[rows_bc]
            for cidx in top_cands:
                doc_id = ft_cids[cidx]
                bi = bge_cid2i.get(doc_id)
                if bi is not None:
                    sc[cidx] += W_BC * float((qc_bge @ bge_c[bi]).max())
                    chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                    if chunk_rows:
                        sc[cidx] += W_CC * float((qc_bge @ cc_embs[chunk_rows].T).max())
        if rows_ec:
            qc_e5 = e5_cite[rows_ec]
            for cidx in top_cands:
                doc_id = ft_cids[cidx]
                ei = e5_cid2i.get(doc_id)
                if ei is not None:
                    sc[cidx] += W_EC * float((qc_e5 @ e5_c[ei]).max())
        scores[:, qi] = sc
        if (qi + 1) % 20 == 0:
            pct = 100 * (qi + 1) / n_q
            log(f"  PROGRESS {pct:.1f}%  q={qi+1}/{n_q}")

    np.savez_compressed(cache, scores=scores, query_ids=ft_qids, corpus_ids=ft_cids)
    log(f"Saved soft scores in {time.time()-t0:.1f}s to {cache}")
    return scores, ft_qids, ft_cids


def get_candidates(d, dom_k=80, cand_n=100):
    """Hard-domain candidate selection for each query.

    `d` is the signals dict returned by
    `models.hard_pipeline_with_cite.load_signals_plus_cite(split)`.
    Returns {query_id: np.array([corpus_idx, ...])}.
    """
    corpus_ids = d["corpus_ids"]; query_ids = d["query_ids"]
    query_domains = d["query_domains"]; dom_to_cidx = d["dom_to_cidx"]
    n_docs = d["n_docs"]; sigs = d["sigs"]
    w_map = {"minilm": 0.074, "bge": 0.0, "bm25_body": 0.07,
             "tfidf_ta": 0.0, "tfidf_ft": 0.16, "gte_mb_ft": 0.506,
             "gte_cmaxsim": 0.19}
    sig_names = list(sigs.keys())
    w = np.array([w_map[s] for s in sig_names], np.float32); w /= w.sum()
    per_query = {}
    for qi, (qid, qdom) in enumerate(zip(query_ids, query_domains)):
        dom_idx = dom_to_cidx.get(qdom, np.arange(n_docs))
        scores_dom = np.zeros(len(dom_idx), np.float32)
        for sname, ww in zip(sig_names, w):
            col = sigs[sname][dom_idx, qi]
            scores_dom += ww * mm(col)
        top_dom = dom_idx[np.argsort(-scores_dom)[:min(dom_k, len(dom_idx))]]
        if len(top_dom) < cand_n:
            scores_global = np.zeros(n_docs, np.float32)
            for sname, ww in zip(sig_names, w):
                scores_global += ww * mm(sigs[sname][:, qi])
            seen = set(top_dom.tolist())
            extra = [j for j in np.argsort(-scores_global) if j not in seen][:cand_n - len(top_dom)]
            cand_idx = list(top_dom) + extra
        else:
            cand_idx = list(top_dom[:cand_n])
        per_query[qid] = np.array(cand_idx, dtype=np.int64)
    return per_query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "held_out"])
    args = p.parse_args()
    if PROGRESS.exists(): PROGRESS.unlink()
    compute_soft_scores(args.split)


if __name__ == "__main__":
    main()
