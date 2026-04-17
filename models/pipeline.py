"""
Full retrieval pipeline, best configuration for now

Signals combined in score space:
  0.5  * cosine(ft_small_query,  ft_small_doc)       (finetuned bge-small)
  0.5  * cosine(bge_large_query, bge_large_doc)       (bge-large-en-v1.5)
  0.35 * mm(BM25)^1.2                                 (BM25 with power scaling)
  0.38 * (query_domain == doc_domain)                 (hard domain boost)
  0.15 * cosine(pooled_cite_bge, bge_large_doc)       (pooled citation context)
  0.95 * max_i cosine(cite_bge_i, bge_large_doc)      (per-cite bge max)
  0.35 * max_i cosine(cite_e5_i,  e5_large_doc)       (per-cite e5 max)
  0.07 * max_i cosine(cite_bge_i, bge_chunk_doc)      (chunk-level cite signal)
(The actual fully detailed pipeline is on the presentation slides)

Usage:
    python models/pipeline.py --split train
    python models/pipeline.py --split held_out --zip
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR      = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
CACHE_DIR     = DATA_DIR / "bm25_cache"

EMB_FT  = EMBEDDINGS_DIR / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE = EMBEDDINGS_DIR / "BAAI_bge-large-en-v1.5"
EMB_E5  = EMBEDDINGS_DIR / "intfloat_e5-large-v2"

# Best weights
W_FT     = 0.5
W_BGE    = 0.5
W_BM25   = 0.35
W_DOMAIN = 0.38
W_POOL   = 0.15
W_BC     = 0.95
W_EC     = 0.35
W_CC     = 0.07
BM25_EXP = 1.2
CITE_TOP_K = 500


def mm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def retrieve(split: str, top_k: int = 100,
             w_ft: float = W_FT, w_bge: float = W_BGE,
             w_bm25: float = W_BM25, bm25_exp: float = BM25_EXP,
             w_domain: float = W_DOMAIN, w_pool: float = W_POOL,
             w_bc: float = W_BC, w_ec: float = W_EC, w_cc: float = W_CC,
             cite_top_k: int = CITE_TOP_K) -> dict:
    print(f"Loading corpus embeddings ...")
    ft_c = np.load(EMB_FT / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_FT / "corpus_ids.json") as f: ft_cids = json.load(f)

    bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)
    bge_cid2i = {c: i for i, c in enumerate(bge_cids)}

    e5_c = np.load(EMB_E5 / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)
    e5_cid2i = {c: i for i, c in enumerate(e5_cids)}

    # bge corpus aligned to ft ordering
    bge_c_al = bge_c[[bge_cid2i[c] for c in ft_cids]]

    # chunk embeddings (bge)
    cc_embs = np.load(EMB_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_chunk_doc_ids.json") as f: cc_doc_ids = json.load(f)
    doc_to_chunk_rows: dict = defaultdict(list)
    for row_idx, doc_id in enumerate(cc_doc_ids):
        doc_to_chunk_rows[doc_id].append(row_idx)

    print(f"Loading query embeddings ({split}) ...")
    ft_q = np.load(EMB_FT / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_FT / f"{split}/query_ids.json") as f: ft_qids = json.load(f)

    bge_q = np.load(EMB_BGE / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/query_ids.json") as f: bge_qids = json.load(f)
    bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
    bge_q_al = np.array([
        bge_q[bge_qid2i[q]] if q in bge_qid2i else np.zeros(bge_q.shape[1], dtype=np.float32)
        for q in ft_qids
    ])

    print(f"Loading cite-context embeddings ({split}) ...")
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

    print(f"Loading BM25 scores ({split}) ...")
    bm25_key = "train" if split == "train" else "held"
    bm25_sc = np.load(CACHE_DIR / f"bm25_{bm25_key}_scores.npy")
    with open(CACHE_DIR / f"bm25_{bm25_key}_query_ids.json") as f: bm25_qids = json.load(f)
    bm25_qi = {q: i for i, q in enumerate(bm25_qids)}
    with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
    bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
    bm25_to_ft = np.array([bm25_cid2i.get(c, 0) for c in ft_cids])

    print(f"Loading metadata ...")
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    cdmap = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
    cdom_arr = np.array([cdmap.get(c, "") for c in ft_cids])
    dmasks = {d: cdom_arr == d for d in np.unique(cdom_arr) if d}

    query_file = DATA_DIR / ("queries.parquet" if split == "train" else "held_out_queries.parquet")
    queries_df = load_queries(query_file)
    qdmap = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

    print(f"Retrieving ({len(ft_qids)} queries) ...")
    submission = {}
    for qi, qid in enumerate(ft_qids):
        sc = w_ft * (ft_c @ ft_q[qi]) + w_bge * (bge_c_al @ bge_q_al[qi])
        if qid in bm25_qi:
            sc += w_bm25 * np.power(mm(bm25_sc[bm25_qi[qid]][bm25_to_ft]), bm25_exp)
        qd = qdmap.get(qid, "")
        if qd in dmasks:
            sc += w_domain * dmasks[qd].astype(np.float32)
        if qid in pool_qi:
            sc += w_pool * (bge_pool[pool_qi[qid]] @ bge_c_al.T)

        top_cands = np.argsort(-sc)[:cite_top_k]
        rows_bc = q2bc.get(qid, [])
        rows_ec = q2ec.get(qid, [])
        if rows_bc:
            qc_bge = bge_cite[rows_bc]
            for cidx in top_cands:
                doc_id = ft_cids[cidx]
                bi = bge_cid2i.get(doc_id)
                if bi is not None:
                    sc[cidx] += w_bc * float((qc_bge @ bge_c[bi]).max())
                    if w_cc > 0:
                        chunk_rows = doc_to_chunk_rows.get(doc_id, [])
                        if chunk_rows:
                            sc[cidx] += w_cc * float((qc_bge @ cc_embs[chunk_rows].T).max())
        if rows_ec:
            qc_e5 = e5_cite[rows_ec]
            for cidx in top_cands:
                doc_id = ft_cids[cidx]
                ei = e5_cid2i.get(doc_id)
                if ei is not None:
                    sc[cidx] += w_ec * float((qc_e5 @ e5_c[ei]).max())

        submission[qid] = [ft_cids[i] for i in np.argsort(-sc)[:top_k]]
        if (qi + 1) % 20 == 0:
            print(f"  {qi+1}/{len(ft_qids)}", flush=True)

    return submission, qdmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    submission, qdmap = retrieve(args.split, top_k=args.top_k)

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        results = evaluate(submission, qrels, ks=[10], query_domains=qdmap, verbose=True)
        save_results(
            results, RESULTS_DIR / "pipeline.csv",
            hyperparameters={
                "w_ft": W_FT, "w_bge": W_BGE, "w_bm25": W_BM25,
                "w_domain": W_DOMAIN, "w_pool": W_POOL,
                "w_bc": W_BC, "w_ec": W_EC, "w_cc": W_CC,
                "bm25_exp": BM25_EXP, "split": args.split,
            },
        )

    name = args.output or f"pipeline_{args.split}"
    out_path = SUBMISSIONS_DIR / (name if name.endswith(".json") else f"{name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {out_path}")

    if args.zip:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, "submission_data.json")
        print(f"Zipped -> {zip_path}")


if __name__ == "__main__":
    main()
