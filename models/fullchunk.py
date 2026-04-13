"""
Full-chunk retrieval — use body chunks on BOTH query and corpus sides.

For each (query, corpus_doc) pair, computes similarity across all combinations
of representations (title+abstract and body chunks) and takes the max.

Score(Q, C) = alpha * sim(q_ta, c_ta)
            + (1 - alpha) * max over all (q_repr, c_repr) pairs of sim(q_repr, c_repr)

Where q_repr includes query TA + query body chunks,
and   c_repr includes corpus TA + corpus body chunks.

Usage:
    python models/fullchunk.py --model BAAI/bge-large-en-v1.5
    python models/fullchunk.py --model BAAI/bge-large-en-v1.5 --alpha 0.4
    python models/fullchunk.py --model BAAI/bge-large-en-v1.5 --alpha 0.5 --split train
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, load_embeddings, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"


def retrieve(query_embs, corpus_embs, q_ids, c_ids,
             q_chunk_embs, q_chunk_query_ids,
             c_chunk_embs, c_chunk_doc_ids,
             alpha: float = 0.5, top_k: int = 100) -> dict:
    """
    Retrieve using max similarity across all representation pairs.

    alpha: weight on TA-vs-TA score. (1-alpha) on the max across all chunk pairs.
    """
    n_corpus = corpus_embs.shape[0]

    # TA-vs-TA similarity: (n_queries, n_corpus)
    ta_sim = query_embs @ corpus_embs.T

    # Build chunk indices
    cid_to_idx = {cid: i for i, cid in enumerate(c_ids)}

    qid_to_chunk_rows = defaultdict(list)
    for row_idx, qid in enumerate(q_chunk_query_ids):
        qid_to_chunk_rows[qid].append(row_idx)

    cid_to_chunk_rows = defaultdict(list)
    for row_idx, cid in enumerate(c_chunk_doc_ids):
        cid_to_chunk_rows[cid].append(row_idx)

    # Map each corpus chunk row -> corpus index for vectorized max-pooling
    c_chunk_corpus_idx = np.array([cid_to_idx[cid] for cid in c_chunk_doc_ids], dtype=np.int32) \
        if c_chunk_doc_ids else np.array([], dtype=np.int32)

    submission = {}
    for i, qid in enumerate(q_ids):
        # Start with TA-vs-TA as the baseline
        ta_scores = ta_sim[i]  # (n_corpus,)
        max_chunk_scores = np.copy(ta_scores)

        q_rows = qid_to_chunk_rows.get(qid, [])

        # Query TA vs corpus chunks → max-pool per corpus doc
        if len(c_chunk_doc_ids) > 0:
            qta_vs_cchunks = query_embs[i] @ c_chunk_embs.T  # (n_corpus_chunks,)
            # Scatter-max: for each corpus doc, take max across its chunks
            np.maximum.at(max_chunk_scores, c_chunk_corpus_idx, qta_vs_cchunks)

        # Query chunks vs corpus TA → max across query chunks per corpus doc
        if q_rows:
            qchunks_vs_cta = q_chunk_embs[q_rows] @ corpus_embs.T  # (n_q_chunks, n_corpus)
            q_chunk_max = np.max(qchunks_vs_cta, axis=0)  # (n_corpus,)
            np.maximum(max_chunk_scores, q_chunk_max, out=max_chunk_scores)

        # Query chunks vs corpus chunks → max across all pairs per corpus doc
        if q_rows and len(c_chunk_doc_ids) > 0:
            qchunks_vs_cchunks = q_chunk_embs[q_rows] @ c_chunk_embs.T  # (n_q_chunks, n_c_chunks)
            # Max across query chunks for each corpus chunk
            max_per_c_chunk = np.max(qchunks_vs_cchunks, axis=0)  # (n_c_chunks,)
            # Scatter-max per corpus doc
            np.maximum.at(max_chunk_scores, c_chunk_corpus_idx, max_per_c_chunk)

        final_scores = alpha * ta_scores + (1.0 - alpha) * max_chunk_scores

        top_indices = np.argsort(-final_scores)[:top_k]
        submission[qid] = [c_ids[j] for j in top_indices]

    return submission


def main():
    parser = argparse.ArgumentParser(description="Full-chunk retrieval (query + corpus chunks)")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                        help="Model slug (must match a folder in data/embeddings/)")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on TA-vs-TA score (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    QUERY_FILES = {
        "train":    DATA_DIR / "queries.parquet",
        "held_out": DATA_DIR / "held_out_queries.parquet",
    }

    model_slug  = args.model.replace("/", "_").replace("\\", "_")
    model_dir   = EMBEDDINGS_DIR / model_slug
    queries_dir = model_dir / args.split
    output_path = args.output or SUBMISSIONS_DIR / f"fullchunk_{model_slug}_{args.split}.json"

    # Load TA embeddings
    print(f"Loading TA embeddings ...")
    query_embs, q_ids = load_embeddings(queries_dir / "query_embeddings.npy",
                                        queries_dir / "query_ids.json")
    corpus_embs, c_ids = load_embeddings(model_dir / "corpus_embeddings.npy",
                                         model_dir / "corpus_ids.json")
    print(f"  Query TA:  {query_embs.shape}")
    print(f"  Corpus TA: {corpus_embs.shape}")

    # Load query chunks
    q_chunk_emb_path = queries_dir / "chunk_embeddings.npy"
    q_chunk_ids_path = queries_dir / "chunk_query_ids.json"
    if q_chunk_emb_path.exists() and q_chunk_ids_path.exists():
        print(f"Loading query chunks ...")
        q_chunk_embs = np.load(q_chunk_emb_path).astype(np.float32)
        with open(q_chunk_ids_path) as f:
            q_chunk_query_ids = json.load(f)
        print(f"  Query chunks: {q_chunk_embs.shape} ({len(set(q_chunk_query_ids))} queries)")
    else:
        print("No query chunk embeddings found — using TA only for queries.")
        q_chunk_embs = np.empty((0, query_embs.shape[1]), dtype=np.float32)
        q_chunk_query_ids = []

    # Load corpus chunks
    c_chunk_emb_path = model_dir / "corpus_chunk_embeddings.npy"
    c_chunk_ids_path = model_dir / "corpus_chunk_doc_ids.json"
    if c_chunk_emb_path.exists() and c_chunk_ids_path.exists():
        print(f"Loading corpus chunks ...")
        c_chunk_embs = np.load(c_chunk_emb_path).astype(np.float32)
        with open(c_chunk_ids_path) as f:
            c_chunk_doc_ids = json.load(f)
        print(f"  Corpus chunks: {c_chunk_embs.shape} ({len(set(c_chunk_doc_ids))} docs)")
    else:
        print("No corpus chunk embeddings found.")
        print("  Run: python scripts/embed_corpus_chunks.py --model", args.model)
        c_chunk_embs = np.empty((0, corpus_embs.shape[1]), dtype=np.float32)
        c_chunk_doc_ids = []

    # Retrieve
    print(f"\nRetrieving with alpha={args.alpha} ...")
    submission = retrieve(query_embs, corpus_embs, q_ids, c_ids,
                          q_chunk_embs, q_chunk_query_ids,
                          c_chunk_embs, c_chunk_doc_ids,
                          alpha=args.alpha, top_k=args.top_k)

    # Evaluate
    if not args.no_eval:
        queries = load_queries(QUERY_FILES[args.split])
        qrels   = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "fullchunk.csv",
                     hyperparameters={"model": args.model, "alpha": args.alpha,
                                      "split": args.split})

    # Save
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
