"""
Multi-chunk retrieval — combine title+abstract similarity with body chunk similarity.

For each query paper:
  1. Compute cosine similarity of its title+abstract embedding vs all corpus docs
  2. Compute cosine similarity of each body chunk vs all corpus docs, take the max per doc
  3. Final score = alpha * ta_score + (1 - alpha) * max_chunk_score

Documents are ranked by final score.  Queries without chunks fall back to ta-only.

Usage:
    python models/multichunk.py --model BAAI/bge-large-en-v1.5
    python models/multichunk.py --model BAAI/bge-large-en-v1.5 --alpha 0.5
    python models/multichunk.py --model BAAI/bge-large-en-v1.5 --alpha 0.7 --split train
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
             chunk_embs, chunk_query_ids,
             alpha: float = 0.6, top_k: int = 100) -> dict:
    """
    Retrieve using title+abstract scores combined with max-chunk scores.

    alpha: weight on title+abstract score (1.0 = ta only, 0.0 = chunks only)
    """
    n_corpus = corpus_embs.shape[0]

    # Pre-compute ta similarity: (n_queries, n_corpus)
    ta_sim = query_embs @ corpus_embs.T

    # Build chunk index: query_id -> list of chunk row indices
    qid_to_chunk_rows = defaultdict(list)
    for row_idx, qid in enumerate(chunk_query_ids):
        qid_to_chunk_rows[qid].append(row_idx)

    submission = {}
    for i, qid in enumerate(q_ids):
        ta_scores = ta_sim[i]  # (n_corpus,)

        chunk_rows = qid_to_chunk_rows.get(qid, [])
        if chunk_rows and alpha < 1.0:
            # Compute chunk similarities: (n_chunks_for_query, n_corpus)
            q_chunk_embs = chunk_embs[chunk_rows]
            chunk_sim = q_chunk_embs @ corpus_embs.T
            # Max-pool across chunks for each corpus doc
            max_chunk_scores = chunk_sim.max(axis=0)  # (n_corpus,)
            final_scores = alpha * ta_scores + (1.0 - alpha) * max_chunk_scores
        else:
            final_scores = ta_scores

        top_indices = np.argsort(-final_scores)[:top_k]
        submission[qid] = [c_ids[j] for j in top_indices]

    return submission


def main():
    parser = argparse.ArgumentParser(description="Multi-chunk retrieval")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                        help="Model slug (must match a folder in data/embeddings/)")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Weight on title+abstract score (default: 0.6). "
                             "0.0 = chunks only, 1.0 = ta only")
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
    output_path = args.output or SUBMISSIONS_DIR / f"multichunk_{model_slug}_{args.split}.json"

    # Load title+abstract embeddings (from embed.py)
    print(f"Loading TA embeddings from {queries_dir} ...")
    query_embs, q_ids = load_embeddings(queries_dir / "query_embeddings.npy",
                                        queries_dir / "query_ids.json")
    corpus_embs, c_ids = load_embeddings(model_dir / "corpus_embeddings.npy",
                                         model_dir / "corpus_ids.json")
    print(f"  Query  embeddings : {query_embs.shape}")
    print(f"  Corpus embeddings : {corpus_embs.shape}")

    # Load chunk embeddings (from embed_chunks.py)
    chunk_emb_path = queries_dir / "chunk_embeddings.npy"
    chunk_ids_path = queries_dir / "chunk_query_ids.json"
    if chunk_emb_path.exists() and chunk_ids_path.exists():
        print(f"Loading chunk embeddings from {queries_dir} ...")
        chunk_embs = np.load(chunk_emb_path).astype(np.float32)
        with open(chunk_ids_path) as f:
            chunk_query_ids = json.load(f)
        print(f"  Chunk embeddings  : {chunk_embs.shape}")
        print(f"  Queries with chunks: {len(set(chunk_query_ids))}")
    else:
        print(f"WARNING: No chunk embeddings found at {queries_dir}/")
        print("  Run: python scripts/embed_chunks.py --model {args.model} --split {args.split}")
        print("  Falling back to TA-only retrieval.")
        chunk_embs = np.empty((0, query_embs.shape[1]), dtype=np.float32)
        chunk_query_ids = []

    # Retrieve
    print(f"\nRetrieving with alpha={args.alpha} ...")
    submission = retrieve(query_embs, corpus_embs, q_ids, c_ids,
                          chunk_embs, chunk_query_ids,
                          alpha=args.alpha, top_k=args.top_k)

    # Evaluate
    if not args.no_eval:
        queries = load_queries(QUERY_FILES[args.split])
        qrels   = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "multichunk.csv",
                     hyperparameters={"model": args.model, "alpha": args.alpha,
                                      "split": args.split})

    # Save
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
