"""
Dense retrieval baseline — cosine similarity over pre-computed embeddings.

Embeddings are L2-normalised, so cosine similarity = dot product.
To encode with a different model, run:
    python scripts/embed.py --model <model-name>

Usage:
    python models/dense.py
    python models/dense.py --model BAAI/bge-base-en-v1.5
    python models/dense.py --output submissions/dense_minilm.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, load_embeddings, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve(query_embs, corpus_embs, q_ids, c_ids, top_k: int = 100) -> dict:
    sim_matrix  = query_embs @ corpus_embs.T                        # (n_queries, n_corpus)
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]

    return {qid: [c_ids[j] for j in top_indices[i]]
            for i, qid in enumerate(q_ids)}


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval baseline")
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help="Model slug (must match a folder in data/embeddings/)")
    parser.add_argument("--split",   default="held_out", choices=["train", "held_out"],
                        help="Which query embeddings to use (default: held_out)")
    parser.add_argument("--queries", default=None,
                        help="Override queries parquet path (for --no-eval this is unused)")
    parser.add_argument("--corpus",  default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    parser.add_argument("--output",  default=SUBMISSIONS_DIR / "dense.json",
                        help="Output path (default: submissions/dense.json)")
    parser.add_argument("--top-k",   type=int, default=100)
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    QUERY_FILES = {
        "train":    DATA_DIR / "queries.parquet",
        "held_out": DATA_DIR / "held_out_queries.parquet",
    }

    model_slug  = args.model.replace("/", "_").replace("\\", "_")
    model_dir   = EMBEDDINGS_DIR / model_slug
    queries_dir = model_dir / args.split

    print(f"Loading embeddings from {model_dir} ...")
    query_embs, q_ids = load_embeddings(queries_dir / "query_embeddings.npy",
                                        queries_dir / "query_ids.json")
    corpus_embs, c_ids = load_embeddings(model_dir / "corpus_embeddings.npy",
                                         model_dir / "corpus_ids.json")
    print(f"  Query  embeddings : {query_embs.shape}")
    print(f"  Corpus embeddings : {corpus_embs.shape}")

    print("Running dense retrieval...")
    submission = retrieve(query_embs, corpus_embs, q_ids, c_ids, top_k=args.top_k)

    if not args.no_eval:
        queries_path = args.queries if args.queries else QUERY_FILES[args.split]
        queries = load_queries(queries_path)
        qrels   = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "dense.csv")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
