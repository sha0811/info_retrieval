"""
BM25 baseline — sparse retrieval over title + abstract.


Default parameters (Okapi BM25 standard values):
    k1 = 1.5  — term frequency saturation
    b  = 0.75 — document length normalization

Usage:
    python models/bm25.py
    python models/bm25.py --k1 1.2 --b 0.8
    python models/bm25.py --output submissions/bm25.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, format_text, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"


def tokenize(text: str) -> list:
    return text.lower().split()


def build_index(corpus, k1: float = 1.5, b: float = 0.75):
    """Build BM25 index from corpus. Only depends on corpus, not queries."""
    corpus_ids   = corpus["doc_id"].tolist()
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]

    print("Tokenizing corpus...")
    tokenized_corpus = [tokenize(t) for t in corpus_texts]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b) 

    return bm25, corpus_ids


def retrieve(queries, bm25, corpus_ids, top_k: int = 100) -> dict:
    query_ids   = queries["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries.iterrows()]

    print("Retrieving...")
    submission = {}
    for qid, query_text in zip(query_ids, query_texts):
        scores      = bm25.get_scores(tokenize(query_text))
        top_indices = np.argsort(-scores)[:top_k]
        submission[qid] = [corpus_ids[i] for i in top_indices]

    return submission


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval baseline")
    parser.add_argument("--queries", default=DATA_DIR / "queries.parquet")
    parser.add_argument("--corpus",  default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    parser.add_argument("--output",  default=SUBMISSIONS_DIR / "bm25.json")
    parser.add_argument("--top-k",   type=int,   default=100)
    parser.add_argument("--k1",      type=float, default=1.5)
    parser.add_argument("--b",       type=float, default=0.75)
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    corpus  = load_corpus(args.corpus)

    print(f"Queries : {len(queries)}")
    print(f"Corpus  : {len(corpus)}")
    print(f"k1={args.k1}  b={args.b}")

    bm25, corpus_ids = build_index(corpus, k1=args.k1, b=args.b)
    submission = retrieve(queries, bm25, corpus_ids, top_k=args.top_k)

    if not args.no_eval:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "bm25.csv", hyperparameters={"k1": args.k1, "b": args.b})

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
