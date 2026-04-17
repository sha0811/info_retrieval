"""
Reciprocal Rank Fusion — combine multiple ranked submission files.

RRF score: sum of 1 / (k + rank) across all systems, where rank is 1-indexed.
Documents not present in a system's ranking are ignored (not penalized).

Usage:
    python models/rrf.py
    python models/rrf.py --inputs submissions/bm25.json submissions/dense.json
    python models/rrf.py --inputs submissions/bm25.json submissions/tfidf.json submissions/dense.json --k 60 --output submissions/rrf.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"

DEFAULT_INPUTS = [
    SUBMISSIONS_DIR / "bm25.json",
    SUBMISSIONS_DIR / "tfidf.json",
    SUBMISSIONS_DIR / "dense.json",
]


def rrf_fusion(submissions: list[dict], k: int = 60, top_k: int = 100) -> dict:
    """
    Fuse ranked lists using Reciprocal Rank Fusion.

    submissions : list of {query_id: [doc_id, ...]} dicts
    k           : RRF constant (default 60, standard value from Cormack et al.)
    top_k       : number of documents to keep per query
    """
    query_ids = set()
    for sub in submissions:
        query_ids.update(sub.keys())

    fused = {}
    for qid in query_ids:
        scores = defaultdict(float)
        for sub in submissions:
            ranked = sub.get(qid, [])
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        fused[qid] = [doc_id for doc_id, _ in
                      sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return fused


def main():
    parser = argparse.ArgumentParser(description="Reciprocal Rank Fusion")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS,
                        help="Submission JSON files to fuse")
    parser.add_argument("--k",      type=int, default=60,
                        help="RRF constant (default: 60)")
    parser.add_argument("--top-k",  type=int, default=100)
    parser.add_argument("--output", default=SUBMISSIONS_DIR / "rrf.json")
    parser.add_argument("--queries", default=DATA_DIR / "held_out_queries.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    submissions = []
    for path in args.inputs:
        with open(path) as f:
            submissions.append(json.load(f))
        print(f"Loaded {path}")

    print(f"\nFusing {len(submissions)} systems with k={args.k} ...")
    fused = rrf_fusion(submissions, k=args.k, top_k=args.top_k)

    if not args.no_eval:
        queries = load_queries(args.queries)
        qrels   = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(fused, qrels, ks=[10, 100], query_domains=query_domains)
        systems = "+".join(Path(p).stem for p in args.inputs)
        save_results(results, RESULTS_DIR / "rrf.csv", hyperparameters={"systems": systems, "k": args.k})

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fused, f)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
