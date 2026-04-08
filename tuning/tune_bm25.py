"""
Grid search over BM25 k1 and b parameters.

Each run is appended to results/bm25.csv with a date and hyperparameters column.
At the end, the top 5 combinations are printed sorted by the chosen metric.

Usage:
    python tuning/tune_bm25.py
    python tuning/tune_bm25.py --metric Recall@10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.bm25 import build_index, retrieve
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

K1_VALUES = [0.5, 1.0, 1.2, 1.5, 1.8, 2.0]
B_VALUES  = [0.25, 0.5, 0.75, 1.0]


def main():
    parser = argparse.ArgumentParser(description="Grid search for BM25 k1 and b")
    parser.add_argument("--queries", default=DATA_DIR / "queries.parquet")
    parser.add_argument("--corpus",  default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    parser.add_argument("--metric",  default="NDCG@10", help="Metric to sort results by")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    corpus  = load_corpus(args.corpus)
    qrels   = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    total = len(K1_VALUES) * len(B_VALUES)
    print(f"Grid search: {len(K1_VALUES)} k1 values × {len(B_VALUES)} b values = {total} runs\n")

    run_results = []
    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            run = (i * len(B_VALUES)) + j + 1
            print(f"[{run}/{total}] k1={k1}  b={b}")

            bm25, corpus_ids = build_index(corpus, k1=k1, b=b)
            submission       = retrieve(queries, bm25, corpus_ids)
            results          = evaluate(submission, qrels, ks=[10, 100],
                                        query_domains=query_domains, verbose=False)

            hp = {"k1": k1, "b": b}
            save_results(results, RESULTS_DIR / "bm25.csv", hyperparameters=hp)

            score = results["overall"].get(args.metric, 0)
            run_results.append((k1, b, score))
            print(f"    {args.metric}: {round(score, 4)}")

    run_results.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop 5 by {args.metric}:")
    print(f"  {'k1':<6} {'b':<6} {args.metric}")
    for k1, b, score in run_results[:5]:
        print(f"  {k1:<6} {b:<6} {round(score, 4)}")


if __name__ == "__main__":
    main()
