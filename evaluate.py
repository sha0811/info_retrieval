import argparse
import json
from pathlib import Path

from helpers import load_qrels, load_queries, evaluate

DATA_DIR = Path("data")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a retrieval submission")
    parser.add_argument("submission", help="Path to submission JSON file")
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--queries", default=DATA_DIR / "queries.parquet")
    parser.add_argument("--domain", action="store_true", help="Show per-domain breakdown")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 100])
    args = parser.parse_args()

    with open(args.submission) as f:
        submission = json.load(f)

    qrels = load_qrels(args.qrels)

    query_domains = None
    if args.domain:
        queries = load_queries(args.queries)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    print(f"\nEvaluating: {args.submission}")
    evaluate(submission, qrels, ks=args.ks, query_domains=query_domains)

if __name__ == "__main__":
    main()
