"""
BM25 title-to-title retrieval.

Uses query paper title as BM25 query against corpus titles.
Rationale: a paper's title is the most concentrated signal of its topic.
Papers with overlapping title vocabulary are very likely citation targets.
This is a pure lexical bridge that dense models systematically underweight.

Usage:
    python models/bm25_title.py --split train
    python models/bm25_title.py --split held_out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}


def tokenize(text: str) -> list[str]:
    return (text or "").lower().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--corpus", default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    output_path = SUBMISSIONS_DIR / f"bm25_title_{args.split}.json"

    print(f"Loading corpus ...")
    corpus = load_corpus(args.corpus)
    corpus_ids = corpus["doc_id"].tolist()
    corpus_titles = [str(r.get("title", "") or "") for _, r in corpus.iterrows()]

    print("Tokenizing corpus titles ...")
    tokenized = [tokenize(t) for t in corpus_titles]
    print("Building BM25 index over titles ...")
    bm25 = BM25Okapi(tokenized, k1=args.k1, b=args.b)

    print(f"Loading queries ({args.split}) ...")
    queries_df = load_queries(QUERY_FILES[args.split])
    print(f"  {len(queries_df)} queries")

    print("Retrieving ...")
    submission = {}
    for _, row in queries_df.iterrows():
        qid = row["doc_id"]
        q_title = str(row.get("title", "") or "")
        q_tokens = tokenize(q_title)

        if q_tokens:
            scores = bm25.get_scores(q_tokens)
        else:
            scores = np.zeros(len(corpus_ids))

        top_indices = np.argsort(-scores)[: args.top_k]
        submission[qid] = [corpus_ids[i] for i in top_indices]

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(
            results,
            RESULTS_DIR / "bm25_title.csv",
            hyperparameters={"split": args.split, "k1": args.k1, "b": args.b},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
