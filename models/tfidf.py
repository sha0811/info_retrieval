"""
TF-IDF baseline — sparse retrieval over title + abstract.

Usage:
    python models/tfidf.py
    python models/tfidf.py --output submissions/tfidf_baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, format_text, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def retrieve(queries, corpus, top_k: int = 100) -> dict:
    query_ids    = queries["doc_id"].tolist()
    corpus_ids   = corpus["doc_id"].tolist()
    query_texts  = [format_text(row) for _, row in queries.iterrows()]
    corpus_texts = [format_text(row) for _, row in corpus.iterrows()]

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 1),
        stop_words="english",
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix  = vectorizer.transform(query_texts)

    sim_matrix = cosine_similarity(query_matrix, corpus_matrix)

    submission = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(-sim_matrix[i])[:top_k]
        submission[qid] = [corpus_ids[j] for j in top_indices]

    return submission


def main():
    parser = argparse.ArgumentParser(description="TF-IDF retrieval baseline")
    parser.add_argument("--queries", default=DATA_DIR / "queries.parquet")
    parser.add_argument("--corpus",  default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    parser.add_argument("--output",  default=SUBMISSIONS_DIR / "tfidf.json")
    parser.add_argument("--top-k",   type=int, default=100)
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    corpus  = load_corpus(args.corpus)

    print(f"Queries : {len(queries)}")
    print(f"Corpus  : {len(corpus)}")
    print("Running TF-IDF retrieval...")

    submission = retrieve(queries, corpus, top_k=args.top_k)

    if not args.no_eval:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "tfidf.csv")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
