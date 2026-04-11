"""
Cross-encoder reranker — rerank any submission JSON using a cross-encoder model.

Takes a retrieval submission (e.g. from RRF, dense, BM25) as input and reorders
the top-k candidates per query using a cross-encoder that jointly scores (query, document) pairs.

Usage:
    python models/reranker.py --input submissions/rrf.json
    python models/reranker.py --input submissions/dense_bge.json --output submissions/reranked_bge.json
    python models/reranker.py --input submissions/rrf.json --model cross-encoder/ms-marco-MiniLM-L-6-v2 --top-k-rerank 50
    python models/reranker.py --input submissions/rrf.json --split train --no-eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, format_text, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(submission: dict, queries_df, corpus_df, model: CrossEncoder,
           top_k_rerank: int = 100, query_title_only: bool = False) -> dict:
    """
    Rerank each query's candidate list using a cross-encoder.

    submission       : {query_id: [doc_id, ...]}  — input ranked list (e.g. top-100 from RRF)
    queries_df       : DataFrame with doc_id, title, abstract columns
    corpus_df        : DataFrame with doc_id, title, abstract columns
    model            : CrossEncoder instance
    top_k_rerank     : number of candidates to rerank (reranks all candidates if <= 0)
    query_title_only : if True, use only the title as query text (reduces token length)
    """
    def get_query_text(row):
        return str(row.get("title", "") or "").strip() if query_title_only else format_text(row)

    query_texts  = {row["doc_id"]: get_query_text(row) for _, row in queries_df.iterrows()}
    corpus_texts = {row["doc_id"]: format_text(row) for _, row in corpus_df.iterrows()}

    reranked = {}
    for i, (qid, candidates) in enumerate(submission.items()):
        if i % 10 == 0:
            print(f"  Reranking query {i+1}/{len(submission)} ...")

        pool = candidates[:top_k_rerank] if top_k_rerank > 0 else candidates
        query_text = query_texts.get(qid, "")

        pairs  = [(query_text, corpus_texts.get(doc_id, "")) for doc_id in pool]
        scores = model.predict(pairs, show_progress_bar=False)

        order = np.argsort(-scores)
        reranked_pool = [pool[j] for j in order]

        # append any candidates beyond top_k_rerank unchanged
        reranked[qid] = reranked_pool + candidates[top_k_rerank:]

    return reranked


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder reranker")
    parser.add_argument("--input",        required=True,
                        help="Input submission JSON to rerank (e.g. submissions/rrf.json)")
    parser.add_argument("--output",       default=None,
                        help="Output path (default: submissions/reranked_<input_stem>.json)")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help="Cross-encoder model name (HuggingFace hub)")
    parser.add_argument("--top-k-rerank", type=int, default=100,
                        help="Number of candidates to rerank per query (default: 100)")
    parser.add_argument("--split",        default="train", choices=["train", "held_out"],
                        help="Which query set to evaluate against (default: train)")
    parser.add_argument("--corpus",       default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",        default=DATA_DIR / "qrels.json")
    parser.add_argument("--query-title-only", action="store_true",
                        help="Use only the title (not abstract) as query text for the cross-encoder")
    parser.add_argument("--no-eval",      action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    QUERY_FILES = {
        "train":    DATA_DIR / "queries.parquet",
        "held_out": DATA_DIR / "held_out_queries.parquet",
    }

    output_path = args.output or SUBMISSIONS_DIR / f"reranked_{Path(args.input).stem}.json"

    # Load input submission
    print(f"Loading submission from {args.input} ...")
    with open(args.input) as f:
        submission = json.load(f)
    print(f"  {len(submission)} queries")

    # Load queries and corpus
    queries_path = QUERY_FILES[args.split]
    print(f"Loading queries ({args.split}) ...")
    queries_df = load_queries(queries_path)

    print(f"Loading corpus ...")
    corpus_df = load_corpus(args.corpus)

    # Load cross-encoder
    print(f"\nLoading cross-encoder: {args.model} ...")
    model = CrossEncoder(args.model)

    # Rerank
    print(f"\nReranking (top-{args.top_k_rerank} candidates per query) ...")
    reranked = rerank(submission, queries_df, corpus_df, model,
                      top_k_rerank=args.top_k_rerank,
                      query_title_only=args.query_title_only)

    # Evaluate
    if not args.no_eval:
        queries  = load_queries(queries_path)
        qrels    = load_qrels(args.qrels)
        query_domains = dict(zip(queries["doc_id"], queries["domain"]))
        results  = evaluate(reranked, qrels, ks=[10, 100], query_domains=query_domains)
        input_stem = Path(args.input).stem
        model_slug = args.model.replace("/", "_")
        save_results(results, RESULTS_DIR / "reranker.csv",
                     hyperparameters={"input": input_stem, "model": model_slug,
                                      "top_k_rerank": args.top_k_rerank})

    # Save
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reranked, f)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
