"""
Domain-filtered dense retrieval.

For each query, retrieves top-K candidates from same-domain corpus papers only,
then fills remaining slots from full-corpus retrieval.

Rationale: citations are overwhelmingly within-domain. Restricting the search
pool to same-domain papers increases density of relevant papers in the retrieved
set, improving Recall@100 — the hard ceiling for any downstream reranker.

Two modes combined via RRF inside this script:
  - domain-only:  top-100 within same domain
  - full-corpus:  top-100 from all corpus (catches cross-domain refs)

Usage:
    python models/domain_dense.py --model ftsmall --split train
    python models/domain_dense.py --model ftsmall --split held_out
    python models/domain_dense.py --model bge_large --split train
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}

MODEL_CONFIGS = {
    "ftsmall": {
        "emb_dir": EMBEDDINGS_DIR / "data_finetuned_models_BAAI_bge-small-en-v1.5",
        "corpus_emb": "corpus_embeddings.npy",
        "corpus_ids": "corpus_ids.json",
        "query_emb": "{split}/query_embeddings.npy",
        "query_ids": "{split}/query_ids.json",
    },
    "bge_large": {
        "emb_dir": EMBEDDINGS_DIR / "BAAI_bge-large-en-v1.5",
        "corpus_emb": "corpus_embeddings.npy",
        "corpus_ids": "corpus_ids.json",
        "query_emb": "{split}/query_embeddings.npy",
        "query_ids": "{split}/query_ids.json",
    },
}


def rrf_merge(list_a: list, list_b: list, k: int = 60, top_k: int = 100) -> list:
    """Merge two ranked lists with RRF."""
    scores = defaultdict(float)
    for rank, doc in enumerate(list_a, 1):
        scores[doc] += 1.0 / (k + rank)
    for rank, doc in enumerate(list_b, 1):
        scores[doc] += 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def retrieve_domain_filtered(
    query_embs: np.ndarray,
    query_ids: list,
    corpus_embs: np.ndarray,
    corpus_ids: list,
    corpus_domains: list,
    query_domains: dict,
    top_k: int = 100,
    min_domain_size: int = 50,
) -> dict:
    """
    For each query:
      1. Retrieve top-K from same-domain corpus papers.
      2. Retrieve top-K from full corpus.
      3. Merge via RRF.
    """
    submission = {}
    n_corpus = len(corpus_ids)

    for i, qid in enumerate(query_ids):
        q_emb = query_embs[i]
        q_domain = query_domains.get(qid, "")

        # Full-corpus cosine sim (embeddings are normalized)
        full_scores = corpus_embs @ q_emb
        full_ranked = np.argsort(-full_scores)[:top_k].tolist()
        full_docs = [corpus_ids[j] for j in full_ranked]

        # Domain-filtered cosine sim
        if q_domain:
            domain_mask = np.array([d == q_domain for d in corpus_domains])
            domain_count = domain_mask.sum()
            if domain_count >= min_domain_size:
                domain_indices = np.where(domain_mask)[0]
                domain_scores = full_scores[domain_mask]
                top_domain = np.argsort(-domain_scores)[:top_k]
                domain_docs = [corpus_ids[domain_indices[j]] for j in top_domain]
                # Merge full + domain via RRF
                merged = rrf_merge(domain_docs, full_docs, top_k=top_k)
            else:
                # Domain too small — just use full corpus
                merged = full_docs
        else:
            merged = full_docs

        submission[qid] = merged

    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ftsmall", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-domain-size", type=int, default=50,
                        help="Min domain corpus size to apply domain filter")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    emb_dir = cfg["emb_dir"]
    output_path = SUBMISSIONS_DIR / f"domain_dense_{args.model}_{args.split}.json"

    print(f"Loading corpus ...")
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_ids_list = corpus["doc_id"].tolist()
    corpus_domains_list = corpus["domain"].fillna("").tolist()

    print(f"Loading corpus embeddings ({args.model}) ...")
    corpus_embs = np.load(emb_dir / cfg["corpus_emb"]).astype(np.float32)
    with open(emb_dir / cfg["corpus_ids"]) as f:
        corpus_ids_emb = json.load(f)

    # Align corpus order with embeddings
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids_emb)}
    emb_order = [id_to_idx[cid] for cid in corpus_ids_list if cid in id_to_idx]
    if len(emb_order) != len(corpus_ids_list):
        print(f"  Warning: {len(corpus_ids_list) - len(emb_order)} corpus docs missing from embeddings")
        # Filter to those that have embeddings
        corpus_ids_list = [cid for cid in corpus_ids_list if cid in id_to_idx]
        corpus_domains_list = [corpus_domains_list[corpus["doc_id"].tolist().index(cid)] for cid in corpus_ids_list]
    corpus_embs = corpus_embs[emb_order]
    print(f"  Corpus embeddings shape: {corpus_embs.shape}")

    print(f"Loading query embeddings ({args.split}) ...")
    query_emb_path = emb_dir / cfg["query_emb"].format(split=args.split)
    query_ids_path = emb_dir / cfg["query_ids"].format(split=args.split)
    query_embs = np.load(query_emb_path).astype(np.float32)
    with open(query_ids_path) as f:
        query_ids = json.load(f)
    print(f"  Query embeddings shape: {query_embs.shape}")

    print(f"Loading queries ({args.split}) ...")
    queries_df = load_queries(QUERY_FILES[args.split])
    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

    domain_counts = queries_df["domain"].value_counts()
    corpus_domain_counts = corpus["domain"].value_counts()
    print(f"  Domain filter stats:")
    for d, n in domain_counts.items():
        c = corpus_domain_counts.get(d, 0)
        print(f"    {d:<28}: {n} queries, {c} corpus docs")

    print(f"\nRetrieving with domain-filtered dense (model={args.model}) ...")
    submission = retrieve_domain_filtered(
        query_embs, query_ids, corpus_embs, corpus_ids_list,
        corpus_domains_list, query_domains,
        top_k=args.top_k, min_domain_size=args.min_domain_size,
    )

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(
            results,
            RESULTS_DIR / "domain_dense.csv",
            hyperparameters={"model": args.model, "split": args.split,
                             "min_domain_size": args.min_domain_size},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
