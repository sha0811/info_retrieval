"""
Citation-context retrieval — use citation sentence embeddings as query expansion.

Combines three signals:
  1. TA similarity:      sim(query_ta, corpus_ta)
  2. Pooled cite context: sim(mean_cite_context, corpus_ta)
  3. Max cite context:    max over citation sentences of sim(cite_sent, corpus_ta)

Final score = alpha * ta_score
            + beta  * pooled_cite_score
            + gamma * max_cite_score

Queries without citation contexts fall back to TA-only retrieval.

Usage:
    python models/cite_context.py --model BAAI/bge-large-en-v1.5 --split train
    python models/cite_context.py --model BAAI/bge-large-en-v1.5 --split train --alpha 0.4 --beta 0.3 --gamma 0.3
    python models/cite_context.py --model BAAI/bge-large-en-v1.5 --split train --sweep
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, load_embeddings, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def retrieve(query_embs, corpus_embs, q_ids, c_ids,
             pooled_cite_embs, has_context,
             cite_chunk_embs, cite_chunk_query_ids,
             alpha=0.5, beta=0.3, gamma=0.2,
             top_k=100) -> dict:
    """
    Retrieve using TA + citation context signals.

    alpha: weight on TA similarity
    beta:  weight on pooled (mean) citation context similarity
    gamma: weight on max-pooled individual citation sentence similarity
    """
    # TA similarity
    ta_sim = query_embs @ corpus_embs.T  # (n_queries, n_corpus)

    # Pooled citation context similarity
    pooled_sim = pooled_cite_embs @ corpus_embs.T  # (n_queries, n_corpus)

    # Build chunk index for max-pool
    qid_to_chunk_rows = defaultdict(list)
    for row_idx, qid in enumerate(cite_chunk_query_ids):
        qid_to_chunk_rows[qid].append(row_idx)

    submission = {}
    for i, qid in enumerate(q_ids):
        ta_scores = ta_sim[i]

        if has_context[i]:
            pooled_scores = pooled_sim[i]

            # Max-pool over individual citation sentences
            chunk_rows = qid_to_chunk_rows.get(qid, [])
            if chunk_rows and gamma > 0:
                chunk_sims = cite_chunk_embs[chunk_rows] @ corpus_embs.T
                max_chunk_scores = chunk_sims.max(axis=0)
            else:
                max_chunk_scores = pooled_scores

            final_scores = (alpha * ta_scores
                            + beta * pooled_scores
                            + gamma * max_chunk_scores)
        else:
            # No citation context — fall back to TA only
            final_scores = ta_scores

        top_indices = np.argsort(-final_scores)[:top_k]
        submission[qid] = [c_ids[j] for j in top_indices]

    return submission


def main():
    parser = argparse.ArgumentParser(description="Citation-context retrieval")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on TA similarity (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight on pooled citation context (default: 0.3)")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Weight on max citation sentence (default: 0.2)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Grid search over alpha/beta/gamma weights")
    args = parser.parse_args()

    QUERY_FILES = {
        "train": DATA_DIR / "queries.parquet",
        "held_out": DATA_DIR / "held_out_queries.parquet",
    }

    model_slug = args.model.replace("/", "_").replace("\\", "_")
    model_dir = EMBEDDINGS_DIR / model_slug
    queries_dir = model_dir / args.split

    # Load TA embeddings
    print("Loading TA embeddings ...")
    query_embs, q_ids = load_embeddings(
        queries_dir / "query_embeddings.npy", queries_dir / "query_ids.json"
    )
    corpus_embs, c_ids = load_embeddings(
        model_dir / "corpus_embeddings.npy", model_dir / "corpus_ids.json"
    )
    print(f"  Query TA:  {query_embs.shape}")
    print(f"  Corpus TA: {corpus_embs.shape}")

    # Load citation context embeddings
    print("Loading citation context embeddings ...")
    pooled_cite_embs = np.load(queries_dir / "cite_context_pooled_embeddings.npy").astype(np.float32)
    with open(queries_dir / "cite_context_pooled_query_ids.json") as f:
        pooled_query_ids = json.load(f)
    with open(queries_dir / "cite_context_has_context.json") as f:
        has_context = json.load(f)

    cite_chunk_embs = np.load(queries_dir / "cite_context_embeddings.npy").astype(np.float32)
    with open(queries_dir / "cite_context_query_ids.json") as f:
        cite_chunk_query_ids = json.load(f)

    n_with = sum(has_context)
    print(f"  Pooled cite:  {pooled_cite_embs.shape}")
    print(f"  Cite chunks:  {cite_chunk_embs.shape}")
    print(f"  Queries with context: {n_with}/{len(q_ids)}")

    # Load eval data
    queries = load_queries(QUERY_FILES[args.split])
    qrels = load_qrels(args.qrels)
    query_domains = dict(zip(queries["doc_id"], queries["domain"]))

    if args.sweep:
        # Grid search
        print("\n=== GRID SEARCH ===")
        best_ndcg = 0
        best_params = None

        alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        betas = [0.0, 0.1, 0.2, 0.3, 0.4]
        gammas = [0.0, 0.1, 0.2, 0.3, 0.4]

        for a, b, g in itertools.product(alphas, betas, gammas):
            if abs(a + b + g - 1.0) > 0.01:
                continue

            sub = retrieve(query_embs, corpus_embs, q_ids, c_ids,
                           pooled_cite_embs, has_context,
                           cite_chunk_embs, cite_chunk_query_ids,
                           alpha=a, beta=b, gamma=g, top_k=args.top_k)

            results = evaluate(sub, qrels, verbose=False)
            ndcg = results["overall"]["NDCG@10"]
            map_score = results["overall"]["MAP"]

            marker = " ***" if ndcg > best_ndcg else ""
            print(f"  a={a:.1f} b={b:.1f} g={g:.1f}  NDCG@10={ndcg:.4f}  MAP={map_score:.4f}{marker}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_params = (a, b, g)

        print(f"\nBest: alpha={best_params[0]:.1f} beta={best_params[1]:.1f} "
              f"gamma={best_params[2]:.1f}  NDCG@10={best_ndcg:.4f}")
        return

    # Single run
    output_path = args.output or SUBMISSIONS_DIR / f"cite_context_{model_slug}_{args.split}.json"

    print(f"\nRetrieving with alpha={args.alpha} beta={args.beta} gamma={args.gamma} ...")
    submission = retrieve(query_embs, corpus_embs, q_ids, c_ids,
                          pooled_cite_embs, has_context,
                          cite_chunk_embs, cite_chunk_query_ids,
                          alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                          top_k=args.top_k)

    if not args.no_eval:
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(results, RESULTS_DIR / "cite_context.csv",
                     hyperparameters={"model": args.model, "alpha": args.alpha,
                                      "beta": args.beta, "gamma": args.gamma,
                                      "split": args.split})

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
