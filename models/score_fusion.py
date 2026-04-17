"""
Score-fusion retrieval: ftsmall dense + domain boost + cite-context similarity.

Three signals combined directly in score space (not RRF):
  1. Finetuned bge-small cosine similarity  (base dense signal)
  2. +W_DOMAIN * (query_domain == corpus_domain)  (hard domain boost)
  3. +W_CITE * max_i(cosine(cite_sentence_i, corpus_doc))  (cite-context signal)

Why score-level rather than RRF:
  RRF equalizes signal contributions by rank; domain and cite signals have
  complementary absolute ranges, so direct score addition outperforms rank fusion.

Best weights found on train split:
  W_DOMAIN = 0.25
  W_CITE   = 1.0

Train NDCG@10: 0.7111  (previous best RRF: 0.6029)
Codabench:     0.69

Usage:
    python models/score_fusion.py --split train
    python models/score_fusion.py --split held_out
    python models/score_fusion.py --split held_out --output submissions/submission_data.json
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Embedding directories
EMB_FT = EMBEDDINGS_DIR / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE = EMBEDDINGS_DIR / "BAAI_bge-large-en-v1.5"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}

# Best weights (tuned on train split)
W_DOMAIN = 0.25
W_CITE = 1.0


def retrieve(
    ft_corpus_embs: np.ndarray,
    ft_corpus_ids: list,
    ft_query_embs: np.ndarray,
    ft_query_ids: list,
    cite_embs: np.ndarray,
    cite_qids: list,
    bge_corpus_embs: np.ndarray,
    bge_corpus_ids: list,
    query_domains: dict,
    corpus_domains: dict,
    w_domain: float = W_DOMAIN,
    w_cite: float = W_CITE,
    cite_top_k: int = 500,
    top_k: int = 100,
) -> dict:
    """
    For each query:
      score(doc) = cosine(q_ft, doc_ft)
                 + w_domain * (query_domain == doc_domain)
                 + w_cite * max_over_cite_sents(cosine(cite_sent_bge, doc_bge))

    cite_top_k: number of top candidates (by base score) on which cite signal is computed.
    """
    # Pre-compute domain masks aligned with ft_corpus_ids
    unique_domains = set(corpus_domains.values())
    domain_masks = {
        d: np.array([corpus_domains.get(cid, "") == d for cid in ft_corpus_ids], dtype=np.float32)
        for d in unique_domains
        if d
    }

    # Group cite embeddings by query id
    qid_to_cite_rows: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(cite_qids):
        qid_to_cite_rows[qid].append(idx)

    bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}

    submission = {}
    for qidx, qid in enumerate(ft_query_ids):
        q_emb = ft_query_embs[qidx]
        q_domain = query_domains.get(qid, "")
        cite_rows = qid_to_cite_rows.get(qid, [])

        # 1. Base dense score
        scores = ft_corpus_embs @ q_emb  # shape: (n_corpus,)

        # 2. Domain boost
        if w_domain > 0 and q_domain in domain_masks:
            scores = scores + w_domain * domain_masks[q_domain]

        # 3. Cite-context signal (max cosine over all citation sentences)
        if w_cite > 0 and cite_rows:
            q_cite_embs = cite_embs[cite_rows]  # (n_sents, dim)
            # Only score top candidates for efficiency
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bge_idx = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bge_idx is not None:
                    cite_sim = float((q_cite_embs @ bge_corpus_embs[bge_idx]).max())
                    scores[cidx] += w_cite * cite_sim

        top_idx = np.argsort(-scores)[:top_k]
        submission[qid] = [ft_corpus_ids[i] for i in top_idx]

    return submission


def main():
    parser = argparse.ArgumentParser(description="Score-fusion retrieval")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--w-domain", type=float, default=W_DOMAIN)
    parser.add_argument("--w-cite", type=float, default=W_CITE)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--cite-top-k", type=int, default=500,
                        help="Candidates on which cite signal is computed (for speed)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: submissions/score_fusion_<split>.json)")
    parser.add_argument("--zip", action="store_true",
                        help="Also produce a .zip with submission_data.json inside")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else SUBMISSIONS_DIR / f"score_fusion_{args.split}.json"

    # ── Load corpus ──────────────────────────────────────────────────────────
    print("Loading corpus ...")
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_domains = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

    # ── Load ft-small embeddings ─────────────────────────────────────────────
    print("Loading finetuned bge-small embeddings ...")
    ft_corpus_embs = np.load(EMB_FT / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_FT / "corpus_ids.json") as f:
        ft_corpus_ids = json.load(f)
    ft_query_embs = np.load(EMB_FT / f"{args.split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_FT / f"{args.split}/query_ids.json") as f:
        ft_query_ids = json.load(f)
    print(f"  Corpus: {ft_corpus_embs.shape}, Queries: {ft_query_embs.shape}")

    # ── Load bge-large cite-context embeddings ───────────────────────────────
    print("Loading bge-large cite-context embeddings ...")
    cite_embs = np.load(EMB_BGE / f"{args.split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{args.split}/cite_context_query_ids.json") as f:
        cite_qids = json.load(f)
    bge_corpus_embs = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_ids.json") as f:
        bge_corpus_ids = json.load(f)
    print(f"  Cite-context: {cite_embs.shape} ({len(set(cite_qids))} queries with contexts)")
    print(f"  BGE corpus:   {bge_corpus_embs.shape}")

    # ── Load query metadata ──────────────────────────────────────────────────
    queries_df = load_queries(QUERY_FILES[args.split])
    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

    # ── Retrieve ─────────────────────────────────────────────────────────────
    print(f"\nRetrieving (w_domain={args.w_domain}, w_cite={args.w_cite}) ...")
    submission = retrieve(
        ft_corpus_embs, ft_corpus_ids,
        ft_query_embs, ft_query_ids,
        cite_embs, cite_qids,
        bge_corpus_embs, bge_corpus_ids,
        query_domains, corpus_domains,
        w_domain=args.w_domain,
        w_cite=args.w_cite,
        cite_top_k=args.cite_top_k,
        top_k=args.top_k,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(
            results,
            RESULTS_DIR / "score_fusion.csv",
            hyperparameters={"w_domain": args.w_domain, "w_cite": args.w_cite, "split": args.split},
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {output_path}")

    if args.zip:
        zip_path = output_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(output_path, "submission_data.json")
        print(f"Zipped -> {zip_path}")


if __name__ == "__main__":
    main()
