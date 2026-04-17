"""
Diagnose retrieval failures — find where a submission struggles and why.

Prints:
  1. Per-query metrics sorted by NDCG@10 (worst first)
  2. For the N worst queries: retrieved vs relevant docs, missed docs, similarity scores
  3. Aggregate failure patterns: by domain, by n_relevant bucket, by similarity gap

Usage:
    python scripts/diagnose.py --submission submissions/dense_bge_large.json
    python scripts/diagnose.py --submission submissions/dense_bge_large.json --worst 20
    python scripts/diagnose.py --submission submissions/dense_bge_large.json --embeddings BAAI/bge-large-en-v1.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import (
    load_corpus,
    load_embeddings,
    load_qrels,
    load_queries,
    ndcg_at_k,
    recall_at_k,
    average_precision,
    mrr_at_k,
)

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


# ── Per-query metrics ────────────────────────────────────────

def compute_per_query(submission, qrels, ks=(10, 100)):
    rows = []
    for qid, rel_list in qrels.items():
        relevant = set(rel_list)
        ranked = submission.get(qid, [])
        row = {"qid": qid, "n_relevant": len(relevant)}
        for k in ks:
            row[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
            row[f"Recall@{k}"] = recall_at_k(ranked, relevant, k)
            row[f"MRR@{k}"] = mrr_at_k(ranked, relevant, k)
        row["AP"] = average_precision(ranked, relevant)

        # Rank of each relevant doc (None if not in top-100)
        rank_map = {doc: r + 1 for r, doc in enumerate(ranked)}
        found = {d: rank_map[d] for d in relevant if d in rank_map}
        missed = relevant - set(found)
        row["n_found_100"] = len(found)
        row["n_missed"] = len(missed)
        row["found_ranks"] = sorted(found.values())
        row["missed_ids"] = list(missed)
        row["ranked"] = ranked
        rows.append(row)
    return rows


# ── Similarity analysis for missed docs ──────────────────────

def load_similarity_data(model_slug, split="train"):
    """Load embeddings and return (query_embs, q_ids, corpus_embs, c_ids)."""
    model_dir = EMBEDDINGS_DIR / model_slug
    queries_dir = model_dir / split
    query_embs, q_ids = load_embeddings(
        queries_dir / "query_embeddings.npy", queries_dir / "query_ids.json"
    )
    corpus_embs, c_ids = load_embeddings(
        model_dir / "corpus_embeddings.npy", model_dir / "corpus_ids.json"
    )
    return query_embs, q_ids, corpus_embs, c_ids


def get_similarity_scores(query_emb, corpus_embs, c_ids, doc_ids):
    """Return {doc_id: cosine_similarity} for the given doc_ids."""
    cid_to_idx = {cid: i for i, cid in enumerate(c_ids)}
    scores = {}
    for did in doc_ids:
        idx = cid_to_idx.get(did)
        if idx is not None:
            scores[did] = float(query_emb @ corpus_embs[idx])
    return scores


# ── Printing ─────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_query_detail(row, queries_df, corpus_df, query_embs, q_ids, corpus_embs, c_ids, rank):
    qid = row["qid"]
    q_row = queries_df[queries_df["doc_id"] == qid].iloc[0]

    print(f"\n{'─' * 80}")
    print(f"  #{rank}  NDCG@10={row['NDCG@10']:.4f}  Recall@10={row['Recall@10']:.4f}  "
          f"AP={row['AP']:.4f}  MRR@10={row['MRR@10']:.4f}")
    print(f"  Domain: {q_row['domain']}  |  Year: {q_row['year']}  |  "
          f"Relevant: {row['n_relevant']}  |  Found in top-100: {row['n_found_100']}")
    print(f"  Title: {q_row['title'][:120]}")
    print(f"{'─' * 80}")

    ranked = row["ranked"]

    # Top-10 retrieved
    print(f"\n  Top-10 retrieved:")
    qrels_set = set()
    with open(DATA_DIR / "qrels.json") as f:
        qrels_set = set(json.load(f).get(qid, []))

    for r, doc_id in enumerate(ranked[:10], 1):
        is_rel = "✓" if doc_id in qrels_set else "✗"
        c_row = corpus_df[corpus_df["doc_id"] == doc_id]
        title = c_row.iloc[0]["title"][:80] if len(c_row) > 0 else "???"
        # Get similarity score if embeddings available
        score_str = ""
        if query_embs is not None:
            qi = q_ids.index(qid) if qid in q_ids else None
            if qi is not None:
                sim = get_similarity_scores(query_embs[qi], corpus_embs, c_ids, [doc_id])
                if doc_id in sim:
                    score_str = f"  sim={sim[doc_id]:.4f}"
        print(f"    {r:>3}. [{is_rel}] {title}{score_str}")

    # Missed relevant docs
    missed_ids = row["missed_ids"]
    if missed_ids:
        print(f"\n  Missed relevant docs ({len(missed_ids)} not in top-100):")
        # Compute similarities if possible
        missed_sims = {}
        if query_embs is not None:
            qi = q_ids.index(qid) if qid in q_ids else None
            if qi is not None:
                missed_sims = get_similarity_scores(
                    query_embs[qi], corpus_embs, c_ids, missed_ids
                )

        # Sort missed docs by similarity (highest first) to see near-misses
        missed_with_sim = [(did, missed_sims.get(did, -1)) for did in missed_ids]
        missed_with_sim.sort(key=lambda x: -x[1])

        for did, sim in missed_with_sim[:10]:
            c_row = corpus_df[corpus_df["doc_id"] == did]
            title = c_row.iloc[0]["title"][:70] if len(c_row) > 0 else "???"
            # Find actual rank in full corpus
            sim_str = f"sim={sim:.4f}" if sim >= 0 else "sim=N/A"
            print(f"         {title}  ({sim_str})")

        if len(missed_ids) > 10:
            print(f"         ... and {len(missed_ids) - 10} more")

    # Found relevant docs with their ranks
    found_ranks = row["found_ranks"]
    if found_ranks:
        ranks_str = ", ".join(str(r) for r in found_ranks[:15])
        if len(found_ranks) > 15:
            ranks_str += f", ... ({len(found_ranks)} total)"
        print(f"\n  Relevant doc ranks: [{ranks_str}]")


def print_domain_breakdown(per_query, queries_df):
    print_header("PER-DOMAIN BREAKDOWN")
    qid_to_domain = dict(zip(queries_df["doc_id"], queries_df["domain"]))

    domain_metrics = defaultdict(list)
    for row in per_query:
        domain = qid_to_domain.get(row["qid"], "Unknown")
        domain_metrics[domain].append(row)

    print(f"\n  {'Domain':<25} {'n':>3}  {'NDCG@10':>8}  {'Recall@10':>10}  "
          f"{'MAP':>6}  {'Miss%':>6}")
    print(f"  {'-' * 70}")

    domain_rows = []
    for domain, rows in domain_metrics.items():
        n = len(rows)
        ndcg = np.mean([r["NDCG@10"] for r in rows])
        recall = np.mean([r["Recall@10"] for r in rows])
        ap = np.mean([r["AP"] for r in rows])
        miss_pct = np.mean([r["n_missed"] / r["n_relevant"] for r in rows]) * 100
        domain_rows.append((domain, n, ndcg, recall, ap, miss_pct))

    for domain, n, ndcg, recall, ap, miss_pct in sorted(domain_rows, key=lambda x: x[2]):
        print(f"  {domain:<25} {n:>3}  {ndcg:>8.4f}  {recall:>10.4f}  "
              f"{ap:>6.4f}  {miss_pct:>5.1f}%")


def print_nrelevant_breakdown(per_query):
    print_header("BREAKDOWN BY NUMBER OF RELEVANT DOCS")

    buckets = {"2": [], "3-5": [], "6-10": [], "11-20": [], "21+": []}
    for row in per_query:
        n = row["n_relevant"]
        if n <= 2:
            buckets["2"].append(row)
        elif n <= 5:
            buckets["3-5"].append(row)
        elif n <= 10:
            buckets["6-10"].append(row)
        elif n <= 20:
            buckets["11-20"].append(row)
        else:
            buckets["21+"].append(row)

    print(f"\n  {'Bucket':<10} {'n':>3}  {'NDCG@10':>8}  {'Recall@10':>10}  "
          f"{'Recall@100':>11}  {'MAP':>6}")
    print(f"  {'-' * 60}")

    for bucket, rows in buckets.items():
        if not rows:
            continue
        n = len(rows)
        ndcg = np.mean([r["NDCG@10"] for r in rows])
        recall10 = np.mean([r["Recall@10"] for r in rows])
        recall100 = np.mean([r["Recall@100"] for r in rows])
        ap = np.mean([r["AP"] for r in rows])
        print(f"  {bucket:<10} {n:>3}  {ndcg:>8.4f}  {recall10:>10.4f}  "
              f"{recall100:>11.4f}  {ap:>6.4f}")


def print_similarity_gap_analysis(per_query, query_embs, q_ids, corpus_embs, c_ids):
    """Analyze the similarity gap between retrieved and missed relevant docs."""
    if query_embs is None:
        return

    print_header("SIMILARITY GAP ANALYSIS")
    print("\n  For each query: compare similarity of found vs missed relevant docs.\n")

    gaps = []
    for row in per_query:
        qid = row["qid"]
        qi = q_ids.index(qid) if qid in q_ids else None
        if qi is None:
            continue

        ranked = row["ranked"]
        missed_ids = row["missed_ids"]
        if not missed_ids:
            continue

        # Sim of the 100th ranked doc (the threshold)
        threshold_sims = get_similarity_scores(
            query_embs[qi], corpus_embs, c_ids, [ranked[-1]] if ranked else []
        )
        threshold = list(threshold_sims.values())[0] if threshold_sims else 0

        # Sim of missed relevant docs
        missed_sims = get_similarity_scores(
            query_embs[qi], corpus_embs, c_ids, missed_ids
        )
        if not missed_sims:
            continue

        max_missed = max(missed_sims.values())
        mean_missed = np.mean(list(missed_sims.values()))

        # How many missed docs have sim > some top-100 docs?
        near_misses = sum(1 for s in missed_sims.values() if s > threshold * 0.95)

        gaps.append({
            "qid": qid,
            "n_missed": len(missed_ids),
            "threshold_sim": threshold,
            "max_missed_sim": max_missed,
            "mean_missed_sim": mean_missed,
            "gap": threshold - max_missed,
            "near_misses": near_misses,
        })

    if not gaps:
        print("  No missed docs to analyze.")
        return

    gaps.sort(key=lambda x: x["gap"])  # smallest gap first = closest near-misses

    print(f"  {'Query':<44} {'Miss':>4} {'Thresh':>7} {'Best miss':>10} "
          f"{'Gap':>6} {'Near':>5}")
    print(f"  {'-' * 80}")
    for g in gaps[:20]:
        print(f"  {g['qid'][:42]:<44} {g['n_missed']:>4} {g['threshold_sim']:>7.4f} "
              f"{g['max_missed_sim']:>10.4f} {g['gap']:>6.4f} {g['near_misses']:>5}")

    # Summary stats
    all_gaps = [g["gap"] for g in gaps]
    all_near = [g["near_misses"] for g in gaps]
    print(f"\n  Gap (threshold - best missed sim):  "
          f"mean={np.mean(all_gaps):.4f}  median={np.median(all_gaps):.4f}  "
          f"min={min(all_gaps):.4f}  max={max(all_gaps):.4f}")
    print(f"  Queries with near-misses (>95% of threshold): "
          f"{sum(1 for n in all_near if n > 0)}/{len(gaps)}")
    total_near = sum(all_near)
    print(f"  Total near-miss relevant docs: {total_near}")


def print_overall_summary(per_query):
    print_header("OVERALL SUMMARY")
    n = len(per_query)
    metrics = {
        "NDCG@10": np.mean([r["NDCG@10"] for r in per_query]),
        "NDCG@100": np.mean([r["NDCG@100"] for r in per_query]),
        "Recall@10": np.mean([r["Recall@10"] for r in per_query]),
        "Recall@100": np.mean([r["Recall@100"] for r in per_query]),
        "MAP": np.mean([r["AP"] for r in per_query]),
        "MRR@10": np.mean([r["MRR@10"] for r in per_query]),
    }
    print(f"\n  Queries: {n}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    # Distribution of NDCG@10
    ndcgs = sorted([r["NDCG@10"] for r in per_query])
    zero_queries = sum(1 for x in ndcgs if x == 0)
    low_queries = sum(1 for x in ndcgs if 0 < x <= 0.3)
    mid_queries = sum(1 for x in ndcgs if 0.3 < x <= 0.6)
    high_queries = sum(1 for x in ndcgs if 0.6 < x <= 0.8)
    great_queries = sum(1 for x in ndcgs if x > 0.8)
    print(f"\n  NDCG@10 distribution:")
    print(f"    = 0.0        : {zero_queries:>3} queries")
    print(f"    (0.0, 0.3]   : {low_queries:>3} queries")
    print(f"    (0.3, 0.6]   : {mid_queries:>3} queries")
    print(f"    (0.6, 0.8]   : {high_queries:>3} queries")
    print(f"    (0.8, 1.0]   : {great_queries:>3} queries")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose retrieval failures")
    parser.add_argument("--submission", required=True,
                        help="Path to submission JSON")
    parser.add_argument("--worst", type=int, default=10,
                        help="Number of worst queries to show in detail (default: 10)")
    parser.add_argument("--embeddings", default=None,
                        help="Model slug for similarity analysis (e.g. BAAI/bge-large-en-v1.5). "
                             "If omitted, similarity scores are not shown.")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--corpus", default=DATA_DIR / "corpus.parquet")
    args = parser.parse_args()

    QUERY_FILES = {
        "train": DATA_DIR / "queries.parquet",
        "held_out": DATA_DIR / "held_out_queries.parquet",
    }

    queries_path = args.queries or QUERY_FILES[args.split]

    # Load data
    print("Loading data ...")
    with open(args.submission) as f:
        submission = json.load(f)
    qrels = load_qrels(args.qrels)
    queries_df = load_queries(queries_path)
    corpus_df = load_corpus(args.corpus)

    # Load embeddings if requested
    query_embs, q_ids, corpus_embs, c_ids = None, None, None, None
    if args.embeddings:
        model_slug = args.embeddings.replace("/", "_").replace("\\", "_")
        print(f"Loading embeddings ({model_slug}) ...")
        query_embs, q_ids, corpus_embs, c_ids = load_similarity_data(model_slug, args.split)
        print(f"  Query: {query_embs.shape}  Corpus: {corpus_embs.shape}")

    # Compute per-query metrics
    print("Computing per-query metrics ...")
    per_query = compute_per_query(submission, qrels)

    # Add domain info
    qid_to_domain = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    for row in per_query:
        row["domain"] = qid_to_domain.get(row["qid"], "Unknown")

    # Sort by NDCG@10 ascending (worst first)
    per_query.sort(key=lambda x: x["NDCG@10"])

    # ── Print reports ──────────────────────────────

    print_overall_summary(per_query)
    print_domain_breakdown(per_query, queries_df)
    print_nrelevant_breakdown(per_query)

    if query_embs is not None:
        print_similarity_gap_analysis(per_query, query_embs, q_ids, corpus_embs, c_ids)

    # Detailed worst queries
    print_header(f"WORST {args.worst} QUERIES (by NDCG@10)")
    for rank, row in enumerate(per_query[:args.worst], 1):
        print_query_detail(
            row, queries_df, corpus_df,
            query_embs, q_ids, corpus_embs, c_ids,
            rank,
        )

    # Full per-query table
    print_header("ALL QUERIES (sorted by NDCG@10)")
    print(f"\n  {'#':>3}  {'NDCG@10':>8}  {'R@10':>6}  {'R@100':>6}  "
          f"{'AP':>6}  {'Rel':>4}  {'Miss':>4}  {'Domain':<20}  Title")
    print(f"  {'-' * 110}")
    for i, row in enumerate(per_query, 1):
        q_row = queries_df[queries_df["doc_id"] == row["qid"]]
        title = q_row.iloc[0]["title"][:45] if len(q_row) > 0 else "???"
        print(f"  {i:>3}  {row['NDCG@10']:>8.4f}  {row['Recall@10']:>6.4f}  "
              f"{row['Recall@100']:>6.4f}  {row['AP']:>6.4f}  {row['n_relevant']:>4}  "
              f"{row['n_missed']:>4}  {row['domain']:<20}  {title}")


if __name__ == "__main__":
    main()
