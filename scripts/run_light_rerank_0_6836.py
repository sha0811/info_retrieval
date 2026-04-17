"""
Reproducible lightweight reranker for the 0.6836 local setup.

What it does:
1) Loads a base top-100 submission.
2) Reranks each query's 100 candidates using cheap features:
   - base rank prior
   - title token overlap
   - same-domain boost
3) Optionally evaluates on local qrels (if query IDs match).
4) Saves reranked predictions JSON.

Default use cases:
- Local eval run:
    python scripts/run_light_rerank_0_6836.py
- Held-out/Codabench run:
    python scripts/run_light_rerank_0_6836.py \
      --queries held_out_queries.parquet \
      --base-submission submissions/submission_data.json \
      --output submissions/submission_data_light_rerank_0.6836style.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Best lightweight rerank weights found in search
W_TITLE = 0.2
W_ABSTRACT = 0.0
W_DOMAIN = 0.15
W_YEAR = 0.0


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def overlap_ratio(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / (len(query_tokens) + 1e-9)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def evaluate_ndcg10(submission: dict[str, list[str]], qrels: dict[str, list[str]]) -> float:
    return float(np.mean([ndcg_at_k(submission.get(qid, []), set(rel), 10) for qid, rel in qrels.items()]))


def build_maps(queries: pd.DataFrame, corpus: pd.DataFrame) -> tuple[dict, dict]:
    qmap = {}
    for _, row in queries.iterrows():
        qid = row["doc_id"]
        qmap[qid] = {
            "title_tokens": tokenize(str(row.get("title", "") or "")),
            "abs_tokens": tokenize(str(row.get("abstract", "") or "")),
            "domain": row.get("domain", ""),
            "year": int(row.get("year", 0) or 0),
        }

    cmap = {}
    for _, row in corpus.iterrows():
        cid = row["doc_id"]
        cmap[cid] = {
            "title_tokens": tokenize(str(row.get("title", "") or "")),
            "abs_tokens": tokenize(str(row.get("abstract", "") or "")),
            "domain": row.get("domain", ""),
            "year": int(row.get("year", 0) or 0),
        }
    return qmap, cmap


def rerank(
    base_submission: dict[str, list[str]],
    qmap: dict,
    cmap: dict,
) -> dict[str, list[str]]:
    out = {}
    for qid, docs in base_submission.items():
        q = qmap.get(qid)
        if q is None:
            out[qid] = docs
            continue

        scored = []
        for rank, cid in enumerate(docs):
            d = cmap.get(cid)
            if d is None:
                scored.append((1.0 / (rank + 1), cid))
                continue

            base_rank = 1.0 / (rank + 1)
            title_ov = overlap_ratio(q["title_tokens"], d["title_tokens"])
            abs_ov = overlap_ratio(q["abs_tokens"], d["abs_tokens"])
            same_domain = 1.0 if q["domain"] and q["domain"] == d["domain"] else 0.0

            yq, yd = q["year"], d["year"]
            year_bonus = 0.0 if not (yq and yd) else 1.0 / (1.0 + abs(yq - yd))

            score = (
                base_rank
                + W_TITLE * title_ov
                + W_ABSTRACT * abs_ov
                + W_DOMAIN * same_domain
                + W_YEAR * year_bonus
            )
            scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        out[qid] = [cid for _, cid in scored[:100]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--queries", type=Path, default=Path("data/queries.parquet"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.parquet"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path("submissions/specter2_score_fusion_domainboost.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/light_rerank_0.6836.json"),
    )
    args = parser.parse_args()

    root = args.root
    queries = pd.read_parquet(root / args.queries)
    corpus = pd.read_parquet(root / args.corpus)
    with open(root / args.base_submission) as f:
        base = json.load(f)

    qmap, cmap = build_maps(queries, corpus)
    reranked = rerank(base, qmap, cmap)

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reranked, f)

    print(f"Saved reranked output to: {out_path}")

    qrels_path = root / args.qrels
    if qrels_path.exists():
        with open(qrels_path) as f:
            qrels = json.load(f)
        if set(qrels.keys()).issubset(set(reranked.keys())):
            base_score = evaluate_ndcg10(base, qrels)
            new_score = evaluate_ndcg10(reranked, qrels)
            print(f"Base NDCG@10: {base_score:.4f}")
            print(f"New  NDCG@10: {new_score:.4f}")
        else:
            print("Skipped local eval: qrels/query IDs do not match this submission.")


if __name__ == "__main__":
    main()

