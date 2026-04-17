"""
Extract and embed citation context sentences from query papers.

Citation contexts are sentences that contain citation markers ([1], (Author et al., 2020), etc.).
The markers themselves are stripped, leaving pure descriptive text about *why* the paper cites.

This gives a query representation that captures the concepts/findings the paper
builds on. We chose to implement this because it is much closer to citation relevance than title+abstract alone.

Two outputs per query:
  - Individual chunk embeddings (one per citation sentence)  → for max-pooling retrieval
  - A single concatenated embedding (all contexts joined)    → for direct retrieval

Usage:
    python scripts/embed_cite_contexts.py --model BAAI/bge-large-en-v1.5 --split train
    python scripts/embed_cite_contexts.py --model BAAI/bge-large-en-v1.5 --split held_out
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}

# Citation marker patterns
BRACKET_PAT = re.compile(r"\[\d[\d,\s\-–;]*\]")
PAREN_CITE_PAT = re.compile(
    r"\("
    r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?"
    r"\s*(?:et\s+al\.?)?"
    r"\s*,?\s*\d{4}"
    r"[^)]*\)"
)
# Author et al. (Year) or Author (Year) — inline style
INLINE_CITE_PAT = re.compile(
    r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?"
    r"\s*(?:et\s+al\.?)?"
    r"\s*\(\d{4}[^)]*\)"
)
# Vancouver numeric style: (1), (2,3), (1;2;3)
# Guards:
#   (?<![a-zA-Z(])  — not preceded by letter/open-paren: avoids f(1), equation(1)
#   (?<!: )(?<!; )  — not preceded by colon/semicolon+space: avoids enumerated lists (1) First...
#   lookahead includes ( so consecutive (15)(16)(17) all get stripped, not just the last
NUMERIC_PAREN_PAT = re.compile(
    r"(?<![a-zA-Z(])(?<!: )(?<!; )"
    r"\(\d{1,3}(?:[,;]\s*\d{1,3})*\)"
    r"(?=[\s,\.;)(]|$)"
)
# Alphanumeric bracket style: [Kar99], [HS18], [KLM89]
ALPHA_BRACKET_PAT = re.compile(r"\[[A-Z][A-Za-z]{0,5}\d{2,4}(?:,\s*[A-Z][A-Za-z]{0,5}\d{2,4})*\]")

ALL_CITE_PATS = [BRACKET_PAT, PAREN_CITE_PAT, INLINE_CITE_PAT, NUMERIC_PAREN_PAT, ALPHA_BRACKET_PAT]

# Model-specific query prefixes
MODEL_PREFIXES = {
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-large-v2": "query: ",
    "intfloat/e5-small-v2": "query: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def extract_citation_sentences(full_text: str, min_chars: int = 30) -> list[str]:
    """
    Extract sentences containing citation markers, with markers removed.

    Returns cleaned, deduplicated sentences that describe *why* the paper cites other work.

    Handles four citation styles:
      - Numeric bracket:      [1], [1,2], [1-3]
      - Author-year paren:    (Smith et al., 2020)
      - Author-year inline:   Smith et al. (2020)
      - Vancouver numeric:    (1), (2,3)          ← previously missed
      - Alphanumeric bracket: [Kar99], [HS18]     ← previously missed
    Consecutive markers on the same sentence (e.g. [1][2][3]) are deduplicated
    so the same sentence is not embedded multiple times.
    """
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    seen_cleaned = set()
    cite_sents = []

    for s in sentences:
        has_cite = any(pat.search(s) for pat in ALL_CITE_PATS)
        if not has_cite:
            continue

        # Remove all citation markers
        clean = s
        for pat in ALL_CITE_PATS:
            clean = pat.sub("", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if len(clean) < min_chars:
            continue

        # Deduplicate: skip if same cleaned text already added (consecutive markers)
        if clean in seen_cleaned:
            continue
        seen_cleaned.add(clean)
        cite_sents.append(clean)

    return cite_sents


def main():
    parser = argparse.ArgumentParser(description="Embed citation context sentences")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=30,
                        help="Minimum chars for a citation sentence to be kept")
    parser.add_argument("--max-sents", type=int, default=50,
                        help="Maximum citation sentences per query (longest first)")
    args = parser.parse_args()

    queries_path = QUERY_FILES[args.split]
    model_slug = args.model.replace("/", "_").replace("\\", "_")
    model_dir = EMBEDDINGS_DIR / model_slug
    queries_dir = model_dir / args.split
    os.makedirs(queries_dir, exist_ok=True)

    q_prefix = MODEL_PREFIXES.get(args.model, "")
    if q_prefix:
        print(f"Using query prefix: {q_prefix!r}")

    # Load queries
    print(f"Loading queries ({args.split}) ...")
    queries_df = pd.read_parquet(queries_path)
    print(f"  {len(queries_df)} queries")

    # Extract citation contexts
    print("Extracting citation context sentences ...")
    all_texts = []       # flat list of texts to embed
    all_query_ids = []   # parallel list of query IDs
    per_query_stats = []

    for _, row in queries_df.iterrows():
        qid = row["doc_id"]
        sents = extract_citation_sentences(row["full_text"], min_chars=args.min_chars)

        # Keep up to max_sents, preferring longer sentences (more context)
        sents.sort(key=len, reverse=True)
        sents = sents[: args.max_sents]

        per_query_stats.append((qid, len(sents)))

        for s in sents:
            all_texts.append(q_prefix + s)
            all_query_ids.append(qid)

    n_with = sum(1 for _, n in per_query_stats if n > 0)
    print(f"  Queries with citation contexts: {n_with}/{len(queries_df)}")
    print(f"  Total citation sentences: {len(all_texts)}")
    avg = len(all_texts) / max(n_with, 1)
    print(f"  Average per query (with contexts): {avg:.1f}")

    if not all_texts:
        print("No citation contexts found. Exiting.")
        return

    # Load model and encode
    print(f"\nLoading model: {args.model} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    print(f"  Device: {device}")

    print("Encoding citation contexts ...")
    embeddings = model.encode(
        all_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save individual chunk embeddings (for max-pool retrieval)
    np.save(queries_dir / "cite_context_embeddings.npy", embeddings)
    with open(queries_dir / "cite_context_query_ids.json", "w") as f:
        json.dump(all_query_ids, f)

    # Also compute and save a single mean-pooled embedding per query
    # (average of all citation context embeddings for that query)
    print("Computing mean-pooled citation context embeddings ...")
    query_ids_ordered = queries_df["doc_id"].tolist()
    qid_to_rows = {}
    for idx, qid in enumerate(all_query_ids):
        qid_to_rows.setdefault(qid, []).append(idx)

    pooled_embs = np.zeros((len(query_ids_ordered), embeddings.shape[1]), dtype=np.float32)
    pooled_query_ids = []
    has_context = []

    for i, qid in enumerate(query_ids_ordered):
        rows = qid_to_rows.get(qid, [])
        if rows:
            emb = embeddings[rows].mean(axis=0)
            emb /= np.linalg.norm(emb) + 1e-9  # re-normalize
            pooled_embs[i] = emb
            has_context.append(True)
        else:
            has_context.append(False)
        pooled_query_ids.append(qid)

    np.save(queries_dir / "cite_context_pooled_embeddings.npy", pooled_embs)
    with open(queries_dir / "cite_context_pooled_query_ids.json", "w") as f:
        json.dump(pooled_query_ids, f)
    with open(queries_dir / "cite_context_has_context.json", "w") as f:
        json.dump(has_context, f)

    print(f"\nSaved to {queries_dir}/:")
    print(f"  cite_context_embeddings.npy          ({embeddings.shape})")
    print(f"  cite_context_query_ids.json          ({len(all_query_ids)} entries)")
    print(f"  cite_context_pooled_embeddings.npy   ({pooled_embs.shape})")
    print(f"  cite_context_pooled_query_ids.json   ({len(pooled_query_ids)} entries)")
    print(f"  cite_context_has_context.json")
    print("Done.")


if __name__ == "__main__":
    main()
