"""
embed_chunks.py — Encode query body chunks with a sentence-transformers model.

For each query paper, extracts body sections (intro, methods, related work, etc.)
from full_text using chunk_meta, then encodes each chunk independently.

Saves:
  <output>/<model_slug>/<split>/chunk_embeddings.npy   (total_chunks, emb_dim)
  <output>/<model_slug>/<split>/chunk_query_ids.json    list of query_id per chunk row

Usage:
  python scripts/embed_chunks.py --model BAAI/bge-large-en-v1.5 --split train
  python scripts/embed_chunks.py --model BAAI/bge-large-en-v1.5 --split held_out
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR      = os.path.join(CHALLENGE_DIR, "data")

sys.path.insert(0, CHALLENGE_DIR)
from helpers import get_body_chunks

QUERY_FILES = {
    "train":    os.path.join(DATA_DIR, "queries.parquet"),
    "held_out": os.path.join(DATA_DIR, "held_out_queries.parquet"),
}

DEFAULT_OUTPUT = os.path.join(DATA_DIR, "embeddings")
DEFAULT_MODEL  = "BAAI/bge-large-en-v1.5"

# Same prefix table as embed.py
MODEL_PREFIXES = {
    "intfloat/e5-base-v2":          {"query": "query: "},
    "intfloat/e5-large-v2":         {"query": "query: "},
    "intfloat/e5-small-v2":         {"query": "query: "},
    "BAAI/bge-base-en-v1.5":        {"query": "Represent this sentence for searching relevant passages: "},
    "BAAI/bge-large-en-v1.5":       {"query": "Represent this sentence for searching relevant passages: "},
    "BAAI/bge-small-en-v1.5":       {"query": "Represent this sentence for searching relevant passages: "},
}


def main():
    parser = argparse.ArgumentParser(description="Embed query body chunks")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Minimum chunk length in characters (default: 100)")
    args = parser.parse_args()

    queries_path = QUERY_FILES[args.split]
    q_prefix = MODEL_PREFIXES.get(args.model, {}).get("query", "")

    # Build output paths
    model_slug  = args.model.replace("/", "_").replace("\\", "_")
    queries_dir = os.path.join(args.output, model_slug, args.split)
    os.makedirs(queries_dir, exist_ok=True)

    # Load queries
    print(f"Loading queries ({args.split}) from {queries_path} ...")
    queries_df = pd.read_parquet(queries_path)
    print(f"  {len(queries_df)} queries")

    # Extract chunks
    print(f"Extracting body chunks (min_chars={args.min_chars}) ...")
    chunk_texts = []
    chunk_query_ids = []

    for _, row in queries_df.iterrows():
        qid = row["doc_id"]
        body_chunks = get_body_chunks(row, min_chars=args.min_chars)
        for text in body_chunks:
            chunk_texts.append(q_prefix + text)
            chunk_query_ids.append(qid)

    print(f"  {len(chunk_texts)} total chunks from {len(queries_df)} queries")
    n_queries_with_chunks = len(set(chunk_query_ids))
    print(f"  {n_queries_with_chunks} queries have at least one chunk")
    if chunk_texts:
        lens = [len(t) for t in chunk_texts]
        print(f"  Chunk lengths: mean={np.mean(lens):.0f}, median={np.median(lens):.0f}, "
              f"max={max(lens)}")

    if not chunk_texts:
        print("No chunks found — nothing to embed.")
        return

    # Load model
    print(f"\nLoading model: {args.model} ...")
    model = SentenceTransformer(args.model)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Encode
    print(f"\nEncoding {len(chunk_texts)} chunks ...")
    chunk_embs = model.encode(
        chunk_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  chunk_embeddings shape: {chunk_embs.shape}")

    # Save
    np.save(os.path.join(queries_dir, "chunk_embeddings.npy"), chunk_embs)
    with open(os.path.join(queries_dir, "chunk_query_ids.json"), "w") as f:
        json.dump(chunk_query_ids, f)

    print(f"\nSaved:")
    print(f"  {queries_dir}/chunk_embeddings.npy")
    print(f"  {queries_dir}/chunk_query_ids.json")
    print("Done.")


if __name__ == "__main__":
    main()
