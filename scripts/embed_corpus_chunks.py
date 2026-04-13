"""
embed_corpus_chunks.py — Encode corpus body chunks with a sentence-transformers model.

For each corpus document, extracts body sections from full_text using chunk_meta,
then encodes each chunk independently.

Saves:
  <output>/<model_slug>/corpus_chunk_embeddings.npy   — (total_chunks, emb_dim)
  <output>/<model_slug>/corpus_chunk_doc_ids.json      — list of doc_id per chunk row

Usage:
  python scripts/embed_corpus_chunks.py --model BAAI/bge-large-en-v1.5
  python scripts/embed_corpus_chunks.py --model BAAI/bge-large-en-v1.5 --batch-size 64
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

DEFAULT_CORPUS = os.path.join(DATA_DIR, "corpus.parquet")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "embeddings")
DEFAULT_MODEL  = "BAAI/bge-large-en-v1.5"

# Corpus docs use the doc prefix (empty for BGE, "passage: " for E5)
MODEL_PREFIXES = {
    "intfloat/e5-base-v2":          {"doc": "passage: "},
    "intfloat/e5-large-v2":         {"doc": "passage: "},
    "intfloat/e5-small-v2":         {"doc": "passage: "},
    "BAAI/bge-base-en-v1.5":        {"doc": ""},
    "BAAI/bge-large-en-v1.5":       {"doc": ""},
    "BAAI/bge-small-en-v1.5":       {"doc": ""},
}


def main():
    parser = argparse.ArgumentParser(description="Embed corpus body chunks")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Minimum chunk length in characters (default: 100)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=10,
                        help="Max chunks to keep per document (default: 10, longest first)")
    args = parser.parse_args()

    doc_prefix = MODEL_PREFIXES.get(args.model, {}).get("doc", "")
    if doc_prefix:
        print(f"Using doc prefix: {doc_prefix!r}")

    # Build output paths
    model_slug = args.model.replace("/", "_").replace("\\", "_")
    model_dir  = os.path.join(args.output, model_slug)
    os.makedirs(model_dir, exist_ok=True)

    # Load corpus
    print(f"Loading corpus from {args.corpus} ...")
    corpus_df = pd.read_parquet(args.corpus)
    print(f"  {len(corpus_df)} documents")

    # Extract chunks
    print(f"Extracting body chunks (min_chars={args.min_chars}, max_per_doc={args.max_chunks_per_doc}) ...")
    chunk_texts = []
    chunk_doc_ids = []
    docs_with_chunks = 0

    for _, row in corpus_df.iterrows():
        doc_id = row["doc_id"]
        body_chunks = get_body_chunks(row, min_chars=args.min_chars)

        if not body_chunks:
            continue

        # Keep the longest chunks (most content) up to max_chunks_per_doc
        body_chunks.sort(key=len, reverse=True)
        body_chunks = body_chunks[:args.max_chunks_per_doc]

        docs_with_chunks += 1
        for text in body_chunks:
            chunk_texts.append(doc_prefix + text)
            chunk_doc_ids.append(doc_id)

    print(f"  {len(chunk_texts)} total chunks from {docs_with_chunks}/{len(corpus_df)} documents")
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
    print(f"  corpus_chunk_embeddings shape: {chunk_embs.shape}")

    # Save
    np.save(os.path.join(model_dir, "corpus_chunk_embeddings.npy"), chunk_embs)
    with open(os.path.join(model_dir, "corpus_chunk_doc_ids.json"), "w") as f:
        json.dump(chunk_doc_ids, f)

    print(f"\nSaved:")
    print(f"  {model_dir}/corpus_chunk_embeddings.npy")
    print(f"  {model_dir}/corpus_chunk_doc_ids.json")
    print("Done.")


if __name__ == "__main__":
    main()
