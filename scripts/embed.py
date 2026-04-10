"""
embed.py — Encode queries and corpus with a sentence-transformers model.

Query embeddings are stored in a split-specific subfolder to avoid overwriting:
  <output>/<model_slug>/train/query_embeddings.npy   (--split train)
  <output>/<model_slug>/held_out/query_embeddings.npy (--split held_out)
  <output>/<model_slug>/corpus_embeddings.npy         (shared)

Usage:
  python embed.py                            # train queries + corpus
  python embed.py --split held_out           # held-out queries only (corpus skipped if already exists)
  python embed.py --model allenai/specter2_base --split held_out --batch-size 64
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR      = os.path.join(CHALLENGE_DIR, "data")

QUERY_FILES = {
    "train":    os.path.join(DATA_DIR, "queries.parquet"),
    "held_out": os.path.join(DATA_DIR, "held_out_queries.parquet"),
}

DEFAULT_CORPUS = os.path.join(DATA_DIR, "corpus.parquet")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "embeddings")
DEFAULT_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def format_text(row: pd.Series) -> str:
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


def encode(model: SentenceTransformer, texts: list, batch_size: int,
           desc: str = "Encoding") -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Embed queries and corpus")
    parser.add_argument("--split", default="train", choices=["train", "held_out"],
                        help="Which query set to embed: 'train' (queries.parquet) or "
                             "'held_out' (held_out_queries.parquet)")
    parser.add_argument("--corpus",     default=DEFAULT_CORPUS)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,
                        help="Root embeddings directory (default: data/embeddings/)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queries-only", action="store_true",
                        help="Only encode queries, skip corpus encoding")
    args = parser.parse_args()

    queries_path = QUERY_FILES[args.split]

    # Build paths:
    #   corpus + model info → data/embeddings/<model_slug>/
    #   query embeddings    → data/embeddings/<model_slug>/<split>/
    model_slug  = args.model.replace("/", "_").replace("\\", "_")
    model_dir   = os.path.join(args.output, model_slug)
    queries_dir = os.path.join(model_dir, args.split)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    # Load queries
    print(f"Loading queries ({args.split}) from {queries_path} …")
    queries_df = pd.read_parquet(queries_path)
    print(f"  {len(queries_df)} queries")
    query_ids   = queries_df["doc_id"].tolist()
    query_texts = [format_text(row) for _, row in queries_df.iterrows()]

    # Load corpus (needed even for queries-only to keep ids consistent)
    corpus_emb_path = os.path.join(model_dir, "corpus_embeddings.npy")
    corpus_ids_path = os.path.join(model_dir, "corpus_ids.json")
    corpus_already_exists = os.path.isfile(corpus_emb_path)

    if not args.queries_only:
        print(f"Loading corpus from {args.corpus} …")
        corpus_df  = pd.read_parquet(args.corpus)
        print(f"  {len(corpus_df)} documents")
        corpus_ids   = corpus_df["doc_id"].tolist()
        corpus_texts = [format_text(row) for _, row in corpus_df.iterrows()]

    # Load model
    print(f"\nLoading model: {args.model} …")
    model = SentenceTransformer(args.model)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Encode queries
    print(f"\nEncoding queries ({args.split}) …")
    query_embs = encode(model, query_texts, args.batch_size)
    print(f"  query_embeddings shape: {query_embs.shape}")
    np.save(os.path.join(queries_dir, "query_embeddings.npy"), query_embs)
    with open(os.path.join(queries_dir, "query_ids.json"), "w") as f:
        json.dump(query_ids, f)

    # Encode corpus
    if args.queries_only:
        if corpus_already_exists:
            print(f"\nSkipping corpus encoding (--queries-only, corpus already at {model_dir}/)")
        else:
            print(f"\nWarning: --queries-only set but no corpus embeddings found at {model_dir}/")
            print("  Run without --queries-only to encode the corpus.")
    else:
        print("\nEncoding corpus …")
        corpus_embs = encode(model, corpus_texts, args.batch_size)
        print(f"  corpus_embeddings shape: {corpus_embs.shape}")
        np.save(corpus_emb_path, corpus_embs)
        with open(corpus_ids_path, "w") as f:
            json.dump(corpus_ids, f)

    # Save model info
    with open(os.path.join(model_dir, "model_info.txt"), "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"embedding_dim: {query_embs.shape[1]}\n")
        f.write(f"normalize_embeddings: True\n")
        f.write(f"text_format: title + ' ' + abstract\n")

    print(f"\nSaved:")
    print(f"  {queries_dir}/query_embeddings.npy")
    print(f"  {queries_dir}/query_ids.json")
    if not args.queries_only:
        print(f"  {model_dir}/corpus_embeddings.npy")
        print(f"  {model_dir}/corpus_ids.json")
    print(f"  {model_dir}/model_info.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
