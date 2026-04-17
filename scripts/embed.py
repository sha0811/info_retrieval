"""
embed.py — Encode queries and corpus with a sentence-transformers model.

Query embeddings are stored in a split-specific subfolder to avoid overwriting:
  <output>/<model_slug>/train/query_embeddings.npy   (--split train)
  <output>/<model_slug>/held_out/query_embeddings.npy (--split held_out)
  <output>/<model_slug>/corpus_embeddings.npy         (shared)

Supports SPECTER2 adapter models (requires: pip install adapters):
  python embed.py --model allenai/specter2 --split train --batch-size 16
  python embed.py --model allenai/specter2_adhoc_query --split train --batch-size 16

Usage:
  python embed.py                            # train queries + corpus
  python embed.py --split held_out           # held-out queries only (corpus skipped if already exists)
  python embed.py --model BAAI/bge-base-en-v1.5 --split train --batch-size 32
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
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

# SPECTER2 adapter models and their corresponding base model
SPECTER2_ADAPTERS = {
    "allenai/specter2":             "allenai/specter2_base",
    "allenai/specter2_adhoc_query": "allenai/specter2_base",
}

# Model-specific prefixes applied before encoding.
# query_prefix : prepended to every query text
# doc_prefix   : prepended to every corpus document text
MODEL_PREFIXES = {
    "intfloat/e5-base-v2":          {"query": "query: ",    "doc": "passage: "},
    "intfloat/e5-large-v2":         {"query": "query: ",    "doc": "passage: "},
    "intfloat/e5-small-v2":         {"query": "query: ",    "doc": "passage: "},
    "BAAI/bge-base-en-v1.5":        {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "BAAI/bge-large-en-v1.5":       {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "BAAI/bge-small-en-v1.5":       {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
}


def format_text(row: pd.Series, prefix: str = "") -> str:
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    text = (title + " " + abstract) if (title and abstract) else (title or abstract)
    return prefix + text


# SentenceTransformer encoding (standard models)

def encode(model: SentenceTransformer, texts: list, batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# SPECTER2 adapter encoding

def load_specter2(adapter_name: str):
    """Load a SPECTER2 adapter model. Requires the `adapters` package."""
    try:
        from adapters import AutoAdapterModel
    except ImportError:
        raise ImportError(
            "The `adapters` package is required for SPECTER2 adapter models.\n"
            "Install it with: pip install adapters"
        )
    from transformers import AutoTokenizer

    base_name = SPECTER2_ADAPTERS[adapter_name]
    print(f"  Loading base model: {base_name} …")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model     = AutoAdapterModel.from_pretrained(base_name)

    print(f"  Loading adapter: {adapter_name} …")
    model.load_adapter(adapter_name, source="hf", load_as="specter2", set_active=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Running on: {device}")

    return model, tokenizer


def encode_specter2(model, tokenizer, texts: list, batch_size: int) -> np.ndarray:
    """Encode texts with a SPECTER2 adapter model using mean pooling."""
    device = next(model.parameters()).device
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # mean pool over token dimension, mask padding
        token_embeddings = outputs.last_hidden_state
        attention_mask   = encoded["attention_mask"].unsqueeze(-1).float()
        sum_embeddings   = (token_embeddings * attention_mask).sum(dim=1)
        sum_mask         = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings       = sum_embeddings / sum_mask

        # L2 normalise
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{len(texts)}]")

    return np.vstack(all_embeddings).astype(np.float32)


# Main

def main():
    parser = argparse.ArgumentParser(description="Embed queries and corpus")
    parser.add_argument("--split", default="train", choices=["train", "held_out"],
                        help="Which query set to embed: 'train' (queries.parquet) or "
                             "'held_out' (held_out_queries.parquet)")
    parser.add_argument("--corpus",       default=DEFAULT_CORPUS)
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT,
                        help="Root embeddings directory (default: data/embeddings/)")
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--queries-only", action="store_true",
                        help="Only encode queries, skip corpus encoding")
    args = parser.parse_args()

    queries_path = QUERY_FILES[args.split]
    is_specter2  = args.model in SPECTER2_ADAPTERS

    # Look up model-specific prefixes (empty string = no prefix)
    prefixes    = MODEL_PREFIXES.get(args.model, {"query": "", "doc": ""})
    q_prefix    = prefixes["query"]
    doc_prefix  = prefixes["doc"]
    if q_prefix or doc_prefix:
        print(f"Using prefixes — query: {q_prefix!r}  doc: {doc_prefix!r}")

    # Build paths
    model_slug  = args.model.replace("/", "_").replace("\\", "_")
    model_dir   = os.path.join(args.output, model_slug)
    queries_dir = os.path.join(model_dir, args.split)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    # Load queries
    print(f"Loading queries ({args.split}) from {queries_path} …")
    queries_df  = pd.read_parquet(queries_path)
    print(f"  {len(queries_df)} queries")
    query_ids   = queries_df["doc_id"].tolist()
    query_texts = [format_text(row, prefix=q_prefix) for _, row in queries_df.iterrows()]

    # Load corpus
    corpus_emb_path      = os.path.join(model_dir, "corpus_embeddings.npy")
    corpus_ids_path      = os.path.join(model_dir, "corpus_ids.json")
    corpus_already_exists = os.path.isfile(corpus_emb_path)

    if not args.queries_only:
        print(f"Loading corpus from {args.corpus} …")
        corpus_df    = pd.read_parquet(args.corpus)
        print(f"  {len(corpus_df)} documents")
        corpus_ids   = corpus_df["doc_id"].tolist()
        corpus_texts = [format_text(row, prefix=doc_prefix) for _, row in corpus_df.iterrows()]

    # Load model
    print(f"\nLoading model: {args.model} …")
    if is_specter2:
        model, tokenizer = load_specter2(args.model)
        emb_dim = model.config.hidden_size
    else:
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        model    = SentenceTransformer(args.model, device=device)
        emb_dim  = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Device: {next(model.parameters()).device if not is_specter2 else device}")

    # Encode queries
    print(f"\nEncoding queries ({args.split}) …")
    if is_specter2:
        query_embs = encode_specter2(model, tokenizer, query_texts, args.batch_size)
    else:
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
        if is_specter2:
            corpus_embs = encode_specter2(model, tokenizer, corpus_texts, args.batch_size)
        else:
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
