"""
Fine-tune a bi-encoder on train qrels for citation prediction.

Uses hard negatives from an existing retrieval run (top-ranked but not cited docs)
and MultipleNegativesRankingLoss from sentence-transformers.

Supports iterative hard negative mining: after each iteration, the current model
re-embeds the corpus and queries, retrieves fresh top-100 candidates, and uses
those as harder negatives for the next iteration.

Usage:
  python scripts/finetune_biencoder.py
  python scripts/finetune_biencoder.py --base-model BAAI/bge-large-en-v1.5 --epochs 3
  python scripts/finetune_biencoder.py --hard-negatives submissions/dense_bge_large.json --n-negatives 10
  python scripts/finetune_biencoder.py --iterations 3 --epochs-per-iter 2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_BASE_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "finetuned_models" / "bge-large-finetuned"
DEFAULT_HARD_NEG = PROJECT_DIR / "submissions" / "dense_bge_large.json"

# Query prefix per model family (doc prefix is always empty for BGE)
MODEL_QUERY_PREFIXES = {
    "BAAI/bge-large-en-v1.5":  "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5":   "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5":  "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2":     "query: ",
    "intfloat/e5-large-v2":    "query: ",
}


def build_text(row) -> str:
    """Title + abstract as a single string."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


def build_training_examples(
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    qrels: dict,
    hard_neg_submission: dict,
    n_negatives: int = 5,
) -> list:
    """
    Build training examples as (anchor, positive, hard_negative) triplets.

    For each (query, positive_doc) pair, sample n_negatives hard negatives
    from the retrieval run (top-ranked docs that are NOT relevant).
    """
    corpus_texts = {row["doc_id"]: build_text(row) for _, row in corpus_df.iterrows()}
    query_texts  = {row["doc_id"]: build_text(row) for _, row in queries_df.iterrows()}

    examples = []
    skipped = 0

    for qid, relevant_ids in qrels.items():
        if qid not in query_texts:
            continue

        anchor_text  = query_texts[qid]
        relevant_set = set(relevant_ids)

        retrieved  = hard_neg_submission.get(qid, [])
        hard_negs  = [did for did in retrieved if did not in relevant_set and did in corpus_texts]

        for pos_id in relevant_ids:
            if pos_id not in corpus_texts:
                skipped += 1
                continue

            pos_text = corpus_texts[pos_id]

            if hard_negs:
                sampled = random.sample(hard_negs, min(n_negatives, len(hard_negs)))
                for neg_id in sampled:
                    examples.append(InputExample(
                        texts=[anchor_text, pos_text, corpus_texts[neg_id]]
                    ))
            else:
                examples.append(InputExample(texts=[anchor_text, pos_text]))

    if skipped:
        print(f"  Skipped {skipped} positive docs not found in corpus")

    return examples


def embed_texts(model: SentenceTransformer, texts: list, batch_size: int) -> np.ndarray:
    """Encode texts and return L2-normalised float32 embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def mine_hard_negatives(
    model: SentenceTransformer,
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    query_prefix: str,
    batch_size: int,
    top_k: int = 100,
) -> dict:
    """
    Re-embed queries and corpus with the current model, retrieve top-k,
    and return a submission dict {query_id: [doc_id, ...]}.
    """
    print("  Embedding queries ...")
    query_texts  = [query_prefix + build_text(row) for _, row in queries_df.iterrows()]
    query_ids    = queries_df["doc_id"].tolist()
    query_embs   = embed_texts(model, query_texts, batch_size)

    print("  Embedding corpus ...")
    corpus_texts = [build_text(row) for _, row in corpus_df.iterrows()]
    corpus_ids   = corpus_df["doc_id"].tolist()
    corpus_embs  = embed_texts(model, corpus_texts, batch_size)

    print("  Retrieving top candidates ...")
    sim_matrix   = query_embs @ corpus_embs.T          # (n_queries, n_corpus)
    top_indices  = np.argsort(-sim_matrix, axis=1)[:, :top_k]

    return {qid: [corpus_ids[j] for j in top_indices[i]]
            for i, qid in enumerate(query_ids)}


def train_one_iteration(
    model: SentenceTransformer,
    examples: list,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    output_dir: str,
):
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss       = losses.MultipleNegativesRankingLoss(model)

    total_steps  = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"  Batch size:    {batch_size}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Learning rate: {lr}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output_dir,
        show_progress_bar=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder for citation retrieval")
    parser.add_argument("--base-model",     default=DEFAULT_BASE_MODEL,
                        help="HuggingFace model to fine-tune (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--output",         default=None,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--hard-negatives", default=DEFAULT_HARD_NEG,
                        help="Path to initial submission JSON for hard negatives (iteration 0)")
    parser.add_argument("--n-negatives",    type=int, default=5,
                        help="Number of hard negatives per positive pair (default: 5)")
    # Iterative mining
    parser.add_argument("--iterations",     type=int, default=1,
                        help="Number of training iterations with hard-negative re-mining (default: 1, i.e. no re-mining)")
    parser.add_argument("--epochs-per-iter",type=int, default=None,
                        help="Epochs per iteration (default: --epochs / --iterations)")
    parser.add_argument("--epochs",         type=int, default=3,
                        help="Total epochs (used only when --epochs-per-iter is not set)")
    # Training hyper-params
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",   type=float, default=0.1)
    parser.add_argument("--embed-batch",    type=int, default=256,
                        help="Batch size for embedding during hard-negative mining (default: 256)")
    parser.add_argument("--seed",           type=int, default=42)
    # Data paths
    parser.add_argument("--queries", default=DATA_DIR / "queries.parquet")
    parser.add_argument("--corpus",  default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",   default=DATA_DIR / "qrels.json")
    args = parser.parse_args()

    random.seed(args.seed)

    epochs_per_iter = args.epochs_per_iter or max(1, args.epochs // args.iterations)

    output_dir = args.output or (PROJECT_DIR / "data" / "finetuned_models" /
                                  args.base_model.replace("/", "_").replace("\\", "_"))

    query_prefix = MODEL_QUERY_PREFIXES.get(args.base_model, "")
    if query_prefix:
        print(f"Using query prefix: {query_prefix!r}")

    # Load data
    print("Loading data ...")
    queries_df = pd.read_parquet(args.queries)
    corpus_df  = pd.read_parquet(args.corpus)

    with open(args.qrels) as f:
        qrels = json.load(f)

    with open(args.hard_negatives) as f:
        hard_neg_submission = json.load(f)

    print(f"  Queries: {len(queries_df)}")
    print(f"  Corpus:  {len(corpus_df)}")
    print(f"  Qrels:   {len(qrels)} queries, {sum(len(v) for v in qrels.values())} total relevant pairs")
    print(f"  Iterations: {args.iterations}  |  Epochs per iteration: {epochs_per_iter}")

    # Load model
    print(f"\nLoading base model: {args.base_model} ...")
    model = SentenceTransformer(args.base_model)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Iterative training loop
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1} / {args.iterations}")
        print(f"{'='*60}")

        iter_output = str(output_dir) if iteration == args.iterations - 1 \
                      else str(output_dir) + f"_iter{iteration + 1}"

        # Build training examples from current hard negatives
        print(f"\nBuilding training examples (n_negatives={args.n_negatives}) ...")
        examples = build_training_examples(
            queries_df, corpus_df, qrels, hard_neg_submission,
            n_negatives=args.n_negatives,
        )
        random.shuffle(examples)
        print(f"  Training examples: {len(examples)}")

        # Train
        print(f"\nTraining for {epochs_per_iter} epoch(s) ...")
        train_one_iteration(
            model, examples,
            epochs=epochs_per_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            output_dir=iter_output,
        )
        print(f"  Checkpoint saved → {iter_output}")

        # Re-mine hard negatives with current model (skip on last iteration)
        if iteration < args.iterations - 1:
            print(f"\nMining new hard negatives with updated model ...")
            # Reload from checkpoint to ensure we use the saved weights
            model = SentenceTransformer(iter_output)
            hard_neg_submission = mine_hard_negatives(
                model, queries_df, corpus_df,
                query_prefix=query_prefix,
                batch_size=args.embed_batch,
                top_k=100,
            )
            print(f"  Mined {len(hard_neg_submission)} query candidate lists")

    print(f"\nFine-tuned model saved to: {output_dir}")


if __name__ == "__main__":
    main()
