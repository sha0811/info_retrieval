"""
Fine-tune a cross-encoder on train qrels for citation prediction.

Takes a first-stage retrieval submission (e.g. your best RRF top-100) as the
candidate pool. For each query, positives are cited papers; hard negatives are
top-ranked retrieved papers that are NOT cited.

Training uses BinaryCrossEntropyLoss with in-batch positives and hard negatives.

Usage:
    python scripts/finetune_crossencoder.py
    python scripts/finetune_crossencoder.py --candidates submissions/rrf_cite_fullchunk_ftsmall_train.json
    python scripts/finetune_crossencoder.py --epochs 3 --n-negatives 10
    python scripts/finetune_crossencoder.py --base-model cross-encoder/ms-marco-MiniLM-L-12-v2
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"


def build_text(row) -> str:
    title    = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


def build_training_samples(queries_df, corpus_df, qrels, candidates, n_negatives):
    """
    Build (query_text, doc_text, label) samples.
    label=1 for cited docs, label=0 for hard negatives.
    """
    corpus_texts = {r["doc_id"]: build_text(r) for _, r in corpus_df.iterrows()}
    query_texts  = {r["doc_id"]: build_text(r) for _, r in queries_df.iterrows()}

    samples = []
    for qid, relevant_ids in qrels.items():
        if qid not in query_texts:
            continue
        q_text      = query_texts[qid]
        relevant_set = set(relevant_ids)

        # Hard negatives: top-ranked non-relevant docs from the candidate pool
        retrieved  = candidates.get(qid, [])
        hard_negs  = [d for d in retrieved if d not in relevant_set and d in corpus_texts]
        sampled_negs = random.sample(hard_negs, min(n_negatives, len(hard_negs)))

        for pos_id in relevant_ids:
            if pos_id not in corpus_texts:
                continue
            samples.append(InputExample(texts=[q_text, corpus_texts[pos_id]], label=1.0))

        for neg_id in sampled_negs:
            samples.append(InputExample(texts=[q_text, corpus_texts[neg_id]], label=0.0))

    return samples


def build_eval_samples(queries_df, corpus_df, qrels, candidates):
    """Build evaluation samples (all positives + all hard negatives)."""
    corpus_texts = {r["doc_id"]: build_text(r) for _, r in corpus_df.iterrows()}
    query_texts  = {r["doc_id"]: build_text(r) for _, r in queries_df.iterrows()}

    samples = []
    for qid, relevant_ids in qrels.items():
        if qid not in query_texts:
            continue
        q_text       = query_texts[qid]
        relevant_set = set(relevant_ids)
        retrieved    = candidates.get(qid, [])

        for doc_id in retrieved[:50]:   # evaluate on top-50 candidates only
            if doc_id not in corpus_texts:
                continue
            label = 1.0 if doc_id in relevant_set else 0.0
            samples.append(InputExample(texts=[q_text, corpus_texts[doc_id]], label=label))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder for citation reranking")
    parser.add_argument("--base-model",   default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="HuggingFace cross-encoder to fine-tune")
    parser.add_argument("--candidates",   default=None,
                        help="Submission JSON used as the candidate pool (default: best available)")
    parser.add_argument("--output",       default=None,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--n-negatives",  type=int, default=10,
                        help="Hard negatives per positive (default: 10)")
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--queries",      default=DATA_DIR / "queries.parquet")
    parser.add_argument("--corpus",       default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels",        default=DATA_DIR / "qrels.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Default candidate pool: best RRF submission available
    if args.candidates is None:
        for candidate in [
            SUBMISSIONS_DIR / "rrf_cite_fullchunk_ftsmall_train.json",
            SUBMISSIONS_DIR / "rrf_finetuned_small_bge_large_e5.json",
            SUBMISSIONS_DIR / "dense_bge_large.json",
        ]:
            if Path(candidate).exists():
                args.candidates = candidate
                break

    output_dir = args.output or str(
        DATA_DIR / "finetuned_models" /
        ("crossencoder_" + args.base_model.split("/")[-1])
    )

    # Load data
    print("Loading data ...")
    queries_df = pd.read_parquet(args.queries)
    corpus_df  = pd.read_parquet(args.corpus)
    with open(args.qrels) as f:
        qrels = json.load(f)
    with open(args.candidates) as f:
        candidates = json.load(f)

    print(f"  Queries:    {len(queries_df)}")
    print(f"  Corpus:     {len(corpus_df)}")
    print(f"  Qrels:      {sum(len(v) for v in qrels.values())} positive pairs")
    print(f"  Candidates: {Path(args.candidates).name}")

    # Build training samples
    print(f"\nBuilding training samples (n_negatives={args.n_negatives}) ...")
    train_samples = build_training_samples(
        queries_df, corpus_df, qrels, candidates, args.n_negatives
    )
    random.shuffle(train_samples)
    n_pos = sum(1 for s in train_samples if s.label == 1.0)
    n_neg = sum(1 for s in train_samples if s.label == 0.0)
    print(f"  Positives: {n_pos}  |  Negatives: {n_neg}  |  Total: {len(train_samples)}")

    # Build eval samples (small subset, just to monitor training)
    eval_samples = build_eval_samples(queries_df, corpus_df, qrels, candidates)
    print(f"  Eval samples: {len(eval_samples)}")

    # Load model
    print(f"\nLoading cross-encoder: {args.base_model} ...")
    model = CrossEncoder(args.base_model, num_labels=1)

    # Training setup
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    total_steps  = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        eval_samples, name="train-eval"
    )

    print(f"\nTraining:")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR:           {args.lr}")
    print(f"  Output:       {output_dir}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True,
    )

    # Always save final model weights regardless of evaluator outcome
    model.save(output_dir)
    print(f"\nFine-tuned cross-encoder saved to: {output_dir}")
    print(f"\nNext step — rerank your best submission:")
    print(f"  python models/reranker.py \\")
    print(f"    --input submissions/rrf_cite_fullchunk_ftsmall_train.json \\")
    print(f"    --model {output_dir} \\")
    print(f"    --split train")


if __name__ == "__main__":
    main()
