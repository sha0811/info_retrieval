"""
Cross-encoder reranking on hard-domain candidates.

Computes and caches CE scores for (query, candidate) pairs produced by the
hard-domain 7-signal retrieval. Caches to data/ce_cache/.

Primarily a cache-builder: the resulting .npz feeds the v7 reranker.
The sweep section is kept for the experimental log.

Usage:
    python scripts/crossencoder_rerank_v2.py --split train --model BAAI/bge-reranker-base
    python scripts/crossencoder_rerank_v2.py --split held_out --model BAAI/bge-reranker-base
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import load_qrels, load_corpus, load_queries
from models.hard_pipeline_with_cite import load_signals_plus_cite, _fast_eval
from scripts.compute_soft_scores import get_candidates

DATA_DIR = ROOT / "data"
CACHE    = DATA_DIR / "ce_cache"
CACHE.mkdir(exist_ok=True)
PROGRESS = Path(__file__).parent / "crossencoder_rerank_v2_progress.txt"


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _mm(x):
    x = np.asarray(x, np.float32)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo + 1e-9)


def rerank_with_ce(d, per_query_cands, ce_scores, w_ce=0.5, w_bc=0.22):
    corpus_ids = d["corpus_ids"]
    submission = {}
    for qid, cand_idx in per_query_cands.items():
        n_cand = len(cand_idx)
        rr = 1.0 / (np.arange(n_cand) + 1).astype(np.float32)
        ce = ce_scores.get(qid, np.zeros(n_cand, np.float32))
        ce = _mm(ce)
        rows_bc = d["q2bc"].get(qid, [])
        bc = np.zeros(n_cand, np.float32)
        if rows_bc:
            qc_bge = d["bge_cite"][rows_bc]
            bge_rows = np.array([d["bge_cid2i"].get(corpus_ids[ci], 0) for ci in cand_idx])
            sims = qc_bge @ d["bge_c"][bge_rows].T
            bc = sims.max(axis=0).astype(np.float32)
        bc = _mm(bc) if bc.max() > 0 else bc
        sc = rr + w_bc * bc + w_ce * ce
        reranked = np.argsort(-sc)[:100]
        submission[qid] = [corpus_ids[cand_idx[k]] for k in reranked]
    return submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "held_out"])
    p.add_argument("--model", default="BAAI/bge-reranker-base")
    p.add_argument("--cand-n", type=int, default=100)
    p.add_argument("--dom-k", type=int, default=80)
    args = p.parse_args()

    if PROGRESS.exists(): PROGRESS.unlink()

    qrels = load_qrels(DATA_DIR / "qrels.json") if args.split == "train" else None
    d = load_signals_plus_cite(args.split)

    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qf = DATA_DIR / ("queries.parquet" if args.split == "train" else "held_out_queries.parquet")
    queries = load_queries(qf)

    def fmt_text(row):
        t = str(row.get("title", "") or "").strip()
        a = str(row.get("abstract", "") or "").strip()
        return (t + " " + a).strip() if t or a else ""

    doc_texts = {r["doc_id"]: fmt_text(r) for _, r in corpus.iterrows()}
    query_texts = {r["doc_id"]: fmt_text(r) for _, r in queries.iterrows()}
    log(f"Loaded {len(doc_texts)} corpus texts, {len(query_texts)} query texts")

    per_query_cands = get_candidates(d, dom_k=args.dom_k, cand_n=args.cand_n)
    corpus_ids = d["corpus_ids"]
    per_query_cands_ids = {qid: [corpus_ids[i] for i in cand_idx]
                           for qid, cand_idx in per_query_cands.items()}

    cache_file = CACHE / f"ce_scores_{args.split}_{args.model.replace('/', '_')}_ta.npz"
    if cache_file.exists():
        log(f"Loading cached {cache_file}")
        ce_scores = np.load(cache_file, allow_pickle=True)["scores"].item()
    else:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading CE model {args.model} on {device}...")
        model = CrossEncoder(args.model, device=device, max_length=512)

        ce_scores = {}
        qids = list(per_query_cands_ids.keys())
        total = sum(len(v) for v in per_query_cands_ids.values())
        done = 0
        t0 = time.time()
        for qi, qid in enumerate(qids):
            ids = per_query_cands_ids[qid]
            qt = query_texts[qid]
            pairs = [[qt, doc_texts[did]] for did in ids]
            out = model.predict(pairs, batch_size=64, show_progress_bar=False,
                                activation_fn=None, convert_to_numpy=True)
            ce_scores[qid] = np.asarray(out, dtype=np.float32)
            done += len(pairs)
            if qi % 5 == 0 or qi == len(qids) - 1:
                elapsed = time.time() - t0
                pct = 100 * done / total
                eta = elapsed / max(done, 1) * (total - done)
                log(f"  PROGRESS {pct:.1f}%  q={qi+1}/{len(qids)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
        np.savez_compressed(cache_file, scores=ce_scores)
        log(f"Saved cache: {cache_file}")

    if qrels:
        sub0 = rerank_with_ce(d, per_query_cands, ce_scores, w_ce=0.0, w_bc=0.22)
        log(f"\nBaseline (w_ce=0, w_bc=0.22): {_fast_eval(sub0, qrels):.4f}")

        log("\n=== Sweep w_ce (w_bc=0.22) ===")
        best = 0.0; best_cfg = None
        for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            sub = rerank_with_ce(d, per_query_cands, ce_scores, w_ce=w, w_bc=0.22)
            sc = _fast_eval(sub, qrels)
            log(f"  w_ce={w}: {sc:.4f}")
            if sc > best: best, best_cfg = sc, w
        log(f"Best: {best:.4f} at w_ce={best_cfg}")


if __name__ == "__main__":
    main()
