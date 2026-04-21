"""
Reranker v7 : the final 4-feature reranker (train NDCG@10 = 0.7603).

Given hard-domain candidates (top-100 from the 7-signal pipeline), re-score
each candidate as:

    score = rr^p  +  w_bc · mm(bge_cite_max)
                  +  w_ce · mm(bge_reranker_base)
                  +  w_soft · mm(soft_pipeline_score)

Best config (from a 4-D grid search on train):
    rr_pow=1.00, w_bc=0.44, w_ce=0.04, w_soft=0.25

Caches consumed (auto-built on the fly for held-out if missing):
    data/ce_cache/ce_scores_{split}_BAAI_bge-reranker-base_ta.npz
    data/soft_scores/soft_scores_{split}.npz

Usage:
    python models/reranker_v7.py --split train
    python models/reranker_v7.py --split train --grid      # re-run the 4-D grid search
    python models/reranker_v7.py --split held_out          # produces the submission JSON
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_corpus, load_queries, evaluate, save_results
from models.hard_pipeline_with_cite import load_signals_plus_cite, _fast_eval
from scripts.compute_soft_scores import compute_soft_scores, get_candidates

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"
CE_CACHE        = DATA_DIR / "ce_cache"
PROGRESS        = Path(__file__).parent / "reranker_v7_progress.txt"

# Best config found by the 4-D grid search
W_BC   = 0.44
W_CE   = 0.04
W_SOFT = 0.25
RR_POW = 1.0
DOM_K  = 80
CAND_N = 100


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _mm(x):
    x = np.asarray(x, np.float32)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo + 1e-9)


def rerank(d, per_query_cands, ce_scores, soft_per_q,
           w_bc=W_BC, w_ce=W_CE, w_soft=W_SOFT, rr_pow=RR_POW):
    corpus_ids = d["corpus_ids"]
    submission = {}
    for qid, cand_idx in per_query_cands.items():
        n_cand = len(cand_idx)
        rr = (1.0 / (np.arange(n_cand) + 1).astype(np.float32)) ** rr_pow

        ce = ce_scores.get(qid, np.zeros(n_cand, np.float32))
        ce = _mm(np.asarray(ce, np.float32)) if len(ce) and np.max(ce) != np.min(ce) else np.zeros(n_cand, np.float32)

        if qid in soft_per_q:
            soft = soft_per_q[qid][cand_idx]
            soft = _mm(soft) if soft.max() > soft.min() else np.zeros(n_cand, np.float32)
        else:
            soft = np.zeros(n_cand, np.float32)

        rows_bc = d["q2bc"].get(qid, [])
        bc = np.zeros(n_cand, np.float32)
        if rows_bc:
            qc_bge = d["bge_cite"][rows_bc]
            bge_rows = np.array([d["bge_cid2i"].get(corpus_ids[ci], 0) for ci in cand_idx])
            sims = qc_bge @ d["bge_c"][bge_rows].T
            bc = sims.max(axis=0).astype(np.float32)
        bc = _mm(bc) if bc.max() > 0 else bc

        sc = rr + w_bc * bc + w_ce * ce + w_soft * soft
        reranked = np.argsort(-sc)[:100]
        submission[qid] = [corpus_ids[cand_idx[k]] for k in reranked]
    return submission


def _build_ce_cache(split, per_query_cands, d, model_name="BAAI/bge-reranker-base"):
    """Compute + cache CE scores for a split (used when the cache is missing)."""
    CE_CACHE.mkdir(exist_ok=True)
    cache_file = CE_CACHE / f"ce_scores_{split}_{model_name.replace('/', '_')}_ta.npz"

    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    qf = DATA_DIR / ("queries.parquet" if split == "train" else "held_out_queries.parquet")
    queries = load_queries(qf)

    def fmt(row):
        t = str(row.get("title", "") or "").strip()
        a = str(row.get("abstract", "") or "").strip()
        return (t + " " + a).strip()

    doc_texts = {r["doc_id"]: fmt(r) for _, r in corpus.iterrows()}
    query_texts = {r["doc_id"]: fmt(r) for _, r in queries.iterrows()}
    corpus_ids = d["corpus_ids"]

    from sentence_transformers import CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CE {model_name} on {device}...")
    model = CrossEncoder(model_name, device=device, max_length=512)

    scores = {}
    qids = list(per_query_cands.keys())
    total = sum(len(v) for v in per_query_cands.values())
    done = 0
    t0 = time.time()
    for qi, qid in enumerate(qids):
        cand_idx = per_query_cands[qid]
        qt = query_texts[qid]
        pairs = [[qt, doc_texts[corpus_ids[ci]]] for ci in cand_idx]
        out = model.predict(pairs, batch_size=64, show_progress_bar=False,
                            activation_fn=None, convert_to_numpy=True)
        scores[qid] = np.asarray(out, dtype=np.float32)
        done += len(pairs)
        if qi % 5 == 0 or qi == len(qids) - 1:
            elapsed = time.time() - t0
            pct = 100 * done / total
            eta = elapsed / max(done, 1) * (total - done)
            log(f"  PROGRESS {pct:.1f}%  q={qi+1}/{len(qids)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    np.savez_compressed(cache_file, scores=scores)
    log(f"Saved {cache_file}")
    return scores


def _load_features(split, d, per_query_cands=None):
    """Load CE + soft-pipeline features for a split, aligned to hard-pipeline ordering.

    If the CE cache is missing, build it on the fly (requires `per_query_cands`).
    """
    ce_cache = CE_CACHE / f"ce_scores_{split}_BAAI_bge-reranker-base_ta.npz"
    if not ce_cache.exists():
        if per_query_cands is None:
            raise FileNotFoundError(
                f"Missing CE cache {ce_cache} and no candidates provided to build it."
            )
        log(f"CE cache missing, computing {ce_cache.name}...")
        ce_scores = _build_ce_cache(split, per_query_cands, d)
    else:
        ce_scores = np.load(ce_cache, allow_pickle=True)["scores"].item()
        log(f"Loaded CE cache {ce_cache}")

    scores, q_ids, c_ids = compute_soft_scores(split)
    hard_corpus_ids = d["corpus_ids"]
    if c_ids == hard_corpus_ids:
        soft_per_q = {qid: scores[:, qi] for qi, qid in enumerate(q_ids)}
    else:
        cid2ri_soft = {c: i for i, c in enumerate(c_ids)}
        reorder = np.array([cid2ri_soft[c] for c in hard_corpus_ids])
        soft_per_q = {qid: scores[reorder, qi] for qi, qid in enumerate(q_ids)}
    log(f"Loaded soft scores {scores.shape}")
    return ce_scores, soft_per_q


def retrieve(split, top_k=100,
             w_bc=W_BC, w_ce=W_CE, w_soft=W_SOFT, rr_pow=RR_POW,
             dom_k=DOM_K, cand_n=CAND_N):
    d = load_signals_plus_cite(split)
    per_query_cands = get_candidates(d, dom_k=dom_k, cand_n=cand_n)
    ce_scores, soft_per_q = _load_features(split, d, per_query_cands)
    submission = rerank(d, per_query_cands, ce_scores, soft_per_q,
                        w_bc=w_bc, w_ce=w_ce, w_soft=w_soft, rr_pow=rr_pow)
    if top_k != 100:
        submission = {q: ranked[:top_k] for q, ranked in submission.items()}
    qdmap = dict(zip(d["query_ids"], d["query_domains"]))
    return submission, qdmap, d, per_query_cands, ce_scores, soft_per_q


def grid_search(d, per_query_cands, ce_scores, soft_per_q, qrels):
    log("\n=== 4D grid search (w_bc, w_ce, w_soft, rr_pow) ===")
    best = 0.0; best_cfg = None
    count = 0
    for rp in [0.5, 0.75, 1.0, 1.25]:
        for wb in [0.15, 0.22, 0.30, 0.40, 0.50]:
            for wc in [0.0, 0.05, 0.10, 0.15, 0.20]:
                for ws in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
                    sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                                 w_bc=wb, w_ce=wc, w_soft=ws, rr_pow=rp)
                    sc = _fast_eval(sub, qrels)
                    count += 1
                    if sc > best:
                        best, best_cfg = sc, (rp, wb, wc, ws)
                        log(f"  NEW BEST: rr^{rp} w_bc={wb} w_ce={wc} w_soft={ws}: {sc:.4f}")
    log(f"\nEvaluated {count} configs. Best: {best:.4f} at cfg={best_cfg}")

    if best_cfg is not None:
        rp_b, wb_b, wc_b, ws_b = best_cfg
        log(f"\n=== Fine grid around {best_cfg} ===")
        for rp in np.arange(max(0.3, rp_b - 0.3), rp_b + 0.31, 0.1):
            for wb in np.arange(max(0.05, wb_b - 0.1), wb_b + 0.11, 0.02):
                for wc in np.arange(max(0.0, wc_b - 0.05), wc_b + 0.06, 0.02):
                    for ws in np.arange(max(0.0, ws_b - 0.10), ws_b + 0.11, 0.03):
                        sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                                     w_bc=float(wb), w_ce=float(wc), w_soft=float(ws),
                                     rr_pow=float(rp))
                        sc = _fast_eval(sub, qrels)
                        if sc > best:
                            best, best_cfg = sc, (float(rp), float(wb), float(wc), float(ws))
                            log(f"  NEW BEST FINE: rr^{rp:.2f} w_bc={wb:.2f} w_ce={wc:.2f} w_soft={ws:.2f}: {sc:.4f}")
        log(f"\nFinal best (fine): {best:.4f} at {best_cfg}")
    return best, best_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--w-bc", type=float, default=W_BC)
    parser.add_argument("--w-ce", type=float, default=W_CE)
    parser.add_argument("--w-soft", type=float, default=W_SOFT)
    parser.add_argument("--rr-pow", type=float, default=RR_POW)
    parser.add_argument("--grid", action="store_true",
                        help="Train-only: re-run the 4-D grid search before retrieving.")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if PROGRESS.exists(): PROGRESS.unlink()

    d = load_signals_plus_cite(args.split)
    per_query_cands = get_candidates(d, dom_k=DOM_K, cand_n=CAND_N)
    ce_scores, soft_per_q = _load_features(args.split, d, per_query_cands)

    if args.grid:
        if args.split != "train":
            log("--grid requires --split train")
            return
        qrels = load_qrels(DATA_DIR / "qrels.json")
        _, best_cfg = grid_search(d, per_query_cands, ce_scores, soft_per_q, qrels)
        rr_pow, w_bc, w_ce, w_soft = best_cfg
    else:
        rr_pow, w_bc, w_ce, w_soft = args.rr_pow, args.w_bc, args.w_ce, args.w_soft

    submission = rerank(d, per_query_cands, ce_scores, soft_per_q,
                        w_bc=w_bc, w_ce=w_ce, w_soft=w_soft, rr_pow=rr_pow)

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        qdmap = dict(zip(d["query_ids"], d["query_domains"]))
        results = evaluate(submission, qrels, ks=[10], query_domains=qdmap, verbose=True)
        save_results(
            results, RESULTS_DIR / "reranker_v7.csv",
            hyperparameters={
                "w_bc": w_bc, "w_ce": w_ce, "w_soft": w_soft, "rr_pow": rr_pow,
                "dom_k": DOM_K, "cand_n": CAND_N, "top_k": args.top_k,
                "split": args.split, "grid": bool(args.grid),
            },
        )

    name = args.output or f"reranker_v7_{args.split}"
    out_path = SUBMISSIONS_DIR / (name if name.endswith(".json") else f"{name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    log(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
