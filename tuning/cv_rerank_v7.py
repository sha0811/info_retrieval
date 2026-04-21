"""
5-fold CV for reranker v7 to check overfitting of the 0.7603 config.

Best config from full-train grid: rr^1.0, w_bc=0.44, w_ce=0.04, w_soft=0.25 → 0.7603

Split 100 queries into 5 folds of 20. For each fold:
  - Grid search on 80 train queries (smaller grid for speed)
  - Evaluate the fold's best config on the 20 held-out queries
  - Also evaluate the global best config on the held-out 20 for comparison

If the global config generalises at least as well as the per-fold best, the
0.7603 gain is not overfitting.

Usage:
    python tuning/cv_rerank_v7.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import load_qrels
from models.hard_pipeline_with_cite import load_signals_plus_cite, _fast_eval, _ndcg10
from models.reranker_v7 import rerank
from scripts.compute_soft_scores import compute_soft_scores, get_candidates

DATA_DIR = ROOT / "data"
CE_CACHE = DATA_DIR / "ce_cache"
PROGRESS = Path(__file__).parent / "cv_rerank_v7_progress.txt"


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def eval_subset(sub, qrels, subset_qids):
    qd = {q: set(v) for q, v in qrels.items() if q in subset_qids}
    return float(np.mean([_ndcg10(sub.get(q, []), qd.get(q, set())) for q in qd]))


def main():
    if PROGRESS.exists(): PROGRESS.unlink()

    qrels = load_qrels(DATA_DIR / "qrels.json")
    d = load_signals_plus_cite("train")
    per_query_cands = get_candidates(d, dom_k=80, cand_n=100)

    cache_ce = CE_CACHE / "ce_scores_train_BAAI_bge-reranker-base_ta.npz"
    ce_scores = np.load(cache_ce, allow_pickle=True)["scores"].item()
    scores, q_ids, c_ids = compute_soft_scores("train")
    hard_corpus_ids = d["corpus_ids"]
    if c_ids == hard_corpus_ids:
        soft_per_q = {qid: scores[:, qi] for qi, qid in enumerate(q_ids)}
    else:
        cid2ri_soft = {c: i for i, c in enumerate(c_ids)}
        reorder = np.array([cid2ri_soft[c] for c in hard_corpus_ids])
        soft_per_q = {qid: scores[reorder, qi] for qi, qid in enumerate(q_ids)}

    all_qids = list(qrels.keys())
    log(f"Got {len(all_qids)} queries")

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_qids))
    shuffled = [all_qids[i] for i in perm]
    folds = [set(shuffled[i*20:(i+1)*20]) for i in range(5)]

    grid = []
    for rp in [0.75, 1.0, 1.25]:
        for wb in [0.30, 0.40, 0.50]:
            for wc in [0.0, 0.05, 0.10]:
                for ws in [0.10, 0.20, 0.30]:
                    grid.append((rp, wb, wc, ws))
    log(f"Grid size: {len(grid)}")

    global_best_cfg = (1.0, 0.44, 0.04, 0.25)

    oof_scores = []
    trainwise_scores = []
    cv_picks = []

    for fi, held_out in enumerate(folds):
        train_qids = set(all_qids) - held_out
        best_sc = -1; best_cfg = None
        for (rp, wb, wc, ws) in grid:
            sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                         w_bc=wb, w_ce=wc, w_soft=ws, rr_pow=rp)
            sc = eval_subset(sub, qrels, train_qids)
            if sc > best_sc: best_sc, best_cfg = sc, (rp, wb, wc, ws)
        rp, wb, wc, ws = best_cfg
        sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                     w_bc=wb, w_ce=wc, w_soft=ws, rr_pow=rp)
        ho_sc = eval_subset(sub, qrels, held_out)
        oof_scores.append(ho_sc)
        trainwise_scores.append(best_sc)
        cv_picks.append(best_cfg)
        log(f"  Fold {fi}: train_best={best_sc:.4f} (cfg={best_cfg}) -> held_out={ho_sc:.4f}")

        rp, wb, wc, ws = global_best_cfg
        sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                     w_bc=wb, w_ce=wc, w_soft=ws, rr_pow=rp)
        ho_fixed = eval_subset(sub, qrels, held_out)
        log(f"           global cfg  {global_best_cfg} -> held_out={ho_fixed:.4f}")

    mean_oof = float(np.mean(oof_scores))
    mean_train = float(np.mean(trainwise_scores))
    log(f"\n=== Summary ===")
    log(f"Mean train-fold best: {mean_train:.4f}")
    log(f"Mean held-out: {mean_oof:.4f}")
    log(f"OOF scores: {[f'{s:.4f}' for s in oof_scores]}")

    sub = rerank(d, per_query_cands, ce_scores, soft_per_q, w_bc=0.22)
    baseline = _fast_eval(sub, qrels)
    log(f"Baseline on full train (w_bc=0.22): {baseline:.4f}")

    rp, wb, wc, ws = global_best_cfg
    sub = rerank(d, per_query_cands, ce_scores, soft_per_q,
                 w_bc=wb, w_ce=wc, w_soft=ws, rr_pow=rp)
    gbs = _fast_eval(sub, qrels)
    log(f"Global best on full train {global_best_cfg}: {gbs:.4f}")


if __name__ == "__main__":
    main()
