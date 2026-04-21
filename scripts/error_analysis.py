"""
Error analysis for the current best hard-cite pipeline (0.7552).
Goals:
  - Per-query NDCG@10 breakdown to find worst queries
  - For worst queries: where do the relevant docs rank in the final list?
  - Are relevant docs in the candidate pool (cand_n=100)?
  - Do they appear in the top-90 domain filter?
"""
import sys, json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import load_corpus, load_qrels, load_queries

DATA_DIR = ROOT / "data"

# Import the run function
sys.path.insert(0, str(ROOT / "_claude"))
from hard_pipeline_with_cite import run, eval_sub, load_signals_plus_cite, ndcg_at_10

qrels = load_qrels(DATA_DIR / "qrels.json")

# Run the best config
print("Running best config w_bc=0.22, cand_n=100, dom_k=80...", flush=True)
sub = run("train", w_bc=0.22, w_ec=0.0, w_cc=0.0, dom_k=80, cand_n=100)
overall = eval_sub(sub, qrels)
print(f"Overall NDCG@10 = {overall:.4f}", flush=True)

# Per-query breakdown
per_q = {}
for qid, rel_list in qrels.items():
    relevant = set(rel_list)
    ranked = sub.get(qid, [])
    per_q[qid] = {
        "ndcg10": ndcg_at_10(ranked, relevant),
        "n_rel": len(relevant),
        "ranked": ranked,
        "relevant": relevant,
    }

# Sort by NDCG ascending
worst = sorted(per_q.items(), key=lambda x: x[1]["ndcg10"])

# Query metadata
qdf = load_queries(DATA_DIR / "queries.parquet")
qid2dom = dict(zip(qdf["doc_id"], qdf["domain"]))
qid2title = dict(zip(qdf["doc_id"], qdf["title"]))

# Per-domain
dom_scores = defaultdict(list)
for qid, info in per_q.items():
    dom_scores[qid2dom[qid]].append(info["ndcg10"])

print("\n=== Per-domain NDCG@10 ===")
for dom, scs in sorted(dom_scores.items(), key=lambda x: np.mean(x[1])):
    print(f"  {dom:<20} n={len(scs):>3}  mean={np.mean(scs):.4f}  min={min(scs):.4f}")

print("\n=== 20 Worst queries ===")
print(f"{'qid':<12} {'dom':<18} {'ndcg10':<8} {'n_rel':<5} {'rel_in_top100':<12} {'title'}")
for qid, info in worst[:20]:
    in_top100 = sum(1 for d in info["ranked"] if d in info["relevant"])
    title = str(qid2title.get(qid, ""))[:50]
    print(f"{qid[:10]:<12} {qid2dom[qid]:<18} {info['ndcg10']:.4f}  {info['n_rel']:<5} {in_top100:<12} {title}")

# Where do relevant docs end up?
print("\n=== For worst 10 queries: ranks of relevant docs in final top-100 ===")
for qid, info in worst[:10]:
    ranks = []
    for d in info["relevant"]:
        try:
            ranks.append(info["ranked"].index(d))
        except ValueError:
            ranks.append(-1)
    ranks_str = ", ".join(str(r) if r >= 0 else "MISS" for r in sorted([r for r in ranks if r >= 0]) + [r for r in ranks if r < 0])
    print(f"  {qid[:10]}  ({qid2dom[qid]}): {ranks_str}")

# Room for improvement: if we had perfect reranking within top-100, how much could we gain?
print("\n=== Potential headroom if we reordered within top-100 ===")
# If all relevant docs in top-100 appeared at ranks 1..|rel|, what would NDCG@10 be?
oracle_scores = []
original_scores = []
for qid, info in per_q.items():
    in_top100 = [d for d in info["ranked"] if d in info["relevant"]]
    # Oracle: move all relevant to the front
    if len(in_top100) > 0:
        oracle_ranked = in_top100 + [d for d in info["ranked"] if d not in info["relevant"]]
        oracle_scores.append(ndcg_at_10(oracle_ranked, info["relevant"]))
    else:
        oracle_scores.append(0.0)
    original_scores.append(info["ndcg10"])
print(f"  Current mean NDCG@10: {np.mean(original_scores):.4f}")
print(f"  Oracle (perfect reorder within top-100): {np.mean(oracle_scores):.4f}")
print(f"  Headroom: +{np.mean(oracle_scores) - np.mean(original_scores):.4f}")

# Also: how many queries have all relevant docs in top-100?
all_in_top = sum(1 for qid, info in per_q.items()
                 if all(d in info["ranked"] for d in info["relevant"]))
missing = defaultdict(int)
for qid, info in per_q.items():
    miss = sum(1 for d in info["relevant"] if d not in info["ranked"])
    missing[miss] += 1
print(f"\nQueries with ALL relevant in top-100: {all_in_top}/100")
print(f"  # missing docs → # queries: {dict(sorted(missing.items()))}")
