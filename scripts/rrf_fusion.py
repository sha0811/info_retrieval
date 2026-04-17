"""
RRF fusion of existing submission JSONs. Instant — no embeddings needed.

Combines:
  A: dual_cite_v2_train.json       (ft+bge+cite, 0.7170 train / 0.69 held-out)
  B: specter2_score_fusion_domainboost.json  (BM25+e5, 0.668 train / 0.70 held-out)

RRF score for doc d: sum(1 / (k + rank_in_list_i(d))) across all lists
"""
import json, sys, zipfile
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate

DATA_DIR       = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels      = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

def rrf(lists, k=60):
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

def load(path):
    with open(path) as f: return json.load(f)

def eval_sub(sub, label):
    res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
    print(f"  {label:<50}  NDCG@10={res['overall']['NDCG@10']:.4f}  R@100={res['overall']['Recall@100']:.4f}")
    return res["overall"]["NDCG@10"]

# Load train submissions
cite  = load(SUBMISSIONS_DIR / "dual_cite_v2_train.json")
bm25e = load(SUBMISSIONS_DIR / "specter2_score_fusion_domainboost.json")

print("Individual scores:")
eval_sub(cite,  "dual_cite (ft+bge+cite)")
eval_sub(bm25e, "BM25+e5 (retrieve_advanced)")

print("\nRRF sweep (k parameter):")
best_ndcg, best_k, best_sub = 0.0, 60, None
for k in [10, 20, 30, 40, 60, 80, 100]:
    qids = list(cite.keys())
    sub = {qid: rrf([cite.get(qid,[]), bm25e.get(qid,[])], k=k)[:100] for qid in qids}
    res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    mark = " <--" if ndcg > best_ndcg else ""
    print(f"  RRF k={k:<4}  NDCG@10={ndcg:.4f}  R@100={rec:.4f}{mark}")
    if ndcg > best_ndcg:
        best_ndcg, best_k, best_sub = ndcg, k, sub

print(f"\nBest RRF k={best_k} -> {best_ndcg:.4f}")

# Also try weighted RRF (give cite more weight by repeating it)
print("\nWeighted RRF (cite repeated N times):")
for n_cite in [2, 3]:
    for k in [20, 40, 60]:
        sub = {qid: rrf([cite.get(qid,[])]*n_cite + [bm25e.get(qid,[])], k=k)[:100] for qid in qids}
        res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
        ndcg = res["overall"]["NDCG@10"]
        mark = " <--" if ndcg > best_ndcg else ""
        print(f"  cite×{n_cite} + bm25×1  k={k}  NDCG@10={ndcg:.4f}{mark}")
        if ndcg > best_ndcg:
            best_ndcg, best_k = ndcg, k
            best_sub = sub

print(f"\n{'='*60}")
print(f"BEST FUSION -> {best_ndcg:.4f}  (previous best 0.7170)")
if best_ndcg > 0.717:
    out = SUBMISSIONS_DIR / "rrf_fusion_train.json"
    with open(out, "w") as f: json.dump(best_sub, f)
    print(f"Saved -> {out}")
    print("*** IMPROVEMENT! Submit held-out version. ***")
else:
    print("No train improvement. But held-out might still be better (0.70 ceiling).")
    # Save anyway since fusion might help held-out
    out = SUBMISSIONS_DIR / "rrf_fusion_train.json"
    with open(out, "w") as f: json.dump(best_sub, f)
