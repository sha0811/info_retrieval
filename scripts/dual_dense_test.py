"""
Test dual-dense score fusion: ftsmall + bge-large query embeddings.
Both models are normalized; we combine in score space:
  score(doc) = (1-a)*cosine(q_ft, doc_ft) + a*cosine(q_bge, doc_bge)
             + w_domain * (query_domain == doc_domain)
             + w_cite * max_i(cosine(cite_sent_bge, doc_bge))

Hypothesis: bge-large captures different semantic aspects than the finetuned
bge-small. Mixing them at score level should be complementary.
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_FT = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"

print("Loading embeddings ...")
ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)
ft_query_embs = np.load(EMB_DIR_FT / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_query_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)
bge_query_embs = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_query_ids = json.load(f)

cite_embs = np.load(EMB_DIR_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/cite_context_query_ids.json") as f: cite_qids = json.load(f)

queries = load_queries(DATA_DIR / "queries.parquet")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries["doc_id"], queries["domain"].fillna("")))
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

# Align bge-large corpus with ft corpus ordering
bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
# bge-large corpus scores aligned to ft corpus order
bge_corpus_embs_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]

# Align bge query embeddings to ft query order
bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_query_ids)}

# Domain masks (aligned to ft_corpus_ids)
corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

# Cite context
qid_to_cite_rows = defaultdict(list)
for idx, qid in enumerate(cite_qids): qid_to_cite_rows[qid].append(idx)

print(f"ft_corpus: {ft_corpus_embs.shape}, bge_corpus_aligned: {bge_corpus_embs_aligned.shape}")
print(f"ft_queries: {ft_query_embs.shape}, bge_queries: {bge_query_embs.shape}")


def retrieve(alpha, w_domain, w_cite, cite_top_k=500, top_k=100):
    """
    alpha: weight for bge-large dense (1-alpha for ftsmall)
    """
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        q_ft = ft_query_embs[qidx]
        q_bge_idx = bge_qid_to_idx.get(qid)
        q_domain = query_domains.get(qid, "")
        cite_rows = qid_to_cite_rows.get(qid, [])

        ft_scores = ft_corpus_embs @ q_ft
        if q_bge_idx is not None:
            q_bge = bge_query_embs[q_bge_idx]
            bge_scores = bge_corpus_embs_aligned @ q_bge
            scores = (1.0 - alpha) * ft_scores + alpha * bge_scores
        else:
            scores = ft_scores

        if w_domain > 0 and q_domain in domain_masks:
            scores = scores + w_domain * domain_masks[q_domain].astype(np.float32)

        if w_cite > 0 and cite_rows:
            q_cite_embs = cite_embs[cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bge_idx = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bge_idx is not None:
                    scores[cidx] += w_cite * float((q_cite_embs @ bge_corpus_embs[bge_idx]).max())

        top_idx = np.argsort(-scores)[:top_k]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


print()
print("%-65s  NDCG@10  R@100" % "Config")
print("-" * 90)

baseline_ndcg = 0.7111
best_ndcg = baseline_ndcg
best_sub = None
best_name = ""

combos = [
    # (name, alpha, w_domain, w_cite)
    ("baseline (alpha=0, wd=0.25, wc=1.0)",     0.0,  0.25, 1.0),
    ("alpha=0.1",                                0.1,  0.25, 1.0),
    ("alpha=0.2",                                0.2,  0.25, 1.0),
    ("alpha=0.3",                                0.3,  0.25, 1.0),
    ("alpha=0.4",                                0.4,  0.25, 1.0),
    ("alpha=0.5",                                0.5,  0.25, 1.0),
    ("alpha=0.1 wd=0.3",                         0.1,  0.30, 1.0),
    ("alpha=0.2 wd=0.3",                         0.2,  0.30, 1.0),
    ("alpha=0.3 wd=0.3",                         0.3,  0.30, 1.0),
    ("alpha=0.2 wd=0.25 wc=1.5",                 0.2,  0.25, 1.5),
    ("alpha=0.3 wd=0.25 wc=1.5",                 0.3,  0.25, 1.5),
    ("alpha=0.2 wd=0.3 wc=1.5",                  0.2,  0.30, 1.5),
]

for name, alpha, wd, wc in combos:
    sub = retrieve(alpha, wd, wc)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-65s  %.4f   %.4f%s" % (name[:65], ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

print()
if best_sub:
    print("NEW BEST: %s -> %.4f" % (best_name, best_ndcg))
    out_path = SUBMISSIONS_DIR / "dual_dense_train.json"
    with open(out_path, "w") as f:
        json.dump(best_sub, f)
    print(f"Saved -> {out_path}")
else:
    print("No improvement over baseline 0.7111")
