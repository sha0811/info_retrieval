"""
Chunk-level reranking on top of dual-dense best result.

For top-K candidates, compute max cosine(q_bge, chunk_i) over all chunks of each doc.
This finds papers where the relevant content is in the body, not the abstract.

Best dual-dense: alpha=0.5, wd=0.30, wc_max=1.0 -> NDCG@10=0.7146
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

# Chunk embeddings
print("Loading chunk embeddings ...")
chunk_embs = np.load(EMB_DIR_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_chunk_doc_ids.json") as f: chunk_doc_ids = json.load(f)
print(f"  {chunk_embs.shape} chunks across {len(set(chunk_doc_ids))} docs")

# Build doc -> chunk index map
doc_to_chunk_idxs = defaultdict(list)
for idx, doc_id in enumerate(chunk_doc_ids):
    doc_to_chunk_idxs[doc_id].append(idx)

queries = load_queries(DATA_DIR / "queries.parquet")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries["doc_id"], queries["domain"].fillna("")))
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
bge_corpus_embs_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]
bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_query_ids)}

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

qid_to_cite_rows = defaultdict(list)
for idx, qid in enumerate(cite_qids): qid_to_cite_rows[qid].append(idx)

# ft_corpus_ids index map
ft_cid_to_idx = {cid: i for i, cid in enumerate(ft_corpus_ids)}


def retrieve(alpha, w_domain, w_cite_max, w_chunk, chunk_rerank_k=50, cite_top_k=500, top_k=100):
    """
    w_chunk: weight for max(cosine(q_bge, chunk_i)) reranking signal
    chunk_rerank_k: how many top candidates to apply chunk reranking on
    """
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        q_ft = ft_query_embs[qidx]
        q_bge_idx = bge_qid_to_idx.get(qid)
        q_domain = query_domains.get(qid, "")
        cite_rows = qid_to_cite_rows.get(qid, [])

        ft_scores = ft_corpus_embs @ q_ft
        q_bge = None
        if q_bge_idx is not None:
            q_bge = bge_query_embs[q_bge_idx]
            bge_scores = bge_corpus_embs_aligned @ q_bge
            scores = (1.0 - alpha) * ft_scores + alpha * bge_scores
        else:
            scores = ft_scores

        if w_domain > 0 and q_domain in domain_masks:
            scores = scores + w_domain * domain_masks[q_domain].astype(np.float32)

        if w_cite_max > 0 and cite_rows:
            q_cite_embs = cite_embs[cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bge_idx = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bge_idx is not None:
                    scores[cidx] += w_cite_max * float((q_cite_embs @ bge_corpus_embs[bge_idx]).max())

        # Chunk reranking: max chunk similarity for top-K candidates
        if w_chunk > 0 and q_bge is not None:
            top_cands = np.argsort(-scores)[:chunk_rerank_k]
            for cidx in top_cands:
                doc_id = ft_corpus_ids[cidx]
                chunk_idxs = doc_to_chunk_idxs.get(doc_id, [])
                if chunk_idxs:
                    doc_chunks = chunk_embs[chunk_idxs]
                    max_sim = float((doc_chunks @ q_bge).max())
                    scores[cidx] += w_chunk * max_sim

        top_idx = np.argsort(-scores)[:top_k]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


print()
print("%-75s  NDCG@10  R@100" % "Config")
print("-" * 100)

best_ndcg = 0.7146
best_sub = None
best_name = ""

combos = [
    # (name, alpha, w_domain, w_cite_max, w_chunk, chunk_rerank_k)
    ("baseline: alpha=0.5 wd=0.30 wc=1.0 wchunk=0",    0.50, 0.30, 1.0, 0.0, 50),
    ("chunk top50 w=0.1",                                0.50, 0.30, 1.0, 0.1, 50),
    ("chunk top50 w=0.2",                                0.50, 0.30, 1.0, 0.2, 50),
    ("chunk top50 w=0.3",                                0.50, 0.30, 1.0, 0.3, 50),
    ("chunk top50 w=0.5",                                0.50, 0.30, 1.0, 0.5, 50),
    ("chunk top100 w=0.1",                               0.50, 0.30, 1.0, 0.1, 100),
    ("chunk top100 w=0.2",                               0.50, 0.30, 1.0, 0.2, 100),
    ("chunk top100 w=0.3",                               0.50, 0.30, 1.0, 0.3, 100),
    ("chunk top30 w=0.3",                                0.50, 0.30, 1.0, 0.3, 30),
    ("chunk top30 w=0.5",                                0.50, 0.30, 1.0, 0.5, 30),
    # Replace cite with chunk
    ("no_cite chunk top100 w=0.3",                       0.50, 0.30, 0.0, 0.3, 100),
    ("no_cite chunk top100 w=0.5",                       0.50, 0.30, 0.0, 0.5, 100),
    ("no_cite chunk top100 w=1.0",                       0.50, 0.30, 0.0, 1.0, 100),
    # Cite + chunk complementary
    ("cite wc=0.7 chunk top100 w=0.3",                   0.50, 0.30, 0.7, 0.3, 100),
    ("cite wc=0.5 chunk top100 w=0.5",                   0.50, 0.30, 0.5, 0.5, 100),
]

for name, alpha, wd, wcm, wchunk, crk in combos:
    sub = retrieve(alpha, wd, wcm, wchunk, crk)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-75s  %.4f   %.4f%s" % (name[:75], ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

print()
if best_sub:
    print("NEW BEST: %s -> %.4f" % (best_name, best_ndcg))
    out_path = SUBMISSIONS_DIR / "dual_dense_chunk_train.json"
    with open(out_path, "w") as f:
        json.dump(best_sub, f)
    print(f"Saved -> {out_path}")
else:
    print("No improvement over 0.7146.")
