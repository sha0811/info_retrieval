"""
BM25 on query title+abstract as additional retrieval signal.
Fuse BM25(query_TA, corpus_TA) with our best dense model scores.

Also try: BM25 only on relevant portions (query paper body w/o citations).
"""
import json, sys, re
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_qrels, load_queries, evaluate
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_FT = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5 = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)
ft_query_embs = np.load(EMB_DIR_FT / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_query_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)
bge_query_embs = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_query_ids = json.load(f)

bge_cite_embs = np.load(EMB_DIR_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)
e5_cite_embs = np.load(EMB_DIR_E5 / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

queries = load_queries(DATA_DIR / "queries.parquet")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries["doc_id"], queries["domain"].fillna("")))
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx = {cid: i for i, cid in enumerate(e5_corpus_ids)}
bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]
bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_query_ids)}

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

qid_to_bge_cite = defaultdict(list)
for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
qid_to_e5_cite = defaultdict(list)
for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

def tokenize(text):
    text = (text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

# Build BM25 on corpus title+abstract
print('Building BM25 on corpus TA ...')
corpus_ta = [str(r.get('ta', '') or '') for _, r in corpus.iterrows()]
corpus_bm25_ids = corpus['doc_id'].tolist()
bm25_ta = BM25Okapi([tokenize(t) for t in corpus_ta])
cid_to_bm25_idx = {cid: i for i, cid in enumerate(corpus_bm25_ids)}

# Build query TA lookup
query_ta_map = dict(zip(queries["doc_id"], queries["ta"].fillna("")))

print('BM25-only baseline ...')
bm25_sub = {}
for qidx, qid in enumerate(ft_query_ids):
    q_ta = query_ta_map.get(qid, "")
    scores = bm25_ta.get_scores(tokenize(q_ta))
    top = np.argsort(-scores)[:100]
    bm25_sub[qid] = [corpus_bm25_ids[i] for i in top]
res = evaluate(bm25_sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
print(f'  BM25(TA): NDCG@10={res["overall"]["NDCG@10"]:.4f} R@100={res["overall"]["Recall@100"]:.4f}')


def retrieve(w_bm25, bm25_k, w_bge_cite=0.65, w_e5_cite=0.35, cite_top_k=1000, top_k=100):
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        q_ft = ft_query_embs[qidx]
        q_domain = query_domains.get(qid, "")
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        e5_cite_rows = qid_to_e5_cite.get(qid, [])

        ft_scores = ft_corpus_embs @ q_ft
        q_bge_idx = bge_qid_to_idx.get(qid)
        if q_bge_idx is not None:
            scores = 0.5 * ft_scores + 0.5 * (bge_corpus_aligned @ bge_query_embs[q_bge_idx])
        else:
            scores = ft_scores.copy()

        if q_domain in domain_masks:
            scores = scores + 0.30 * domain_masks[q_domain].astype(np.float32)

        if w_bge_cite > 0 and bge_cite_rows:
            q_cite = bge_cite_embs[bge_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bi = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bi is not None:
                    scores[cidx] += w_bge_cite * float((q_cite @ bge_corpus_embs[bi]).max())

        if w_e5_cite > 0 and e5_cite_rows:
            q_cite_e5 = e5_cite_embs[e5_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                ei = e5_cid_to_idx.get(ft_corpus_ids[cidx])
                if ei is not None:
                    scores[cidx] += w_e5_cite * float((q_cite_e5 @ e5_corpus_embs[ei]).max())

        # BM25 on query TA
        if w_bm25 > 0:
            q_ta = query_ta_map.get(qid, "")
            bm25_scores = bm25_ta.get_scores(tokenize(q_ta))

            # Apply to top-bm25_k candidates
            top_cands = np.argsort(-scores)[:bm25_k]
            bm25_local = np.array([bm25_scores[cid_to_bm25_idx.get(ft_corpus_ids[cidx], 0)] for cidx in top_cands])
            bmax = bm25_local.max()
            if bmax > 0:
                bm25_local /= bmax
                for j, cidx in enumerate(top_cands):
                    scores[cidx] += w_bm25 * bm25_local[j]

        top_idx = np.argsort(-scores)[:top_k]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


print()
print("%-65s  NDCG@10  R@100" % "Config")
print("-" * 90)

best_ndcg = 0.7170
best_sub = None
best_name = ""

combos = [
    (0.05, 100), (0.1, 100), (0.2, 100), (0.3, 100),
    (0.05, 200), (0.1, 200), (0.2, 200),
    (0.1, 500), (0.2, 500), (0.3, 500),
    (0.5, 1000), (1.0, 1000),
]

for w_bm25, bm25_k in combos:
    sub = retrieve(w_bm25, bm25_k)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    name = f"bm25_TA w={w_bm25} k={bm25_k}"
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-65s  %.4f   %.4f%s" % (name[:65], ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

print()
if best_sub:
    print("NEW BEST: %s -> %.4f" % (best_name, best_ndcg))
    out_path = SUBMISSIONS_DIR / "bm25_ta_fusion_train.json"
    with open(out_path, "w") as f:
        json.dump(best_sub, f)
    print(f"Saved -> {out_path}")
else:
    print("No improvement over 0.7170.")
