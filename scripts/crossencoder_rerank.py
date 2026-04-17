"""
Experiment: cross-encoder reranking grid search (train split only, not integrated into pipeline).

Tested CE reranking on top of bi-encoder retrieval, it did not manage to beat our best submission at the time
Uses finetuned cross-encoder: data/finetuned_models/crossencoder_ms-marco-MiniLM-L-6-v2/
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
EMB_DIR_E5 = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
CE_DIR = DATA_DIR / "finetuned_models" / "crossencoder_ms-marco-MiniLM-L-6-v2"

# Load sentence_transformers CrossEncoder
try:
    from sentence_transformers import CrossEncoder
    print("Loading cross-encoder ...")
    ce = CrossEncoder(str(CE_DIR), max_length=512)
    print(f"  Loaded: {CE_DIR.name}")
except Exception as e:
    print(f"ERROR loading cross-encoder: {e}")
    sys.exit(1)

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
query_ta_map = dict(zip(queries["doc_id"], queries["ta"].fillna("")))
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
corpus_ta_map = dict(zip(corpus["doc_id"], corpus["ta"].fillna("")))

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


def get_bi_encoder_scores(qid, qidx):
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

    if bge_cite_rows:
        q_cite = bge_cite_embs[bge_cite_rows]
        top_cands = np.argsort(-scores)[:1000]
        for cidx in top_cands:
            bi = bge_cid_to_idx.get(ft_corpus_ids[cidx])
            if bi is not None:
                scores[cidx] += 0.65 * float((q_cite @ bge_corpus_embs[bi]).max())

    if e5_cite_rows:
        q_cite_e5 = e5_cite_embs[e5_cite_rows]
        top_cands = np.argsort(-scores)[:1000]
        for cidx in top_cands:
            ei = e5_cid_to_idx.get(ft_corpus_ids[cidx])
            if ei is not None:
                scores[cidx] += 0.35 * float((q_cite_e5 @ e5_corpus_embs[ei]).max())

    return scores


print()
print("%-65s  NDCG@10  R@100" % "Config")
print("-" * 90)

best_ndcg = 0.7170
best_sub = None
best_name = ""

# Test CE reranking with different top-k
for ce_topk in [20, 30, 50]:
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        scores = get_bi_encoder_scores(qid, qidx)
        top_cands = np.argsort(-scores)[:200]  # get top-200 first

        # CE rerank top-k of those
        q_ta = query_ta_map.get(qid, "")[:500]  # use title+abstract as query
        ce_cands = top_cands[:ce_topk]
        ce_pairs = [(q_ta, corpus_ta_map.get(ft_corpus_ids[cidx], "")[:500]) for cidx in ce_cands]
        ce_scores = ce.predict(ce_pairs, show_progress_bar=False)

        # Normalize CE scores to [0, 1]
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_scores_norm = np.zeros_like(ce_scores)

        # Combine: add normalized CE score back
        final_scores = scores.copy()
        for j, cidx in enumerate(ce_cands):
            final_scores[cidx] += 1.0 * ce_scores_norm[j]  # weight=1.0 for CE

        top_idx = np.argsort(-final_scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]

    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    name = f"CE rerank top-{ce_topk} w=1.0"
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-65s  %.4f   %.4f%s" % (name, ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

# Also test different CE weights
for ce_topk, ce_w in [(30, 0.5), (30, 2.0), (50, 2.0), (50, 0.5)]:
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        scores = get_bi_encoder_scores(qid, qidx)
        top_cands = np.argsort(-scores)[:200]
        q_ta = query_ta_map.get(qid, "")[:500]
        ce_cands = top_cands[:ce_topk]
        ce_pairs = [(q_ta, corpus_ta_map.get(ft_corpus_ids[cidx], "")[:500]) for cidx in ce_cands]
        ce_scores = ce.predict(ce_pairs, show_progress_bar=False)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_scores_norm = np.zeros_like(ce_scores)
        final_scores = scores.copy()
        for j, cidx in enumerate(ce_cands):
            final_scores[cidx] += ce_w * ce_scores_norm[j]
        top_idx = np.argsort(-final_scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]

    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    name = f"CE rerank top-{ce_topk} w={ce_w}"
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-65s  %.4f   %.4f%s" % (name, ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

print()
if best_sub:
    print("NEW BEST: %s -> %.4f" % (best_name, best_ndcg))
    out_path = SUBMISSIONS_DIR / "ce_rerank_train.json"
    with open(out_path, "w") as f:
        json.dump(best_sub, f)
    print(f"Saved -> {out_path}")
else:
    print("No improvement over 0.7170.")
