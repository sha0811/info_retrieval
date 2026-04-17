"""
Full fusion: BM25(nonlinear) + ft-small + bge-large + e5-large-v2 + cite context (bge + e5) + domain boost.

The BM25 pipeline (retrieve_advanced.py) scored 0.70 on codabench.
The dual-dense + cite pipeline scored 0.717 train / 0.69 codabench.
This script fuses all signals to get the best of both.

Usage:
    python scripts/fusion_bm25_dense.py                    # sweep weights, print best
    python scripts/fusion_bm25_dense.py --build-heldout    # also generate held-out submission
"""
import json, sys, zipfile, re
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

EMB_DIR_FT    = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE   = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5    = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

# ── Load corpus embeddings ────────────────────────────────────────────────────
print("Loading corpus embeddings...")
ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)

e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)

# Align bge/e5 to ft index order
bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx  = {cid: i for i, cid in enumerate(e5_corpus_ids)}
bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]
e5_corpus_aligned  = e5_corpus_embs[[e5_cid_to_idx[cid]  for cid in ft_corpus_ids]]

# ── Load train query embeddings ───────────────────────────────────────────────
print("Loading train query embeddings...")
ft_q_embs = np.load(EMB_DIR_FT / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_q_ids = json.load(f)

bge_q_embs = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_q_ids = json.load(f)

e5_q_embs = np.load(EMB_DIR_E5 / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "train/query_ids.json") as f: e5_q_ids = json.load(f)

bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_q_ids)}
e5_qid_to_idx  = {qid: i for i, qid in enumerate(e5_q_ids)}

# ── Cite context embeddings ───────────────────────────────────────────────────
bge_cite_embs = np.load(EMB_DIR_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)

e5_cite_embs = np.load(EMB_DIR_E5 / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

qid_to_bge_cite = defaultdict(list)
for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
qid_to_e5_cite = defaultdict(list)
for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

# ── Metadata ──────────────────────────────────────────────────────────────────
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")

corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

# ── BM25 on corpus title+abstract+full_text ───────────────────────────────────
print("Building BM25 on title+abstract...")
corpus_ids_list = corpus["doc_id"].tolist()
corpus_reindex = {cid: i for i, cid in enumerate(corpus_ids_list)}

# BM25 on title+abstract only (fast)
corpus_ta = (corpus["title"].fillna("") + " " + corpus["abstract"].fillna("")).tolist()
bm25 = BM25Okapi([t.lower().split() for t in corpus_ta], k1=1.8, b=0.75)

# BM25 is indexed over corpus_ids_list; we need scores aligned to ft_corpus_ids
ft_to_bm25_idx = [corpus_reindex.get(cid, 0) for cid in ft_corpus_ids]

# Query BM25 text: title+abstract
query_text_map = {}
for _, row in queries_df.iterrows():
    qid = row["doc_id"]
    t = str(row.get("title", "") or "").strip()
    a = str(row.get("abstract", "") or "").strip()
    query_text_map[qid] = (t + " " + a).strip()

print(f"BM25 built. Corpus size: {len(corpus_ta)}, queries: {len(ft_q_ids)}")


def minmax(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-9)


def retrieve_train(
    w_ft=0.5, w_bge=0.5, w_e5=0.0,
    w_bm25=0.6,
    w_domain=0.30,
    w_bge_cite=0.65, w_e5_cite=0.35,
    cite_top_k=1000,
    bm25_nonlinear=True,
    dense_nonlinear=True,
):
    sub = {}
    for qidx, qid in enumerate(ft_q_ids):
        q_ft  = ft_q_embs[qidx]
        q_domain = query_domains.get(qid, "")

        # Dense scores
        ft_scores  = ft_corpus_embs @ q_ft

        bge_idx = bge_qid_to_idx.get(qid)
        bge_scores = bge_corpus_aligned @ bge_q_embs[bge_idx] if bge_idx is not None else np.zeros_like(ft_scores)

        e5_idx = e5_qid_to_idx.get(qid)
        e5_scores = e5_corpus_aligned @ e5_q_embs[e5_idx] if e5_idx is not None else np.zeros_like(ft_scores)

        # BM25 scores aligned to ft order
        q_text = query_text_map.get(qid, "")
        bm25_raw = bm25.get_scores(q_text.lower().split())
        bm25_aligned = bm25_raw[ft_to_bm25_idx]

        # Normalize
        bm25_norm  = minmax(bm25_aligned)
        ft_norm    = minmax(ft_scores)
        bge_norm   = minmax(bge_scores)
        e5_norm    = minmax(e5_scores)

        if bm25_nonlinear:
            bm25_contrib = w_bm25 * np.power(bm25_norm, 1.2)
        else:
            bm25_contrib = w_bm25 * bm25_norm

        if dense_nonlinear:
            dense_contrib = w_ft * np.power(ft_norm, 0.9) + w_bge * np.power(bge_norm, 0.9) + w_e5 * np.power(e5_norm, 0.9)
        else:
            dense_contrib = w_ft * ft_norm + w_bge * bge_norm + w_e5 * e5_norm

        # Domain boost
        domain_contrib = np.zeros(len(ft_corpus_ids), dtype=np.float32)
        if w_domain > 0 and q_domain in domain_masks:
            domain_contrib = w_domain * domain_masks[q_domain].astype(np.float32)

        scores = bm25_contrib + dense_contrib + domain_contrib

        # Cite context re-scoring
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        if w_bge_cite > 0 and bge_cite_rows:
            q_cite = bge_cite_embs[bge_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                bi = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bi is not None:
                    scores[cidx] += w_bge_cite * float((q_cite @ bge_corpus_embs[bi]).max())

        e5_cite_rows = qid_to_e5_cite.get(qid, [])
        if w_e5_cite > 0 and e5_cite_rows:
            q_cite_e5 = e5_cite_embs[e5_cite_rows]
            top_cands = np.argsort(-scores)[:cite_top_k]
            for cidx in top_cands:
                ei = e5_cid_to_idx.get(ft_corpus_ids[cidx])
                if ei is not None:
                    scores[cidx] += w_e5_cite * float((q_cite_e5 @ e5_corpus_embs[ei]).max())

        top_idx = np.argsort(-scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


# ── Step 1: baseline configs ──────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 1: Baseline configs")
print("="*80)
print(f"{'Config':<60}  NDCG@10   R@100")
print("-"*80)

configs_baseline = [
    # name, w_ft, w_bge, w_e5, w_bm25, w_domain, w_bge_cite, w_e5_cite
    ("dual-dense+cite (current best)",   0.5,  0.5,  0.0,  0.0,  0.30,  0.65,  0.35),
    ("e5only+bm25 (retrieve_advanced)",  0.0,  0.0,  1.0,  0.6,  0.20,  0.0,   0.0),
    ("all3dense+bm25+cite",              0.33, 0.33, 0.33, 0.6,  0.30,  0.65,  0.35),
    ("ft+bge+e5+bm25+cite",              0.4,  0.4,  0.2,  0.5,  0.30,  0.65,  0.35),
    ("ft+bge+e5+bm25+cite v2",           0.3,  0.4,  0.3,  0.5,  0.30,  0.65,  0.35),
    ("ft+bge+e5+bm25+cite v3",           0.25, 0.35, 0.4,  0.6,  0.30,  0.65,  0.35),
    ("ft+bge+e5+bm25+cite v4",           0.2,  0.3,  0.5,  0.6,  0.30,  0.65,  0.35),
    ("e5+bge+bm25+cite (no ft)",         0.0,  0.4,  0.6,  0.6,  0.30,  0.65,  0.35),
    ("e5+bge+bm25+cite v2",              0.0,  0.3,  0.7,  0.6,  0.30,  0.65,  0.35),
    ("e5+bge+bm25+cite v3",              0.0,  0.5,  0.5,  0.6,  0.30,  0.65,  0.35),
    ("e5+bm25+cite (no bge,no ft)",      0.0,  0.0,  1.0,  0.6,  0.30,  0.65,  0.35),
    ("ft+e5+bm25+cite (no bge)",         0.3,  0.0,  0.7,  0.6,  0.30,  0.65,  0.35),
]

best_ndcg = 0.0
best_cfg = None
best_sub = None

for name, w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec in configs_baseline:
    sub = retrieve_train(w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    marker = " <--" if ndcg > best_ndcg else ""
    print(f"  {name:<58}  {ndcg:.4f}   {rec:.4f}{marker}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg = (w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec, name)
        best_sub = sub

print(f"\nBest so far: {best_cfg[-1]} -> {best_ndcg:.4f}")

# ── Step 2: fine-tune bm25 weight around best ─────────────────────────────────
print("\n" + "="*80)
print("STEP 2: BM25 weight sweep around best config")
print("="*80)

w_ft0, w_bge0, w_e50, _, w_dom0, w_bc0, w_ec0, _ = best_cfg

print(f"{'bm25_w':<12}  NDCG@10   R@100")
print("-"*40)
for w_bm25 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
    sub = retrieve_train(w_ft0, w_bge0, w_e50, w_bm25, w_dom0, w_bc0, w_ec0)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    marker = " <--" if ndcg > best_ndcg else ""
    print(f"  bm25={w_bm25:<8.2f}  {ndcg:.4f}   {rec:.4f}{marker}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg = (w_ft0, w_bge0, w_e50, w_bm25, w_dom0, w_bc0, w_ec0, f"bm25={w_bm25}")
        best_sub = sub

# ── Step 3: domain weight sweep ───────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 3: Domain weight sweep")
print("="*80)

w_ft0, w_bge0, w_e50, w_bm250, w_dom0, w_bc0, w_ec0, _ = best_cfg
print(f"{'domain_w':<12}  NDCG@10   R@100")
print("-"*40)
for w_dom in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    sub = retrieve_train(w_ft0, w_bge0, w_e50, w_bm250, w_dom, w_bc0, w_ec0)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    marker = " <--" if ndcg > best_ndcg else ""
    print(f"  domain={w_dom:<6.2f}  {ndcg:.4f}   {rec:.4f}{marker}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg = (w_ft0, w_bge0, w_e50, w_bm250, w_dom, w_bc0, w_ec0, f"domain={w_dom}")
        best_sub = sub

# ── Step 4: cite weight sweep ─────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 4: Cite weight sweep")
print("="*80)

w_ft0, w_bge0, w_e50, w_bm250, w_dom0, _, _, _ = best_cfg
print(f"{'bge_cite':<10}  {'e5_cite':<10}  NDCG@10   R@100")
print("-"*50)
for w_bc, w_ec in [
    (0.0, 0.0), (0.3, 0.2), (0.5, 0.3), (0.65, 0.35),
    (0.8, 0.4), (1.0, 0.5), (0.5, 0.5), (0.7, 0.3),
    (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.0, 0.5),
    (1.2, 0.6), (1.5, 0.75),
]:
    sub = retrieve_train(w_ft0, w_bge0, w_e50, w_bm250, w_dom0, w_bc, w_ec)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    marker = " <--" if ndcg > best_ndcg else ""
    print(f"  bge={w_bc:<7.2f}  e5={w_ec:<7.2f}  {ndcg:.4f}   {rec:.4f}{marker}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg = (w_ft0, w_bge0, w_e50, w_bm250, w_dom0, w_bc, w_ec, f"cite bge={w_bc} e5={w_ec}")
        best_sub = sub

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"BEST CONFIG: {best_cfg[-1]}")
print(f"  w_ft={best_cfg[0]}, w_bge={best_cfg[1]}, w_e5={best_cfg[2]}")
print(f"  w_bm25={best_cfg[3]}, w_domain={best_cfg[4]}")
print(f"  w_bge_cite={best_cfg[5]}, w_e5_cite={best_cfg[6]}")
print(f"  NDCG@10 = {best_ndcg:.4f}")
print("="*80)

# Save best train result
out_path = SUBMISSIONS_DIR / "fusion_bm25_dense_train.json"
with open(out_path, "w") as f:
    json.dump(best_sub, f)
print(f"\nSaved best train submission -> {out_path}")

# ── Held-out generation ───────────────────────────────────────────────────────
import sys
if "--build-heldout" in sys.argv:
    print("\nGenerating held-out submission...")

    w_ft0, w_bge0, w_e50, w_bm250, w_dom0, w_bc0, w_ec0, _ = best_cfg

    # Load held-out embeddings
    ft_hq_embs = np.load(EMB_DIR_FT / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_FT / "held_out/query_ids.json") as f: ft_hq_ids = json.load(f)

    bge_hq_embs = np.load(EMB_DIR_BGE / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_BGE / "held_out/query_ids.json") as f: bge_hq_ids = json.load(f)

    e5_hq_embs = np.load(EMB_DIR_E5 / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_E5 / "held_out/query_ids.json") as f: e5_hq_ids = json.load(f)

    bge_hqid_to_idx = {qid: i for i, qid in enumerate(bge_hq_ids)}
    e5_hqid_to_idx  = {qid: i for i, qid in enumerate(e5_hq_ids)}

    bge_hcite_embs = np.load(EMB_DIR_BGE / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_BGE / "held_out/cite_context_query_ids.json") as f: bge_hcite_qids = json.load(f)

    e5_hcite_embs = np.load(EMB_DIR_E5 / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_E5 / "held_out/cite_context_query_ids.json") as f: e5_hcite_qids = json.load(f)

    qid_to_bge_hcite = defaultdict(list)
    for idx, qid in enumerate(bge_hcite_qids): qid_to_bge_hcite[qid].append(idx)
    qid_to_e5_hcite = defaultdict(list)
    for idx, qid in enumerate(e5_hcite_qids): qid_to_e5_hcite[qid].append(idx)

    held_out_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    held_domains = dict(zip(held_out_df["doc_id"], held_out_df["domain"].fillna("")))

    # Build held-out query texts (title+abstract, matching BM25 corpus)
    hq_text_map = {}
    for _, row in held_out_df.iterrows():
        qid = row["doc_id"]
        t = str(row.get("title", "") or "").strip()
        a = str(row.get("abstract", "") or "").strip()
        hq_text_map[qid] = (t + " " + a).strip()

    sub_held = {}
    for qidx, qid in enumerate(ft_hq_ids):
        q_ft = ft_hq_embs[qidx]
        q_domain = held_domains.get(qid, "")

        ft_scores  = ft_corpus_embs @ q_ft

        bge_idx = bge_hqid_to_idx.get(qid)
        bge_scores = bge_corpus_aligned @ bge_hq_embs[bge_idx] if bge_idx is not None else np.zeros_like(ft_scores)

        e5_idx = e5_hqid_to_idx.get(qid)
        e5_scores = e5_corpus_aligned @ e5_hq_embs[e5_idx] if e5_idx is not None else np.zeros_like(ft_scores)

        q_text = hq_text_map.get(qid, "")
        bm25_raw = bm25.get_scores(q_text.lower().split())
        bm25_aligned = bm25_raw[ft_to_bm25_idx]

        bm25_norm = minmax(bm25_aligned)
        ft_norm   = minmax(ft_scores)
        bge_norm  = minmax(bge_scores)
        e5_norm   = minmax(e5_scores)

        bm25_contrib  = w_bm250 * np.power(bm25_norm, 1.2)
        dense_contrib = w_ft0 * np.power(ft_norm, 0.9) + w_bge0 * np.power(bge_norm, 0.9) + w_e50 * np.power(e5_norm, 0.9)

        domain_contrib = np.zeros(len(ft_corpus_ids), dtype=np.float32)
        if w_dom0 > 0 and q_domain in domain_masks:
            domain_contrib = w_dom0 * domain_masks[q_domain].astype(np.float32)

        scores = bm25_contrib + dense_contrib + domain_contrib

        bge_cite_rows = qid_to_bge_hcite.get(qid, [])
        if w_bc0 > 0 and bge_cite_rows:
            q_cite = bge_hcite_embs[bge_cite_rows]
            top_cands = np.argsort(-scores)[:1000]
            for cidx in top_cands:
                bi = bge_cid_to_idx.get(ft_corpus_ids[cidx])
                if bi is not None:
                    scores[cidx] += w_bc0 * float((q_cite @ bge_corpus_embs[bi]).max())

        e5_cite_rows = qid_to_e5_hcite.get(qid, [])
        if w_ec0 > 0 and e5_cite_rows:
            q_cite_e5 = e5_hcite_embs[e5_cite_rows]
            top_cands = np.argsort(-scores)[:1000]
            for cidx in top_cands:
                ei = e5_cid_to_idx.get(ft_corpus_ids[cidx])
                if ei is not None:
                    scores[cidx] += w_ec0 * float((q_cite_e5 @ e5_corpus_embs[ei]).max())

        top_idx = np.argsort(-scores)[:100]
        sub_held[qid] = [ft_corpus_ids[i] for i in top_idx]

    json_path = SUBMISSIONS_DIR / "fusion_bm25_dense_held_out.json"
    zip_path  = SUBMISSIONS_DIR / "fusion_bm25_dense_held_out.zip"
    with open(json_path, "w") as f:
        json.dump(sub_held, f)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="submission_data.json")
    print(f"Saved held-out -> {zip_path}")
