"""
Fast sweep: ft-small + bge-large + e5-large-v2 + BM25 + domain, NO cite context.

Rationale:
- cite context overfits train (0.717 train / 0.69 held-out)
- BM25 + e5-large-v2 alone = 0.70 held-out
- Adding ft-small + bge-large (stronger dense) should push further
- No cite loop = each config runs in ~10 seconds

Run: python scripts/triple_dense_bm25.py
     python scripts/triple_dense_bm25.py --build-heldout
"""
import json, sys, zipfile
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
EMB_DIR_FT  = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5  = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

print("Loading embeddings...")
ft_c  = np.load(EMB_DIR_FT  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_cids = json.load(f)

bge_c = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)

e5_c  = np.load(EMB_DIR_E5  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5  / "corpus_ids.json") as f: e5_cids = json.load(f)

# Align bge and e5 to ft index order
bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
e5_cid2i  = {c: i for i, c in enumerate(e5_cids)}
bge_c_al  = bge_c[[bge_cid2i[c] for c in ft_cids]]
e5_c_al   = e5_c[[e5_cid2i[c]  for c in ft_cids]]

ft_q  = np.load(EMB_DIR_FT  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_qids = json.load(f)

bge_q = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_qids = json.load(f)

e5_q  = np.load(EMB_DIR_E5  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5  / "train/query_ids.json") as f: e5_qids = json.load(f)

bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
e5_qid2i  = {q: i for i, q in enumerate(e5_qids)}

corpus     = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels      = load_qrels(DATA_DIR / "qrels.json")

corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
query_domains     = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(c, "") for c in ft_cids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

print("Building BM25 on title+abstract...")
corpus_ta    = (corpus["title"].fillna("") + " " + corpus["abstract"].fillna("")).tolist()
bm25         = BM25Okapi([t.lower().split() for t in corpus_ta], k1=1.8, b=0.75)
cids_list    = corpus["doc_id"].tolist()
c_reindex    = {c: i for i, c in enumerate(cids_list)}
ft_to_bm25   = np.array([c_reindex.get(c, 0) for c in ft_cids])
query_ta     = dict(zip(queries_df["doc_id"],
    (queries_df["title"].fillna("") + " " + queries_df["abstract"].fillna("")).tolist()))

print(f"Ready. {len(ft_qids)} queries, corpus {len(ft_cids)}\n")

def mm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def retrieve(w_ft, w_bge, w_e5, w_bm25, w_dom, nonlinear=True):
    sub = {}
    for qi, qid in enumerate(ft_qids):
        s_ft  = ft_c  @ ft_q[qi]
        s_bge = bge_c_al @ bge_q[bge_qid2i[qid]] if qid in bge_qid2i else s_ft
        s_e5  = e5_c_al  @ e5_q[e5_qid2i[qid]]   if qid in e5_qid2i  else s_ft

        bm25_raw = bm25.get_scores(query_ta.get(qid, "").lower().split())
        s_bm25   = bm25_raw[ft_to_bm25]

        if nonlinear:
            score = (w_ft  * np.power(mm(s_ft),   0.9)
                   + w_bge * np.power(mm(s_bge),  0.9)
                   + w_e5  * np.power(mm(s_e5),   0.9)
                   + w_bm25* np.power(mm(s_bm25), 1.2))
        else:
            score = w_ft*mm(s_ft) + w_bge*mm(s_bge) + w_e5*mm(s_e5) + w_bm25*mm(s_bm25)

        q_dom = query_domains.get(qid, "")
        if w_dom > 0 and q_dom in domain_masks:
            score += w_dom * domain_masks[q_dom].astype(np.float32)

        top = np.argsort(-score)[:100]
        sub[qid] = [ft_cids[i] for i in top]
    return sub

PREV_BEST = 0.7170
best_ndcg, best_cfg, best_sub = 0.0, None, None

configs = [
    # name,                          w_ft,  w_bge, w_e5,  w_bm25, w_dom
    ("ft+bge only (baseline)",        0.5,   0.5,   0.0,   0.0,    0.30),
    ("ft+bge+e5 no_bm25",             0.33,  0.33,  0.33,  0.0,    0.30),
    ("ft+bge+e5 bm25=0.3",            0.33,  0.33,  0.33,  0.3,    0.30),
    ("ft+bge+e5 bm25=0.5",            0.33,  0.33,  0.33,  0.5,    0.30),
    ("ft+bge+e5 bm25=0.6",            0.33,  0.33,  0.33,  0.6,    0.30),
    ("bge+e5 bm25=0.5 no_ft",         0.0,   0.4,   0.6,   0.5,    0.30),
    ("bge+e5 bm25=0.6 no_ft",         0.0,   0.4,   0.6,   0.6,    0.30),
    ("bge+e5 bm25=0.6 dom=0.2",       0.0,   0.4,   0.6,   0.6,    0.20),
    ("bge+e5 bm25=0.6 dom=0.0",       0.0,   0.4,   0.6,   0.6,    0.00),
    ("ft+bge+e5 bm25=0.6 dom=0.2",    0.25,  0.25,  0.5,   0.6,    0.20),
    ("ft+bge+e5 bm25=0.6 dom=0.0",    0.25,  0.25,  0.5,   0.6,    0.00),
    ("e5 only bm25=0.6 dom=0.2",      0.0,   0.0,   1.0,   0.6,    0.20),
    ("e5 only bm25=0.6 dom=0.3",      0.0,   0.0,   1.0,   0.6,    0.30),
    ("bge only bm25=0.6 dom=0.3",     0.0,   1.0,   0.0,   0.6,    0.30),
    ("ft+bge bm25=0.6 dom=0.3",       0.5,   0.5,   0.0,   0.6,    0.30),
    ("ft+bge bm25=0.6 dom=0.2",       0.5,   0.5,   0.0,   0.6,    0.20),
    ("ft+e5 bm25=0.6 dom=0.2",        0.5,   0.0,   0.5,   0.6,    0.20),
    ("ft+e5 bm25=0.6 dom=0.3",        0.5,   0.0,   0.5,   0.6,    0.30),
]

print(f"{'Config':<42}  NDCG@10   R@100")
print("-"*65)
for name, wft, wbge, we5, wbm, wdom in configs:
    sub = retrieve(wft, wbge, we5, wbm, wdom)
    res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec  = res["overall"]["Recall@100"]
    mark = " <-- BEST" if ndcg > best_ndcg else ""
    print(f"  {name:<40}  {ndcg:.4f}   {rec:.4f}{mark}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_cfg  = (wft, wbge, we5, wbm, wdom, name)
        best_sub  = sub

print(f"\n{'='*65}")
print(f"BEST: {best_cfg[-1]} -> {best_ndcg:.4f}")
if best_ndcg > PREV_BEST:
    print(f"*** IMPROVEMENT +{best_ndcg-PREV_BEST:.4f} over {PREV_BEST} ***")
    p = SUBMISSIONS_DIR / "triple_dense_bm25_train.json"
    with open(p, "w") as f: json.dump(best_sub, f)
    print(f"Saved -> {p}")
else:
    print(f"No improvement over {PREV_BEST}.")

# ── Held-out ──────────────────────────────────────────────────────────────────
if "--build-heldout" in sys.argv:
    print("\nBuilding held-out submission...")
    wft, wbge, we5, wbm, wdom, _ = best_cfg

    ft_hq  = np.load(EMB_DIR_FT  / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_FT / "held_out/query_ids.json") as f: ft_hqids = json.load(f)

    bge_hq = np.load(EMB_DIR_BGE / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_BGE / "held_out/query_ids.json") as f: bge_hqids = json.load(f)

    e5_hq  = np.load(EMB_DIR_E5  / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_E5  / "held_out/query_ids.json") as f: e5_hqids = json.load(f)

    bge_hqid2i = {q: i for i, q in enumerate(bge_hqids)}
    e5_hqid2i  = {q: i for i, q in enumerate(e5_hqids)}

    held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))
    held_ta   = dict(zip(held_df["doc_id"],
        (held_df["title"].fillna("") + " " + held_df["abstract"].fillna("")).tolist()))

    sub_h = {}
    for qi, qid in enumerate(ft_hqids):
        s_ft  = ft_c  @ ft_hq[qi]
        s_bge = bge_c_al @ bge_hq[bge_hqid2i[qid]] if qid in bge_hqid2i else s_ft
        s_e5  = e5_c_al  @ e5_hq[e5_hqid2i[qid]]   if qid in e5_hqid2i  else s_ft

        bm25_raw = bm25.get_scores(held_ta.get(qid, "").lower().split())
        s_bm25   = bm25_raw[ft_to_bm25]

        score = (wft  * np.power(mm(s_ft),   0.9)
               + wbge * np.power(mm(s_bge),  0.9)
               + we5  * np.power(mm(s_e5),   0.9)
               + wbm  * np.power(mm(s_bm25), 1.2))

        q_dom = held_doms.get(qid, "")
        if wdom > 0 and q_dom in domain_masks:
            score += wdom * domain_masks[q_dom].astype(np.float32)

        top = np.argsort(-score)[:100]
        sub_h[qid] = [ft_cids[i] for i in top]

    jp = SUBMISSIONS_DIR / "triple_dense_bm25_held_out.json"
    zp = SUBMISSIONS_DIR / "triple_dense_bm25_held_out.zip"
    with open(jp, "w") as f: json.dump(sub_h, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"Saved -> {zp}")
