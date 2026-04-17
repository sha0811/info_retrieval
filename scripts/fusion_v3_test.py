"""
Fusion using cached full_text BM25 scores + all dense models + cite context.
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd

DATA_DIR  = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "bm25_cache"
SUBS_DIR  = Path(__file__).parent.parent / "submissions"
EMB_FT    = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_BGE   = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_E5    = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

print("Loading embeddings + BM25 cache...")
ft_c  = np.load(EMB_FT  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "corpus_ids.json") as f: ft_cids = json.load(f)
bge_c = np.load(EMB_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "corpus_ids.json") as f: bge_cids = json.load(f)
e5_c  = np.load(EMB_E5  / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "corpus_ids.json") as f: e5_cids = json.load(f)

bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
e5_cid2i  = {c: i for i, c in enumerate(e5_cids)}
bge_c_al  = bge_c[[bge_cid2i[c] for c in ft_cids]]
e5_c_al   = e5_c[[e5_cid2i[c]  for c in ft_cids]]

ft_q  = np.load(EMB_FT  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_FT  / "train/query_ids.json") as f: ft_qids = json.load(f)
bge_q = np.load(EMB_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/query_ids.json") as f: bge_qids = json.load(f)
e5_q  = np.load(EMB_E5  / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "train/query_ids.json") as f: e5_qids = json.load(f)
bge_qid2i = {q: i for i, q in enumerate(bge_qids)}
e5_qid2i  = {q: i for i, q in enumerate(e5_qids)}

bge_cite = np.load(EMB_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
e5_cite  = np.load(EMB_E5  / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_E5  / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)
qid_to_bge_cite = defaultdict(list)
for i, q in enumerate(bge_cite_qids): qid_to_bge_cite[q].append(i)
qid_to_e5_cite = defaultdict(list)
for i, q in enumerate(e5_cite_qids):  qid_to_e5_cite[q].append(i)

bm25_tr = np.load(CACHE_DIR / "bm25_train_scores.npy")
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tr_qids = json.load(f)
with open(CACHE_DIR / "bm25_corpus_ids.json")       as f: bm25_cids = json.load(f)
bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
bm25_to_ft = np.array([bm25_cid2i.get(c, 0) for c in ft_cids])
bm25_tr_qi = {q: i for i, q in enumerate(bm25_tr_qids)}

corpus     = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels      = load_qrels(DATA_DIR / "qrels.json")
cdmap      = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
qdmap      = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))
cdom_arr   = np.array([cdmap.get(c, "") for c in ft_cids])
dmasks     = {d: cdom_arr == d for d in np.unique(cdom_arr) if d}
print("Ready.\n")

def mm(v): return (v - v.min()) / (v.max() - v.min() + 1e-9)

def run(w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec, ctk=1000):
    sub = {}
    for qi, qid in enumerate(ft_qids):
        s_ft  = ft_c  @ ft_q[qi]
        s_bge = bge_c_al @ bge_q[bge_qid2i[qid]] if qid in bge_qid2i else s_ft
        s_e5  = e5_c_al  @ e5_q[e5_qid2i[qid]]   if qid in e5_qid2i  else s_ft
        # Dense: raw cosine similarities (no normalization) — matches original best pipeline
        sc    = w_ft * s_ft + w_bge * s_bge + w_e5 * s_e5
        # BM25: needs normalization since scale differs from cosine
        bmc   = w_bm25 * np.power(mm(bm25_tr[bm25_tr_qi[qid]][bm25_to_ft]), 1.2) if w_bm25 > 0 and qid in bm25_tr_qi else 0
        sc    = sc + bmc
        qd    = qdmap.get(qid, "")
        if w_dom > 0 and qd in dmasks: sc += w_dom * dmasks[qd].astype(np.float32)
        if w_bc > 0:
            rows = qid_to_bge_cite.get(qid, [])
            if rows:
                qc = bge_cite[rows]
                for cidx in np.argsort(-sc)[:ctk]:
                    bi = bge_cid2i.get(ft_cids[cidx])
                    if bi is not None: sc[cidx] += w_bc * float((qc @ bge_c[bi]).max())
        if w_ec > 0:
            rows = qid_to_e5_cite.get(qid, [])
            if rows:
                qc = e5_cite[rows]
                for cidx in np.argsort(-sc)[:ctk]:
                    ei = e5_cid2i.get(ft_cids[cidx])
                    if ei is not None: sc[cidx] += w_ec * float((qc @ e5_c[ei]).max())
        sub[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]
    return sub

PREV = 0.7170
best_n, best_a, best_s = 0.0, None, None

def tst(nm, *a):
    global best_n, best_a, best_s
    sub = run(*a)
    res = evaluate(sub, qrels, ks=[10,100], query_domains=qdmap, verbose=False)
    n, r = res["overall"]["NDCG@10"], res["overall"]["Recall@100"]
    mk = " <-- BEST" if n > best_n else ""
    print(f"  {nm:<54}  {n:.4f}  {r:.4f}{mk}")
    if n > best_n: best_n, best_a, best_s = n, a, sub

print(f"{'Config':<56}  NDCG@10  R@100")
print("-"*74)
# w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec
for nm, *a in [
    ("baseline ft+bge+cite",             0.5,  0.5,  0.0, 0.0, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.2+cite",             0.5,  0.5,  0.0, 0.2, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.3+cite",             0.5,  0.5,  0.0, 0.3, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.4+cite",             0.5,  0.5,  0.0, 0.4, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.5+cite",             0.5,  0.5,  0.0, 0.5, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.3+cite dom=0.2",     0.5,  0.5,  0.0, 0.3, 0.20, 0.65, 0.35),
    ("ft+bge+bm25=0.4+cite dom=0.2",     0.5,  0.5,  0.0, 0.4, 0.20, 0.65, 0.35),
    ("ft+bge+e5+bm25=0.3+cite",          0.33, 0.33, 0.33,0.3, 0.30, 0.65, 0.35),
    ("ft+bge+e5+bm25=0.4+cite",          0.33, 0.33, 0.33,0.4, 0.30, 0.65, 0.35),
    ("ft+bge+bm25=0.3+nocite",           0.5,  0.5,  0.0, 0.3, 0.30, 0.0,  0.0),
    ("ft+bge+bm25=0.4+nocite",           0.5,  0.5,  0.0, 0.4, 0.30, 0.0,  0.0),
    ("e5only+bm25=0.6 dom=0.2",          0.0,  0.0,  1.0, 0.6, 0.20, 0.0,  0.0),
    ("ft+bge+bm25=0.6 dom=0.2 nocite",   0.5,  0.5,  0.0, 0.6, 0.20, 0.0,  0.0),
]:
    tst(nm, *a)

print(f"\nBest: {best_n:.4f} (prev {PREV})")
if best_n > PREV:
    print(f"*** IMPROVEMENT +{best_n-PREV:.4f} ***")
    with open(SUBS_DIR / "fusion_v3_train.json", "w") as f: json.dump(best_s, f)

    # Held-out
    print("Generating held-out...")
    bm25_he = np.load(CACHE_DIR / "bm25_held_scores.npy")
    with open(CACHE_DIR / "bm25_held_query_ids.json") as f: bm25_he_qids = json.load(f)
    bm25_he_qi = {q: i for i, q in enumerate(bm25_he_qids)}

    ft_hq  = np.load(EMB_FT  / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_FT  / "held_out/query_ids.json") as f: ft_hqids = json.load(f)
    bge_hq = np.load(EMB_BGE / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "held_out/query_ids.json") as f: bge_hqids = json.load(f)
    e5_hq  = np.load(EMB_E5  / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_E5  / "held_out/query_ids.json") as f: e5_hqids = json.load(f)
    bge_hqid2i = {q: i for i, q in enumerate(bge_hqids)}
    e5_hqid2i  = {q: i for i, q in enumerate(e5_hqids)}

    bge_hc = np.load(EMB_BGE / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "held_out/cite_context_query_ids.json") as f: bge_hcqids = json.load(f)
    e5_hc  = np.load(EMB_E5  / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_E5  / "held_out/cite_context_query_ids.json") as f: e5_hcqids = json.load(f)
    q2bghc = defaultdict(list)
    for i, q in enumerate(bge_hcqids): q2bghc[q].append(i)
    q2e5hc = defaultdict(list)
    for i, q in enumerate(e5_hcqids):  q2e5hc[q].append(i)

    held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))
    w_ft, w_bge, w_e5, w_bm25, w_dom, w_bc, w_ec = best_a

    sub_h = {}
    for qi, qid in enumerate(ft_hqids):
        s_ft  = ft_c  @ ft_hq[qi]
        s_bge = bge_c_al @ bge_hq[bge_hqid2i[qid]] if qid in bge_hqid2i else s_ft
        s_e5  = e5_c_al  @ e5_hq[e5_hqid2i[qid]]   if qid in e5_hqid2i  else s_ft
        bmc   = w_bm25 * np.power(mm(bm25_he[bm25_he_qi[qid]][bm25_to_ft]), 1.2) if w_bm25>0 and qid in bm25_he_qi else 0
        sc    = w_ft*s_ft + w_bge*s_bge + w_e5*s_e5 + bmc
        qd    = held_doms.get(qid, "")
        if w_dom > 0 and qd in dmasks: sc += w_dom * dmasks[qd].astype(np.float32)
        if w_bc > 0:
            rows = q2bghc.get(qid, [])
            if rows:
                qc = bge_hc[rows]
                for cidx in np.argsort(-sc)[:1000]:
                    bi = bge_cid2i.get(ft_cids[cidx])
                    if bi is not None: sc[cidx] += w_bc * float((qc @ bge_c[bi]).max())
        if w_ec > 0:
            rows = q2e5hc.get(qid, [])
            if rows:
                qc = e5_hc[rows]
                for cidx in np.argsort(-sc)[:1000]:
                    ei = e5_cid2i.get(ft_cids[cidx])
                    if ei is not None: sc[cidx] += w_ec * float((qc @ e5_c[ei]).max())
        sub_h[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]

    jp = SUBS_DIR / "fusion_v3_held_out.json"
    zp = SUBS_DIR / "fusion_v3_held_out.zip"
    with open(jp, "w") as f: json.dump(sub_h, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"Saved held-out -> {zp}")
else:
    print("No train improvement.")
    print("Generating BM25+cite held-out anyway for codabench test...")
    # Generate the best-looking no-cite BM25 config for held-out testing
    bm25_he = np.load(CACHE_DIR / "bm25_held_scores.npy")
    with open(CACHE_DIR / "bm25_held_query_ids.json") as f: bm25_he_qids = json.load(f)
    bm25_he_qi = {q: i for i, q in enumerate(bm25_he_qids)}
    ft_hq  = np.load(EMB_FT  / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_FT  / "held_out/query_ids.json") as f: ft_hqids = json.load(f)
    bge_hq = np.load(EMB_BGE / "held_out/query_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "held_out/query_ids.json") as f: bge_hqids = json.load(f)
    bge_hqid2i = {q: i for i, q in enumerate(bge_hqids)}
    bge_hc = np.load(EMB_BGE / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "held_out/cite_context_query_ids.json") as f: bge_hcqids = json.load(f)
    e5_hc  = np.load(EMB_E5  / "held_out/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_E5  / "held_out/cite_context_query_ids.json") as f: e5_hcqids = json.load(f)
    q2bghc = defaultdict(list)
    for i, q in enumerate(bge_hcqids): q2bghc[q].append(i)
    q2e5hc = defaultdict(list)
    for i, q in enumerate(e5_hcqids):  q2e5hc[q].append(i)
    held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))

    # Try bm25=0.3+cite on held-out (best train config with BM25)
    for w_bm25, suffix in [(0.3, "bm25=0.3+cite"), (0.4, "bm25=0.4+cite")]:
        sub_h = {}
        for qi, qid in enumerate(ft_hqids):
            s_ft  = ft_c  @ ft_hq[qi]
            s_bge = bge_c_al @ bge_hq[bge_hqid2i[qid]] if qid in bge_hqid2i else s_ft
            bmc   = w_bm25 * np.power(mm(bm25_he[bm25_he_qi[qid]][bm25_to_ft]), 1.2) if qid in bm25_he_qi else 0
            sc    = 0.5*s_ft + 0.5*s_bge + bmc
            qd    = held_doms.get(qid, "")
            if qd in dmasks: sc += 0.3 * dmasks[qd].astype(np.float32)
            rows = q2bghc.get(qid, [])
            if rows:
                qc = bge_hc[rows]
                for cidx in np.argsort(-sc)[:1000]:
                    bi = bge_cid2i.get(ft_cids[cidx])
                    if bi is not None: sc[cidx] += 0.65 * float((qc @ bge_c[bi]).max())
            rows = q2e5hc.get(qid, [])
            if rows:
                qc = e5_hc[rows]
                for cidx in np.argsort(-sc)[:1000]:
                    ei = e5_cid2i.get(ft_cids[cidx])
                    if ei is not None: sc[cidx] += 0.35 * float((qc @ e5_c[ei]).max())
            sub_h[qid] = [ft_cids[i] for i in np.argsort(-sc)[:100]]

        jp = SUBS_DIR / f"fusion_v3_{suffix.replace('=','').replace('+','_')}_held_out.json"
        zp = SUBS_DIR / f"fusion_v3_{suffix.replace('=','').replace('+','_')}_held_out.zip"
        with open(jp, "w") as f: json.dump(sub_h, f)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(jp, arcname="submission_data.json")
        print(f"  Saved {zp.name}")
