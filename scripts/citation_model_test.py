"""
Test citation-specialized models (specter2_base, scincl) as additional dense signals.

specter2_base has BOTH train and held_out query embeddings — key for breaking the 0.69 plateau.
scincl has train only.

Best so far: dual-dense (ftsmall+bge-large) + dual-cite (bge+e5) -> NDCG@10=0.7170 train
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
EMB_DIR_SP2 = DATA_DIR / "embeddings" / "allenai_specter2_base"
EMB_DIR_SC = DATA_DIR / "embeddings" / "malteos_scincl"

print("Loading embeddings ...")
ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)
ft_query_embs = np.load(EMB_DIR_FT / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "train/query_ids.json") as f: ft_query_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)
bge_query_embs = np.load(EMB_DIR_BGE / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/query_ids.json") as f: bge_query_ids = json.load(f)

e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)

bge_cite_embs = np.load(EMB_DIR_BGE / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "train/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
e5_cite_embs = np.load(EMB_DIR_E5 / "train/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "train/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

sp2_corpus_embs = np.load(EMB_DIR_SP2 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SP2 / "corpus_ids.json") as f: sp2_corpus_ids = json.load(f)
sp2_query_embs = np.load(EMB_DIR_SP2 / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SP2 / "train/query_ids.json") as f: sp2_query_ids = json.load(f)

sc_corpus_embs = np.load(EMB_DIR_SC / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SC / "corpus_ids.json") as f: sc_corpus_ids = json.load(f)
sc_query_embs = np.load(EMB_DIR_SC / "train/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SC / "train/query_ids.json") as f: sc_query_ids = json.load(f)

queries = load_queries(DATA_DIR / "queries.parquet")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries["doc_id"], queries["domain"].fillna("")))
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

# Align all corpus embeddings to ft_corpus_ids order
bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx = {cid: i for i, cid in enumerate(e5_corpus_ids)}
sp2_cid_to_idx = {cid: i for i, cid in enumerate(sp2_corpus_ids)}
sc_cid_to_idx = {cid: i for i, cid in enumerate(sc_corpus_ids)}

bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]
e5_corpus_aligned = e5_corpus_embs[[e5_cid_to_idx[cid] for cid in ft_corpus_ids]]
sp2_corpus_aligned = sp2_corpus_embs[[sp2_cid_to_idx[cid] for cid in ft_corpus_ids]]
sc_corpus_aligned = sc_corpus_embs[[sc_cid_to_idx[cid] for cid in ft_corpus_ids]]

bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_query_ids)}
sp2_qid_to_idx = {qid: i for i, qid in enumerate(sp2_query_ids)}
sc_qid_to_idx = {qid: i for i, qid in enumerate(sc_query_ids)}

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

qid_to_bge_cite = defaultdict(list)
for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
qid_to_e5_cite = defaultdict(list)
for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

print(f"ft_corpus: {ft_corpus_embs.shape}")
print(f"sp2_corpus: {sp2_corpus_embs.shape}, sp2_queries: {sp2_query_embs.shape}")
print(f"scincl_corpus: {sc_corpus_embs.shape}, scincl_queries: {sc_query_embs.shape}")


def retrieve(w_ft, w_bge, w_sp2, w_sc, w_domain, w_bge_cite, w_e5_cite, cite_top_k=1000, top_k=100):
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        q_ft = ft_query_embs[qidx]
        q_domain = query_domains.get(qid, "")
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        e5_cite_rows = qid_to_e5_cite.get(qid, [])

        scores = w_ft * (ft_corpus_aligned if False else ft_corpus_embs @ q_ft)
        # Use pre-aligned for all corpus models
        scores = w_ft * (ft_corpus_embs @ q_ft)

        q_bge_idx = bge_qid_to_idx.get(qid)
        if q_bge_idx is not None and w_bge > 0:
            scores = scores + w_bge * (bge_corpus_aligned @ bge_query_embs[q_bge_idx])

        q_sp2_idx = sp2_qid_to_idx.get(qid)
        if q_sp2_idx is not None and w_sp2 > 0:
            scores = scores + w_sp2 * (sp2_corpus_aligned @ sp2_query_embs[q_sp2_idx])

        q_sc_idx = sc_qid_to_idx.get(qid)
        if q_sc_idx is not None and w_sc > 0:
            scores = scores + w_sc * (sc_corpus_aligned @ sc_query_embs[q_sc_idx])

        if w_domain > 0 and q_domain in domain_masks:
            scores = scores + w_domain * domain_masks[q_domain].astype(np.float32)

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

        top_idx = np.argsort(-scores)[:top_k]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


# First: standalone specter2 and scincl performance
print("\n--- Standalone tests ---")
def retrieve_single(corpus_aligned, query_embs, qid_to_idx, corpus_ids_list, w_domain=0.0):
    sub = {}
    for qidx, qid in enumerate(ft_query_ids):
        qi = qid_to_idx.get(qid)
        if qi is None:
            continue
        scores = corpus_aligned @ query_embs[qi]
        if w_domain > 0:
            q_domain = query_domains.get(qid, "")
            if q_domain in domain_masks:
                scores = scores + w_domain * domain_masks[q_domain].astype(np.float32)
        top_idx = np.argsort(-scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub

for name, corpus_al, query_e, qid_map in [
    ("specter2_base", sp2_corpus_aligned, sp2_query_embs, sp2_qid_to_idx),
    ("scincl", sc_corpus_aligned, sc_query_embs, sc_qid_to_idx),
]:
    sub = retrieve_single(corpus_al, query_e, qid_map, ft_corpus_ids)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    print(f"  {name}: NDCG@10={res['overall']['NDCG@10']:.4f} R@100={res['overall']['Recall@100']:.4f}")

print()
print("%-85s  NDCG@10  R@100" % "Config")
print("-" * 110)

best_ndcg = 0.7170
best_sub = None
best_name = ""

combos = [
    # name, w_ft, w_bge, w_sp2, w_sc, w_domain, w_bge_cite, w_e5_cite
    # Baseline: current best (ft=0.5, bge=0.5, no sp2/sc)
    ("baseline: ft=0.5 bge=0.5 wd=0.30 wbc=0.65 we5c=0.35",      0.5, 0.5, 0.0, 0.0, 0.30, 0.65, 0.35),
    # Add specter2 lightly
    ("ft=0.5 bge=0.5 sp2=0.1 wd=0.30 wbc=0.65 we5c=0.35",        0.5, 0.5, 0.1, 0.0, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sp2=0.2 wd=0.30 wbc=0.65 we5c=0.35",        0.5, 0.5, 0.2, 0.0, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sp2=0.3 wd=0.30 wbc=0.65 we5c=0.35",        0.5, 0.5, 0.3, 0.0, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sp2=0.5 wd=0.30 wbc=0.65 we5c=0.35",        0.5, 0.5, 0.5, 0.0, 0.30, 0.65, 0.35),
    # Add scincl lightly
    ("ft=0.5 bge=0.5 sc=0.1 wd=0.30 wbc=0.65 we5c=0.35",         0.5, 0.5, 0.0, 0.1, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sc=0.2 wd=0.30 wbc=0.65 we5c=0.35",         0.5, 0.5, 0.0, 0.2, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sc=0.3 wd=0.30 wbc=0.65 we5c=0.35",         0.5, 0.5, 0.0, 0.3, 0.30, 0.65, 0.35),
    # sp2 + sc together
    ("ft=0.5 bge=0.5 sp2=0.2 sc=0.1 wd=0.30 wbc=0.65 we5c=0.35", 0.5, 0.5, 0.2, 0.1, 0.30, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sp2=0.2 sc=0.2 wd=0.30 wbc=0.65 we5c=0.35", 0.5, 0.5, 0.2, 0.2, 0.30, 0.65, 0.35),
    # sp2-dominant (it generalizes better)
    ("ft=0.4 bge=0.4 sp2=0.4 wd=0.30 wbc=0.65 we5c=0.35",        0.4, 0.4, 0.4, 0.0, 0.30, 0.65, 0.35),
    ("ft=0.3 bge=0.3 sp2=0.5 wd=0.30 wbc=0.65 we5c=0.35",        0.3, 0.3, 0.5, 0.0, 0.30, 0.65, 0.35),
    # sp2 only + domain + cite
    ("ft=0.5 sp2=0.5 wd=0.30 wbc=0.65 we5c=0.35",                0.5, 0.0, 0.5, 0.0, 0.30, 0.65, 0.35),
    # sp2 no domain (check if domain hurts generalization)
    ("ft=0.5 bge=0.5 sp2=0.2 wd=0.0 wbc=0.65 we5c=0.35",         0.5, 0.5, 0.2, 0.0, 0.00, 0.65, 0.35),
    ("ft=0.5 bge=0.5 sp2=0.2 wd=0.15 wbc=0.65 we5c=0.35",        0.5, 0.5, 0.2, 0.0, 0.15, 0.65, 0.35),
]

for cfg in combos:
    name, w_ft, w_bge, w_sp2, w_sc, w_domain, w_bge_c, w_e5_c = cfg
    sub = retrieve(w_ft, w_bge, w_sp2, w_sc, w_domain, w_bge_c, w_e5_c)
    res = evaluate(sub, qrels, ks=[10, 100], query_domains=query_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    marker = " <-- BEST" if ndcg > best_ndcg else ""
    print("%-85s  %.4f   %.4f%s" % (name[:85], ndcg, rec, marker))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_name = name
        best_sub = sub

print()
if best_sub:
    print("NEW BEST: %s -> %.4f" % (best_name, best_ndcg))
    out_path = SUBMISSIONS_DIR / "citation_model_fusion_train.json"
    with open(out_path, "w") as f:
        json.dump(best_sub, f)
    print(f"Saved -> {out_path}")
else:
    print("No improvement over 0.7170.")
