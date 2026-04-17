"""
Generate best held_out submission using the current best pipeline:
- dual-dense: ft-small (w=0.5) + bge-large (w=0.5)
- domain boost: w=0.30
- dual cite context: bge-large cite (w=0.65) + e5-large cite (w=0.35)
- cite_top_k=1000

Also test variants with no domain boost, reduced domain boost, etc.
to understand what generalizes to held_out.

Also test scincl addition (tiny improvement on train: 0.7173).
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_FT = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5 = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
EMB_DIR_SC = DATA_DIR / "embeddings" / "malteos_scincl"

print("Loading corpus embeddings ...")
ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)

e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)

sc_corpus_embs = np.load(EMB_DIR_SC / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_SC / "corpus_ids.json") as f: sc_corpus_ids = json.load(f)

# Align to ft order
bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx = {cid: i for i, cid in enumerate(e5_corpus_ids)}
sc_cid_to_idx = {cid: i for i, cid in enumerate(sc_corpus_ids)}

bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]
e5_corpus_aligned = e5_corpus_embs[[e5_cid_to_idx[cid] for cid in ft_corpus_ids]]
sc_corpus_aligned = sc_corpus_embs[[sc_cid_to_idx[cid] for cid in ft_corpus_ids]]

queries_train = load_queries(DATA_DIR / "queries.parquet")
queries_held = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))

train_domains = dict(zip(queries_train["doc_id"], queries_train["domain"].fillna("")))
held_domains = dict(zip(queries_held["doc_id"], queries_held["domain"].fillna("")))

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

def load_query_embeddings(split):
    """Load all query embeddings for a split ('train' or 'held_out')."""
    ft_q_embs = np.load(EMB_DIR_FT / split / "query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_FT / split / "query_ids.json") as f: ft_q_ids = json.load(f)

    bge_q_embs = np.load(EMB_DIR_BGE / split / "query_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_BGE / split / "query_ids.json") as f: bge_q_ids = json.load(f)

    bge_cite_embs = np.load(EMB_DIR_BGE / split / "cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_BGE / split / "cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)

    e5_cite_embs = np.load(EMB_DIR_E5 / split / "cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_DIR_E5 / split / "cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

    bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_q_ids)}
    qid_to_bge_cite = defaultdict(list)
    for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
    qid_to_e5_cite = defaultdict(list)
    for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

    return {
        "ft_q_embs": ft_q_embs, "ft_q_ids": ft_q_ids,
        "bge_q_embs": bge_q_embs, "bge_qid_to_idx": bge_qid_to_idx,
        "bge_cite_embs": bge_cite_embs, "qid_to_bge_cite": qid_to_bge_cite,
        "e5_cite_embs": e5_cite_embs, "qid_to_e5_cite": qid_to_e5_cite,
    }


def retrieve(embs, query_domains_map, w_domain, w_bge_cite, w_e5_cite, w_sc=0.0, cite_top_k=1000, top_k=100):
    ft_q_embs = embs["ft_q_embs"]
    ft_q_ids = embs["ft_q_ids"]
    bge_q_embs = embs["bge_q_embs"]
    bge_qid_to_idx = embs["bge_qid_to_idx"]
    bge_cite_embs = embs["bge_cite_embs"]
    qid_to_bge_cite = embs["qid_to_bge_cite"]
    e5_cite_embs = embs["e5_cite_embs"]
    qid_to_e5_cite = embs["qid_to_e5_cite"]

    sub = {}
    for qidx, qid in enumerate(ft_q_ids):
        q_ft = ft_q_embs[qidx]
        q_domain = query_domains_map.get(qid, "")
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        e5_cite_rows = qid_to_e5_cite.get(qid, [])

        ft_scores = ft_corpus_embs @ q_ft

        q_bge_idx = bge_qid_to_idx.get(qid)
        if q_bge_idx is not None:
            bge_scores = bge_corpus_aligned @ bge_q_embs[q_bge_idx]
            scores = 0.5 * ft_scores + 0.5 * bge_scores
        else:
            scores = ft_scores.copy()

        if w_sc > 0:
            # scincl doesn't have held_out query embeddings, skip for held_out
            pass

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


print("Loading train query embeddings ...")
train_embs = load_query_embeddings("train")
print("Loading held_out query embeddings ...")
held_embs = load_query_embeddings("held_out")

print()
print("%-65s  Train NDCG@10" % "Config")
print("-" * 85)

best_train_ndcg = 0.0
best_held_sub = None
best_name = ""

configs = [
    # name, w_domain, w_bge_cite, w_e5_cite
    ("baseline wd=0.30 wbc=0.65 we5c=0.35",           0.30, 0.65, 0.35),
    ("wd=0.20 wbc=0.65 we5c=0.35",                    0.20, 0.65, 0.35),
    ("wd=0.10 wbc=0.65 we5c=0.35",                    0.10, 0.65, 0.35),
    ("wd=0.0 wbc=0.65 we5c=0.35 (no domain)",         0.00, 0.65, 0.35),
    ("wd=0.30 wbc=1.0 we5c=0.0 (bge cite only)",      0.30, 1.0,  0.0),
    ("wd=0.30 wbc=0.0 we5c=1.0 (e5 cite only)",       0.30, 0.0,  1.0),
    ("wd=0.30 wbc=0.5 we5c=0.5",                      0.30, 0.5,  0.5),
    ("wd=0.30 wbc=0.0 we5c=0.0 (no cite)",            0.30, 0.0,  0.0),
    ("wd=0.0 wbc=0.0 we5c=0.0 (pure dual-dense)",     0.00, 0.0,  0.0),
]

for name, wd, wbc, we5c in configs:
    train_sub = retrieve(train_embs, train_domains, wd, wbc, we5c)
    res = evaluate(train_sub, qrels, ks=[10, 100], query_domains=train_domains, verbose=False)
    ndcg = res["overall"]["NDCG@10"]
    rec = res["overall"]["Recall@100"]
    print("%-65s  %.4f  R@100=%.4f" % (name[:65], ndcg, rec))
    if ndcg > best_train_ndcg:
        best_train_ndcg = ndcg
        best_name = name
        # Generate held_out for this config
        best_held_sub = retrieve(held_embs, held_domains, wd, wbc, we5c)

print()
print(f"Best train config: {best_name} -> {best_train_ndcg:.4f}")

# Also generate held_out for the exact current best config
print("\nGenerating held_out submissions for all configs ...")
for name, wd, wbc, we5c in configs:
    held_sub = retrieve(held_embs, held_domains, wd, wbc, we5c)
    safe_name = name.replace(" ", "_").replace("=", "").replace(".", "p").replace("(", "").replace(")", "").replace("/", "_")[:50]
    out_json = SUBMISSIONS_DIR / f"held_{safe_name}.json"
    out_zip = SUBMISSIONS_DIR / f"held_{safe_name}.zip"
    with open(out_json, "w") as f:
        json.dump(held_sub, f)
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(out_json, arcname="submission_data.json")
    print(f"  Saved {out_zip.name}")
