"""
Final held_out submission combining best signals.
Also generates a variant with body chunk multi-vector (higher R@100).
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_FT = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
EMB_DIR_E5 = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"

ft_corpus_embs = np.load(EMB_DIR_FT / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "corpus_ids.json") as f: ft_corpus_ids = json.load(f)

bge_corpus_embs = np.load(EMB_DIR_BGE / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "corpus_ids.json") as f: bge_corpus_ids = json.load(f)

e5_corpus_embs = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_corpus_ids = json.load(f)

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
held_out = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
corpus_domain_map = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
held_domains = dict(zip(held_out["doc_id"], held_out["domain"].fillna("")))

bge_cid_to_idx = {cid: i for i, cid in enumerate(bge_corpus_ids)}
e5_cid_to_idx = {cid: i for i, cid in enumerate(e5_corpus_ids)}
bge_corpus_aligned = bge_corpus_embs[[bge_cid_to_idx[cid] for cid in ft_corpus_ids]]

corpus_domains_arr = np.array([corpus_domain_map.get(cid, "") for cid in ft_corpus_ids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

print("Loading held_out embeddings...")
ft_q_embs = np.load(EMB_DIR_FT / "held_out/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_FT / "held_out/query_ids.json") as f: ft_q_ids = json.load(f)

bge_q_embs = np.load(EMB_DIR_BGE / "held_out/query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "held_out/query_ids.json") as f: bge_q_ids = json.load(f)

bge_cite_embs = np.load(EMB_DIR_BGE / "held_out/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_BGE / "held_out/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)

e5_cite_embs = np.load(EMB_DIR_E5 / "held_out/cite_context_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "held_out/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)

bge_qid_to_idx = {qid: i for i, qid in enumerate(bge_q_ids)}
qid_to_bge_cite = defaultdict(list)
for idx, qid in enumerate(bge_cite_qids): qid_to_bge_cite[qid].append(idx)
qid_to_e5_cite = defaultdict(list)
for idx, qid in enumerate(e5_cite_qids): qid_to_e5_cite[qid].append(idx)

# Check if held_out has chunk embeddings (query body chunks)
held_chunk_path = EMB_DIR_BGE / "held_out" / "chunk_embeddings.npy"
if held_chunk_path.exists():
    print("Loading held_out chunk embeddings...")
    held_chunk_embs = np.load(held_chunk_path).astype(np.float32)
    with open(EMB_DIR_BGE / "held_out" / "chunk_query_ids.json") as f:
        held_chunk_qids = json.load(f)
    qid_to_held_chunks = defaultdict(list)
    for idx, qid in enumerate(held_chunk_qids): qid_to_held_chunks[qid].append(idx)
    print(f"  chunks: {held_chunk_embs.shape}, {len(set(held_chunk_qids))} queries")
else:
    print("No held_out chunk embeddings found.")
    held_chunk_embs = None
    qid_to_held_chunks = defaultdict(list)


def retrieve_held(w_domain, w_bge_cite, w_e5_cite, w_chunk=0.0, chunk_topk=500, cite_top_k=1000):
    sub = {}
    for qidx, qid in enumerate(ft_q_ids):
        q_ft = ft_q_embs[qidx]
        q_domain = held_domains.get(qid, "")
        bge_cite_rows = qid_to_bge_cite.get(qid, [])
        e5_cite_rows = qid_to_e5_cite.get(qid, [])

        ft_scores = ft_corpus_embs @ q_ft
        q_bge_idx = bge_qid_to_idx.get(qid)
        if q_bge_idx is not None:
            scores = 0.5 * ft_scores + 0.5 * (bge_corpus_aligned @ bge_q_embs[q_bge_idx])
        else:
            scores = ft_scores.copy()

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

        if w_chunk > 0 and held_chunk_embs is not None:
            chunk_rows = qid_to_held_chunks.get(qid, [])
            if chunk_rows:
                q_chunks = held_chunk_embs[chunk_rows]
                top_cands = np.argsort(-scores)[:chunk_topk]
                valid_cands = [c for c in top_cands if ft_corpus_ids[c] in bge_cid_to_idx]
                if valid_cands:
                    cand_embs = np.array([bge_corpus_embs[bge_cid_to_idx[ft_corpus_ids[c]]] for c in valid_cands])
                    mv_scores = (q_chunks @ cand_embs.T).max(axis=0)
                    for j, cidx in enumerate(valid_cands):
                        scores[cidx] += w_chunk * mv_scores[j]

        top_idx = np.argsort(-scores)[:100]
        sub[qid] = [ft_corpus_ids[i] for i in top_idx]
    return sub


print("\nGenerating final held_out submissions...")
configs = [
    ("final_best_wd030_wbc065_we5c035", 0.30, 0.65, 0.35, 0.0, 500),
]

for name, wd, wbc, we5c, wc, ctk in configs:
    print(f"  {name} ...", end=" ", flush=True)
    sub = retrieve_held(wd, wbc, we5c, wc, ctk)

    json_path = SUBMISSIONS_DIR / f"{name}.json"
    zip_path = SUBMISSIONS_DIR / f"{name}.zip"
    with open(json_path, "w") as f:
        json.dump(sub, f)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="submission_data.json")
    print(f"saved")

print("\nDone!")
