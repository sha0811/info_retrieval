"""
Hard-domain 7-signal retrieval (Stage 1 of the final pipeline).

Signals combined by weighted sum of min-max-normalized columns, with hard-domain
filter (top dom_k in same domain, then fill to total_k from global ranking):
  1. MiniLM cosine sim
  2. BGE-large cosine sim
  3. BM25 body (reciprocal rank)
  4. TF-IDF TA
  5. TF-IDF full-text (10k features)
  6. GTE-ModernBERT full-text @8192 cosine sim
  7. GTE-large cMaxSim (query TA vs 213k body chunks, 1024-dim)

Same start as the winning competitors solution, a
Dirichlet search (--dirichlet N) re-tunes the weights.

Usage:
    python models/hard_domain_retrieval.py --split train
    python models/hard_domain_retrieval.py --split train --dirichlet 5000
    python models/hard_domain_retrieval.py --split held_out --zip
"""

import argparse
import json
import sys
import time
import zipfile
from collections import defaultdict
from itertools import product as ip
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR        = Path(__file__).parent.parent / "data"
EMB_DIR         = DATA_DIR / "embeddings"
BM25_DIR        = DATA_DIR / "bm25_cache"
TFIDF_DIR       = DATA_DIR / "tfidf_cache"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR     = Path(__file__).parent.parent / "results"

EMB_BGE  = EMB_DIR / "BAAI_bge-large-en-v1.5"
EMB_MINI = EMB_DIR / "sentence-transformers_all-MiniLM-L6-v2"
EMB_GMB  = EMB_DIR / "gte_modernbert_fulltext_8k"
EMB_GTEL = EMB_DIR / "gte_large_chunks"
EMB_E5   = EMB_DIR / "intfloat_e5-large-v2"

#weights
COMP_W_MAP = {"minilm": 0.092, "bge": 0.061, "bm25_body": 0.084,
              "tfidf_ta": 0.045, "tfidf_ft": 0.121, "gte_mb_ft": 0.539,
              "gte_cmaxsim": 0.059}

DEFAULT_DOM_K = 90
DEFAULT_TOTAL_K = 100


def mm(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)


def load_signals(split):
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_ids = corpus["doc_id"].tolist()
    corp_domains = corpus["domain"].tolist()
    n_docs = len(corpus_ids)

    qfile = DATA_DIR / ("queries.parquet" if split == "train" else "held_out_queries.parquet")
    qdf = load_queries(qfile)
    query_ids = qdf["doc_id"].tolist()
    query_domains = qdf["domain"].tolist()
    n_q = len(query_ids)

    dom_to_cidx = defaultdict(list)
    for i, d in enumerate(corp_domains):
        dom_to_cidx[d].append(i)
    dom_to_cidx = {k: np.array(v) for k, v in dom_to_cidx.items()}

    print(f"Split={split}  corpus={n_docs}  queries={n_q}", flush=True)
    sigs = {}

    def align_corpus(embs, eids):
        id2i = {c: i for i, c in enumerate(eids)}
        return embs[[id2i[c] for c in corpus_ids]]

    def align_query(embs, eids):
        id2i = {q: i for i, q in enumerate(eids)}
        return np.array([embs[id2i[q]] if q in id2i else np.zeros(embs.shape[1], np.float32)
                         for q in query_ids])

    def load_emb(path_stem, sub=""):
        base = Path(path_stem)
        emb = np.load(base / ("corpus_embeddings.npy" if not sub else f"{sub}/query_embeddings.npy")).astype(np.float32)
        with open(base / ("corpus_ids.json" if not sub else f"{sub}/query_ids.json")) as f:
            ids = json.load(f)
        return emb, ids

    # MiniLM signal
    mini_q_path = EMB_MINI / f"{split}" / "query_embeddings.npy"
    if mini_q_path.exists():
        mini_c, mini_cids = load_emb(EMB_MINI)
        mini_q, mini_qids = load_emb(EMB_MINI, split)
        sigs["minilm"] = (align_corpus(mini_c, mini_cids) @ align_query(mini_q, mini_qids).T).astype(np.float32)
        print(f"  minilm: {sigs['minilm'].shape}", flush=True)
    else:
        print(f"  minilm: SKIP", flush=True)

    # BGE-large
    bge_c, bge_cids = load_emb(EMB_BGE)
    bge_q, bge_qids = load_emb(EMB_BGE, split)
    bge_c_al = align_corpus(bge_c, bge_cids)
    bge_q_al = align_query(bge_q, bge_qids)
    sigs["bge"] = (bge_c_al @ bge_q_al.T).astype(np.float32)
    print(f"  bge_large: {sigs['bge'].shape}", flush=True)

    # BM25 body (reciprocal-rank)
    bm25_key = "train" if split == "train" else "held"
    bm25_body_raw = np.load(BM25_DIR / f"bm25_body_{bm25_key}_scores.npy").astype(np.float32)
    with open(BM25_DIR / f"bm25_body_{bm25_key}_query_ids.json") as f: bm25_body_qids = json.load(f)
    with open(BM25_DIR / "bm25_body_corpus_ids.json") as f: bm25_body_cids = json.load(f)
    bm25b_cid2i = {c: i for i, c in enumerate(bm25_body_cids)}
    bm25b_qi = {q: i for i, q in enumerate(bm25_body_qids)}
    corpus_to_bm25b = np.array([bm25b_cid2i.get(c, 0) for c in corpus_ids])
    bm25_body_mat = np.zeros((n_docs, n_q), np.float32)
    for qi, qid in enumerate(query_ids):
        if qid in bm25b_qi:
            raw = bm25_body_raw[corpus_to_bm25b, bm25b_qi[qid]]
            order = np.argsort(-raw)
            rr = np.zeros(n_docs, np.float32)
            rr[order] = 1.0 / (np.arange(n_docs) + 1)
            bm25_body_mat[:, qi] = rr
    sigs["bm25_body"] = bm25_body_mat
    print(f"  bm25_body (RR): {sigs['bm25_body'].shape}", flush=True)

    # TF-IDF TA
    ta_key = split if split == "train" else "held"
    tfidf_ta = np.load(TFIDF_DIR / f"tfidf_ta_{ta_key}_scores.npy").T
    with open(TFIDF_DIR / "tfidf_ta_corpus_ids.json") as f: ta_cids = json.load(f)
    ta_cid2i = {c: i for i, c in enumerate(ta_cids)}
    ta_to_corpus = np.array([ta_cid2i.get(c, 0) for c in corpus_ids])
    sigs["tfidf_ta"] = tfidf_ta[ta_to_corpus, :].astype(np.float32)
    print(f"  tfidf_ta: {sigs['tfidf_ta'].shape}", flush=True)

    # TF-IDF FT
    tfidf_ft = np.load(TFIDF_DIR / f"tfidf_ft10k_{ta_key}_scores.npy").T
    with open(TFIDF_DIR / "tfidf_ft10k_corpus_ids.json") as f: ft_cids2 = json.load(f)
    ft10k_cid2i = {c: i for i, c in enumerate(ft_cids2)}
    ft_to_corpus = np.array([ft10k_cid2i.get(c, 0) for c in corpus_ids])
    sigs["tfidf_ft"] = tfidf_ft[ft_to_corpus, :].astype(np.float32)
    print(f"  tfidf_ft: {sigs['tfidf_ft'].shape}", flush=True)

    # GTE-ModernBERT @8192
    gmb_c, gmb_cids = load_emb(EMB_GMB)
    gmb_q, gmb_qids = load_emb(EMB_GMB, split)
    sigs["gte_mb_ft"] = (align_corpus(gmb_c, gmb_cids) @ align_query(gmb_q, gmb_qids).T).astype(np.float32)
    print(f"  gte_mb_ft_8k: {sigs['gte_mb_ft'].shape}", flush=True)

    # GTE-large cMaxSim (query TA vs body chunks)
    gtel_chunks = np.load(EMB_GTEL / "corpus_chunk_embeddings.npy").astype(np.float32)
    with open(EMB_GTEL / "corpus_chunk_doc_ids.json") as f: gtel_doc_ids = json.load(f)
    gtel_q = np.load(EMB_GTEL / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_GTEL / f"{split}/query_ids.json") as f: gtel_qids = json.load(f)
    doc2chunks_gtel = defaultdict(list)
    for ri, did in enumerate(gtel_doc_ids):
        doc2chunks_gtel[did].append(ri)
    gtel_qi = {q: i for i, q in enumerate(gtel_qids)}
    gtel_q_al = np.array([gtel_q[gtel_qi[q]] if q in gtel_qi else np.zeros(gtel_q.shape[1], np.float32)
                          for q in query_ids])
    print(f"  gte_large chunks: {gtel_chunks.shape}, computing cMaxSim...", flush=True)
    t0 = time.time()
    sim_all = gtel_q_al @ gtel_chunks.T
    ms = np.zeros((n_docs, n_q), np.float32)
    for ci, did in enumerate(corpus_ids):
        rows = doc2chunks_gtel.get(did, [])
        if rows:
            ms[ci, :] = sim_all[:, rows].max(axis=1)
    sigs["gte_cmaxsim"] = ms
    print(f"  gte_cmaxsim: {ms.shape}  in {time.time()-t0:.1f}s", flush=True)

    # Cite context (BGE + E5) for optional re-scoring
    cite = {}
    bge_cite_embs = np.load(EMB_BGE / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
    q2bc = defaultdict(list)
    for ri, qid in enumerate(bge_cite_qids): q2bc[qid].append(ri)
    bge_cid2i = {c: i for i, c in enumerate(bge_cids)}

    e5_c, e5_cids = load_emb(EMB_E5)
    e5_cite_embs = np.load(EMB_E5 / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_E5 / f"{split}/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)
    q2ec = defaultdict(list)
    for ri, qid in enumerate(e5_cite_qids): q2ec[qid].append(ri)
    e5_cid2i = {c: i for i, c in enumerate(e5_cids)}

    cite.update(dict(bge_cite=bge_cite_embs, q2bc=dict(q2bc), bge_c=bge_c, bge_cid2i=bge_cid2i,
                     e5_cite=e5_cite_embs, q2ec=dict(q2ec), e5_c=e5_c, e5_cid2i=e5_cid2i))

    meta = dict(corpus_ids=corpus_ids, query_ids=query_ids,
                query_domains=query_domains, dom_to_cidx=dom_to_cidx,
                n_docs=n_docs, n_q=n_q)
    print(f"All signals loaded: {list(sigs.keys())}", flush=True)
    return sigs, cite, meta


def retrieve(sigs, cite, meta, weights,
             dom_k=DEFAULT_DOM_K, total_k=DEFAULT_TOTAL_K,
             w_bc=0.0, w_ec=0.0):
    corpus_ids    = meta["corpus_ids"]
    query_ids     = meta["query_ids"]
    query_domains = meta["query_domains"]
    dom_to_cidx   = meta["dom_to_cidx"]
    n_docs        = meta["n_docs"]
    sig_list = list(sigs.values())
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()

    result = {}
    for qi, (qid, qdom) in enumerate(zip(query_ids, query_domains)):
        dom_idx = dom_to_cidx.get(qdom, np.arange(n_docs))

        scores_dom = np.zeros(len(dom_idx), np.float32)
        for arr, ww in zip(sig_list, w):
            col = arr[dom_idx, qi]
            scores_dom += ww * mm(col)

        top_dom = dom_idx[np.argsort(-scores_dom)[:min(dom_k, len(dom_idx))]]

        if len(top_dom) < total_k:
            scores_global = np.zeros(n_docs, np.float32)
            for arr, ww in zip(sig_list, w):
                scores_global += ww * mm(arr[:, qi])
            seen  = set(top_dom.tolist())
            extra = [j for j in np.argsort(-scores_global) if j not in seen][:total_k - len(top_dom)]
            cand_idx = list(top_dom) + extra
        else:
            cand_idx = list(top_dom[:total_k])

        if w_bc > 0 or w_ec > 0:
            sc = np.zeros(len(cand_idx), np.float32)
            for arr, ww in zip(sig_list, w):
                col_full = arr[:, qi]
                lo, hi = col_full.min(), col_full.max()
                if hi > lo:
                    sc += ww * (col_full[cand_idx] - lo) / (hi - lo)

            rows_bc = cite["q2bc"].get(qid, [])
            rows_ec = cite["q2ec"].get(qid, [])
            if rows_bc and w_bc > 0:
                qc_bge = cite["bge_cite"][rows_bc]
                for k_ci, ci in enumerate(cand_idx):
                    doc_id = corpus_ids[ci]
                    bi = cite["bge_cid2i"].get(doc_id)
                    if bi is not None:
                        sc[k_ci] += w_bc * float((qc_bge @ cite["bge_c"][bi]).max())
            if rows_ec and w_ec > 0:
                qc_e5 = cite["e5_cite"][rows_ec]
                for k_ci, ci in enumerate(cand_idx):
                    doc_id = corpus_ids[ci]
                    ei = cite["e5_cid2i"].get(doc_id)
                    if ei is not None:
                        sc[k_ci] += w_ec * float((qc_e5 @ cite["e5_c"][ei]).max())

            reranked = np.argsort(-sc)[:total_k]
            result[qid] = [corpus_ids[cand_idx[k]] for k in reranked]
        else:
            result[qid] = [corpus_ids[cand_idx[k]] for k in range(min(total_k, len(cand_idx)))]

    return result


def _ndcg10(ranked, relevant):
    dcg  = sum(1/np.log2(r+1) for r, d in enumerate(ranked[:10], 1) if d in relevant)
    idcg = sum(1/np.log2(r+1) for r in range(1, min(len(relevant), 10)+1))
    return dcg / idcg if idcg else 0.0


def _fast_eval(sub, query_ids, qrels_dict):
    scores = [_ndcg10(sub.get(qid, []), qrels_dict.get(qid, set()))
              for qid in query_ids if qid in qrels_dict]
    return float(np.mean(scores))


def dirichlet_search(sigs, cite, meta, qrels, n_samples=5000, dom_k=DEFAULT_DOM_K):
    n_sigs = len(sigs)
    sig_names = list(sigs.keys())
    rng = np.random.default_rng(42)
    qrels_dict = {q: set(v) for q, v in qrels.items()}
    samples = rng.dirichlet(np.ones(n_sigs), size=n_samples)
    best_score, best_w = 0.0, None
    t0 = time.time()
    print(f"Dirichlet search: {n_samples} samples  signals={sig_names}", flush=True)
    for si, ws in enumerate(samples):
        sub = retrieve(sigs, cite, meta, ws, dom_k=dom_k)
        sc = _fast_eval(sub, meta["query_ids"], qrels_dict)
        if sc > best_score:
            best_score, best_w = sc, ws.copy()
        if (si + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (si+1) * (n_samples - si - 1)
            pct = 100 * (si+1) / n_samples
            print(f"  [{pct:.0f}%] [{si+1}/{n_samples}] best={best_score:.4f}  "
                  f"ETA={eta:.0f}s  {dict(zip(sig_names, best_w.round(3)))}", flush=True)
    print("Fine-tuning around best...", flush=True)
    delta = 0.04
    for combo in ip(*[[max(0, best_w[i]-delta), best_w[i], min(1, best_w[i]+delta)]
                      for i in range(n_sigs)]):
        ws = np.array(combo, np.float32)
        if ws.sum() < 1e-6: continue
        ws /= ws.sum()
        sub = retrieve(sigs, cite, meta, ws, dom_k=dom_k)
        sc = _fast_eval(sub, meta["query_ids"], qrels_dict)
        if sc > best_score:
            best_score, best_w = sc, ws.copy()
    print(f"Best: {best_score:.4f}  {dict(zip(sig_names, best_w.round(3)))}", flush=True)
    return best_w, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOTAL_K)
    parser.add_argument("--dom-k", type=int, default=DEFAULT_DOM_K)
    parser.add_argument("--dirichlet", type=int, default=0,
                        help="If >0 and split=train, run Dirichlet weight search with N samples")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sigs, cite, meta = load_signals(args.split)
    sig_names = list(sigs.keys())

    # Determine weights
    if args.split == "train" and args.dirichlet > 0:
        qrels = load_qrels(DATA_DIR / "qrels.json")
        best_w, _ = dirichlet_search(sigs, cite, meta, qrels,
                                     n_samples=args.dirichlet, dom_k=args.dom_k)
        weights = best_w
        weight_source = f"dirichlet_{args.dirichlet}"
    else:
        weights = np.array([COMP_W_MAP.get(s, 0.05) for s in sig_names], np.float32)
        weights /= weights.sum()
        weight_source = "competitor_best"

    submission = retrieve(sigs, cite, meta, weights,
                          dom_k=args.dom_k, total_k=args.top_k)

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        qdmap = dict(zip(meta["query_ids"], meta["query_domains"]))
        results = evaluate(submission, qrels, ks=[10], query_domains=qdmap, verbose=True)
        hp = {f"w_{s}": round(float(weights[i]), 4) for i, s in enumerate(sig_names)}
        hp.update({"dom_k": args.dom_k, "top_k": args.top_k,
                   "weight_source": weight_source, "split": args.split})
        save_results(results, RESULTS_DIR / "hard_domain_retrieval.csv", hyperparameters=hp)

    name = args.output or f"hard_domain_retrieval_{args.split}"
    out_path = SUBMISSIONS_DIR / (name if name.endswith(".json") else f"{name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved -> {out_path}")

    if args.zip:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, "submission_data.json")
        print(f"Zipped -> {zip_path}")


if __name__ == "__main__":
    main()
