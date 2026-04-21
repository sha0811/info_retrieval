"""
Hard-domain 7-signal pipeline + cite-context re-ranking on top candidates.

The previous pipeline (hard_domain_retrieval.py) reaches around 0.74 without cite-context. This model adds
cite-context re-scoring on the hard-domain candidate pool (0.7552 on train).

Pipeline:
  1. Hard-domain candidate selection: restrict to same-domain documents, score them with
     7 signals (MiniLM, BGE-large, BM25, TF-IDF-TA, TF-IDF-FT, GTE-ModernBERT-FT,
     GTE-cMaxSim), take the top dom_k. Fill up to cand_n from the global ranking if
     there are too few in-domain docs.
  2. Cite-context re-ranking: the query paper's citation sentences (extracted from its
     full text) are encoded and matched against each candidate. The final score is:

       score(q, d) = 1/(rank+1)  --> reciprocal rank of the first step
                   + w_bc * mm(max_i cosine(bge_cite_i(q), bge_large(d)))
                   + w_ec * mm(max_i cosine(e5_cite_i(q),  e5_large(d)))
                   + w_cc * mm(max_i cosine(bge_cite_i(q), bge_chunks(d)))

     where mm() is per-query min-max normalisation over the candidate pool,
     bge_cite_i / e5_cite_i are the per-sentence cite-context embeddings, and
     bge_chunks are BGE embeddings of the document's body chunks.

Usage:
    python models/hard_pipeline_with_cite.py --split train
    python models/hard_pipeline_with_cite.py --split train --sweep
    python models/hard_pipeline_with_cite.py --split held_out --w-bc 0.5 --w-ec 0.2 --w-cc 0.05 --zip
"""

import argparse
import json
import sys
import time
import zipfile
from collections import defaultdict
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
EMB_E5   = EMB_DIR / "intfloat_e5-large-v2"
EMB_MINI = EMB_DIR / "sentence-transformers_all-MiniLM-L6-v2"
EMB_GMB  = EMB_DIR / "gte_modernbert_fulltext_8k"
EMB_GTEL = EMB_DIR / "gte_large_chunks"

PROGRESS_FILE = Path(__file__).parent / "hard_cite_progress.txt"

# Best 7-signal weights from prior Dirichlet search
SIG_W_MAP = {"minilm": 0.074, "bge": 0.0, "bm25_body": 0.07,
             "tfidf_ta": 0.0, "tfidf_ft": 0.16, "gte_mb_ft": 0.506,
             "gte_cmaxsim": 0.19}

# Best cite-context weights found via --sweep
DEFAULT_W_BC = 0.5
DEFAULT_W_EC = 0.2
DEFAULT_W_CC = 0.05
DEFAULT_DOM_K = 90
DEFAULT_CAND_N = 200


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS_FILE, "a") as f: f.write(msg + "\n")


def mm(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo + 1e-9)


_cached = {}


def load_signals_plus_cite(split):
    if split in _cached: return _cached[split]
    log(f"Loading all signals + cite context ({split})...")
    t0 = time.time()

    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_ids = corpus["doc_id"].tolist()
    corp_domains = corpus["domain"].tolist()
    n_docs = len(corpus_ids)

    qf = DATA_DIR / ("queries.parquet" if split == "train" else "held_out_queries.parquet")
    qdf = load_queries(qf)
    query_ids = qdf["doc_id"].tolist()
    query_domains = qdf["domain"].tolist()
    n_q = len(query_ids)

    dom_to_cidx = defaultdict(list)
    for i, d in enumerate(corp_domains): dom_to_cidx[d].append(i)
    dom_to_cidx = {k: np.array(v) for k, v in dom_to_cidx.items()}

    def align_corpus(embs, eids):
        id2i = {c: i for i, c in enumerate(eids)}
        return embs[[id2i[c] for c in corpus_ids]]

    def align_query(embs, eids):
        id2i = {q: i for i, q in enumerate(eids)}
        return np.array([embs[id2i[q]] if q in id2i else np.zeros(embs.shape[1], np.float32)
                         for q in query_ids])

    def load_emb(stem, sub=""):
        emb = np.load(stem / ("corpus_embeddings.npy" if not sub else f"{sub}/query_embeddings.npy")).astype(np.float32)
        with open(stem / ("corpus_ids.json" if not sub else f"{sub}/query_ids.json")) as f:
            ids = json.load(f)
        return emb, ids

    sigs = {}
    mini_c, mini_cids = load_emb(EMB_MINI); mini_q, mini_qids = load_emb(EMB_MINI, split)
    sigs["minilm"] = (align_corpus(mini_c, mini_cids) @ align_query(mini_q, mini_qids).T).astype(np.float32)

    bge_c, bge_cids = load_emb(EMB_BGE); bge_q, bge_qids = load_emb(EMB_BGE, split)
    bge_cid2i = {c: i for i, c in enumerate(bge_cids)}
    sigs["bge"] = (align_corpus(bge_c, bge_cids) @ align_query(bge_q, bge_qids).T).astype(np.float32)

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

    ta_key = split if split == "train" else "held"
    tfidf_ta = np.load(TFIDF_DIR / f"tfidf_ta_{ta_key}_scores.npy").T
    with open(TFIDF_DIR / "tfidf_ta_corpus_ids.json") as f: ta_cids = json.load(f)
    ta_cid2i = {c: i for i, c in enumerate(ta_cids)}
    ta_to_corpus = np.array([ta_cid2i.get(c, 0) for c in corpus_ids])
    sigs["tfidf_ta"] = tfidf_ta[ta_to_corpus, :].astype(np.float32)

    tfidf_ft = np.load(TFIDF_DIR / f"tfidf_ft10k_{ta_key}_scores.npy").T
    with open(TFIDF_DIR / "tfidf_ft10k_corpus_ids.json") as f: ft_cids2 = json.load(f)
    ft10k_cid2i = {c: i for i, c in enumerate(ft_cids2)}
    ft_to_corpus = np.array([ft10k_cid2i.get(c, 0) for c in corpus_ids])
    sigs["tfidf_ft"] = tfidf_ft[ft_to_corpus, :].astype(np.float32)

    gmb_c, gmb_cids = load_emb(EMB_GMB); gmb_q, gmb_qids = load_emb(EMB_GMB, split)
    sigs["gte_mb_ft"] = (align_corpus(gmb_c, gmb_cids) @ align_query(gmb_q, gmb_qids).T).astype(np.float32)

    gtel_chunks = np.load(EMB_GTEL / "corpus_chunk_embeddings.npy").astype(np.float32)
    with open(EMB_GTEL / "corpus_chunk_doc_ids.json") as f: gtel_doc_ids = json.load(f)
    gtel_q = np.load(EMB_GTEL / f"{split}/query_embeddings.npy").astype(np.float32)
    with open(EMB_GTEL / f"{split}/query_ids.json") as f: gtel_qids = json.load(f)
    gtel_qi = {q: i for i, q in enumerate(gtel_qids)}
    gtel_q_al = np.array([gtel_q[gtel_qi[q]] if q in gtel_qi else np.zeros(gtel_q.shape[1], np.float32)
                          for q in query_ids])
    doc2chunks_gtel = defaultdict(list)
    for ri, did in enumerate(gtel_doc_ids): doc2chunks_gtel[did].append(ri)
    sim_all = gtel_q_al @ gtel_chunks.T
    ms = np.zeros((n_docs, n_q), np.float32)
    for ci, did in enumerate(corpus_ids):
        rows = doc2chunks_gtel.get(did, [])
        if rows: ms[ci, :] = sim_all[:, rows].max(axis=1)
    sigs["gte_cmaxsim"] = ms

    # Cite context
    bge_cite = np.load(EMB_BGE / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / f"{split}/cite_context_query_ids.json") as f: bge_cite_qids = json.load(f)
    e5_c, e5_cids = load_emb(EMB_E5)
    e5_cid2i = {c: i for i, c in enumerate(e5_cids)}
    e5_cite = np.load(EMB_E5 / f"{split}/cite_context_embeddings.npy").astype(np.float32)
    with open(EMB_E5 / f"{split}/cite_context_query_ids.json") as f: e5_cite_qids = json.load(f)
    q2bc = defaultdict(list)
    for i, q in enumerate(bge_cite_qids): q2bc[q].append(i)
    q2ec = defaultdict(list)
    for i, q in enumerate(e5_cite_qids): q2ec[q].append(i)

    cc_embs = np.load(EMB_BGE / "corpus_chunk_embeddings.npy").astype(np.float32)
    with open(EMB_BGE / "corpus_chunk_doc_ids.json") as f: cc_doc_ids = json.load(f)
    doc_to_chunk_rows = defaultdict(list)
    for ri, did in enumerate(cc_doc_ids): doc_to_chunk_rows[did].append(ri)

    data = dict(
        corpus_ids=corpus_ids, query_ids=query_ids, query_domains=query_domains,
        dom_to_cidx=dom_to_cidx, n_docs=n_docs, n_q=n_q, sigs=sigs,
        bge_c=bge_c, bge_cid2i=bge_cid2i, bge_cite=bge_cite, q2bc=dict(q2bc),
        e5_c=e5_c, e5_cid2i=e5_cid2i, e5_cite=e5_cite, q2ec=dict(q2ec),
        cc_embs=cc_embs, doc_to_chunk_rows=dict(doc_to_chunk_rows),
    )
    _cached[split] = data
    log(f"Loaded in {time.time()-t0:.1f}s")
    return data


def retrieve(split, w_bc=DEFAULT_W_BC, w_ec=DEFAULT_W_EC, w_cc=DEFAULT_W_CC,
             dom_k=DEFAULT_DOM_K, cand_n=DEFAULT_CAND_N, top_k=100):
    d = load_signals_plus_cite(split)
    corpus_ids = d["corpus_ids"]; query_ids = d["query_ids"]
    query_domains = d["query_domains"]; dom_to_cidx = d["dom_to_cidx"]
    n_docs = d["n_docs"]; sigs = d["sigs"]

    sig_names = list(sigs.keys())
    w = np.array([SIG_W_MAP[s] for s in sig_names], np.float32)
    w /= w.sum()

    submission = {}
    qdmap = dict(zip(query_ids, query_domains))
    for qi, (qid, qdom) in enumerate(zip(query_ids, query_domains)):
        dom_idx = dom_to_cidx.get(qdom, np.arange(n_docs))

        scores_dom = np.zeros(len(dom_idx), np.float32)
        for sname, ww in zip(sig_names, w):
            col = sigs[sname][dom_idx, qi]
            scores_dom += ww * mm(col)

        top_dom = dom_idx[np.argsort(-scores_dom)[:min(dom_k, len(dom_idx))]]
        if len(top_dom) < cand_n:
            scores_global = np.zeros(n_docs, np.float32)
            for sname, ww in zip(sig_names, w):
                scores_global += ww * mm(sigs[sname][:, qi])
            seen = set(top_dom.tolist())
            extra = [j for j in np.argsort(-scores_global) if j not in seen][:cand_n - len(top_dom)]
            cand_idx = list(top_dom) + extra
        else:
            cand_idx = list(top_dom[:cand_n])

        if w_bc > 0 or w_ec > 0 or w_cc > 0:
            # base score: reciprocal rank of candidates within the 7-signal ranking
            sc = np.array([1.0 / (rk + 1) for rk in range(len(cand_idx))], np.float32)

            rows_bc = d["q2bc"].get(qid, [])
            rows_ec = d["q2ec"].get(qid, [])
            bc_scores = np.zeros(len(cand_idx), np.float32)
            ec_scores = np.zeros(len(cand_idx), np.float32)
            cc_scores = np.zeros(len(cand_idx), np.float32)
            if rows_bc and (w_bc > 0 or w_cc > 0):
                qc_bge = d["bge_cite"][rows_bc]
                for k_ci, ci in enumerate(cand_idx):
                    doc_id = corpus_ids[ci]
                    if w_bc > 0:
                        bi = d["bge_cid2i"].get(doc_id)
                        if bi is not None:
                            bc_scores[k_ci] = float((qc_bge @ d["bge_c"][bi]).max())
                    if w_cc > 0:
                        chunk_rows = d["doc_to_chunk_rows"].get(doc_id, [])
                        if chunk_rows:
                            cc_scores[k_ci] = float((qc_bge @ d["cc_embs"][chunk_rows].T).max())
            if rows_ec and w_ec > 0:
                qc_e5 = d["e5_cite"][rows_ec]
                for k_ci, ci in enumerate(cand_idx):
                    doc_id = corpus_ids[ci]
                    ei = d["e5_cid2i"].get(doc_id)
                    if ei is not None:
                        ec_scores[k_ci] = float((qc_e5 @ d["e5_c"][ei]).max())

            if w_bc > 0 and bc_scores.max() > 0: sc += w_bc * mm(bc_scores)
            if w_ec > 0 and ec_scores.max() > 0: sc += w_ec * mm(ec_scores)
            if w_cc > 0 and cc_scores.max() > 0: sc += w_cc * mm(cc_scores)

            reranked = np.argsort(-sc)[:top_k]
            submission[qid] = [corpus_ids[cand_idx[k]] for k in reranked]
        else:
            submission[qid] = [corpus_ids[c] for c in cand_idx[:top_k]]

    return submission, qdmap


def _ndcg10(ranked, relevant):
    dcg  = sum(1/np.log2(r+1) for r, d in enumerate(ranked[:10], 1) if d in relevant)
    idcg = sum(1/np.log2(r+1) for r in range(1, min(len(relevant), 10)+1))
    return dcg / idcg if idcg else 0.0


def _fast_eval(sub, qrels):
    qd = {q: set(v) for q, v in qrels.items()}
    return float(np.mean([_ndcg10(sub.get(q, []), qd.get(q, set())) for q in qd]))


def run_sweep(qrels, dom_k):
    log("\n=== Phase A: baseline (7-signal only, no cite) ===")
    for cand_n in [100, 200, 500]:
        sub, _ = retrieve("train", w_bc=0.0, w_ec=0.0, w_cc=0.0, dom_k=dom_k, cand_n=cand_n)
        log(f"  cand_n={cand_n}  NDCG@10={_fast_eval(sub, qrels):.4f}")

    log("\n=== Phase B: sweep w_bc (cand_n=200) ===")
    best_bc, best_sc_B = 0.0, 0.0
    for wb in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95, 1.2, 1.5, 2.0]:
        sub, _ = retrieve("train", w_bc=wb, cand_n=200, dom_k=dom_k)
        sc = _fast_eval(sub, qrels)
        log(f"PROGRESS w_bc={wb:.2f}  NDCG@10={sc:.4f}")
        if sc > best_sc_B: best_sc_B, best_bc = sc, wb
    log(f"Phase B best: w_bc={best_bc}  NDCG={best_sc_B:.4f}")

    log(f"\n=== Phase C: w_bc={best_bc} + sweep w_ec ===")
    best_ec, best_sc_C = 0.0, best_sc_B
    for we in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        sub, _ = retrieve("train", w_bc=best_bc, w_ec=we, cand_n=200, dom_k=dom_k)
        sc = _fast_eval(sub, qrels)
        log(f"PROGRESS w_ec={we:.2f}  NDCG@10={sc:.4f}")
        if sc > best_sc_C: best_sc_C, best_ec = sc, we
    log(f"Phase C best: (w_bc={best_bc}, w_ec={best_ec})  NDCG={best_sc_C:.4f}")

    log(f"\n=== Phase D: + sweep w_cc ===")
    best_cc, best_sc_D = 0.0, best_sc_C
    for wc in [0.0, 0.03, 0.07, 0.15, 0.3]:
        sub, _ = retrieve("train", w_bc=best_bc, w_ec=best_ec, w_cc=wc, cand_n=200, dom_k=dom_k)
        sc = _fast_eval(sub, qrels)
        log(f"PROGRESS w_cc={wc:.2f}  NDCG@10={sc:.4f}")
        if sc > best_sc_D: best_sc_D, best_cc = sc, wc
    log(f"\n*** SWEEP BEST: w_bc={best_bc}, w_ec={best_ec}, w_cc={best_cc}  NDCG={best_sc_D:.4f} ***")
    return best_bc, best_ec, best_cc, best_sc_D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "held_out"])
    p.add_argument("--sweep", action="store_true",
                   help="Train-only: sweep w_bc/w_ec/w_cc, then retrieve with best")
    p.add_argument("--w-bc", type=float, default=DEFAULT_W_BC)
    p.add_argument("--w-ec", type=float, default=DEFAULT_W_EC)
    p.add_argument("--w-cc", type=float, default=DEFAULT_W_CC)
    p.add_argument("--dom-k", type=int, default=DEFAULT_DOM_K)
    p.add_argument("--cand-n", type=int, default=DEFAULT_CAND_N)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--zip", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if PROGRESS_FILE.exists(): PROGRESS_FILE.unlink()

    if args.sweep:
        if args.split != "train":
            log("--sweep requires --split train")
            return
        qrels = load_qrels(DATA_DIR / "qrels.json")
        w_bc, w_ec, w_cc, _ = run_sweep(qrels, dom_k=args.dom_k)
    else:
        w_bc, w_ec, w_cc = args.w_bc, args.w_ec, args.w_cc

    submission, qdmap = retrieve(args.split, w_bc=w_bc, w_ec=w_ec, w_cc=w_cc,
                                 dom_k=args.dom_k, cand_n=args.cand_n, top_k=args.top_k)

    if not args.no_eval and args.split == "train":
        qrels = load_qrels(DATA_DIR / "qrels.json")
        results = evaluate(submission, qrels, ks=[10], query_domains=qdmap, verbose=True)
        save_results(
            results, RESULTS_DIR / "hard_pipeline_with_cite.csv",
            hyperparameters={
                "w_bc": w_bc, "w_ec": w_ec, "w_cc": w_cc,
                "dom_k": args.dom_k, "cand_n": args.cand_n, "top_k": args.top_k,
                "split": args.split, "swept": bool(args.sweep),
            },
        )

    name = args.output or f"hard_pipeline_with_cite_{args.split}"
    out_path = SUBMISSIONS_DIR / (name if name.endswith(".json") else f"{name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    log(f"Saved -> {out_path}")

    if args.zip:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, "submission_data.json")
        log(f"Zipped -> {zip_path}")

    log("ALL DONE.")


if __name__ == "__main__":
    main()
