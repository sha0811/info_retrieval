from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


# Tuned defaults (can be overridden via CLI flags).
WB = 0.6
WD = 0.3
DB = 0.2
K1 = 1.8
B = 0.75
EMBEDDING_DIR = "intfloat_e5-large"
EMBEDDING_MODEL_NAME = "intfloat/e5-large"
QUERY_PREFIX = "Represent this scientific paper for retrieval: "
DOC_PREFIX = "Represent this scientific paper for retrieval: "
WK = 0.06
TOPK_INITIAL = 200
TOPK_FINAL = 100
TOPK_RERANK = 100
BM25_K1_GRID = [1.5, 1.8, 2.1, 2.5, 3.0]
BM25_B_GRID = [0.5, 0.75, 1.0]


def to_embedding_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def format_query_text(df: pd.DataFrame, for_embedding: bool = False, prefix: str = QUERY_PREFIX) -> list[str]:
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip().tolist()
    if for_embedding:
        return [prefix + t for t in texts]
    return texts


def format_corpus_text(
    corpus: pd.DataFrame,
    for_embedding: bool = False,
    prefix: str = DOC_PREFIX,
) -> list[str]:
    texts = []
    for _, row in corpus.iterrows():
        title = str(row.get("title", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        full_text = str(row.get("full_text", "") or "").strip()
        text = (title + " " + abstract + " " + full_text).strip()
        if for_embedding:
            text = prefix + text
        texts.append(text)
    return texts


def minmax(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def evaluate_ndcg10(submission: dict[str, list[str]], qrels: dict[str, list[str]]) -> float:
    vals = []
    for qid, rel_docs in qrels.items():
        vals.append(ndcg_at_k(submission.get(qid, []), set(rel_docs), 10))
    return float(np.mean(vals))


def load_or_encode_embeddings(
    root: Path,
    model_name: str,
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    use_gpu: bool,
    show_progress: bool,
    force_recompute_corpus: bool = False,
    use_query_prefix: bool = True,
    use_doc_prefix: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    emb_dir = root / "data" / "embeddings" / to_embedding_dir_name(model_name)
    emb_dir.mkdir(parents=True, exist_ok=True)

    q_path = emb_dir / "query_embeddings.npy"
    c_path = emb_dir / "corpus_embeddings.npy"
    qids_path = emb_dir / "query_ids.json"
    cids_path = emb_dir / "corpus_ids.json"

    query_ids = queries_df["doc_id"].tolist()
    corpus_ids = corpus_df["doc_id"].tolist()

    device = "cuda" if use_gpu else "cpu"
    model = SentenceTransformer(model_name, device=device)

    encode_query_texts = format_query_text(
        queries_df,
        for_embedding=use_query_prefix,
        prefix=QUERY_PREFIX,
    )
    encode_corpus_texts = format_corpus_text(
        corpus_df,
        for_embedding=use_doc_prefix,
        prefix=DOC_PREFIX,
    )

    # Held-out / alternate query sets use different doc_ids than the public cache.
    q_path_heldout = emb_dir / "query_embeddings_heldout.npy"
    qids_path_heldout = emb_dir / "query_ids_heldout.json"

    def encode_queries_and_save(save_path: Path, ids_path: Path) -> np.ndarray:
        print(f"Encoding query embeddings with {model_name} on {device}...")
        emb = model.encode(
            encode_query_texts,
            batch_size=32,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(save_path, emb)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(query_ids, f)
        return emb

    if q_path.exists() and qids_path.exists():
        with open(qids_path, encoding="utf-8") as f:
            q_ids_emb = json.load(f)
        qid2i = {qid: i for i, qid in enumerate(q_ids_emb)}
        if all(qid in qid2i for qid in query_ids):
            q_emb = np.load(q_path).astype(np.float32)
            q_emb = q_emb[[qid2i[qid] for qid in query_ids]]
        elif q_path_heldout.exists() and qids_path_heldout.exists():
            with open(qids_path_heldout, encoding="utf-8") as f:
                q_ids_h = json.load(f)
            if q_ids_h == query_ids:
                q_emb = np.load(q_path_heldout).astype(np.float32)
            else:
                q_emb = encode_queries_and_save(q_path_heldout, qids_path_heldout)
        else:
            q_emb = encode_queries_and_save(q_path_heldout, qids_path_heldout)
    else:
        q_emb = encode_queries_and_save(q_path, qids_path)

    need_corpus_encode = force_recompute_corpus or (not c_path.exists()) or (not cids_path.exists())
    if need_corpus_encode:
        print(f"Encoding corpus embeddings with {model_name} on {device}...")
        c_emb = model.encode(
            encode_corpus_texts,
            batch_size=16,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(c_path, c_emb)
        with open(cids_path, "w", encoding="utf-8") as f:
            json.dump(corpus_ids, f)
    else:
        c_emb = np.load(c_path).astype(np.float32)
        with open(cids_path, encoding="utf-8") as f:
            c_ids_emb = json.load(f)
        cid2i = {cid: i for i, cid in enumerate(c_ids_emb)}
        c_emb = c_emb[[cid2i[cid] for cid in corpus_ids]]

    if q_emb.shape[1] != c_emb.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {model_name}: "
            f"query dim={q_emb.shape[1]}, corpus dim={c_emb.shape[1]}"
        )

    return q_emb, c_emb


def fuse_scores_nonlinear(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    domain_match: np.ndarray,
    wb: float,
    wd: float,
    db: float,
) -> np.ndarray:
    bm25_norm = minmax(bm25_scores)
    dense_norm = minmax(dense_scores)
    domain_norm = domain_match.astype(np.float32)
    combined = wb * np.power(bm25_norm, 1.2) + wd * np.power(dense_norm, 0.9) + db * domain_norm
    return combined


def boost_keyword_overlap(query_text: str, candidate_texts: list[str]) -> np.ndarray:
    query_keywords = set(w.lower() for w in query_text.split() if len(w) > 3)
    if not query_keywords:
        return np.zeros(len(candidate_texts), dtype=np.float32)
    keyword_scores = np.array(
        [
            len(query_keywords & set(text.lower().split())) / (len(query_keywords) + 1e-9)
            for text in candidate_texts
        ],
        dtype=np.float32,
    )
    return keyword_scores


def rerank_with_cross_encoder(
    query_text: str,
    candidate_ids: np.ndarray,
    corpus_id_to_text: dict[str, str],
    model,
    batch_size: int = 64,
) -> list[str]:
    pairs = [(query_text, corpus_id_to_text[cid]) for cid in candidate_ids]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    order = np.argsort(-scores)
    return [candidate_ids[i] for i in order]


def build_submission(
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    q_emb: np.ndarray,
    c_emb: np.ndarray,
    wb: float,
    wd: float,
    db: float,
    wk: float,
    bm25_k1: float,
    bm25_b: float,
    use_nonlinear_fusion: bool,
    use_cross_encoder: bool,
    cross_encoder_model_name: str,
    topk_initial: int,
    topk_final: int,
    topk_rerank: int,
    use_gpu: bool,
    show_progress: bool,
) -> dict[str, list[str]]:
    query_ids = queries_df["doc_id"].tolist()
    corpus_ids = corpus_df["doc_id"].tolist()
    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    corpus_domains = np.array(corpus_df["domain"].tolist())

    query_texts_plain = format_query_text(queries_df, for_embedding=False)
    corpus_texts_plain = format_corpus_text(corpus_df, for_embedding=False)
    corpus_id_to_text = dict(zip(corpus_ids, corpus_texts_plain))

    bm25 = BM25Okapi([t.lower().split() for t in corpus_texts_plain], k1=bm25_k1, b=bm25_b)

    cross_encoder = None
    if use_cross_encoder:
        from sentence_transformers import CrossEncoder

        device = "cuda" if use_gpu else "cpu"
        cross_encoder = CrossEncoder(cross_encoder_model_name, device=device)

    submission: dict[str, list[str]] = {}
    iterator = enumerate(query_ids)
    if show_progress:
        iterator = enumerate(tqdm(query_ids, desc="Ranking queries", unit="query"))

    topk_initial = min(topk_initial, len(corpus_ids))
    topk_rerank = min(topk_rerank, topk_initial)
    topk_final = min(topk_final, topk_rerank)

    for i, qid in iterator:
        q_text = query_texts_plain[i]
        bm25_scores = bm25.get_scores(q_text.lower().split())
        dense_scores = q_emb[i] @ c_emb.T
        domain_match = corpus_domains == query_domains[qid]

        if use_nonlinear_fusion:
            score = fuse_scores_nonlinear(
                bm25_scores=bm25_scores,
                dense_scores=dense_scores,
                domain_match=domain_match,
                wb=wb,
                wd=wd,
                db=db,
            )
        else:
            score = wb * minmax(bm25_scores) + wd * minmax(dense_scores) + db * domain_match

        top = np.argpartition(-score, topk_initial)[:topk_initial]
        top = top[np.argsort(-score[top])]
        top_ids = np.array([corpus_ids[j] for j in top], dtype=object)

        # Keyword overlap boost before cross-encoder reranking.
        keyword_scores = boost_keyword_overlap(q_text, [corpus_id_to_text[cid] for cid in top_ids])
        pre_ce_score = score[top] + wk * keyword_scores
        top2_idx = np.argsort(-pre_ce_score)[:topk_rerank]
        candidate_ids = top_ids[top2_idx]

        if cross_encoder is not None:
            reranked_ids = rerank_with_cross_encoder(
                q_text,
                candidate_ids=candidate_ids,
                corpus_id_to_text=corpus_id_to_text,
                model=cross_encoder,
            )
        else:
            reranked_ids = candidate_ids.tolist()

        submission[qid] = reranked_ids[:topk_final]

    return submission


def grid_search_bm25(
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    qrels: dict[str, list[str]],
    q_emb: np.ndarray,
    c_emb: np.ndarray,
    wb: float,
    wd: float,
    db: float,
    wk: float,
    k1_values: list[float],
    b_values: list[float],
    use_nonlinear_fusion: bool,
    show_progress: bool,
    use_gpu: bool,
) -> tuple[float, float, float]:
    best_ndcg = -1.0
    best_k1 = K1
    best_b = B

    configs = [(k1, bval) for k1 in k1_values for bval in b_values]
    total = len(configs)

    for idx, (k1, bval) in enumerate(configs):
        print(f"\n[BM25 grid {idx + 1}/{total}] k1={k1}, b={bval} (evaluating...)")
        submission = build_submission(
            queries_df=queries_df,
            corpus_df=corpus_df,
            q_emb=q_emb,
            c_emb=c_emb,
            wb=wb,
            wd=wd,
            db=db,
            wk=wk,
            bm25_k1=k1,
            bm25_b=bval,
            use_nonlinear_fusion=use_nonlinear_fusion,
            use_cross_encoder=False,
            cross_encoder_model_name="",
            topk_initial=TOPK_INITIAL,
            topk_final=TOPK_FINAL,
            topk_rerank=TOPK_RERANK,
            use_gpu=use_gpu,
            show_progress=show_progress,
        )
        ndcg = evaluate_ndcg10(submission, qrels)
        print(f"[BM25 grid {idx + 1}/{total}] k1={k1}, b={bval} -> NDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_k1 = k1
            best_b = bval

    return best_k1, best_b, best_ndcg


def run_local(
    root: Path,
    output_path: Path,
    wb: float,
    wd: float,
    db: float,
    wk: float,
    k1: float,
    b: float,
    model_name: str,
    use_gpu: bool,
    grid_search_bm25_first: bool,
    use_nonlinear_fusion: bool,
    use_cross_encoder: bool,
    cross_encoder_model_name: str,
    topk_initial: int,
    topk_final: int,
    topk_rerank: int,
    show_progress: bool,
) -> tuple[float, float, float]:
    queries = pd.read_parquet(root / "data/queries.parquet")
    corpus = pd.read_parquet(root / "data/corpus.parquet")
    with open(root / "data/qrels.json", encoding="utf-8") as f:
        qrels = json.load(f)

    q_emb, c_emb = load_or_encode_embeddings(
        root=root,
        model_name=model_name,
        queries_df=queries,
        corpus_df=corpus,
        use_gpu=use_gpu,
        show_progress=show_progress,
        force_recompute_corpus=False,
    )

    best_k1 = k1
    best_b = b
    if grid_search_bm25_first:
        best_k1, best_b, best_score = grid_search_bm25(
            queries_df=queries,
            corpus_df=corpus,
            qrels=qrels,
            q_emb=q_emb,
            c_emb=c_emb,
            wb=wb,
            wd=wd,
            db=db,
            wk=wk,
            k1_values=BM25_K1_GRID,
            b_values=BM25_B_GRID,
            use_nonlinear_fusion=use_nonlinear_fusion,
            show_progress=show_progress,
            use_gpu=use_gpu,
        )
        print(f"Best BM25 from grid-search -> k1={best_k1}, b={best_b}, ndcg@10={best_score:.4f}")

    submission = build_submission(
        queries_df=queries,
        corpus_df=corpus,
        q_emb=q_emb,
        c_emb=c_emb,
        wb=wb,
        wd=wd,
        db=db,
        wk=wk,
        bm25_k1=best_k1,
        bm25_b=best_b,
        use_nonlinear_fusion=use_nonlinear_fusion,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_model_name=cross_encoder_model_name,
        topk_initial=topk_initial,
        topk_final=topk_final,
        topk_rerank=topk_rerank,
        use_gpu=use_gpu,
        show_progress=show_progress,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    return evaluate_ndcg10(submission, qrels), best_k1, best_b


def build_heldout_submission(
    root: Path,
    output_path: Path,
    wb: float,
    wd: float,
    db: float,
    wk: float,
    k1: float,
    b: float,
    model_name: str,
    use_gpu: bool,
    use_nonlinear_fusion: bool,
    use_cross_encoder: bool,
    cross_encoder_model_name: str,
    topk_initial: int,
    topk_final: int,
    topk_rerank: int,
    show_progress: bool,
) -> None:
    heldout = pd.read_parquet(root / "data/held_out_queries.parquet")
    corpus = pd.read_parquet(root / "data/corpus.parquet")
    q_emb, c_emb = load_or_encode_embeddings(
        root=root,
        model_name=model_name,
        queries_df=heldout,
        corpus_df=corpus,
        use_gpu=use_gpu,
        show_progress=show_progress,
        force_recompute_corpus=False,
    )

    submission = build_submission(
        queries_df=heldout,
        corpus_df=corpus,
        q_emb=q_emb,
        c_emb=c_emb,
        wb=wb,
        wd=wd,
        db=db,
        wk=wk,
        bm25_k1=k1,
        bm25_b=b,
        use_nonlinear_fusion=use_nonlinear_fusion,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_model_name=cross_encoder_model_name,
        topk_initial=topk_initial,
        topk_final=topk_final,
        topk_rerank=topk_rerank,
        use_gpu=use_gpu,
        show_progress=show_progress,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root path")
    parser.add_argument("--local-output", type=Path, default=None, help="Output JSON for local query set")
    parser.add_argument(
        "--build-heldout",
        action="store_true",
        help="Also build submissions/submission_data.json for held-out queries",
    )
    parser.add_argument("--heldout-output", type=Path, default=None, help="Output JSON path for held-out submission")
    parser.add_argument("--wb", type=float, default=WB, help="BM25 weight")
    parser.add_argument("--wd", type=float, default=WD, help="Dense weight")
    parser.add_argument("--db", type=float, default=DB, help="Same-domain additive boost")
    parser.add_argument("--wk", type=float, default=WK, help="Keyword overlap boost")
    parser.add_argument("--k1", type=float, default=K1, help="BM25 k1")
    parser.add_argument("--b", type=float, default=B, help="BM25 b")
    parser.add_argument(
        "--model-name",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="Sentence-transformers model name for dense retrieval",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA for embedding/cross-encoder inference")
    parser.add_argument("--grid-search-bm25", action="store_true", help="Run BM25 (k1,b) grid-search before ranking")
    parser.add_argument("--disable-nonlinear-fusion", action="store_true", help="Use legacy linear fusion instead")
    parser.add_argument("--disable-cross-encoder", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Cross-encoder model name for top-k reranking",
    )
    parser.add_argument("--topk-initial", type=int, default=TOPK_INITIAL, help="Initial candidate pool size")
    parser.add_argument("--topk-rerank", type=int, default=TOPK_RERANK, help="Candidates passed to cross-encoder")
    parser.add_argument("--topk-final", type=int, default=TOPK_FINAL, help="Final docs per query in output")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args()

    root = args.root
    local_output = args.local_output or (root / "submissions/specter2_score_fusion_domainboost.json")
    heldout_output = args.heldout_output or (root / "submissions/submission_data.json")
    use_nonlinear_fusion = not args.disable_nonlinear_fusion
    use_cross_encoder = not args.disable_cross_encoder

    ndcg10, k1_used, b_used = run_local(
        root,
        local_output,
        args.wb,
        args.wd,
        args.db,
        args.wk,
        args.k1,
        args.b,
        args.model_name,
        args.use_gpu,
        args.grid_search_bm25,
        use_nonlinear_fusion,
        use_cross_encoder,
        args.cross_encoder_model,
        args.topk_initial,
        args.topk_final,
        args.topk_rerank,
        show_progress=not args.no_progress,
    )
    print(f"Local NDCG@10: {ndcg10:.4f}")
    print(f"Saved local predictions to: {local_output}")
    if args.grid_search_bm25:
        print(f"Using BM25 params for held-out (if any): k1={k1_used}, b={b_used}")

    if args.build_heldout:
        build_heldout_submission(
            root,
            heldout_output,
            args.wb,
            args.wd,
            args.db,
            args.wk,
            k1_used,
            b_used,
            args.model_name,
            args.use_gpu,
            use_nonlinear_fusion,
            use_cross_encoder,
            args.cross_encoder_model,
            args.topk_initial,
            args.topk_final,
            args.topk_rerank,
            show_progress=not args.no_progress,
        )
        print(f"Saved held-out submission to: {heldout_output}")


if __name__ == "__main__":
    main()
