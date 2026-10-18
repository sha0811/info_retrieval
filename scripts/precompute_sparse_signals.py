"""
Precompute the three sparse lexical signals used by the Stage 1 domain-filtered pipeline
(models/domain_retrieval.py, models/domain_retrieval_cite.py, and so models/final_reranker.py):

  1. BM25 body       BM25 (rank_bm25 Okapi, default k1/b) over the body of each corpus paper,
                     i.e. full_text[500:10500] (skips the abstract area; falls back to the first
                     3000 chars of title+abstract when there is no full text).
                     Query = title+abstract followed by the title.
                     Stored as reciprocal rank: score[doc, query] = 1 / (rank + 1).
  2. TF-IDF TA       TF-IDF cosine similarity on title+abstract (query and corpus).
  3. TF-IDF FT 10k   TF-IDF cosine similarity on full text (first 50k chars, query and corpus),
                     vocabulary capped at 10,000 terms.

Both TF-IDF vectorizers are fit on the corpus (sublinear tf, min_df=2) and applied to queries.

Outputs (split key "train" = queries.parquet, "held" = held_out_queries.parquet):
  data/bm25_cache/bm25_body_{train,held}_scores.npy        (n_docs, n_queries) float32
  data/bm25_cache/bm25_body_{train,held}_query_ids.json    column order
  data/bm25_cache/bm25_body_corpus_ids.json                row order
  data/tfidf_cache/tfidf_{ta,ft10k}_{train,held}_scores.npy      (n_queries, n_docs) float32,
                                                                 rows in queries-parquet order
  data/tfidf_cache/tfidf_{ta,ft10k}_{train,held}_query_ids.json  row order
  data/tfidf_cache/tfidf_{ta,ft10k}_corpus_ids.json              column order

Usage:
    python scripts/precompute_sparse_signals.py
    python scripts/precompute_sparse_signals.py --signals tfidf
    python scripts/precompute_sparse_signals.py --out-dir /tmp/check   # write elsewhere
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_queries, format_text

DATA_DIR = Path(__file__).parent.parent / "data"
SPLITS = {"train": "queries.parquet", "held": "held_out_queries.parquet"}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


# BM25 body

def tokenize(text):
    return re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower()).split()


def body_text(row):
    full_text = str(row.get("full_text", "") or "")
    if not full_text.strip():
        return str(row.get("ta", "") or "")[:3000]
    return full_text[500:10500]


def build_bm25_body(corpus, queries, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_ids = corpus["doc_id"].tolist()
    n_docs = len(corpus_ids)

    print("BM25 body: tokenizing corpus bodies...", flush=True)
    tokens = [tokenize(body_text(row)) for _, row in tqdm(corpus.iterrows(), total=n_docs)]
    bm25 = BM25Okapi(tokens)
    del tokens

    for key, qdf in queries.items():
        query_ids = qdf["doc_id"].tolist()
        query_texts = (qdf["ta"].fillna("") + " " + qdf["title"].fillna("")).tolist()
        mat = np.zeros((n_docs, len(query_ids)), dtype=np.float32)
        for qi, text in enumerate(tqdm(query_texts, desc=f"BM25 body {key}")):
            scores = bm25.get_scores(tokenize(text)).astype(np.float32)
            mat[np.argsort(-scores), qi] = 1.0 / (np.arange(n_docs) + 1.0)
        np.save(out_dir / f"bm25_body_{key}_scores.npy", mat)
        save_json(query_ids, out_dir / f"bm25_body_{key}_query_ids.json")
        print(f"  saved bm25_body_{key}: {mat.shape}", flush=True)

    save_json(corpus_ids, out_dir / "bm25_body_corpus_ids.json")


# TF-IDF

def build_tfidf(name, corpus_texts, corpus_ids, query_texts, query_ids, out_dir, max_features=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"TF-IDF {name}: fitting on corpus (max_features={max_features})...", flush=True)
    vect = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2, ngram_range=(1, 1))
    corpus_vecs = vect.fit_transform(corpus_texts)

    for key in tqdm(query_texts, desc=f"TF-IDF {name}"):
        sims = (vect.transform(query_texts[key]) @ corpus_vecs.T).toarray().astype(np.float32)
        np.save(out_dir / f"tfidf_{name}_{key}_scores.npy", sims)
        save_json(query_ids[key], out_dir / f"tfidf_{name}_{key}_query_ids.json")
        print(f"  saved tfidf_{name}_{key}: {sims.shape}", flush=True)

    save_json(corpus_ids, out_dir / f"tfidf_{name}_corpus_ids.json")


def main():
    p = argparse.ArgumentParser(description="Precompute BM25-body and TF-IDF caches for Stage 1")
    p.add_argument("--signals", choices=["all", "bm25", "tfidf"], default="all")
    p.add_argument("--out-dir", type=Path, default=DATA_DIR,
                   help="Root folder for bm25_cache/ and tfidf_cache/ (default: data/)")
    args = p.parse_args()

    print("Loading corpus and queries...", flush=True)
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    queries = {key: load_queries(DATA_DIR / fname) for key, fname in SPLITS.items()}
    corpus_ids = corpus["doc_id"].tolist()
    query_ids = {key: qdf["doc_id"].tolist() for key, qdf in queries.items()}

    if args.signals in ("all", "bm25"):
        build_bm25_body(corpus, queries, args.out_dir / "bm25_cache")

    if args.signals in ("all", "tfidf"):
        tfidf_dir = args.out_dir / "tfidf_cache"
        full_text = lambda df: [str(t or "")[:50000] for t in df["full_text"]]
        build_tfidf("ta",
                    [format_text(row) for _, row in corpus.iterrows()], corpus_ids,
                    {k: [format_text(row) for _, row in q.iterrows()] for k, q in queries.items()},
                    query_ids, tfidf_dir)
        build_tfidf("ft10k", full_text(corpus), corpus_ids,
                    {k: full_text(q) for k, q in queries.items()},
                    query_ids, tfidf_dir, max_features=10000)

    print("Done.")


if __name__ == "__main__":
    main()
