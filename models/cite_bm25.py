"""
Citation-sentence BM25 retrieval.

Uses citation sentences from the query paper's full text as BM25 queries against
the corpus (title + abstract). For each corpus document, the final score is the
MAX BM25 score across all citation sentences of the query.

Rationale
---------
Dense embeddings perform semantic matching — they find papers that are topically
similar to the query. But citation relevance requires a different signal: the cited
paper's terminology often appears *verbatim* in the citation sentences of the citing
paper. BM25 on those sentences provides an exact lexical bridge that dense models miss.

38.7% of relevant paper title key-words appear directly in the query's citation
sentences — this model is specifically designed to recover those cases.

Usage:
    python models/cite_bm25.py --split train --output submissions/cite_bm25_train.json
    python models/cite_bm25.py --split held_out --output submissions/cite_bm25_held_out.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_corpus, load_qrels, load_queries, evaluate, save_results

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}

# Citation marker patterns (kept in sync with embed_cite_contexts.py)
BRACKET_PAT = re.compile(r"\[\d[\d,\s\-–;]*\]")
PAREN_CITE_PAT = re.compile(
    r"\("
    r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?"
    r"\s*(?:et\s+al\.?)?"
    r"\s*,?\s*\d{4}"
    r"[^)]*\)"
)
INLINE_CITE_PAT = re.compile(
    r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?"
    r"\s*(?:et\s+al\.?)?"
    r"\s*\(\d{4}[^)]*\)"
)
# Vancouver numeric style: (1), (2,3), (1;2;3)
NUMERIC_PAREN_PAT = re.compile(
    r"(?<![a-zA-Z(])(?<!: )(?<!; )"
    r"\(\d{1,3}(?:[,;]\s*\d{1,3})*\)"
    r"(?=[\s,\.;)(]|$)"
)
# Alphanumeric bracket style: [Kar99], [HS18], [KLM89]
ALPHA_BRACKET_PAT = re.compile(r"\[[A-Z][A-Za-z]{0,5}\d{2,4}(?:,\s*[A-Z][A-Za-z]{0,5}\d{2,4})*\]")

ALL_CITE_PATS = [BRACKET_PAT, PAREN_CITE_PAT, INLINE_CITE_PAT, NUMERIC_PAREN_PAT, ALPHA_BRACKET_PAT]


def extract_citation_sentences(full_text: str, min_chars: int = 30, max_sents: int = 50) -> list[str]:
    """
    Extract and clean citation sentences from full text.

    Handles four citation styles:
      - Numeric bracket:      [1], [1,2], [1-3]
      - Author-year paren:    (Smith et al., 2020)
      - Author-year inline:   Smith et al. (2020)
      - Vancouver numeric:    (1), (2,3)          ← previously missed
      - Alphanumeric bracket: [Kar99], [HS18]     ← previously missed
    Consecutive markers on the same sentence are deduplicated.
    Returns sentences sorted longest-first, capped at max_sents.
    """
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    seen_cleaned = set()
    cite_sents = []

    for s in sentences:
        has_cite = any(pat.search(s) for pat in ALL_CITE_PATS)
        if not has_cite:
            continue

        clean = s
        for pat in ALL_CITE_PATS:
            clean = pat.sub("", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if len(clean) < min_chars:
            continue

        if clean in seen_cleaned:
            continue
        seen_cleaned.add(clean)
        cite_sents.append(clean)

    cite_sents.sort(key=len, reverse=True)
    return cite_sents[:max_sents]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(corpus, k1: float = 1.5, b: float = 0.75):
    """Build BM25 index over corpus title + abstract."""
    corpus_ids = corpus["doc_id"].tolist()
    corpus_texts = []
    for _, row in corpus.iterrows():
        title = str(row.get("title", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        text = (title + " " + abstract) if (title and abstract) else (title or abstract)
        corpus_texts.append(text)

    print("Tokenizing corpus ...")
    tokenized = [tokenize(t) for t in corpus_texts]
    print("Building BM25 index ...")
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)
    return bm25, corpus_ids


def retrieve(queries_df, bm25, corpus_ids, top_k: int = 100,
             min_chars: int = 30, max_sents: int = 50) -> dict:
    """
    For each query: extract citation sentences, score each against the corpus,
    take the max BM25 score per corpus document, return top_k.
    """
    submission = {}
    n_corpus = len(corpus_ids)
    fallback_count = 0

    for _, row in queries_df.iterrows():
        qid = row["doc_id"]
        full_text = str(row.get("full_text", "") or "")
        cite_sents = extract_citation_sentences(full_text, min_chars=min_chars, max_sents=max_sents)

        if cite_sents:
            # Max BM25 score across all citation sentences
            max_scores = np.zeros(n_corpus, dtype=np.float64)
            for sent in cite_sents:
                scores = bm25.get_scores(tokenize(sent))
                np.maximum(max_scores, scores, out=max_scores)
        else:
            # Fallback: use title + abstract as query (same as standard BM25)
            fallback_count += 1
            title = str(row.get("title", "") or "").strip()
            abstract = str(row.get("abstract", "") or "").strip()
            ta = (title + " " + abstract) if (title and abstract) else (title or abstract)
            max_scores = bm25.get_scores(tokenize(ta))

        top_indices = np.argsort(-max_scores)[:top_k]
        submission[qid] = [corpus_ids[i] for i in top_indices]

    if fallback_count:
        print(f"  Fallback to TA BM25 for {fallback_count} queries (no citation sentences found)")

    return submission


def main():
    parser = argparse.ArgumentParser(description="Citation-sentence BM25 retrieval")
    parser.add_argument("--split", default="train", choices=["train", "held_out"])
    parser.add_argument("--corpus", default=DATA_DIR / "corpus.parquet")
    parser.add_argument("--qrels", default=DATA_DIR / "qrels.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: submissions/cite_bm25_<split>.json)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-sents", type=int, default=50,
                        help="Max citation sentences per query (longest first)")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    output_path = args.output or SUBMISSIONS_DIR / f"cite_bm25_{args.split}.json"
    queries_path = QUERY_FILES[args.split]

    print(f"Loading queries ({args.split}) ...")
    queries_df = load_queries(queries_path)
    print(f"  {len(queries_df)} queries")

    print(f"Loading corpus ...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} documents")

    bm25, corpus_ids = build_index(corpus, k1=args.k1, b=args.b)

    print(f"\nRetrieving with citation-sentence BM25 (max_sents={args.max_sents}) ...")
    submission = retrieve(queries_df, bm25, corpus_ids, top_k=args.top_k, max_sents=args.max_sents)

    if not args.no_eval:
        qrels = load_qrels(args.qrels)
        query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
        results = evaluate(submission, qrels, ks=[10, 100], query_domains=query_domains)
        save_results(
            results,
            RESULTS_DIR / "cite_bm25.csv",
            hyperparameters={"split": args.split, "k1": args.k1, "b": args.b,
                             "max_sents": args.max_sents},
        )

    import os
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
