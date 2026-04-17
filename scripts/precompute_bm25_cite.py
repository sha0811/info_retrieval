"""
Precompute BM25 scores using citation sentences as query text.
The standard BM25 uses title+abstract as query. This script uses the
citation-context sentences extracted from full_text, which contain
more specific vocabulary about what was cited.
"""
import json, sys, re
import numpy as np
import scipy.sparse as sp
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "bm25_cache"

K1, B = 1.8, 0.75

print("Loading corpus...")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_ids = corpus["doc_id"].tolist()

# BM25 indexes corpus title+abstract+full_text
def build_corpus_text(row):
    t = str(row.get("title", "") or "").strip()
    a = str(row.get("abstract", "") or "").strip()
    f = str(row.get("full_text", "") or "").strip()
    return " ".join([t, a, f])

print("Building corpus texts (title+abstract+full_text)...")
corpus_texts = [build_corpus_text(row) for _, row in corpus.iterrows()]
print(f"  {len(corpus_texts)} documents")

print("Fitting CountVectorizer on corpus...")
cv = CountVectorizer(
    tokenizer=lambda x: x.lower().split(),
    token_pattern=None,
    lowercase=False,
    max_features=300000
)
tf_matrix = cv.fit_transform(corpus_texts)  # (n_docs, vocab)
print(f"  TF matrix: {tf_matrix.shape}, nnz={tf_matrix.nnz}")

n_docs, vocab_size = tf_matrix.shape
doc_lengths = np.asarray(tf_matrix.sum(axis=1)).flatten()
avgdl = doc_lengths.mean()

df = np.asarray((tf_matrix > 0).sum(axis=0)).flatten()
idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

print(f"  avgdl={avgdl:.1f}, vocab={vocab_size}")

# Build BM25 matrix once
print("Building BM25 matrix...")
tf_coo = tf_matrix.tocoo()
tf_vals = tf_coo.data.astype(np.float32)
rows = tf_coo.row
cols = tf_coo.col
norm = K1 * (1 - B + B * doc_lengths / avgdl)
norm_rows = norm[rows].astype(np.float32)
wtf_vals = tf_vals * (K1 + 1) / (tf_vals + norm_rows)
wtf_matrix = sp.csr_matrix((wtf_vals, (rows, cols)), shape=tf_matrix.shape)
idf_sp = sp.diags(idf.astype(np.float32))
bm25_matrix = (wtf_matrix @ idf_sp)  # (n_docs, vocab)
print(f"  BM25 matrix: {bm25_matrix.shape}")

def bm25_scores_batch(query_texts):
    q_tf = cv.transform(query_texts)
    q_binary = (q_tf > 0).astype(np.float32)
    scores = q_binary @ bm25_matrix.T
    if sp.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float32)

# Citation sentence extraction
BRACKET_PAT = re.compile(r"\[\d[\d,\s\-–;]*\]")
PAREN_CITE_PAT = re.compile(r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s*(?:et\s+al\.?)?\s*,?\s*\d{4}[^)]*\)")
INLINE_CITE_PAT = re.compile(r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s*(?:et\s+al\.?)?\s*\(\d{4}[^)]*\)")
NUMERIC_PAREN_PAT = re.compile(r"(?<![a-zA-Z(])(?<!: )(?<!; )\(\d{1,3}(?:[,;]\s*\d{1,3})*\)(?=[\s,\.;)(]|$)")
ALPHA_BRACKET_PAT = re.compile(r"\[[A-Z][A-Za-z]{0,5}\d{2,4}(?:,\s*[A-Z][A-Za-z]{0,5}\d{2,4})*\]")
ALL_CITE_PATS = [BRACKET_PAT, PAREN_CITE_PAT, INLINE_CITE_PAT, NUMERIC_PAREN_PAT, ALPHA_BRACKET_PAT]

def extract_cite_sents(full_text, min_chars=30, max_sents=50):
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    seen_cleaned = set()
    cite_sents = []
    for s in sentences:
        has_cite = any(pat.search(s) for pat in ALL_CITE_PATS)
        if not has_cite: continue
        clean = s
        for pat in ALL_CITE_PATS:
            clean = pat.sub("", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < min_chars: continue
        if clean in seen_cleaned: continue
        seen_cleaned.add(clean)
        cite_sents.append(clean)
    cite_sents.sort(key=len, reverse=True)
    return cite_sents[:max_sents]

def build_cite_query_text(row):
    """Use citation sentences as BM25 query."""
    sents = extract_cite_sents(str(row.get("full_text", "") or ""))
    if sents:
        return " ".join(sents)
    # Fallback to title+abstract
    t = str(row.get("title", "") or "").strip()
    a = str(row.get("abstract", "") or "").strip()
    return t + " " + a

# Train queries
print("\nScoring train queries (citation sentences)...")
queries_df = load_queries(DATA_DIR / "queries.parquet")
train_cite_texts = [build_cite_query_text(row) for _, row in queries_df.iterrows()]
train_ta_texts = [(str(row.get("title","") or "") + " " + str(row.get("abstract","") or "")).strip() for _, row in queries_df.iterrows()]
train_ids = queries_df["doc_id"].tolist()
print(f"  {len(train_cite_texts)} train queries")
print(f"  Sample cite query (first 200): {train_cite_texts[0][:200]}")
print(f"  Sample TA query (first 200): {train_ta_texts[0][:200]}")

scores_cite_train = bm25_scores_batch(train_cite_texts)
print(f"  Train cite scores shape: {scores_cite_train.shape}")
np.save(CACHE_DIR / "bm25_cite_train_scores.npy", scores_cite_train.astype(np.float32))
with open(CACHE_DIR / "bm25_cite_train_query_ids.json", "w") as f: json.dump(train_ids, f)
print(f"  Saved train cite scores")

# Held-out queries
print("\nScoring held-out queries (citation sentences)...")
held_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_cite_texts = [build_cite_query_text(row) for _, row in held_df.iterrows()]
held_ids = held_df["doc_id"].tolist()
print(f"  {len(held_cite_texts)} held-out queries")

scores_cite_held = bm25_scores_batch(held_cite_texts)
print(f"  Held-out cite scores shape: {scores_cite_held.shape}")
np.save(CACHE_DIR / "bm25_cite_held_scores.npy", scores_cite_held.astype(np.float32))
with open(CACHE_DIR / "bm25_cite_held_query_ids.json", "w") as f: json.dump(held_ids, f)
print(f"  Saved held-out cite scores")

print("\nDone.")
