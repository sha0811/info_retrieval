"""
Fast vectorized BM25 using sklearn sparse matrices.
Builds BM25 on title+abstract+full_text but using C-level sparse ops (sklearn).
Saves scores for train + held-out queries to disk for reuse.

Expected runtime: ~2-3 minutes (vs 30+ min with rank_bm25 Python loop).
"""
import json, sys
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
CACHE_DIR.mkdir(exist_ok=True)

K1, B = 1.8, 0.75

print("Loading corpus...")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_ids = corpus["doc_id"].tolist()

def build_text(df, include_fulltext=True):
    texts = []
    for _, row in df.iterrows():
        t = str(row.get("title", "") or "").strip()
        a = str(row.get("abstract", "") or "").strip()
        parts = [t, a]
        if include_fulltext:
            f = str(row.get("full_text", "") or "").strip()
            if f: parts.append(f)
        texts.append(" ".join(parts))
    return texts

print("Building corpus texts (title+abstract+full_text)...")
corpus_texts = build_text(corpus, include_fulltext=True)
print(f"  {len(corpus_texts)} documents")

# ── Vectorized BM25 ───────────────────────────────────────────────────────────
print("Fitting CountVectorizer on corpus...")
# Use split() tokenizer to match rank_bm25.BM25Okapi behavior exactly
# (preserves hyphenated terms like "COVID-19", "gene-expression", etc.)
cv = CountVectorizer(
    tokenizer=lambda x: x.lower().split(),
    token_pattern=None,
    lowercase=False,   # tokenizer already lowercases
    max_features=300000
)
tf_matrix = cv.fit_transform(corpus_texts)  # (n_docs, vocab)
print(f"  TF matrix: {tf_matrix.shape}, nnz={tf_matrix.nnz}")

n_docs, vocab_size = tf_matrix.shape
doc_lengths = np.asarray(tf_matrix.sum(axis=1)).flatten()  # (n_docs,)
avgdl = doc_lengths.mean()

# IDF: log((N - df + 0.5) / (df + 0.5) + 1)
df = np.asarray((tf_matrix > 0).sum(axis=0)).flatten()  # (vocab,)
idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)   # (vocab,)

print(f"  avgdl={avgdl:.1f}, vocab={vocab_size}")

def bm25_scores_batch(query_texts):
    """Compute BM25 scores for multiple queries at once. Returns (n_queries, n_docs)."""
    q_tf = cv.transform(query_texts)  # (n_q, vocab)

    # BM25 numerator per term: tf_d * (k1 + 1)
    # BM25 denominator per term: tf_d + k1 * (1 - b + b * dl / avgdl)
    # Full: IDF * tf_d*(k1+1) / (tf_d + k1*(1-b+b*dl/avgdl))

    # Compute per-doc normalization factor: k1 * (1 - b + b * dl / avgdl)
    norm = K1 * (1 - B + B * doc_lengths / avgdl)  # (n_docs,)

    # For each non-zero entry in tf_matrix: tf_bm25 = tf*(k1+1) / (tf + norm[doc])
    # We need to weight each term by idf then sum over query terms
    # Use: score(q, d) = sum_t in q [ idf[t] * tf_matrix[d,t]*(k1+1) / (tf_matrix[d,t] + norm[d]) ]

    # Build weighted TF matrix: wtf[d,t] = tf[d,t]*(k1+1) / (tf[d,t] + norm[d])
    # This is O(nnz) sparse operation
    tf_coo = tf_matrix.tocoo()
    tf_vals = tf_coo.data.astype(np.float32)
    rows = tf_coo.row
    cols = tf_coo.col

    norm_rows = norm[rows].astype(np.float32)
    wtf_vals = tf_vals * (K1 + 1) / (tf_vals + norm_rows)

    wtf_matrix = sp.csr_matrix((wtf_vals, (rows, cols)), shape=tf_matrix.shape)  # (n_docs, vocab)

    # Weight by IDF
    idf_sp = sp.diags(idf.astype(np.float32))  # (vocab, vocab)
    bm25_matrix = (wtf_matrix @ idf_sp)  # (n_docs, vocab)

    # For each query, sum bm25_matrix columns where query has non-zero TF
    # score(q,d) = sum_t [q_tf[q,t] > 0] bm25_matrix[d,t]
    # = (q_tf_binary @ bm25_matrix.T)[q, d]
    q_binary = (q_tf > 0).astype(np.float32)  # (n_q, vocab)
    scores = q_binary @ bm25_matrix.T  # (n_q, n_docs) — sparse result
    if sp.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float32)

# ── Score train queries ───────────────────────────────────────────────────────
print("\nScoring train queries...")
queries_df = load_queries(DATA_DIR / "queries.parquet")
# Queries use title+abstract ONLY (matching retrieve_advanced.py behavior)
train_texts = build_text(queries_df, include_fulltext=False)
train_ids = queries_df["doc_id"].tolist()

scores_train = bm25_scores_batch(train_texts)  # (100, 20K)
print(f"  Train scores shape: {scores_train.shape}")

np.save(CACHE_DIR / "bm25_train_scores.npy", scores_train.astype(np.float32))
with open(CACHE_DIR / "bm25_train_query_ids.json", "w") as f: json.dump(train_ids, f)
with open(CACHE_DIR / "bm25_corpus_ids.json", "w") as f: json.dump(corpus_ids, f)
print(f"  Saved train scores -> {CACHE_DIR / 'bm25_train_scores.npy'}")

# ── Score held-out queries ────────────────────────────────────────────────────
print("\nScoring held-out queries...")
held_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
# Queries use title+abstract ONLY (matching retrieve_advanced.py behavior)
held_texts = build_text(held_df, include_fulltext=False)
held_ids = held_df["doc_id"].tolist()

scores_held = bm25_scores_batch(held_texts)
print(f"  Held-out scores shape: {scores_held.shape}")

np.save(CACHE_DIR / "bm25_held_scores.npy", scores_held.astype(np.float32))
with open(CACHE_DIR / "bm25_held_query_ids.json", "w") as f: json.dump(held_ids, f)
print(f"  Saved held-out scores -> {CACHE_DIR / 'bm25_held_scores.npy'}")

print("\nDone. BM25 scores cached. Run fusion_v3.py to use them.")
