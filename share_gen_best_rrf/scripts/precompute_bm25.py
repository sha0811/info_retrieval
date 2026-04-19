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

print("Fitting CountVectorizer on corpus...")
cv = CountVectorizer(
    tokenizer=lambda x: x.lower().split(),
    token_pattern=None,
    lowercase=False,
    max_features=300000
)
tf_matrix = cv.fit_transform(corpus_texts)
print(f"  TF matrix: {tf_matrix.shape}, nnz={tf_matrix.nnz}")

n_docs, vocab_size = tf_matrix.shape
doc_lengths = np.asarray(tf_matrix.sum(axis=1)).flatten()  
avgdl = doc_lengths.mean()

df = np.asarray((tf_matrix > 0).sum(axis=0)).flatten()  
idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)  

print(f"  avgdl={avgdl:.1f}, vocab={vocab_size}")

def bm25_scores_batch(query_texts):
    q_tf = cv.transform(query_texts) 
    norm = K1 * (1 - B + B * doc_lengths / avgdl) 

    tf_coo = tf_matrix.tocoo()
    tf_vals = tf_coo.data.astype(np.float32)
    rows = tf_coo.row
    cols = tf_coo.col

    norm_rows = norm[rows].astype(np.float32)
    wtf_vals = tf_vals * (K1 + 1) / (tf_vals + norm_rows)

    wtf_matrix = sp.csr_matrix((wtf_vals, (rows, cols)), shape=tf_matrix.shape)  

    idf_sp = sp.diags(idf.astype(np.float32)) 
    bm25_matrix = (wtf_matrix @ idf_sp) 

    q_binary = (q_tf > 0).astype(np.float32)  
    scores = q_binary @ bm25_matrix.T 
    if sp.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float32)

print("\nScoring train queries...")
queries_df = load_queries(DATA_DIR / "queries.parquet")
train_texts = build_text(queries_df, include_fulltext=False)
train_ids = queries_df["doc_id"].tolist()

scores_train = bm25_scores_batch(train_texts)  # (100, 20K)
print(f"  Train scores shape: {scores_train.shape}")

np.save(CACHE_DIR / "bm25_train_scores.npy", scores_train.astype(np.float32))
with open(CACHE_DIR / "bm25_train_query_ids.json", "w") as f: json.dump(train_ids, f)
with open(CACHE_DIR / "bm25_corpus_ids.json", "w") as f: json.dump(corpus_ids, f)
print(f"  Saved train scores -> {CACHE_DIR / 'bm25_train_scores.npy'}")

print("\nScoring held-out queries...")
held_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_texts = build_text(held_df, include_fulltext=False)
held_ids = held_df["doc_id"].tolist()

scores_held = bm25_scores_batch(held_texts)
print(f"  Held-out scores shape: {scores_held.shape}")

np.save(CACHE_DIR / "bm25_held_scores.npy", scores_held.astype(np.float32))
with open(CACHE_DIR / "bm25_held_query_ids.json", "w") as f: json.dump(held_ids, f)
print(f"  Saved held-out scores -> {CACHE_DIR / 'bm25_held_scores.npy'}")