import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import CrossEncoder

import torch
import pandas as pd

DATA_DIR = Path(__file__).parent / "../data"
EMB_DIR = DATA_DIR / "embeddings"

print("Loading data...")
corpus_df = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")

with open(EMB_DIR / "specter2" / "corpus_ids.json") as f:
    corpus_ids = json.load(f)
with open(EMB_DIR / "specter2" / "held_out" / "query_ids.json") as f:
    query_ids = json.load(f)

print("\nStage 1: Dense retrieval (Specter-v2)...")

spec_corpus = np.load(EMB_DIR / "specter2" / "corpus_embeddings.npy")
spec_queries = np.load(EMB_DIR / "specter2" / "held_out" / "query_embeddings.npy")

spec_corpus_norm = spec_corpus / (np.linalg.norm(spec_corpus, axis=1, keepdims=True) + 1e-8)
spec_queries_norm = spec_queries / (np.linalg.norm(spec_queries, axis=1, keepdims=True) + 1e-8)

scores = np.dot(spec_queries_norm, spec_corpus_norm.T)  # (100, 20000)

top_k = 1000
top_candidates = {}
for i, qid in enumerate(query_ids):
    top_indices = np.argsort(-scores[i])[:top_k]
    top_candidates[str(qid)] = [(corpus_ids[idx], scores[i][idx]) for idx in top_indices]

print(f"Got top-{top_k} candidates per query")

print("\nStage 2: Cross-encoder re-ranking (qnli-distilroberta-base)...")
print("Loading cross-encoder model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder("cross-encoder/qnli-distilroberta-base", max_length=512, device=device)

print(f"Using device: {device.upper()}")

submission = {}

for qid_str in tqdm(top_candidates.keys(), desc="Re-ranking"):
    qid = qid_str
    query_idx = query_ids.index(qid)
    query_text = queries_df.iloc[query_idx]
    query_str = f"{query_text['title']} {query_text['abstract']}"
    
    candidates = top_candidates[qid_str]
    doc_ids = [cid for cid, _ in candidates]
    
    docs = []
    for doc_id in doc_ids:
        doc_idx = corpus_ids.index(doc_id)
        doc = corpus_df.iloc[doc_idx]
        doc_str = f"{doc['title']} {doc['abstract']}"
        docs.append(doc_str)
    
    pairs = [[query_str, doc] for doc in docs]
    ce_scores = model.predict(pairs, batch_size=64)
    
    dense_scores = np.array([score for _, score in candidates])
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
    ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
    
    hybrid_scores = 0.3 * dense_norm + 0.7 * ce_norm
    
    top_100_indices = np.argsort(-hybrid_scores)[:100]
    top_100_docs = [doc_ids[idx] for idx in top_100_indices]
    
    submission[qid_str] = top_100_docs

print(f"Re-ranked all queries")

# ── Save ───────────────────────────────────────────────────────────────────────
output_path = DATA_DIR / "submission_rerank_qnli.json"
with open(output_path, "w") as f:
    json.dump(submission, f)

print(f" Saved: {output_path}")
print(f"  Total queries: {len(submission)}")
print(f"  Hybrid: 30% dense + 70% cross-encoder")
print(f"  Cross-encoder: qnli-distilroberta-base")