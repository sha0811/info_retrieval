import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "../data"
EMB_DIR = DATA_DIR / "embeddings"

print("Loading embeddings...")

e5_corpus = np.load(EMB_DIR / "intfloat_e5-large-v2" / "corpus_embeddings.npy")
e5_queries = np.load(EMB_DIR / "intfloat_e5-large-v2" / "held_out" / "query_embeddings.npy")
with open(EMB_DIR / "intfloat_e5-large-v2" / "corpus_ids.json") as f:
    corpus_ids = json.load(f)
with open(EMB_DIR / "intfloat_e5-large-v2" / "held_out" / "query_ids.json") as f:
    query_ids = json.load(f)

spec_corpus = np.load(EMB_DIR / "specter2" / "corpus_embeddings.npy")
spec_queries = np.load(EMB_DIR / "specter2" / "held_out" / "query_embeddings.npy")

e5_corpus_norm = e5_corpus / (np.linalg.norm(e5_corpus, axis=1, keepdims=True) + 1e-8)
e5_queries_norm = e5_queries / (np.linalg.norm(e5_queries, axis=1, keepdims=True) + 1e-8)
spec_corpus_norm = spec_corpus / (np.linalg.norm(spec_corpus, axis=1, keepdims=True) + 1e-8)
spec_queries_norm = spec_queries / (np.linalg.norm(spec_queries, axis=1, keepdims=True) + 1e-8)

print(f"Corpus: {e5_corpus.shape[0]} docs, Queries: {e5_queries.shape[0]}")

print("\nComputing hybrid scores...")

e5_scores = np.dot(e5_queries_norm, e5_corpus_norm.T)  # (100, 20000)

spec_scores = np.dot(spec_queries_norm, spec_corpus_norm.T)  # (100, 20000)

scaler = MinMaxScaler()
e5_scores_norm = scaler.fit_transform(e5_scores.T).T
spec_scores_norm = scaler.fit_transform(spec_scores.T).T

alpha = 0.55 
hybrid_scores = alpha * e5_scores_norm + (1 - alpha) * spec_scores_norm

print("Ranking and generating submission...")

submission = {}
for i, qid in enumerate(tqdm(query_ids)):
    scores = hybrid_scores[i]
    top_100_indices = np.argsort(-scores)[:100]
    top_100_docs = [corpus_ids[idx] for idx in top_100_indices]
    submission[str(qid)] = top_100_docs

# ── Save ───────────────────────────────────────────────────────────────────────
output_path = DATA_DIR / "hybrid_e5_specter.json"
with open(output_path, "w") as f:
    json.dump(submission, f)

print(f"\nSaved: {output_path}")
print(f"  Total queries: {len(submission)}")
print(f"  Hybrid weights: E5={alpha:.0%}, Specter-v2={(1-alpha):.0%}")