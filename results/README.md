# Results

All scores are **NDCG@10 on the 100 train queries**. Held-out queries have no public qrels and are scored on Codabench only.

## Score progression

| Model | Train NDCG@10 | Key change | Script |
|-------|---------------|------------|--------|
| TF-IDF baseline | 0.4841 | Lexical bag-of-words | `models/tfidf.py` |
| Dense (MiniLM) | 0.5073 | Single bi-encoder | `models/dense.py` |
| Dense (BGE-large) | 0.6776 | Stronger encoder | `models/dense.py` |
| Weighted-sum pipeline | 0.7401 | 8-signal weighted sum (dense, BM25, domain boost, citation contexts) | `models/weighted_sum_pipeline.py` |
| Domain-filtered retrieval | 0.7377 | Candidates restricted to the query's domain, 7 signals | `models/domain_retrieval.py` |
| Domain-filtered pipeline + cite | 0.7552 | Citation-sentence re-scoring of the hard-domain pool | `models/domain_retrieval_cite.py` |
| **Final reranker** | **0.7603** | 4-feature reranker: reciprocal rank, cite sim, cross-encoder, soft score | `models/final_reranker.py` |

### Held-out (Codabench)

| Submission | Held-out NDCG@10 |
|------------|------------------|
| Weighted-sum pipeline | ~0.71 |

The train → held-out gap (~0.03) comes mainly from two things: weights were selected on train, and only 88% of held-out queries have extractable citation sentences (97% on train), so 12 queries get no citation-context signal.

## What did not help

Measured against the weighted-sum pipeline (0.7401) unless stated otherwise:

- **Extra dense encoders** (SPECTER2, SciNCL, E5, BGE-base) added as signals: every one lowered the score (−0.001 to −0.007).
- **BM25 variants** (title-only or full-text queries, citation sentences as queries, rank or soft-rank normalisation): all worse than BM25 on title + abstract with min-max normalisation and power 1.2.
- **Other metadata and lexical signals** (same-venue boost, TF-IDF title similarity, dense pseudo-relevance feedback): all slightly below the baseline.
- **Rank fusion** (RRF over diverse configurations, `models/rrf.py`): 0.712–0.734, below the score-space weighted sum.
- **Larger first-stage pool** (10,000 instead of 1,000 candidates): no change; the pool was not the bottleneck.
- **Cross-encoder as the sole ranker** (MiniLM, `bge-reranker-base`): 0.27–0.45 on its own. It only became useful as one feature among four in final reranker.

## CSV logs

Every model script appends one row per run to `results/<model>.csv` via `helpers.save_results()`, with NDCG@10, recall and the full hyperparameter string.

| File | Model |
|------|-------|
| `tfidf.csv`, `bm25.csv`, `dense.csv` | Baselines |
| `weighted_sum_pipeline.csv`, `score_fusion.csv`, `rrf.csv` | Weighted-sum pipeline and fusion variants |
| `cite_context.csv`, `cite_bm25.csv`, `chunk_maxsim.csv` | Citation-context and body-chunk retrievers |
| `domain_retrieval.csv`, `domain_retrieval_cite.csv` | Stage 1 |
| `reranker.csv`, `final_reranker.csv` | Stage 2 rerankers |
