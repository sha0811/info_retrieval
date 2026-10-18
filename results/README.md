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
| **Final reranker** | **0.7509** |

The final reranker loses only ~0.01 between train and held-out (0.7603 → 0.7509). The weighted-sum pipeline lost ~0.03 (0.7401 → ~0.71), mainly because its weights were tuned on train and it leans heavily on citation contexts: only 88% of held-out queries have extractable citation sentences (97% on train), so 12 queries get no citation-context signal.

## What did not help

Measured against the weighted-sum pipeline (0.7401) unless stated otherwise:

- **Extra dense encoders** (SPECTER2, SciNCL, E5, BGE-base) added as signals: every one lowered the score (−0.001 to −0.007).
- **BM25 variants** (title-only or full-text queries, citation sentences as queries, rank or soft-rank normalisation): all worse than BM25 on title + abstract with min-max normalisation and power 1.2.
- **Other metadata and lexical signals** (same-venue boost, TF-IDF title similarity, dense pseudo-relevance feedback): all slightly below the baseline.
- **Rank fusion** (RRF over diverse configurations, `models/rrf.py`): 0.712–0.734, below the score-space weighted sum.
- **Larger first-stage pool** (10,000 instead of 1,000 candidates): no change; the pool was not the bottleneck.
- **Cross-encoder as the sole ranker** (MiniLM, `bge-reranker-base`, reranking a dense or RRF list): 0.27–0.55 on its own (`reranker.csv`). It only became useful as one feature among four in the final reranker.

## CSV logs

Every model script appends one row per run to `results/<model>.csv` via `helpers.save_results()`, with NDCG@10, recall and the full hyperparameter string.

| File | Model |
|------|-------|
| `tfidf.csv`, `bm25.csv`, `dense.csv` | Baselines |
| `weighted_sum_pipeline.csv`, `score_fusion.csv`, `rrf.csv` | Weighted-sum pipeline and fusion variants |
| `cite_context.csv`, `chunk_maxsim.csv` | Citation-context and body-chunk retrievers |
| `cite_bm25.csv` | BM25 with citation sentences as queries (script removed) |
| `domain_retrieval.csv`, `domain_retrieval_cite.csv` | Stage 1 |
| `final_reranker.csv` | Stage 2 (final reranker) |
| `reranker.csv` | Early cross-encoder-only reranking experiments (script removed) |
