# Citation Retrieval Pipeline

Academic citation retrieval competition (Codabench). Given a query paper, retrieve a ranked list of 100 corpus documents so that the papers it actually cites appear at the top.

- **Metric**: NDCG@10
- **Corpus**: 20,000 documents
- **Queries**: 100 train (with ground-truth qrels) + 100 held-out (scored on Codabench)
- **Best train score**: 0.7603 (reranker v7)

---

## Quick Start

```bash
pip install -r requirements.txt

# Reproduce the best train score (0.7603) — requires all caches to be present
python models/reranker_v7.py --split train

# Generate the held-out submission
python models/reranker_v7.py --split held_out
```

---

## Architecture Overview

The final pipeline is a two-stage system:

```
┌──────────────────────────────────────────────────────┐
│  STAGE 1 — Hard-domain 7-signal retrieval            │
│  Restrict candidates to the query's domain,          │
│  score with 7 signals, take top 100.                 │
│  → models/hard_pipeline_with_cite.py  (0.7552)       │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 2 — Four-feature reranker                     │
│  score = rr^1.0 + 0.44·bc + 0.04·ce + 0.25·soft     │
│  → models/reranker_v7.py             (0.7603)        │
└──────────────────────────────────────────────────────┘
```

### Score progression

| Model | Train NDCG@10 | Description |
|-------|---------------|-------------|
| TF-IDF baseline | 0.4841 | Lexical bag-of-words |
| Dense (MiniLM) | 0.5073 | Single bi-encoder |
| Dense (BGE-large) | 0.6776 | Stronger encoder |
| Soft pipeline | **0.7401** | 8-signal weighted sum |
| Hard pipeline + cite | **0.7552** | Hard-domain filter + cite re-ranking |
| Reranker v7 | **0.7603** | Joint 4-feature reranker on top of Stage 1 |

---

## Project Structure

```
info_retrieval/
├── helpers.py                   Shared utilities: loaders, evaluate(), save_results(), metrics
│
├── models/                      Retrieval models : each produces submissions/ and logs results/
│   ├── pipeline.py              Soft 8-signal weighted-sum pipeline (0.7401)
│   ├── hard_domain_retrieval.py 7-signal hard-domain pipeline, Stage 1 alone (0.7377)
│   ├── hard_pipeline_with_cite.py Stage 1 + cite-context re-ranking (0.7552)
│   ├── reranker_v7.py           Final 4-feature reranker : best model (0.7603)
│   ├── dense.py                 Standalone bi-encoder dense retriever
│   ├── bm25.py                  BM25 retriever (slow; use precompute_bm25.py for the pipeline)
│   ├── tfidf.py                 TF-IDF baseline
│   ├── rrf.py                   Reciprocal Rank Fusion (tried, did not beat pipeline)
│   ├── score_fusion.py          Early score-fusion experiments
│   ├── cite_context.py          Dense retrieval with citation-context signal
│   └── fullchunk.py             Multi-vector retrieval using body chunks
│
├── scripts/                     Data preparation : run once to build caches
│   ├── embed.py                 Embed corpus/queries with any SentenceTransformer model
│   ├── embed_cite_contexts.py   Extract citation sentences and embed them (BGE + E5)
│   ├── embed_corpus_chunks.py   Embed corpus body chunks with BGE-large
│   ├── embed_gte_ft_8k_gpu.py   Embed corpus/queries with GTE-ModernBERT at 8192 tokens (GPU)
│   ├── embed_gte_large_chunks.py Embed corpus chunks with GTE-large (cMaxSim signal)
│   ├── embed_gte_modernbert.py  Alternative GTE-ModernBERT embedding (standard context)
│   ├── precompute_bm25.py       Precompute and cache BM25 score matrices
│   ├── compute_soft_scores.py   Compute and cache the soft-pipeline score matrix (used by reranker_v7)
│   ├── crossencoder_rerank_v2.py Build CE score cache with BAAI/bge-reranker-base
│   ├── finetune_biencoder.py    Fine-tune BGE-small on (query, gold) pairs with hard negatives
│   ├── error_analysis.py        Per-query NDCG breakdown and oracle-reorder headroom
│   └── deep_error_analysis.py   Soft-pipeline-era error analysis (weak domain diagnostics)
│
├── tuning/                      Hyperparameter search scripts
│   ├── cv_weights_fast.py       5-fold CV grid search over soft-pipeline signal weights
│   ├── tune_bm25.py             BM25 k1/b and score-power grid search
│   └── cv_rerank_v7.py          5-fold CV validating the reranker v7 config (not overfitting)
│
├── results/                     CSV logs of all evaluation runs 
├── submissions/                 Generated submission JSON files
│
├── data/
│   ├── corpus.parquet           20k documents: doc_id, title, abstract, full_text, domain, ...
│   ├── queries.parquet          100 train queries
│   ├── held_out_queries.parquet 100 held-out queries
│   ├── qrels.json               {query_id: [relevant_doc_id, ...]}
│   ├── embeddings/              Pre-computed embeddings, one subdirectory per model
│   ├── bm25_cache/              Pre-computed BM25 score matrices
│   ├── tfidf_cache/             Pre-computed TF-IDF score matrices
│   ├── ce_cache/                Cached cross-encoder scores (train + held-out)
│   ├── soft_scores/             Cached soft-pipeline score matrices (train + held-out)
│   └── finetuned_models/        Fine-tuned model checkpoints
│
└── notebooks/
    └── research_recap.ipynb     Full narrative of the research journey, from baseline to 0.7603
```

---

## Reproducing the Final Pipeline from Scratch

### 1. Pre-compute all caches (one-time, GPU recommended)

```bash
# BM25 and TF-IDF
python scripts/precompute_bm25.py

# Dense embeddings
python scripts/embed.py --model BAAI/bge-large-en-v1.5 --split train
python scripts/embed.py --model BAAI/bge-large-en-v1.5 --split held_out
python scripts/embed.py --model intfloat/e5-large-v2 --split train
python scripts/embed.py --model intfloat/e5-large-v2 --split held_out
python scripts/embed.py --model sentence-transformers/all-MiniLM-L6-v2 --split train
python scripts/embed.py --model sentence-transformers/all-MiniLM-L6-v2 --split held_out

# Citation-context embeddings
python scripts/embed_cite_contexts.py --split train
python scripts/embed_cite_contexts.py --split held_out

# Corpus body chunks
python scripts/embed_corpus_chunks.py

# GTE-ModernBERT full-text at 8192 tokens 
python scripts/embed_gte_ft_8k_gpu.py

# GTE-large chunk embeddings (cMaxSim signal)
python scripts/embed_gte_large_chunks.py

# Domain-finetuned BGE-small
python scripts/finetune_biencoder.py
python scripts/embed.py --model data/finetuned_models/bge-small-finetuned --split train
python scripts/embed.py --model data/finetuned_models/bge-small-finetuned --split held_out

# Soft-pipeline and CE feature caches (needed by reranker_v7)
python scripts/compute_soft_scores.py --split train
python scripts/compute_soft_scores.py --split held_out
python scripts/crossencoder_rerank_v2.py --split train --model BAAI/bge-reranker-base
python scripts/crossencoder_rerank_v2.py --split held_out --model BAAI/bge-reranker-base
```

### 2. Run the final model

```bash
# Train evaluation 
python models/reranker_v7.py --split train

# Held-out submission
python models/reranker_v7.py --split held_out
# → submissions/reranker_v7_held_out.json
```

### 3. Run the soft pipeline (optional, 0.7401)

```bash
python models/pipeline.py --split train
python models/pipeline.py --split held_out
```

---

## Evaluation

```bash
# Evaluate any submission JSON against train qrels
python evaluate.py --submission submissions/reranker_v7_train.json --split train
```

All model scripts also call `helpers.save_results()` which appends a row to `results/<model>.csv` with the NDCG@10, recall, and full hyperparameter string.

---

## Key Design Decisions

**Why hard-domain filtering?** 97.6% of gold citations share the query's domain. Restricting candidates to the same domain then filling with global results cleans up the candidate pool significantly.

**Why citation sentences?** The sentences in a query paper that *mention* citations describe precisely what the cited paper is about, they are far more specific than the query's title or abstract alone.

**Why the 4-feature reranker?** Cross-encoder (CE) and soft-pipeline scores each add only ~+0.001 in isolation, but when combined with a re-tuned cite weight and reciprocal-rank base they interact super-additively, unlocking a broad plateau around 0.760.
