# Citation Retrieval Pipeline

Academic citation retrieval: given a query paper, retrieve a ranked list of 100 relevant corpus documents.

- **Metric**: NDCG@10
- **Corpus**: 20,000 documents
- **Queries**: 100 train (with qrels) + 100 held-out (evaluated on Codabench)
- **Best train score**: 0.7401
- **Competitor target**: 0.7489 (Codabench held-out)

---

## Best Pipeline

Implemented in `models/pipeline.py`. Signals combined in score space:

```
0.5  * cosine(ft_small_query,  ft_small_doc)       finetuned bge-small
0.5  * cosine(bge_large_query, bge_large_doc)       bge-large-en-v1.5
0.35 * mm(BM25)^1.2                                 BM25 (title+abstract query)
0.38 * (query_domain == doc_domain)                 hard domain boost
0.15 * cosine(pooled_cite_bge, bge_large_doc)       pooled citation context
0.95 * max_i cosine(cite_bge_i, bge_large_doc)      per-cite bge max
0.35 * max_i cosine(cite_e5_i,  e5_large_doc)       per-cite e5 max
0.07 * max_i cosine(cite_bge_i, bge_chunk_doc)      chunk-level cite signal
```

```bash
python models/pipeline.py --split train        # evaluate on train
python models/pipeline.py --split held_out --zip  # generate submission
```

---

## Setup

### Prerequisites

Run once before using `models/pipeline.py`:

```bash
# 1. Precompute BM25 score cache
python scripts/precompute_bm25.py

# 2. Embed corpus and queries
python scripts/embed.py --model BAAI/bge-large-en-v1.5 --split train
python scripts/embed.py --model BAAI/bge-large-en-v1.5 --split held_out
python scripts/embed.py --model intfloat/e5-large-v2 --split train
python scripts/embed.py --model intfloat/e5-large-v2 --split held_out

# 3. Extract and embed citation context sentences
python scripts/embed_cite_contexts.py --split train
python scripts/embed_cite_contexts.py --split held_out

# 4. Embed corpus body chunks
python scripts/embed_corpus_chunks.py --model BAAI/bge-large-en-v1.5
```

---

## Project Structure

```
models/          Retrieval models — each writes results/ and submissions/
  pipeline.py      Best full pipeline (train=0.7401)
  dense.py         Bi-encoder dense retrieval
  bm25.py          BM25 retrieval (rank_bm25, slow — use precompute_bm25.py for pipeline)
  tfidf.py         TF-IDF retrieval
  rrf.py           Reciprocal Rank Fusion
  reranker.py      Cross-encoder reranking
  score_fusion.py  Score-level fusion (ft_small + domain + cite)
  cite_bm25.py     BM25 using citation sentences as query
  cite_context.py  Dense retrieval with citation context signal
  fullchunk.py     Multi-vector retrieval using body chunks

scripts/         Data preparation and experiments
  embed.py                  Embed corpus/queries with any SentenceTransformer model
  embed_cite_contexts.py    Extract and embed citation sentences
  embed_cite_ft.py          Embed citation contexts with finetuned model
  embed_chunks.py           Embed query body chunks
  embed_corpus_chunks.py    Embed corpus body chunks
  precompute_bm25.py        Precompute BM25 scores to disk (required by pipeline)
  precompute_bm25_cite.py   Precompute cite-sentence BM25 scores
  gen_train_submission.py   Generate train submission for hard negatives
  finetune_biencoder.py     Fine-tune a bi-encoder with hard negatives
  finetune_crossencoder.py  Fine-tune a cross-encoder reranker
  crossencoder_rerank.py    Experiment: CE reranking grid search (did not beat 0.7401)
  deep_error_analysis.py    Per-query error analysis
  test_specter2_queryexp.py Experiment: Specter2 query expansion

tuning/          Weight search scripts
  tune_bm25.py         Grid search over BM25 k1/b parameters
  cv_weights_fast.py   5-fold CV weight search for pipeline signals

results/         CSV logs of all evaluation runs (appended, not overwritten)
submissions/     Generated submission files (.json + .zip)
data/
  corpus.parquet            20k docs: doc_id, title, abstract, full_text, domain, venue, year
  queries.parquet           100 train queries
  held_out_queries.parquet  100 held-out queries
  qrels.json                {query_id: [relevant_doc_id, ...]}
  bm25_cache/               Pre-computed BM25 score matrices
  embeddings/               Pre-computed embeddings by model name
  finetuned_models/         Fine-tuned model checkpoints
helpers.py       Shared utilities: load_qrels, load_queries, evaluate, save_results
```

---

## Key Findings

- **Local optimum confirmed**: coordinate ascent + exhaustive signal testing shows 0.7401 is a local maximum on train
- **Train/held-out gap (~0.03)**: likely due to 88% cite coverage on held-out vs 97% on train, and slight domain overfitting
- **GPU needed for further gains**: bi-encoder finetuning is infeasible on CPU (15+ hours/epoch for bge-large)
- **All additional signals tested hurt**: extra dense models (Specter2, SciNCL, BGE-base, E5-base), rank normalization, BM25 variants, PRF, venue matching — all reduced train NDCG
