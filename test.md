# Citation Retrieval Pipeline — Comprehensive Roadmap

## Task
Academic citation retrieval: given a query paper, retrieve a ranked list of 100 corpus documents.
- **Metric**: NDCG@10
- **Corpus**: 20,000 documents with pre-computed embeddings
- **Queries**: 100 train queries (with qrels) + 100 held-out queries (no qrels, evaluated on Codabench)
- **Target**: Beat competitor score of **0.7489** on held-out Codabench

---

## Current Best Scores
| Split | NDCG@10 | Notes |
|---|---|---|
| Train | **0.7401** | Full pipeline (confirmed local optimum) |
| Held-out (Codabench) | **~0.71** | cv5fast submission |
| Competitor (held-out) | **0.7489** | Target to beat |

---

## Best Pipeline (Train = 0.7401)

```python
sc = 0.5 * (ft_small_corpus @ ft_small_query)        # finetuned BGE-small
   + 0.5 * (bge_large_corpus @ bge_large_query)       # BGE-large-en-v1.5
   + 0.35 * mm(bm25_scores)^1.2                       # BM25 (title+abstract query)
   + 0.38 * domain_mask                               # same-domain boost
   + 0.15 * (bge_pool @ bge_large_corpus)             # pooled cite context (mean of all cite sentences)
# Then for top-1000 candidates:
   + 0.95 * max(bge_cite_sentences @ bge_large_corpus[doc])   # per-cite-sentence max-sim
   + 0.35 * max(e5_cite_sentences  @ e5_large_corpus[doc])    # per-cite-sentence max-sim
   + 0.07 * max(bge_cite_sentences @ corpus_body_chunks[doc]) # cite vs corpus body chunks
```

Where:
- `ft_small` = `data/finetuned_models/BAAI_bge-small-en-v1.5` (domain-finetuned on train data)
- `bge_large` = `BAAI/bge-large-en-v1.5` (1024-dim)
- `e5_large` = `intfloat/e5-large-v2` (1024-dim)
- `bm25` = BM25 on full corpus text, queried with title+abstract; scores cached in `data/bm25_cache/`
- `domain_mask` = binary mask (1 if corpus doc has same domain as query, 0 otherwise)
- `bge_pool` = mean-pooled embedding of all citation context sentences for the query
- `bge_cite_sentences` = individual embeddings of sentences containing citation markers `[1]`, `(Author et al., 2020)`, etc.
- `corpus_body_chunks` = body text chunks of corpus documents (154,277 chunks total, ~7.7 chunks/doc)

---

## Available Pre-Computed Embeddings

| Model | Corpus | Train | Held-out | Cite Context | Chunks |
|---|---|---|---|---|---|
| `data_finetuned_models_BAAI_bge-small-en-v1.5` | ✓ | ✓ | ✓ | ft_cite (384-dim) | - |
| `BAAI_bge-large-en-v1.5` | ✓ + chunks | ✓ | ✓ | ✓ (pooled + per-sentence) | corpus body chunks |
| `intfloat_e5-large-v2` | ✓ | ✓ | ✓ | ✓ (pooled + per-sentence) | - |
| `allenai_specter2_base` | ✓ | ✓ | ✓ | - | - |
| `malteos_scincl` | ✓ | ✓ | ✓ | ✓ (per-sentence only) | - |
| `BAAI_bge-base-en-v1.5` | ✓ | ✓ | ✓ | - | - |
| `intfloat_e5-base-v2` | ✓ | ✓ | ✓ | - | - |
| `allenai_specter2` | ✓ | ✓ | - | - | - |
| `BAAI_bge-small-en-v1.5` | ✓ | ✓ | - | - | - |
| `sentence-transformers_all-MiniLM-L6-v2` | ✓ | ✓ | - | - | - |
| `data_finetuned_models_BAAI_bge-large-en-v1.5` | - | empty | - | - | - |

Query body chunks also pre-computed: `BAAI_bge-large-en-v1.5/train/chunk_embeddings.npy` (1005 chunks, 100 queries) and held-out.

---

## Cached Signals

| File | Description |
|---|---|
| `data/bm25_cache/bm25_train_scores.npy` | BM25 scores (100 train queries × 20k docs), query=title+abstract |
| `data/bm25_cache/bm25_held_scores.npy` | BM25 scores for 100 held-out queries |
| `data/bm25_cache/bm25_cite_train_scores.npy` | BM25 using citation sentences as query text |
| `data/bm25_cache/bm25_cite_held_scores.npy` | Same for held-out |
| `data/bm25_cache/bm25_corpus_ids.json` | Doc ID order for BM25 matrix |

---

## Everything Tested (With Results)

### Dense Retrieval Signals — All Tested, All Hurt on Train

| Signal | Weight Tested | Best Score | vs Baseline |
|---|---|---|---|
| Specter2_base direct dense | 0.1–0.5 | 0.7380 | **-0.0021** |
| SciNCL direct dense | 0.1–0.5 | 0.7390 | **-0.0011** |
| BGE-base-en direct dense | 0.1–0.5 | 0.7330 | **-0.0071** |
| E5-base-v2 direct dense | 0.1–0.5 | 0.7372 | **-0.0029** |
| E5-large-v2 direct dense query | 0.1–0.5 | 0.7343 | **-0.0058** |
| Specter2 direct dense | 0.1 | 0.7380 | **-0.0021** |
| Query body chunks max-sim | 0.1–0.5 | 0.7366 | **-0.0035** |
| Corpus body chunk max-sim (BGE query) | 0.1–0.5 | 0.7366 | **-0.0035** |

### Cite Context Signals — All Hurt

| Signal | Weight Tested | Best Score | vs Baseline |
|---|---|---|---|
| SciNCL cite context | 0.05–0.5 | 0.7386 | **-0.0015** |
| E5-large pooled cite | 0.05–0.2 | 0.7396 | **-0.0005** |
| FT-small cite context | 0.05–0.5 | 0.7365 | **-0.0036** |
| Query body chunk vs cite candidates | 0.1–0.5 | 0.7358 | **-0.0043** |

### BM25 Variants — All Hurt

| Signal | Best Score | Notes |
|---|---|---|
| BM25-title only (query=title) | 0.7119 | -0.0282 |
| BM25-fulltext (query=full_text) | 0.7079 | -0.0322 |
| BM25-cite (query=citation sentences) | 0.4253 alone, 0.7258 combined | Hurt badly |
| Rank normalization (1/rank) | 0.7201 | Worse than mm^1.2 |
| Softrank (temp=50) | 0.7034 | Much worse |
| Softrank (temp=100) | 0.7073 | Worse |
| BM25 power 0.5 | 0.7219 | Current power=1.2 is optimal |
| BM25 power 1.0 | 0.7296 | Worse |
| BM25 power 1.5 | 0.7293 | Worse |
| BM25 power 2.0 | 0.7227 | Worse |

### Other Signals — All Hurt

| Signal | Best Score | Notes |
|---|---|---|
| Venue matching (same venue boost) | 0.7364 | 21% of relevant docs share venue |
| TF-IDF title similarity | 0.7327 | -0.0074 |
| TF-IDF title+abstract similarity | 0.7362 | -0.0039 |
| Pseudo-relevance feedback (dense PRF) | 0.7393 | n_fb=20, alpha=0.1 |
| Adaptive weights for no-cite queries | 0.7401 | No change (only 3/100 no-cite in train) |

### Ensemble Methods — All Hurt on Train

| Method | Score | Notes |
|---|---|---|
| RRF (k=60) of diverse configs | 0.7121–0.7341 | 8 configs |
| RRF (k=3,5) BM25+E5+FT | 0.7242–0.7303 | Various combinations |

### Weight Optimization

| Method | Score | Notes |
|---|---|---|
| Coordinate ascent (6 weights, ±0.05–0.2) | 0.7401 | No improvement found |
| 5-fold cross-validation grid search | 0.7402 | Found dom=0.40, bc=0.95 (marginal) |

### Candidate Pool Size

| Pool Size | Score | Notes |
|---|---|---|
| 1000 (current) | 0.7401 | Baseline |
| 10000 | 0.7401 | No change — pool is not the bottleneck |

### Model Finetuning

| Attempt | Status | Notes |
|---|---|---|
| BGE-large-en-v1.5 finetune (batch=2, n_neg=3, lr=1e-5) | Did not save | ~15 hours on CPU, process likely died |
| BGE-large-en-v1.5 finetune (batch=8, n_neg=7) | Segfault | OOM |
| BGE-small-v2 finetune (harder negatives, batch=16) | Started, ~8hr CPU | ~35s/step, too slow |

### Cross-Encoder Reranking (Earlier Sessions)

| Config | NDCG@10 |
|---|---|
| MiniLM-L6 reranker top-100 | 0.4479 |
| BGE-reranker-base top-100 | 0.2704–0.4384 |
| Finetuned CE top-100 | 0.5500 |
| Finetuned CE top-20 | 0.5242 |

---

## Key Data Insights

1. **Candidate recall**: 664/736 relevant docs appear in top-100 before cite scoring; only 7 are at rank >1000 (fundamentally unretrieval)
2. **Hardest query**: `680af879` (Philosophy), relevant doc at rank 6633 dense / 15333 BM25 — nothing can fix this
3. **Cite context coverage**: Train: 97/100 queries have cite context; Held-out: 88/100
4. **No-cite pool embeddings**: For the 12 held-out queries without cite context, pooled embedding = zero vector
5. **Domain overlap**: Train and held-out have same domain distribution; no held-out-only domains
6. **Weakest domains** (train): Philosophy (0.1934), Art (0.4415), Business (0.5963), CS (0.6579), Biology (0.6788)
7. **BM25 is already optimal**: title+abstract query is best; full_text and title-only both worse

---

## Held-out Submissions Generated

| File | Train NDCG | Notes |
|---|---|---|
| `full_pipeline_bm035_bc095_ec035_dom038_pool015_cc007.zip` | 0.7401 | **Full pipeline on held-out** |
| `cv5fast_dom0.40_bc0.95_train0.7402.zip` | 0.7402 | 5-fold CV optimized weights; only 10 top-10 diffs from full_pipeline |
| `v5_full_bm035_bc095_ec035_d038_p015_cc007.zip` | 0.7401 | Same as full_pipeline |
| `v5_full_sp2_01.zip` | 0.7380 | + Specter2 w=0.1 (hurts train) |
| `v5_full_dom020.zip` | 0.7310 | Reduced domain weight |
| `v5_full_dom030.zip` | 0.7396 | Slightly reduced domain |
| `v5_full_bc065_ec035.zip` | 0.7356 | Reduced cite weights |
| `ensemble_rrf_k60.zip` | - | RRF of full_pipeline + cv5fast + dual_cite |
| `dual_cite_held_out.zip` | - | Older submission, ~0.71 Codabench |
| `final_best_wd030_wbc065_we5c035.zip` | - | Older, missing BM25 and pooled cite |

**Key finding**: `full_pipeline` and `cv5fast` differ by only 10 top-10 changes across 100 queries. They should score similarly (~0.71) on Codabench.

---

## Why is Held-out ~0.71 but Train is 0.7401?

Investigated causes:
- **NOT missing signals**: cv5fast uses full pipeline (BM25 + pooled cite + chunk cite) on held-out
- **Likely**: 88% cite coverage on held-out vs 97% on train (12 queries get no cite boost)
- **Likely**: Domain signal slightly overfit (train has rare domains: History, Philosophy, Art — never seen at test time)
- **Possibly**: Cite weights (0.95, 0.35) are over-tuned for train characteristics
- **Gap**: 0.7401 - 0.71 ≈ 0.03 — competitor at 0.7489 has 0.04 more than us

---

## Ideas NOT Yet Tried (or Not Feasible on CPU)

1. **BGE-large finetuning** — infeasible on CPU (15+ hours per epoch)
2. **BGE-small finetuning with harder negatives from full pipeline** — ~8 hours on CPU, partial attempt
3. **LLM-based query expansion** (no LLM API available)
4. **ColBERT-style late interaction** — would need GPU
5. **Learning-to-rank with neural features** — insufficient training data (100 queries)
6. **Different BM25 parameters (k1, b)** — standard values likely near-optimal
7. **Abstract-as-cite-context fallback** for queries without citation markers — would need re-embedding
8. **Larger corpus context** — e.g., embedding references section separately

---

## Existing Scripts

| Script | Purpose |
|---|---|
| `scripts/gen_full_held_out.py` | **Generate full-pipeline held-out submission** |
| `scripts/gen_held_variants.py` | Generate multiple held-out variants with different weights |
| `scripts/test_rank_norm.py` | Rank/softrank BM25 normalization + coordinate ascent |
| `scripts/test_new_dense.py` | Test Specter2, SciNCL, BGE-base, E5-base as signals |
| `scripts/test_ft_cite.py` | Test FT-small cite context signal |
| `scripts/test_prf.py` | Pseudo-relevance feedback |
| `scripts/cv_weights_fast.py` | 5-fold CV weight optimization |
| `scripts/gen_train_submission.py` | Generate train submission (dense+BM25+domain) for hard negatives |
| `scripts/gen_full_hard_negatives.py` | Generate hard negatives from full pipeline |
| `scripts/finetune_biencoder.py` | Finetune bi-encoder models |
| `scripts/embed_cite_contexts.py` | Extract and embed citation sentences |
| `scripts/embed.py` | General embedding script |

---

## File Structure

```
data/
  corpus.parquet              # 20k docs: doc_id, title, abstract, ta, full_text, domain, venue, year
  queries.parquet             # 100 train queries (same schema + n_relevant)
  held_out_queries.parquet    # 100 held-out queries
  qrels.json                  # {query_id: [relevant_doc_id, ...]}
  bm25_cache/                 # pre-computed BM25 score matrices
  embeddings/                 # pre-computed embeddings by model name
  finetuned_models/           # BAAI_bge-small-en-v1.5 (finetuned), crossencoder
submissions/                  # all generated submissions (.json + .zip)
results/                      # CSV logs of past evaluations
models/                       # model classes (dense.py, rrf.py, etc.)
helpers.py                    # load_qrels, load_queries, evaluate
```

---

## Constraints

- **CPU only** (no GPU) — severely limits deep learning
- **100 train queries** — insufficient for reliable weight tuning (overfitting risk)
- **Local optimum confirmed**: coordinate ascent + exhaustive signal testing shows 0.7401 is a local maximum on train
- **Held-out ceiling unknown**: competitor at 0.7489 suggests improvement is possible
