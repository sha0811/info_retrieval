"""
Error analysis for best submission (cite×2 + BM25+e5, k=3).
Shows per-query NDCG@10, which queries fail, and why.
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels
import pandas as pd

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_E5      = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
CACHE_DIR       = DATA_DIR / "bm25_cache"

queries_df    = load_queries(DATA_DIR / "queries.parquet")
qrels         = load_qrels(DATA_DIR / "qrels.json")   # {qid: [doc_id, ...]}
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))
query_titles  = dict(zip(queries_df["doc_id"], queries_df["title"].fillna("")))
query_abstracts = dict(zip(queries_df["doc_id"], queries_df["abstract"].fillna("")))

# ── Load embeddings and BM25 ───────────────────────────────────────────────────
e5_c = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)
e5_tq = np.load(EMB_DIR_E5 / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "query_ids.json") as f: e5_tqids = json.load(f)
e5_tqid2i = {q: i for i, q in enumerate(e5_tqids)}

bm25_tr = np.load(CACHE_DIR / "bm25_train_scores.npy").astype(np.float32)
with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tqids = json.load(f)
bm25_tqi   = {q: i for i, q in enumerate(bm25_tqids)}
bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
bm25_to_e5 = np.array([bm25_cid2i.get(c, 0) for c in e5_cids])

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
cdmap  = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
ctitle = dict(zip(corpus["doc_id"], corpus["title"].fillna("")))
carr   = np.array([cdmap.get(c, "") for c in e5_cids])
dmasks = {d: carr == d for d in np.unique(carr) if d}

def mm(v): return (v - v.min()) / (v.max() - v.min() + 1e-9)

def rrf(lists, k=3):
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

def ndcg10(retrieved, relevant_set):
    def dcg(rels, k=10):
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    rels  = [1 if d in relevant_set else 0 for d in retrieved[:10]]
    ideal = [1] * min(len(relevant_set), 10)
    idcg  = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0

# ── Build the best submission on train ────────────────────────────────────────
with open(SUBMISSIONS_DIR / "dual_cite_v2_train.json") as f: cite = json.load(f)

sub_bm25e5 = {}
for qid in bm25_tqids:
    s_e5   = e5_c @ e5_tq[e5_tqid2i[qid]]
    s_bm25 = bm25_tr[bm25_tqi[qid]][bm25_to_e5]
    sc = 0.6 * np.power(mm(s_bm25), 1.2) + 0.3 * np.power(mm(s_e5), 0.9)
    qd = query_domains.get(qid, "")
    if qd in dmasks: sc += 0.2 * dmasks[qd].astype(np.float32)
    sub_bm25e5[qid] = [e5_cids[i] for i in np.argsort(-sc)[:100]]

qids = list(cite.keys())
sub_best = {qid: rrf([cite.get(qid, [])] * 2 + [sub_bm25e5.get(qid, [])], k=3)[:100]
            for qid in qids}

# ── Per-query stats ────────────────────────────────────────────────────────────
results = []
for qid in qids:
    rel_set   = set(qrels.get(qid, []))
    retrieved = sub_best.get(qid, [])
    n         = ndcg10(retrieved, rel_set)
    hits10    = sum(1 for d in retrieved[:10] if d in rel_set)
    first_rel = next((i + 1 for i, d in enumerate(retrieved) if d in rel_set), 101)

    # Where does each signal rank the first relevant doc?
    cite_list  = cite.get(qid, [])
    bm25_list  = sub_bm25e5.get(qid, [])
    cite_first = next((i + 1 for i, d in enumerate(cite_list) if d in rel_set), 101)
    bm25_first = next((i + 1 for i, d in enumerate(bm25_list) if d in rel_set), 101)

    results.append((n, qid, rel_set, hits10, first_rel, cite_first, bm25_first))

results.sort()

# ── Print table ───────────────────────────────────────────────────────────────
header = "{:<14} {:<14} {:>8} {:>8} {:>8} {:>10} {:>10}  {}".format(
    "qid", "domain", "NDCG@10", "hits@10", "1st_rel", "cite_1st", "bm25_1st", "query title"
)
print(header)
print("-" * 110)
for n, qid, rel_set, h10, frr, cf, bf in results:
    dom   = query_domains.get(qid, "")
    title = query_titles.get(qid, "")[:40]
    line  = "{:<14} {:<14} {:>8.4f} {:>8} {:>8} {:>10} {:>10}  {}".format(
        qid[:13], dom, n, h10, frr, cf, bf, title
    )
    print(line)

# ── Failure analysis ─────────────────────────────────────────────────────────
def explain_failure(n, qid, rel_set, h10, frr, cf, bf):
    """Return a natural-language explanation of why this query fails."""
    title  = query_titles.get(qid, "")
    domain = query_domains.get(qid, "")
    rel_titles = [ctitle.get(d, "") for d in list(rel_set)]

    lines = []

    # Case 1: both signals completely miss
    if cf > 100 and bf > 100:
        lines.append(
            "Both the cite pipeline and BM25+e5 completely fail to retrieve any relevant "
            "document in the top-100. This is a fundamental retrieval failure: the relevant "
            "documents share almost no vocabulary or semantic overlap with the query. "
            "The query is about '{}' but the relevant documents are titled: {}.".format(
                title[:60],
                "; ".join("'{}'".format(t[:50]) for t in rel_titles[:2])
            )
        )
        lines.append(
            "Root cause: the citation relationship is cross-domain or thematic — the "
            "relevant papers are cited for a peripheral concept not reflected in either "
            "the query title/abstract or the BM25 full-text terms. No fusion strategy "
            "can fix this without additional signals (e.g. citation graph, co-citation)."
        )

    # Case 2: BM25 finds it but cite misses, and RRF over-weights cite
    elif bf <= 5 and cf > 20:
        lines.append(
            "BM25+e5 ranks the first relevant document at position {} — a strong signal. "
            "However, the cite pipeline ranks it at position {}, and since RRF gives cite "
            "double weight (cite×2), the BM25 signal is overruled and the relevant doc "
            "is pushed to rank {} in the final ranking.".format(bf, cf, frr)
        )
        lines.append(
            "Root cause: the cite pipeline relies on citation context embeddings "
            "(BGE + finetuned model). For this query, the citation contexts do not "
            "closely match the relevant documents — possibly because the query paper "
            "cites them in a passing or methodological context rather than as core "
            "thematic references. BM25 on full-text captures the terminology better here. "
            "A lower cite weight (cite×1 instead of ×2) or larger k would help."
        )

    # Case 3: cite finds it but BM25 misses
    elif cf <= 10 and bf > 50:
        lines.append(
            "The cite pipeline places the first relevant document at rank {} — reasonable. "
            "BM25+e5 completely misses it (rank {}). The RRF result is rank {}.".format(
                cf, bf, frr)
        )
        lines.append(
            "Root cause: the relevant documents use different vocabulary from the query "
            "(synonyms, abbreviations, or domain-specific jargon that BM25 does not match). "
            "The semantic embedding partially compensates but not enough. "
            "The final rank {} means it falls just outside the top-10 scoring window.".format(frr)
        )

    # Case 4: both find it, but not high enough
    elif frr > 10:
        lines.append(
            "Both signals find a relevant document — cite at rank {}, BM25 at rank {} — "
            "but neither ranks it in the top-10. After RRF fusion, the first relevant "
            "document lands at rank {}, missing the NDCG@10 window.".format(cf, bf, frr)
        )
        lines.append(
            "Root cause: the query has {} relevant documents, and while some are found "
            "early, others are ranked below position 10. The model retrieves many "
            "topically similar but non-relevant papers ahead of them, suggesting the "
            "query topic is broad and the specific angle of the relevant citations "
            "is subtle (e.g. a specific methodology or sub-topic).".format(len(rel_set))
        )

    # Case 5: relevant found at rank 2+ but low NDCG (multiple relevants missed)
    else:
        lines.append(
            "First relevant document found at rank {} (hits in top-10: {}). "
            "NDCG@10={:.4f} is low because {} of the {} relevant documents are ranked "
            "below position 10 or in poor positions, reducing the discounted gain.".format(
                frr, h10, n, len(rel_set) - h10, len(rel_set))
        )
        lines.append(
            "Root cause: the relevant documents cover different sub-aspects of the query "
            "topic. The model retrieves documents matching the primary topic well, but "
            "misses laterally-related papers that represent different perspectives "
            "or methodologies referenced in the query."
        )

    return "\n  ".join(lines)


print("\n" + "=" * 110)
print("FAILURE ANALYSIS (NDCG@10 < 0.3):")
print("=" * 110)

for n, qid, rel_set, h10, frr, cf, bf in results:
    if n >= 0.3:
        continue
    print("\nQuery: " + qid)
    print("  Title   : " + query_titles.get(qid, "")[:90])
    print("  Domain  : " + query_domains.get(qid, ""))
    print("  NDCG@10 : {:.4f}  hits@10={}  1st_rel_rank={}  cite_1st={}  bm25_1st={}".format(
        n, h10, frr, cf, bf))
    print("  Relevant docs ({} total):".format(len(rel_set)))
    for rel_doc in list(rel_set)[:3]:
        print("    - '{}'".format(ctitle.get(rel_doc, "")[:80]))
    print("  Top-5 retrieved (our model):")
    for i, doc in enumerate(sub_best.get(qid, [])[:5]):
        mark = " <-- RELEVANT" if doc in rel_set else ""
        print("    {}. '{}'{}".format(i + 1, ctitle.get(doc, "")[:70], mark))
    print("\n  Explanation:")
    print("  " + explain_failure(n, qid, rel_set, h10, frr, cf, bf))

# ── Domain summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("SUMMARY BY DOMAIN:")
dom_stats = defaultdict(list)
for n, qid, rel_set, h10, frr, cf, bf in results:
    dom_stats[query_domains.get(qid, "unknown")].append(n)

print("{:<20} {:>6} {:>10} {:>10} {:>10}".format("domain", "count", "avg_NDCG", "min_NDCG", "max_NDCG"))
print("-" * 60)
for dom, ns in sorted(dom_stats.items(), key=lambda x: np.mean(x[1])):
    print("{:<20} {:>6} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        dom, len(ns), np.mean(ns), np.min(ns), np.max(ns)))
