"""
Multi-way RRF exploration across all available train/held-out submission pairs.

Strategy:
1. Evaluate all train submissions individually
2. Try all pairs + triples with best performers
3. Generate held-out zips for best configs
"""
import json, sys, zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_queries, load_qrels, evaluate
import pandas as pd

DATA_DIR        = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
EMB_DIR_E5      = DATA_DIR / "embeddings" / "intfloat_e5-large-v2"
EMB_DIR_FT      = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
EMB_DIR_BGE     = DATA_DIR / "embeddings" / "BAAI_bge-large-en-v1.5"
CACHE_DIR       = DATA_DIR / "bm25_cache"

queries_df    = load_queries(DATA_DIR / "queries.parquet")
qrels         = load_qrels(DATA_DIR / "qrels.json")
query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"].fillna("")))

def rrf(lists, k=60):
    scores = defaultdict(float)
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])

def load_json(p):
    with open(p) as f: return json.load(f)

def ndcg(sub):
    res = evaluate(sub, qrels, ks=[10,100], query_domains=query_domains, verbose=False)
    return res["overall"]["NDCG@10"], res["overall"]["Recall@100"]

# ── Build BM25+e5 train sub from cache ────────────────────────────────────────
print("Building BM25(full)+e5 train submission from cache...")
e5_c  = np.load(EMB_DIR_E5 / "corpus_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "corpus_ids.json") as f: e5_cids = json.load(f)
e5_tq = np.load(EMB_DIR_E5 / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "query_ids.json") as f: e5_tqids = json.load(f)
e5_tqid2i = {q: i for i, q in enumerate(e5_tqids)}

bm25_train = np.load(CACHE_DIR / "bm25_train_scores.npy").astype(np.float32)
with open(CACHE_DIR / "bm25_corpus_ids.json") as f: bm25_cids = json.load(f)
with open(CACHE_DIR / "bm25_train_query_ids.json") as f: bm25_tqids = json.load(f)
bm25_tqi  = {q: i for i, q in enumerate(bm25_tqids)}
bm25_cid2i = {c: i for i, c in enumerate(bm25_cids)}
bm25_to_e5 = np.array([bm25_cid2i.get(c, 0) for c in e5_cids])

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_domain_map  = dict(zip(corpus["doc_id"], corpus["domain"].fillna("")))
corpus_domains_arr = np.array([corpus_domain_map.get(c, "") for c in e5_cids])
domain_masks = {d: corpus_domains_arr == d for d in np.unique(corpus_domains_arr) if d}

def mm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def make_bm25e5_sub(qids, e5_qembs, e5_qid2i, bm25_scores, bm25_qid2i, doms,
                    WB=0.6, WD=0.3, DB=0.2):
    sub = {}
    for qid in qids:
        s_e5   = e5_c @ e5_qembs[e5_qid2i[qid]]
        s_bm25 = bm25_scores[bm25_qid2i[qid]][bm25_to_e5]
        score  = WB * np.power(mm(s_bm25), 1.2) + WD * np.power(mm(s_e5), 0.9)
        q_dom  = doms.get(qid, "")
        if q_dom in domain_masks:
            score += DB * domain_masks[q_dom].astype(np.float32)
        top = np.argsort(-score)[:100]
        sub[qid] = [e5_cids[i] for i in top]
    return sub

sub_bm25e5_train = make_bm25e5_sub(
    e5_tqids, e5_tq, e5_tqid2i, bm25_train, bm25_tqi, query_domains
)

# ── Also build BM25-only train sub ────────────────────────────────────────────
sub_bm25only_train = {}
for qid in bm25_tqids:
    qi = bm25_tqi[qid]
    scores = bm25_train[qi][bm25_to_e5]
    top = np.argsort(-scores)[:100]
    sub_bm25only_train[qid] = [e5_cids[i] for i in top]

# ── Catalog of named train submissions ────────────────────────────────────────
train_subs = {
    "cite"         : load_json(SUBMISSIONS_DIR / "dual_cite_v2_train.json"),
    "bm25+e5"      : sub_bm25e5_train,
    "bm25only"     : sub_bm25only_train,
}

# Load additional submissions if they exist
optional = {
    "bge_large"    : "dense_bge_large.json",
    "ft_small"     : "dense_finetuned_small_train.json",
    "e5_only"      : "dense_e5.json",
    "bge_base"     : "dense_bge.json",
    "multichunk"   : "multichunk_BAAI_bge-large-en-v1.5_train.json",
    "reranked_bge" : "reranked_dense_bge.json",
}
for name, fname in optional.items():
    p = SUBMISSIONS_DIR / fname
    if p.exists():
        train_subs[name] = load_json(p)

# ── Evaluate all individually ──────────────────────────────────────────────────
print("\nIndividual train scores:")
scores_map = {}
for name, sub in train_subs.items():
    n, r = ndcg(sub)
    scores_map[name] = n
    print(f"  {name:<20}  NDCG@10={n:.4f}  R@100={r:.4f}")

# Sort by NDCG
ranked_subs = sorted(scores_map, key=lambda k: -scores_map[k])
print(f"\nRanked: {ranked_subs}")

# ── Pairwise RRF sweep ─────────────────────────────────────────────────────────
print("\nPairwise RRF (k=5 and k=10, best pairs):")
best_ndcg, best_cfg = scores_map[ranked_subs[0]], {"name": ranked_subs[0], "sub": train_subs[ranked_subs[0]]}
pair_results = []

top_subs = ranked_subs[:6]  # Only try top-6 to keep it fast
for a, b in combinations(top_subs, 2):
    for k in [3, 5, 10, 20]:
        for na in [1, 2, 3]:
            for nb in [1]:
                qids = list(train_subs[a].keys() & train_subs[b].keys())
                sub = {qid: rrf([train_subs[a].get(qid,[])] * na +
                                [train_subs[b].get(qid,[])] * nb, k=k)[:100]
                       for qid in qids}
                n, r = ndcg(sub)
                pair_results.append((n, r, f"{a}×{na}+{b}×{nb} k={k}", sub))

pair_results.sort(reverse=True)
print("Top-10 pairs:")
for n, r, name, sub in pair_results[:10]:
    mark = " <-- BEST" if n > best_ndcg else ""
    print(f"  {name:<45}  NDCG@10={n:.4f}  R@100={r:.4f}{mark}")
    if n > best_ndcg:
        best_ndcg = n
        best_cfg = {"name": name, "sub": sub}

# ── Triple RRF (best pair + next component) ────────────────────────────────────
print("\nTriple RRF (best pairs + extra component):")
triple_results = []
top_pairs = [(n, r, nm, s) for n, r, nm, s in pair_results[:5]]

for _, _, pair_name, pair_sub in top_pairs:
    # Parse which subs are in the pair
    for extra in top_subs:
        for k in [3, 5, 10]:
            qids = list(pair_sub.keys())
            # Try adding the extra component
            sub = {qid: rrf(
                [pair_sub.get(qid, [])] + [train_subs[extra].get(qid, [])],
                k=k)[:100] for qid in qids}
            n, r = ndcg(sub)
            label = f"[{pair_name}]+{extra} k={k}"
            triple_results.append((n, r, label, sub))

triple_results.sort(reverse=True)
print("Top-5 triples:")
for n, r, name, sub in triple_results[:5]:
    mark = " <-- BEST" if n > best_ndcg else ""
    print(f"  {name:<55}  NDCG@10={n:.4f}  R@100={r:.4f}{mark}")
    if n > best_ndcg:
        best_ndcg = n
        best_cfg = {"name": name, "sub": sub}

print(f"\n{'='*70}")
print(f"BEST train NDCG@10 = {best_ndcg:.4f}  ({best_cfg['name']})")

# ── Generate held-out for top configs ─────────────────────────────────────────
print("\nBuilding held-out submissions for top configs...")

# Load held-out embeddings
e5_hq = np.load(EMB_DIR_E5 / "held_out" / "query_embeddings.npy").astype(np.float32)
with open(EMB_DIR_E5 / "held_out" / "query_ids.json") as f: e5_hqids = json.load(f)
e5_hqid2i = {q: i for i, q in enumerate(e5_hqids)}

bm25_held = np.load(CACHE_DIR / "bm25_held_scores.npy").astype(np.float32)
with open(CACHE_DIR / "bm25_held_query_ids.json") as f: bm25_hqids = json.load(f)
bm25_hqi  = {q: i for i, q in enumerate(bm25_hqids)}

held_df   = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_doms = dict(zip(held_df["doc_id"], held_df["domain"].fillna("")))

sub_bm25e5_held = make_bm25e5_sub(
    e5_hqids, e5_hq, e5_hqid2i, bm25_held, bm25_hqi, held_doms
)
sub_bm25only_held = {}
for qid in bm25_hqids:
    qi = bm25_hqi[qid]
    scores = bm25_held[qi][bm25_to_e5]
    top = np.argsort(-scores)[:100]
    sub_bm25only_held[qid] = [e5_cids[i] for i in top]

cite_held = load_json(SUBMISSIONS_DIR / "final_best_wd030_wbc065_we5c035.json")

held_subs = {
    "cite"         : cite_held,
    "bm25+e5"      : sub_bm25e5_held,
    "bm25only"     : sub_bm25only_held,
}
optional_held = {
    "bge_large"    : "dense_bge_large_held_out.json",
    "ft_small"     : "dense_finetuned_small_held_out.json",
    "e5_only"      : "dense_e5_held_out.json",
    "bge_base"     : "dense_bge_held_out.json",
    "multichunk"   : "multichunk_BAAI_bge-large-en-v1.5_held_out.json",
}
for name, fname in optional_held.items():
    p = SUBMISSIONS_DIR / fname
    if p.exists():
        held_subs[name] = load_json(p)

def save_held(sub, name):
    safe = name.replace(" ", "_").replace("+", "P").replace("×", "x").replace("/", "-")[:60]
    jp = SUBMISSIONS_DIR / f"multi_rrf_{safe}.json"
    zp = SUBMISSIONS_DIR / f"multi_rrf_{safe}.zip"
    with open(jp, "w") as f: json.dump(sub, f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="submission_data.json")
    print(f"  Saved: {zp.name}  (train NDCG@10 reference: see above)")
    return zp.name

# Generate held-out for top-5 pair configs
print("\nTop pair configs -> held-out:")
saved = []
qids_held = list(cite_held.keys())

for i, (n, r, pair_name, _) in enumerate(pair_results[:5]):
    # Reconstruct the held-out version from the config name
    # Format: "a×na+b×nb k=k"
    try:
        parts, kpart = pair_name.rsplit(" k=", 1)
        k_val = int(kpart)
        lists_held = []
        for comp in parts.split("+"):
            sname, cnt = comp.rsplit("×", 1)
            sname = sname.strip()
            cnt = int(cnt)
            if sname in held_subs:
                lists_held.extend([held_subs[sname]] * cnt)
            else:
                print(f"  [WARN] No held-out for '{sname}' — skipping this config")
                lists_held = []
                break
        if not lists_held:
            continue
        sub_held = {qid: rrf([s.get(qid, []) for s in lists_held], k=k_val)[:100]
                    for qid in qids_held}
        fname = save_held(sub_held, f"pair{i+1}_train{n:.4f}_{pair_name}")
        saved.append((n, fname))
    except Exception as e:
        print(f"  [WARN] Could not generate held-out for '{pair_name}': {e}")

# Also generate held-out for best triple configs
print("\nTop triple configs -> held-out:")
for i, (n, r, triple_name, _) in enumerate(triple_results[:3]):
    try:
        # Format: "[a×na+b×nb k=X]+extra k=Y"
        inner, rest = triple_name.split("]+", 1)
        inner = inner.lstrip("[")
        extra_part, kpart = rest.rsplit(" k=", 1)
        k_val = int(kpart)

        # Build inner from pair_results
        inner_sub_held = None
        for _, _, pname, _ in pair_results[:5]:
            if inner == pname.rsplit(" k=", 1)[0] + " k=" + inner.rsplit("k=",1)[-1] or inner in pname:
                # Try to match
                pass

        # Simpler: re-parse the inner part directly
        inner_parts, inner_k = inner.rsplit(" k=", 1)
        inner_k = int(inner_k)
        lists_inner_held = []
        for comp in inner_parts.split("+"):
            sname, cnt = comp.rsplit("×", 1)
            sname = sname.strip()
            cnt = int(cnt)
            if sname in held_subs:
                lists_inner_held.extend([held_subs[sname]] * cnt)
            else:
                lists_inner_held = []
                break
        if not lists_inner_held:
            print(f"  [WARN] Missing held-out component for '{triple_name}'")
            continue

        inner_sub = {qid: rrf([s.get(qid, []) for s in lists_inner_held], k=inner_k)[:100]
                     for qid in qids_held}

        if extra_part.strip() not in held_subs:
            print(f"  [WARN] No held-out for extra='{extra_part.strip()}'")
            continue

        sub_held = {qid: rrf([inner_sub.get(qid, []), held_subs[extra_part.strip()].get(qid, [])], k=k_val)[:100]
                    for qid in qids_held}
        fname = save_held(sub_held, f"triple{i+1}_train{n:.4f}_{triple_name[:40]}")
        saved.append((n, fname))
    except Exception as e:
        print(f"  [WARN] Could not generate triple held-out: {e}")

print(f"\n{'='*70}")
print(f"Best train NDCG@10 = {best_ndcg:.4f}")
print("\nGenerated held-out zips (sorted by train score):")
for n, fname in sorted(saved, reverse=True):
    print(f"  train={n:.4f}  {fname}")
