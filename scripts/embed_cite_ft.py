"""
Compute citation context embeddings using finetuned bge-small model.
Saves to data/embeddings/data_finetuned_models_BAAI_bge-small-en-v1.5/{split}/
"""
import json, sys, re, numpy as np
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(__file__).parent.parent / "data"
EMB_DIR  = DATA_DIR / "embeddings" / "data_finetuned_models_BAAI_bge-small-en-v1.5"
MODEL_PATH = DATA_DIR / "finetuned_models" / "BAAI_bge-small-en-v1.5"

QUERY_FILES = {
    "train": DATA_DIR / "queries.parquet",
    "held_out": DATA_DIR / "held_out_queries.parquet",
}

BRACKET_PAT = re.compile(r"\[\d[\d,\s\-\u2013;]*\]")
PAREN_CITE_PAT = re.compile(r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s*(?:et\s+al\.?)?\s*,?\s*\d{4}[^)]*\)")
INLINE_CITE_PAT = re.compile(r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s*(?:et\s+al\.?)?\s*\(\d{4}[^)]*\)")
NUMERIC_PAREN_PAT = re.compile(r"(?<![a-zA-Z(])\(\d{1,3}(?:[,;]\s*\d{1,3})*\)(?=[\s,\.;)(]|$)")
ALPHA_BRACKET_PAT = re.compile(r"\[[A-Z][A-Za-z]{0,5}\d{2,4}(?:,\s*[A-Z][A-Za-z]{0,5}\d{2,4})*\]")
ALL_CITE_PATS = [BRACKET_PAT, PAREN_CITE_PAT, INLINE_CITE_PAT, NUMERIC_PAREN_PAT, ALPHA_BRACKET_PAT]

def extract_cite_sents(full_text, max_sents=50):
    sentences = re.split(r"(?<=[.!?])\s+", full_text or "")
    seen, sents = set(), []
    for s in sentences:
        if not any(p.search(s) for p in ALL_CITE_PATS): continue
        clean = s
        for p in ALL_CITE_PATS: clean = p.sub("", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 30 or clean in seen: continue
        seen.add(clean)
        sents.append(clean)
    sents.sort(key=len, reverse=True)
    return sents[:max_sents]


from sentence_transformers import SentenceTransformer
print(f"Loading finetuned model from {MODEL_PATH}...")
model = SentenceTransformer(str(MODEL_PATH))
print("  Loaded.")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "held_out"])
args = parser.parse_args()

out_dir = EMB_DIR / args.split
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nProcessing {args.split}...")
queries_df = pd.read_parquet(QUERY_FILES[args.split])
print(f"  {len(queries_df)} queries")

all_embs = []
all_qids = []

for _, row in queries_df.iterrows():
    qid = row["doc_id"]
    sents = extract_cite_sents(str(row.get("full_text", "") or ""))
    if not sents:
        continue
    embs = model.encode(sents, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    all_embs.append(embs)
    all_qids.extend([qid] * len(sents))

if all_embs:
    all_embs_arr = np.vstack(all_embs).astype(np.float32)
    np.save(out_dir / "ft_cite_context_embeddings.npy", all_embs_arr)
    with open(out_dir / "ft_cite_context_query_ids.json", "w") as f:
        json.dump(all_qids, f)
    n_queries = len(set(all_qids))
    print(f"  Saved {len(all_qids)} embeddings for {n_queries} queries")
    print(f"  Shape: {all_embs_arr.shape}")
else:
    print("  No embeddings generated!")
