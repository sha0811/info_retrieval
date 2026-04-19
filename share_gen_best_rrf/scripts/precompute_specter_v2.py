"""
Generate specter-v2 embeddings for corpus and queries (train + held-out).
Replaces E5 embeddings. Saves in the same format as E5 for seamless integration.

Expected runtime: ~5-10 minutes (GPU recommended)
"""
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import load_queries

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

DATA_DIR = Path(__file__).parent.parent / "data"
EMB_DIR = DATA_DIR / "embeddings" / "specter2"
EMB_DIR.mkdir(parents=True, exist_ok=True)

print("Loading specter2 base model and retrieval adapter...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.to(device)

print("Loading proximity adapter...")
try:
    model.load_adapter("allenai/specter2_proximity", source="hf", load_as="proximity", set_active=True)
    print("Proximity adapter loaded and activated")
except Exception as e:
    print(f"Warning: Failed to load specter2_proximity adapter: {e}")
    print("  Falling back to base model without adapter")

model.set_active_adapters("proximity")
model.to(device)  
model.eval()

def build_text(df, include_fulltext=True):
    texts = []
    sep_token = tokenizer.sep_token 
    
    for _, row in df.iterrows():
        t = str(row.get("title", "") or "").strip()
        a = str(row.get("abstract", "") or "").strip()
        
        parts = []
        if t: parts.append(t)
        if a: parts.append(a)
        
        if include_fulltext:
            f = str(row.get("full_text", "") or "").strip()
            if f:
                parts.append(f)
                
        texts.append(f" {sep_token} ".join(parts))
    return texts

def encode_batch(texts, batch_size=32):
    """Encode texts in batches using transformers and adapters."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        
        # Max length is 512 for Specter 2
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings).astype(np.float32)

print("\nLoading corpus...")
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
corpus_ids = corpus["doc_id"].tolist()

print(f"Building corpus texts for {len(corpus)} documents...")
corpus_texts = build_text(corpus, include_fulltext=True)

print("Encoding corpus with specter-v2...")
corpus_embeddings = encode_batch(corpus_texts)

# Save corpus embeddings
np.save(EMB_DIR / "corpus_embeddings.npy", corpus_embeddings)
with open(EMB_DIR / "corpus_ids.json", "w") as f:
    json.dump(corpus_ids, f)
print(f"Saved corpus embeddings: {corpus_embeddings.shape}")

print("\nLoading train queries...")
queries_df = load_queries(DATA_DIR / "queries.parquet")
train_ids = queries_df["doc_id"].tolist()

print(f"Building train query texts for {len(queries_df)} queries...")
train_texts = build_text(queries_df, include_fulltext=False)

print("Encoding train queries with specter-v2...")
train_embeddings = encode_batch(train_texts)

train_emb_dir = EMB_DIR / "train"
train_emb_dir.mkdir(exist_ok=True)
np.save(train_emb_dir / "query_embeddings.npy", train_embeddings)
with open(train_emb_dir / "query_ids.json", "w") as f:
    json.dump(train_ids, f)
print(f"Saved train query embeddings: {train_embeddings.shape}")

print("\nLoading held-out queries...")
held_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
held_ids = held_df["doc_id"].tolist()

print(f"Building held-out query texts for {len(held_df)} queries...")
held_texts = build_text(held_df, include_fulltext=False)

print("Encoding held-out queries with specter-v2...")
held_embeddings = encode_batch(held_texts)

held_emb_dir = EMB_DIR / "held_out"
held_emb_dir.mkdir(exist_ok=True)
np.save(held_emb_dir / "query_embeddings.npy", held_embeddings)
with open(held_emb_dir / "query_ids.json", "w") as f:
    json.dump(held_ids, f)

print("Done! specter-v2 embeddings generated and saved.")
print(f"  Location: {EMB_DIR}")
print("  Next: Run gen_best_rrf_held.py to use specter-v2 embeddings")
