"""
Embed corpus body chunks + train/held-out queries with thenlper/gte-large.


Outputs to data/embeddings/gte_large_chunks/:
  corpus_chunk_embeddings.npy       (N_chunks, 1024)
  corpus_chunk_doc_ids.json         list of doc_ids per chunk
  train/query_embeddings.npy        (100, 1024)
  train/query_ids.json
  held_out/query_embeddings.npy     (100, 1024)
  held_out/query_ids.json

"""

import json, time, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import get_body_chunks

DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "embeddings" / "gte_large_chunks"
CHUNK_DIR = OUT_DIR / "chunk_parts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "train").mkdir(exist_ok=True)
(OUT_DIR / "held_out").mkdir(exist_ok=True)

MODEL_NAME = "thenlper/gte-large"
BATCH_SIZE = 32
PART_SIZE  = 2000   # chunks per saved part 

PROGRESS_FILE = Path(__file__).parent / "embed_gte_large_progress.txt"


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS_FILE, "w") as f:
        f.write(msg + "\n")


def part_path(i):
    return CHUNK_DIR / f"part_{i:04d}.npy"


def encode_batch(model, texts):
    with torch.no_grad():
        e = model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )
    return e.astype(np.float32)


def encode_with_resume(model, texts, desc, phase_offset, phase_range):
    """Encode with part-level checkpoints for resume capability."""
    n = len(texts)
    n_parts = (n + PART_SIZE - 1) // PART_SIZE
    existing = [i for i in range(n_parts) if part_path(i).exists()]
    start_i = max(existing) + 1 if existing else 0
    if existing:
        log(f"RESUME: {len(existing)}/{n_parts} parts exist, continuing from part {start_i}")
    t0 = time.time()
    for pi in range(start_i, n_parts):
        s = pi * PART_SIZE
        e = min(s + PART_SIZE, n)
        part_texts = texts[s:e]
        part_embs = []
        for i in range(0, len(part_texts), BATCH_SIZE):
            batch = part_texts[i:i + BATCH_SIZE]
            part_embs.append(encode_batch(model, batch))
        arr = np.concatenate(part_embs, axis=0)
        np.save(part_path(pi), arr)
        elapsed = time.time() - t0
        done_session = pi + 1 - start_i
        eta = elapsed / max(done_session, 1) * (n_parts - pi - 1)
        local_pct = (pi + 1) / n_parts
        global_pct = (phase_offset + local_pct * phase_range) * 100
        log(f"PROGRESS {global_pct:.1f}% | [{desc}] part {pi+1}/{n_parts} "
            f"({e}/{n} chunks) | session {elapsed:.0f}s | ETA {eta:.0f}s")
    log(f"Concatenating {n_parts} parts...")
    all_arrs = [np.load(part_path(i)) for i in range(n_parts)]
    result = np.concatenate(all_arrs, axis=0)
    log(f"DONE [{desc}]: {result.shape}")
    return result


def encode_simple(model, texts, desc, phase_offset, phase_range):
    t0 = time.time()
    n = len(texts)
    all_embs = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        all_embs.append(encode_batch(model, batch))
        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        local_pct = done / n
        global_pct = (phase_offset + local_pct * phase_range) * 100
        if (i // BATCH_SIZE) % 2 == 0 or done == n:
            log(f"PROGRESS {global_pct:.1f}% | [{desc}] {done}/{n} | {elapsed:.0f}s")
    return np.concatenate(all_embs, axis=0)


# Main

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")
if device == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

corpus_emb_path = OUT_DIR / "corpus_chunk_embeddings.npy"
train_emb_path  = OUT_DIR / "train" / "query_embeddings.npy"
held_emb_path   = OUT_DIR / "held_out" / "query_embeddings.npy"

need_model = (not corpus_emb_path.exists()) or (not train_emb_path.exists()) or (not held_emb_path.exists())
if need_model:
    log(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model = model.to(device)
    log(f"Loaded. dim={model.get_sentence_embedding_dimension()}")

#  Corpus body chunks 
if corpus_emb_path.exists():
    log(f"Corpus chunks already exist, skipping.")
else:
    log("Loading corpus...")
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    log(f"Extracting body chunks (min_chars=100, NO per-doc cap)...")
    chunk_texts = []
    chunk_doc_ids = []
    docs_with = 0
    for _, row in corpus.iterrows():
        body_chunks = get_body_chunks(row, min_chars=100)
        if not body_chunks:
            continue
        docs_with += 1
        for text in body_chunks:
            chunk_texts.append(text)
            chunk_doc_ids.append(row["doc_id"])
    log(f"  Total chunks: {len(chunk_texts)} from {docs_with}/{len(corpus)} docs")
    lens = [len(t) for t in chunk_texts]
    log(f"  Chunk chars: mean={np.mean(lens):.0f} median={np.median(lens):.0f} max={max(lens)}")

    embs = encode_with_resume(model, chunk_texts, "corpus_chunks",
                              phase_offset=0.0, phase_range=0.92)
    np.save(corpus_emb_path, embs)
    with open(OUT_DIR / "corpus_chunk_doc_ids.json", "w") as f:
        json.dump(chunk_doc_ids, f)
    log(f"Saved corpus chunks: {embs.shape}")

# Train queries (TA text)
if train_emb_path.exists():
    log(f"Train queries already exist, skipping.")
else:
    log("Loading train queries...")
    q_df = pd.read_parquet(DATA_DIR / "queries.parquet")
    qids = q_df["doc_id"].tolist()
    ta_texts = [str(r.get("ta", "") or r.get("abstract", "")) for _, r in q_df.iterrows()]
    embs = encode_simple(model, ta_texts, "train",
                         phase_offset=0.92, phase_range=0.04)
    np.save(train_emb_path, embs)
    with open(OUT_DIR / "train" / "query_ids.json", "w") as f:
        json.dump(qids, f)
    log(f"Saved train: {embs.shape}")

# Held-out queries (TA text)
if held_emb_path.exists():
    log(f"Held-out queries already exist, skipping.")
else:
    log("Loading held-out queries...")
    h_df = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    hids = h_df["doc_id"].tolist()
    ta_texts = [str(r.get("ta", "") or r.get("abstract", "")) for _, r in h_df.iterrows()]
    embs = encode_simple(model, ta_texts, "held_out",
                         phase_offset=0.96, phase_range=0.04)
    np.save(held_emb_path, embs)
    with open(OUT_DIR / "held_out" / "query_ids.json", "w") as f:
        json.dump(hids, f)
    log(f"Saved held-out: {embs.shape}")

log("ALL DONE.")
