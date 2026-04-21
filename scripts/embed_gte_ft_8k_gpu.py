"""
Embed corpus + train + held-out queries with Alibaba-NLP/gte-modernbert-base
using full_text at max_seq_length=8192.

Saves to data/embeddings/gte_modernbert_fulltext_8k/.
Uses batch_size=4, max_seq_length=8192. ETA ~8-9h on RTX 3060 Ti.

CHUNK-LEVEL CHECKPOINTING: corpus is saved in chunks of CHUNK_SIZE docs.
Kill & restart at any time — it resumes from the last completed chunk.
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "embeddings" / "gte_modernbert_fulltext_8k"
CHUNK_DIR = OUT_DIR / "corpus_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "train").mkdir(exist_ok=True)
(OUT_DIR / "held_out").mkdir(exist_ok=True)

MODEL_NAME  = "Alibaba-NLP/gte-modernbert-base"
MAX_SEQ_LEN = 8192
BATCH_SIZE  = 4
CHUNK_SIZE  = 1000   # save every 1000 docs 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

PROGRESS_FILE = Path(__file__).parent / "embed_gte_ft_8k_progress.txt"

def log_progress(msg):
    print(msg, flush=True)
    with open(PROGRESS_FILE, "w") as f:
        f.write(msg + "\n")


def get_fulltext(row, col="full_text", fallback_col="ta"):
    v = row.get(col)
    if v and str(v).strip():
        return str(v)
    v2 = row.get(fallback_col)
    return str(v2) if v2 else ""


def chunk_path(idx):
    return CHUNK_DIR / f"chunk_{idx:04d}.npy"


def encode_batch(model, texts):
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return embs.astype(np.float32)


def encode_corpus_with_checkpoints(model, texts, phase_offset=0.0, phase_range=0.90):
    """Encode corpus texts in chunks of CHUNK_SIZE, saving each chunk to disk.

    Resumes from the last completed chunk if any exist.
    """
    n = len(texts)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    # detect completed chunks
    completed = []
    for ci in range(n_chunks):
        if chunk_path(ci).exists():
            completed.append(ci)
    if completed:
        last_completed = max(completed)
        docs_done = (last_completed + 1) * CHUNK_SIZE
        docs_done = min(docs_done, n)
        log_progress(f"RESUME: {len(completed)}/{n_chunks} chunks exist ({docs_done}/{n} docs done)")
        start_chunk = last_completed + 1
    else:
        start_chunk = 0
        docs_done = 0

    t0 = time.time()
    for ci in range(start_chunk, n_chunks):
        chunk_start = ci * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, n)
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_embs = []
        for i in range(0, len(chunk_texts), BATCH_SIZE):
            batch = chunk_texts[i:i+BATCH_SIZE]
            embs = encode_batch(model, batch)
            chunk_embs.append(embs)
            # intra-chunk progress every 20 batches 
            if (i // BATCH_SIZE) % 20 == 0 and i > 0:
                intra_done = chunk_start + i + len(batch)
                elapsed = time.time() - t0
                intra_pct = intra_done / n
                intra_global_pct = (phase_offset + intra_pct * phase_range) * 100
                session_done = intra_done - (start_chunk * CHUNK_SIZE)
                eta_session = elapsed / max(session_done, 1) * (n - intra_done)
                msg = (f"PROGRESS {intra_global_pct:.1f}% | [corpus] chunk {ci+1}/{n_chunks} "
                       f"({intra_done}/{n} docs, {100*intra_pct:.1f}%) | "
                       f"session {elapsed:.0f}s | ETA {eta_session:.0f}s")
                log_progress(msg)
        chunk_arr = np.concatenate(chunk_embs, axis=0)
        np.save(chunk_path(ci), chunk_arr)

        # progress
        docs_done = chunk_end
        elapsed = time.time() - t0
        chunks_remaining = n_chunks - (ci + 1)
        chunks_done_this_session = ci + 1 - start_chunk
        eta = (elapsed / max(chunks_done_this_session, 1)) * chunks_remaining
        local_pct = docs_done / n
        global_pct = (phase_offset + local_pct * phase_range) * 100
        msg = (f"PROGRESS {global_pct:.1f}% | [corpus] chunk {ci+1}/{n_chunks} "
               f"({docs_done}/{n} docs, {100*local_pct:.1f}%) | "
               f"session {elapsed:.0f}s | ETA {eta:.0f}s")
        log_progress(msg)

    # concatenate all chunks into single corpus_embeddings.npy
    log_progress(f"Concatenating {n_chunks} chunks into single file...")
    all_arrs = [np.load(chunk_path(ci)) for ci in range(n_chunks)]
    result = np.concatenate(all_arrs, axis=0)
    log_progress(f"DONE [corpus]: {result.shape}")
    return result


def encode_simple(model, texts, desc, phase_offset=0.0, phase_range=0.05):
    t0 = time.time()
    n = len(texts)
    all_embs = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embs = encode_batch(model, batch)
        all_embs.append(embs)
        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        eta = elapsed / done * (n - done) if done > 0 else 0
        local_pct = done / n
        global_pct = (phase_offset + local_pct * phase_range) * 100
        if (i // BATCH_SIZE) % 5 == 0 or done == n:
            msg = (f"PROGRESS {global_pct:.1f}% | [{desc}] {done}/{n} "
                   f"({100*local_pct:.1f}%) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")
            log_progress(msg)
    result = np.concatenate(all_embs, axis=0)
    elapsed = time.time() - t0
    log_progress(f"DONE [{desc}]: {result.shape} in {elapsed:.0f}s")
    return result


# Corpus (chunked)
corpus_npy = OUT_DIR / "corpus_embeddings.npy"
need_model = not corpus_npy.exists() or not (OUT_DIR / "train" / "query_embeddings.npy").exists() \
             or not (OUT_DIR / "held_out" / "query_embeddings.npy").exists()

if need_model:
    print(f"Loading {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LEN
    model = model.to(device)
    print(f"Loaded. max_seq_length={model.max_seq_length}  dim={model.get_sentence_embedding_dimension()}", flush=True)

if corpus_npy.exists():
    print(f"Corpus already exists: {corpus_npy}  (skipping)", flush=True)
else:
    print("Loading corpus...", flush=True)
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    corpus_ids = corpus["doc_id"].tolist()
    texts = [get_fulltext(row) for _, row in corpus.iterrows()]
    print(f"Encoding {len(texts)} corpus docs (full_text, 8192 tokens, batch=4, chunk={CHUNK_SIZE})...", flush=True)
    embs = encode_corpus_with_checkpoints(model, texts, phase_offset=0.0, phase_range=0.90)
    np.save(corpus_npy, embs)
    with open(OUT_DIR / "corpus_ids.json", "w") as f:
        json.dump(corpus_ids, f)
    print(f"Saved corpus: {embs.shape} -> {corpus_npy}", flush=True)

# Train queries
train_npy = OUT_DIR / "train" / "query_embeddings.npy"
if train_npy.exists():
    print(f"Train already exists (skipping)", flush=True)
else:
    print("Loading train queries...", flush=True)
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    qids = queries["doc_id"].tolist()
    texts = [get_fulltext(row) for _, row in queries.iterrows()]
    embs = encode_simple(model, texts, "train", phase_offset=0.90, phase_range=0.05)
    np.save(train_npy, embs)
    with open(OUT_DIR / "train" / "query_ids.json", "w") as f:
        json.dump(qids, f)
    print(f"Saved train: {embs.shape}", flush=True)

# Held-out queries
held_npy = OUT_DIR / "held_out" / "query_embeddings.npy"
if held_npy.exists():
    print(f"Held-out already exists (skipping)", flush=True)
else:
    print("Loading held-out queries...", flush=True)
    held = pd.read_parquet(DATA_DIR / "held_out_queries.parquet")
    hids = held["doc_id"].tolist()
    texts = [get_fulltext(row) for _, row in held.iterrows()]
    embs = encode_simple(model, texts, "held_out", phase_offset=0.95, phase_range=0.05)
    np.save(held_npy, embs)
    with open(OUT_DIR / "held_out" / "query_ids.json", "w") as f:
        json.dump(hids, f)
    print(f"Saved held-out: {embs.shape}", flush=True)

print("ALL DONE.", flush=True)
