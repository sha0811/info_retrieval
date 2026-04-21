"""
Embed corpus and queries with Alibaba-NLP/gte-modernbert-base (768-dim, 8192-token context).


Uses full text (title + abstract + full_text truncated to 8192 tokens via model's tokenizer).
Progress written to data/embeddings/gte_modernbert/progress.log every 10 batches.
"""

import json, sys, time, numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from helpers import load_corpus, load_queries

DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "embeddings" / "gte_modernbert"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "train").mkdir(exist_ok=True)
(OUT_DIR / "held_out").mkdir(exist_ok=True)

LOG_FILE    = OUT_DIR / "progress.log"
STATUS_FILE = OUT_DIR / "status.txt"
MODEL_ID   = "Alibaba-NLP/gte-modernbert-base"
BATCH_SIZE = 32  # TA-only keeps sequences short and uniform


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def update_status(stage, done, total, elapsed, eta):
    pct = done / total * 100
    eta_str = f"{int(eta//60)}m{int(eta%60)}s" if eta > 0 else "?"
    with open(STATUS_FILE, "w") as f:
        f.write(f"Stage:    {stage}\n")
        f.write(f"Progress: {pct:.1f}%  ({done}/{total} docs)\n")
        f.write(f"Elapsed:  {int(elapsed//60)}m{int(elapsed%60)}s\n")
        f.write(f"ETA:      {eta_str}\n")
        f.write(f"Updated:  {time.strftime('%H:%M:%S')}\n")


def build_text(row):
    title    = str(row.get("title",    "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    parts = [p for p in [title, abstract] if p]
    return " ".join(parts)


def embed_and_save(model, texts, ids, out_npy, out_json, label=""):
    n = len(texts)
    all_embs = []
    t0 = time.time()
    batches = list(range(0, n, BATCH_SIZE))
    pbar = tqdm(batches, desc=label, unit="batch", dynamic_ncols=True)
    for bi, i in enumerate(pbar):
        batch = texts[i:i+BATCH_SIZE]
        for attempt in range(3):
            try:
                embs = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True)
                break
            except RuntimeError as e:
                if attempt < 2:
                    import torch, time as _t
                    log(f"CUDA error on batch {bi+1}, attempt {attempt+1}/3 — clearing cache and retrying: {e}")
                    torch.cuda.empty_cache()
                    _t.sleep(5)
                else:
                    raise
        all_embs.append(embs.astype(np.float32))
        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        pbar.set_postfix({"pct": f"{done/n*100:.1f}%", "ETA": f"{int(eta//60)}m{int(eta%60)}s"})
        update_status(label, done, n, elapsed, eta)
        if bi % 10 == 0:
            log(f"{label}: {done/n*100:.1f}% ({done}/{n}) | {int(elapsed//60)}m{int(elapsed%60)}s elapsed | ETA {int(eta//60)}m{int(eta%60)}s")
    embs_all = np.concatenate(all_embs, axis=0)
    np.save(out_npy, embs_all)
    with open(out_json, "w") as f:
        json.dump(ids, f)
    log(f"{label}: DONE — shape={embs_all.shape} -> {out_npy}")
    return embs_all


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {MODEL_ID} on {device}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device=device)
    log(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Corpus — title + abstract + full_text[:10000]
    log("Building corpus texts...")
    corpus = load_corpus(DATA_DIR / "corpus.parquet")
    corpus_texts = [build_text(row) for _, row in corpus.iterrows()]
    corpus_ids   = corpus["doc_id"].tolist()
    log(f"Corpus: {len(corpus_texts)} docs, sample len={len(corpus_texts[0])}")
    embed_and_save(model, corpus_texts, corpus_ids,
                   OUT_DIR / "corpus_embeddings.npy",
                   OUT_DIR / "corpus_ids.json", label="CORPUS")

    # Train queries
    log("Building train query texts...")
    train_df    = load_queries(DATA_DIR / "queries.parquet")
    train_texts = [build_text(row) for _, row in train_df.iterrows()]
    embed_and_save(model, train_texts, train_df["doc_id"].tolist(),
                   OUT_DIR / "train/query_embeddings.npy",
                   OUT_DIR / "train/query_ids.json", label="TRAIN-Q")

    # Held-out queries
    log("Building held-out query texts...")
    held_df    = load_queries(DATA_DIR / "held_out_queries.parquet")
    held_texts = [build_text(row) for _, row in held_df.iterrows()]
    embed_and_save(model, held_texts, held_df["doc_id"].tolist(),
                   OUT_DIR / "held_out/query_embeddings.npy",
                   OUT_DIR / "held_out/query_ids.json", label="HELD-Q")

    log("ALL DONE. Run test_gte_modernbert.py next.")


if __name__ == "__main__":
    main()
