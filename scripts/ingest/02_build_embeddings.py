import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="data/chunks/all_chunks.jsonl",
                    help="Path to JSONL with chunks")
    ap.add_argument("--outdir", type=str, default="embeddings",
                    help="Output dir for index + metadata")
    ap.add_argument("--model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformers model name")
    ap.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of chunks for quick test")
    return ap.parse_args()

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Загрузка модели
    print(f"🔤 Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    texts = []
    meta = []

    # 2) Стримим чанки
    print(f"📥 Reading chunks from: {args.chunks}")
    for i, rec in enumerate(stream_jsonl(args.chunks), start=1):
        texts.append(rec["text"])
        meta.append({
            "id": i-1,
            "chunk_id": rec.get("chunk_id", f"chunk_{i-1}"),
            "category": rec.get("category", ""),
            "language": rec.get("language", ""),
            "source": rec.get("source", "")
        })
        if args.limit and i >= args.limit:
            break

    if not texts:
        raise SystemExit("No chunks found. Make sure JSONL exists and not empty.")

    df_meta = pd.DataFrame(meta)
    print(f"🧾 Chunks collected: {len(df_meta)}")

    # 3) Эмбеддинги батчами
    def batched(lst, n):
        for k in range(0, len(lst), n):
            yield lst[k:k+n]

    all_vecs = []
    for batch in batched(texts, args.batch_size):
        vecs = model.encode(
            batch,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # важно для cosine через inner product
        )
        all_vecs.append(vecs.astype("float32"))

    X = np.vstack(all_vecs)  # shape = (N, D)
    print(f"🔢 Embeddings shape: {X.shape}")

    # 4) FAISS IndexFlatIP (cosine при нормализованных векторах)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(outdir / "index.faiss"))
    print(f"💾 Saved FAISS index → {outdir / 'index.faiss'}")

    # 5) Метаданные
    df_meta.to_parquet(outdir / "metadata.parquet", index=False)
    with open(outdir / "model.txt", "w", encoding="utf-8") as f:
        f.write(args.model + "\n")

    print(f"💾 Saved metadata → {outdir / 'metadata.parquet'}")
    print("✅ Done.")

if __name__ == "__main__":
    main()
