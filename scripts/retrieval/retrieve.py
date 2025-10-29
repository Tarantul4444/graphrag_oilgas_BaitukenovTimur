import argparse
from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embdir", type=str, default="embeddings")
    ap.add_argument("--model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    return ap.parse_args()

def main():
    args = parse_args()
    embdir = Path(args.embdir)

    # Load index + metadata
    index = faiss.read_index(str(embdir / "index.faiss"))
    meta = pd.read_parquet(embdir / "metadata.parquet")
    model = SentenceTransformer(args.model)

    q = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q, args.k)  # cosine ~ inner product на нормализованных векторах
    scores = D[0]
    ids = I[0]

    print(f"\n🔎 Query: {args.query}")
    print("Top results:\n")
    for rank, (idx, sc) in enumerate(zip(ids, scores), start=1):
        row = meta.iloc[idx]
        print(f"{rank:>2}. score={sc:.4f} | {row['category']}/{row['language']} | {row['source']}")
        # легкое превью: постараемся вытащить кусок текста (если нужно — можно прочитать из исходника/JSONL)
        print(f"    chunk_id: {row['chunk_id']}")
    print()

if __name__ == "__main__":
    main()
