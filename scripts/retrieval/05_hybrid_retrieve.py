import argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
import re
TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def tokenize(txt): return [t.lower() for t in TOKEN.findall(txt)]

def minmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0: return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embdir", default="embeddings")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--alpha", type=float, default=0.6, help="weight for FAISS (cosine); BM25 gets 1-alpha")
    args = ap.parse_args()

    # load FAISS + meta
    index = faiss.read_index(str(Path(args.embdir) / "index.faiss"))
    meta = pd.read_parquet(Path(args.embdir) / "metadata.parquet")
    st = SentenceTransformer(args.model)

    # FAISS scores
    qv = st.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, args.k * 10)  # возьмём запас
    faiss_ids, faiss_scores = I[0], D[0]

    # BM25 scores
    with open(Path(args.embdir) / "bm25.pkl", "rb") as f:
        obj = pickle.load(f)
    bm25, tokenized = obj["bm25"], obj["tokenized"]
    bm25_scores = bm25.get_scores(tokenize(args.query))

    # соберём кандидатов: top из объединения
    cand = set(faiss_ids.tolist())
    cand |= set(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:args.k*10])

    # нормируем и смешиваем
    fmap = {i:s for i,s in zip(faiss_ids, faiss_scores)}
    bsel = [bm25_scores[i] for i in cand]
    fsel = [fmap.get(i, 0.0) for i in cand]
    fmm, bmm = minmax(fsel), minmax(bsel)
    hybrid = args.alpha * fmm + (1 - args.alpha) * bmm

    order = [x for _, x in sorted(zip(hybrid, list(cand)), key=lambda t: t[0], reverse=True)][:args.k]

    print(f"\n🔎 Hybrid query: {args.query}  (α={args.alpha})\n")
    for rank, i in enumerate(order, 1):
        row = meta.iloc[i]
        print(f"{rank:>2}. {row['category']}/{row['language']} | {row['source']}")
        print(f"    chunk_id: {row['chunk_id']}")
    print()

if __name__ == "__main__":
    main()
