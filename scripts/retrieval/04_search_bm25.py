import pickle, re
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
def tokenize(txt):
    return [t.lower() for t in TOKEN.findall(txt)]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pkl", default="embeddings/bm25.pkl")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        obj = pickle.load(f)
    bm25: BM25Okapi = obj["bm25"]; tokenized = obj["tokenized"]; meta = obj["meta"]

    scores = bm25.get_scores(tokenize(args.q))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.k]

    print(f"\n🔎 BM25 query: {args.q}\n")
    for rank, i in enumerate(top, 1):
        m = meta[i]
        print(f"{rank:>2}. score={scores[i]:.2f} | {m['category']}/{m['language']} | {m['source']}")
        print(f"    chunk_id: {m['chunk_id']}")
    print()

if __name__ == "__main__":
    main()
