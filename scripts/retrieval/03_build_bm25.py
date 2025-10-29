import json, pickle, re
from pathlib import Path
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

# автозагрузка стоп-слов (один раз)
try:
    _ = stopwords.words("russian")
except:
    nltk.download("stopwords")

RU_SW = set(stopwords.words("russian"))
EN_SW = set(stopwords.words("english"))

def load_chunks(path: Path):
    texts, metas = [], []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append({
                "id": i-1,
                "chunk_id": rec.get("chunk_id", f"chunk_{i-1}"),
                "category": rec.get("category", ""),
                "language": rec.get("language", ""),
                "source": rec.get("source", "")
            })
    return texts, metas

TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def tokenize(txt: str, lang: str):
    toks = [t.lower() for t in TOKEN.findall(txt)]
    sw = RU_SW if lang.startswith("ru") else EN_SW
    return [t for t in toks if t not in sw and len(t) > 2]

def main():
    chunks_path = Path("data/chunks/all_chunks.jsonl")
    out_dir = Path("embeddings"); out_dir.mkdir(exist_ok=True, parents=True)

    texts, metas = load_chunks(chunks_path)
    # язык берём из меты, если есть; иначе пытаемся угадать по кириллице
    lang_guess = lambda s: "ru" if re.search("[А-Яа-яЁё]", s) else "en"
    tokenized = [tokenize(t, metas[i].get("language") or lang_guess(t)) for i, t in enumerate(texts)]

    bm25 = BM25Okapi(tokenized)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized, "meta": metas}, f)
    print(f"✅ BM25 built over {len(texts)} chunks → {out_dir/'bm25.pkl'}")

if __name__ == "__main__":
    main()