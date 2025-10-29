import csv, subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

BASE_DIR = Path(__file__).resolve().parents[2]

def get_answer_from_pipeline(query: str) -> str:
    """
    Запускает rag_pipeline.py отдельным процессом в raw-режиме и возвращает чистый ответ.
    """
    proc = subprocess.run(
        ["python", "scripts/rag/rag_pipeline.py", "--q", query, "--provider", "gemini", "--raw"],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )
    out = (proc.stdout or "").strip()
    if not out:
        err = (proc.stderr or "").strip()
        return f"[no answer]{' ' + err if err else ''}"
    return out

# ── Метрики ──────────────────────────────────────────────────────────────────
rouge = Rouge()

def cosine_sim(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])

def compute_metrics(pred: str, gold: str, embed_model=None):
    m = {}
    try:
        smoothie = SmoothingFunction().method4
        m["bleu"] = sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)
        r = rouge.get_scores(pred, gold)[0]["rouge-l"]["f"]
        m["rougeL"] = r
        if embed_model:
            v1 = embed_model.encode([pred])[0]
            v2 = embed_model.encode([gold])[0]
            m["cosine"] = cosine_sim(v1, v2)
    except Exception as e:
        m["error"] = str(e)
    return m

def main():
    qa_file = BASE_DIR / "eval" / "qa_gold.csv"
    out_file = BASE_DIR / "eval" / "results.csv"
    out_file.parent.mkdir(exist_ok=True)

    # эмбеддинги для cosine
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    rows = list(csv.DictReader(open(qa_file, encoding="utf-8")))
    results = []

    for r in tqdm(rows, desc="Evaluating"):
        q, gold = r["question"], r["answer"]
        pred = get_answer_from_pipeline(q)
        metrics = compute_metrics(pred, gold, embed_model)
        results.append({"question": q, "gold": gold, "pred": pred, **metrics})

    # сохранить CSV
    keys = ["question", "gold", "pred", "bleu", "rougeL", "cosine", "error"]
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # агрегаты
    bleus   = [r["bleu"]   for r in results if "bleu"   in r and isinstance(r["bleu"],   (int,float))]
    rouges  = [r["rougeL"] for r in results if "rougeL" in r and isinstance(r["rougeL"], (int,float))]
    cosines = [r["cosine"] for r in results if "cosine" in r and isinstance(r["cosine"], (int,float))]

    summary = f"BLEU={np.mean(bleus):.3f}  ROUGE-L={np.mean(rouges):.3f}  Cosine={np.mean(cosines):.3f}"
    (BASE_DIR / "eval" / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n✅ Done!")
    print(summary)

if __name__ == "__main__":
    main()
