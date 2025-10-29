import argparse, json, pickle, re
from pathlib import Path
import numpy as np, pandas as pd, faiss, yaml

from sentence_transformers import SentenceTransformer
TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def tokenize(txt): return [t.lower() for t in TOKEN.findall(txt)]

def minmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0: return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

def load_graph(graph_json_path: Path):
    if not graph_json_path.exists(): return None
    data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    # упрощённый представление для быстрых соседей
    adj = {}
    for e in data["edges"]:
        a, b, w = e["source"], e["target"], e.get("weight", 1)
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return {"nodes": {n["id"]: n for n in data["nodes"]}, "adj": adj}

def k_hop_neighbors(adj, seeds, k=1, max_neighbors=8):
    seen = set(seeds)
    frontier = set(seeds)
    out = []
    for _ in range(k):
        nxt = set()
        for s in frontier:
            for (v, w) in sorted(adj.get(s, []), key=lambda x: x[1], reverse=True)[:max_neighbors]:
                if v in seen: continue
                seen.add(v); nxt.add(v); out.append((s, v, w))
        frontier = nxt
        if not frontier: break
    return out, list(seen - set(seeds))

def load_chunks_map(chunks_jsonl: Path):
    # сопоставим chunk_id -> текст (обрезка в build_context)
    mp = {}
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            mp[rec["chunk_id"]] = rec["text"]
    return mp

def build_context(query, cfg):
    # загрузка артефактов
    embdir = Path(".")
    index = faiss.read_index(cfg["paths"]["faiss"])
    meta = pd.read_parquet(cfg["paths"]["meta"])
    with open(cfg["paths"]["bm25"], "rb") as f:
        obj = pickle.load(f)
    bm25, tokenized = obj["bm25"], obj["tokenized"]
    model = SentenceTransformer(cfg["model"]["name"] if "sentence-transformers" in cfg["model"]["name"] else "paraphrase-multilingual-MiniLM-L12-v2")

    # FAISS
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, cfg["retrieval"]["top_k_faiss"])
    faiss_ids, faiss_scores = I[0], D[0]

    # BM25
    scores_bm25 = bm25.get_scores(tokenize(query))
    bm25_top = np.argsort(scores_bm25)[::-1][:cfg["retrieval"]["top_k_bm25"]]

    # объединяем кандидатов и считаем гибридный скор
    cand = set(faiss_ids.tolist()) | set(bm25_top.tolist())
    fmap = {i:s for i,s in zip(faiss_ids, faiss_scores)}
    sel_f = [fmap.get(i, 0.0) for i in cand]
    sel_b = [scores_bm25[i] for i in cand]
    hf, hb = minmax(sel_f), minmax(sel_b)
    hybrid = cfg["retrieval"]["alpha"]*hf + (1-cfg["retrieval"]["alpha"])*hb
    order = [x for _,x in sorted(zip(hybrid, list(cand)), key=lambda t:t[0], reverse=True)][:cfg["retrieval"]["top_k_final"]]

    # топ-чанки
    top_rows = meta.iloc[order].copy()
    # подхват текста чанков
    chunks_map = load_chunks_map(Path(cfg["paths"]["chunks_jsonl"]))
    snip_len = int(cfg["context"]["snippet_chars"])
    items = []
    for _, r in top_rows.iterrows():
        tid = r["chunk_id"]
        txt = (chunks_map.get(tid,"") or "")[:snip_len]
        items.append({
            "chunk_id": tid,
            "category": r["category"],
            "language": r["language"],
            "source": r["source"],
            "text": txt
        })

    # расширение по графу
    facts = []
    if cfg["graph"]["enable"]:
        G = load_graph(Path(cfg["paths"]["graph_json"]))
        if G:
            # семена: попробуем выделить узлы по простому токен-матчу
            seeds = set()
            for it in items:
                t = it["text"].lower()
                for nid, node in G["nodes"].items():
                    label = str(node.get("label","")).lower()
                    if label and label in t:
                        seeds.add(nid)
                        if len(seeds) > 32: break
                if len(seeds) > 32: break
            if seeds:
                edges, neigh = k_hop_neighbors(G["adj"], list(seeds), k=int(cfg["graph"]["k_hop"]), max_neighbors=int(cfg["graph"]["max_neighbors"]))
                # оформим факты
                for (a, b, w) in edges[: int(cfg["graph"]["max_neighbors"])]:
                    la = G["nodes"].get(a,{}).get("label", a)
                    lb = G["nodes"].get(b,{}).get("label", b)
                    facts.append(f"Связь: {la} — {lb} (вес {w})")

    # сборка контекста (лимит по символам)
    ctx_parts = []
    for it in items:
        ctx_parts.append(f"[{it['category']}/{it['language']}] {it['source']}\n{it['text']}\n")
    if facts:
        ctx_parts.append("Граф-факты:\n" + "\n".join(facts))
    ctx = "\n---\n".join(ctx_parts)
    return ctx[: int(cfg["context"]["max_chars"])], items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default=Path(__file__).resolve().parents[2] / "config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ctx, items = build_context(args.query, cfg)
    print("\n===== CONTEXT (preview) =====\n")
    print(ctx[:1200] + ("..." if len(ctx) > 1200 else ""))
    print("\n===== SOURCES =====")
    for it in items:
        print(f"- {it['category']}/{it['language']} | {it['source']} | {it['chunk_id']}")

if __name__ == "__main__":
    main()
