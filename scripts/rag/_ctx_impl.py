# scripts/rag/_ctx_impl.py
import json, pickle, re
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

# ---------------- CACHES (singletons) ----------------
_CACHE = {
    "model_name": None,
    "model": None,
    "index_path": None,
    "index": None,
    "meta_path": None,
    "meta": None,
    "bm25_path": None,
    "bm25": None,
    "bm25_tokenized": None,
    "chunks_path": None,
    "chunks_map": None,
    "graph_path": None,
    "graph": None,
}

FALLBACK_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # RU+EN
    "sentence-transformers/paraphrase-MiniLM-L6-v2",                # EN only
]

TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

def tokenize(txt):
    return [t.lower() for t in TOKEN.findall(txt)]

def minmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

# ---------------- Lazy loaders ----------------
def _ensure_model(cfg):
    want = (cfg.get("model", {}) or {}).get("retrieval_model") \
           or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if any(x in want.lower() for x in ["gpt", "o3", "claude", "openai"]):
        want = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if _CACHE["model"] is not None and _CACHE["model_name"] == want:
        return _CACHE["model"]

    # load fresh
    for m in [want] + [x for x in FALLBACK_MODELS if x != want]:
        try:
            print(f"🔹 Loading retrieval model: {m}")
            _CACHE["model"] = SentenceTransformer(m)
            _CACHE["model_name"] = m
            break
        except Exception as e:
            print(f"⚠️  Cannot load model '{m}': {e}")
            _CACHE["model"] = None
    if _CACHE["model"] is None:
        raise RuntimeError("❌ None of the retrieval models could be loaded.")
    return _CACHE["model"]

def _ensure_index(path: str):
    p = str(Path(path))
    if _CACHE["index"] is not None and _CACHE["index_path"] == p:
        return _CACHE["index"]
    _CACHE["index"] = faiss.read_index(p)
    _CACHE["index_path"] = p
    return _CACHE["index"]

def _ensure_meta(path: str):
    p = str(Path(path))
    if _CACHE["meta"] is not None and _CACHE["meta_path"] == p:
        return _CACHE["meta"]
    _CACHE["meta"] = pd.read_parquet(p)
    _CACHE["meta_path"] = p
    return _CACHE["meta"]

def _ensure_bm25(path: str):
    p = str(Path(path))
    if _CACHE["bm25"] is not None and _CACHE["bm25_path"] == p:
        return _CACHE["bm25"], _CACHE["bm25_tokenized"]
    with open(p, "rb") as f:
        obj = pickle.load(f)
    _CACHE["bm25"] = obj["bm25"]
    _CACHE["bm25_tokenized"] = obj["tokenized"]
    _CACHE["bm25_path"] = p
    return _CACHE["bm25"], _CACHE["bm25_tokenized"]

def _ensure_chunks_map(path: str):
    p = str(Path(path))
    if _CACHE["chunks_map"] is not None and _CACHE["chunks_path"] == p:
        return _CACHE["chunks_map"]
    mp = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            mp[rec["chunk_id"]] = rec["text"]
    _CACHE["chunks_map"] = mp
    _CACHE["chunks_path"] = p
    return _CACHE["chunks_map"]

def _ensure_graph(path: str):
    p = str(Path(path))
    if not Path(p).exists():
        _CACHE["graph"] = None
        _CACHE["graph_path"] = p
        return None
    if _CACHE["graph"] is not None and _CACHE["graph_path"] == p:
        return _CACHE["graph"]
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    adj = {}
    for e in data["edges"]:
        a, b, w = e["source"], e["target"], e.get("weight", 1)
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    _CACHE["graph"] = {"nodes": {n["id"]: n for n in data["nodes"]}, "adj": adj}
    _CACHE["graph_path"] = p
    return _CACHE["graph"]

def k_hop_neighbors(adj, seeds, k=1, max_neighbors=8):
    seen, frontier, out = set(seeds), set(seeds), []
    for _ in range(k):
        nxt = set()
        for s in frontier:
            for (v, w) in sorted(adj.get(s, []), key=lambda x: x[1], reverse=True)[:max_neighbors]:
                if v in seen: continue
                seen.add(v); nxt.add(v); out.append((s, v, w))
        frontier = nxt
        if not frontier: break
    return out, list(seen - set(seeds))

# ---------------- Main ----------------
def build_context(query, cfg):
    # artifacts
    index = _ensure_index(cfg["paths"]["faiss"])
    meta = _ensure_meta(cfg["paths"]["meta"])
    bm25, tokenized = _ensure_bm25(cfg["paths"]["bm25"])
    chunks_map = _ensure_chunks_map(cfg["paths"]["chunks_jsonl"])
    G = _ensure_graph(cfg["paths"]["graph_json"]) if cfg["graph"]["enable"] else None

    # model
    model = _ensure_model(cfg)

    # FAISS
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, int(cfg["retrieval"]["top_k_faiss"]))
    faiss_ids, faiss_scores = I[0], D[0]

    # BM25
    scores_bm25 = bm25.get_scores(tokenize(query))
    bm25_top = np.argsort(scores_bm25)[::-1][:int(cfg["retrieval"]["top_k_bm25"])]

    # Hybrid
    cand = set(faiss_ids.tolist()) | set(bm25_top.tolist())
    fmap = {i: s for i, s in zip(faiss_ids, faiss_scores)}
    sel_f = [fmap.get(i, 0.0) for i in cand]
    sel_b = [scores_bm25[i] for i in cand]
    hf, hb = minmax(sel_f), minmax(sel_b)
    hybrid = float(cfg["retrieval"]["alpha"]) * hf + (1 - float(cfg["retrieval"]["alpha"])) * hb
    order = [x for _, x in sorted(zip(hybrid, list(cand)), key=lambda t: t[0], reverse=True)][:int(cfg["retrieval"]["top_k_final"])]

    # snippets
    snip_len = int(cfg["context"]["snippet_chars"])
    items = []
    for _, r in meta.iloc[order].iterrows():
        tid = r["chunk_id"]
        txt = (chunks_map.get(tid, "") or "")[:snip_len]
        items.append({
            "chunk_id": tid,
            "category": r["category"],
            "language": r["language"],
            "source": r["source"],
            "text": txt
        })

    # graph expansion
    facts = []
    if G:
        seeds = set()
        for it in items:
            t = it["text"].lower()
            for nid, node in G["nodes"].items():
                label = str(node.get("label", "")).lower()
                if label and label in t:
                    seeds.add(nid)
                    if len(seeds) > 32: break
            if len(seeds) > 32: break
        if seeds:
            edges, _ = k_hop_neighbors(G["adj"], list(seeds),
                                       k=int(cfg["graph"]["k_hop"]),
                                       max_neighbors=int(cfg["graph"]["max_neighbors"]))
            for (a, b, w) in edges[:int(cfg["graph"]["max_neighbors"])]:
                la = G["nodes"].get(a, {}).get("label", a)
                lb = G["nodes"].get(b, {}).get("label", b)
                facts.append(f"Связь: {la} — {lb} (вес {w})")

    parts = []
    for it in items:
        parts.append(f"[{it['category']}/{it['language']}] {it['source']}\n{it['text']}\n")
    if facts:
        parts.append("Граф-факты:\n" + "\n".join(facts))
    ctx = "\n---\n".join(parts)

    return ctx[:int(cfg["context"]["max_chars"])], items