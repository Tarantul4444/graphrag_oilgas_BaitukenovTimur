import json, re, itertools, pickle
from pathlib import Path
import networkx as nx

# Примеры словарей (можешь расширять из своих документов)
OIL_FIELDS = {"тенгиз", "караб", "карачага", "кашага", "каламкас", "узен", "жетыбай", "кенкияк", "жиентон", "zhien", "tengiz", "karachaganak", "kashagan", "kalamkas", "uzen", "zhetybai", "kenkiyak"}
ORGS = {"kmg", "казмунайгаз", "samruk", "kpo", "tco", "ncoc", "minenergo", "минэнерго", "iea", "opec", "kazenergy"}
TECH = {"полимерное", "депарафинизация", "деэмульгатор", "мембрана", "флэринг", "газлифт", "waterflooding", "eor", "cfz", "amine"}
NORMS = {"ст рк", "gost", "iso", "astm", "тр тс", "тр еаэс", "adilet", "закон", "постановление", "приказ"}
PROC = {"добыча", "подготовка", "транспорт", "переработка", "хранение", "бурение", "сжигание", "flare", "processing", "transportation"}

TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]+")

def detect_entities(text: str):
    t = text.lower()
    toks = set(TOKEN.findall(t))
    nodes = []
    if any(any(f in tok for f in OIL_FIELDS) for tok in toks): nodes.append(("field", "месторождения"))
    if any(any(o in tok for o in ORGS) for tok in toks):      nodes.append(("org", "организации"))
    if any(any(x in tok for x in TECH) for tok in toks):       nodes.append(("tech", "технологии"))
    if any(any(x in tok for x in NORMS) for tok in toks):      nodes.append(("norm", "нормативы"))
    if any(any(x in tok for x in PROC) for tok in toks):       nodes.append(("process", "процессы"))
    # также попробуем вытащить конкретные названия месторождений и орг (по совпадениям)
    fields = [tok for tok in toks if any(f in tok for f in OIL_FIELDS)]
    orgs   = [tok for tok in toks if any(o in tok for o in ORGS)]
    techs  = [tok for tok in toks if any(o in tok for o in TECH)]
    norms  = [tok for tok in toks if any(o in tok for o in NORMS)]
    procs  = [tok for tok in toks if any(o in tok for o in PROC)]
    return {
        "fields": set(fields), "orgs": set(orgs),
        "techs": set(techs),   "norms": set(norms),
        "procs": set(procs)
    }

def main():
    chunks_path = Path("data/chunks/all_chunks.jsonl")
    G = nx.Graph()
    # тип узла пишем в атрибут 'type'
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            ents = detect_entities(rec["text"])
            buckets = [("field", ents["fields"]), ("org", ents["orgs"]), ("tech", ents["techs"]),
                       ("norm", ents["norms"]), ("process", ents["procs"])]
            # добавляем узлы
            present_nodes = []
            for ntype, vals in buckets:
                for v in vals:
                    node_id = f"{ntype}:{v}"
                    if node_id not in G:
                        G.add_node(node_id, type=ntype, label=v)
                    present_nodes.append(node_id)
            # ко-упоминание в одном чанке → ребро
            for a, b in itertools.combinations(sorted(set(present_nodes)), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    out_dir = Path("graph"); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    # также сохраним json (узлы + ребра) для простых визуализаций
    data = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes()],
        "edges": [{"source": u, "target": v, "weight": G[u][v]["weight"]} for u, v in G.edges()]
    }
    import json as _json
    with (out_dir / "graph.json").open("w", encoding="utf-8") as fp:
        _json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"✅ Graph built: {len(G)} nodes, {G.number_of_edges()} edges → graph/graph.gpickle, graph.json")

if __name__ == "__main__":
    main()
