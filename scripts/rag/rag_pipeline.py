import os, sys, re, argparse
from pathlib import Path
from dotenv import load_dotenv
import yaml

# ── Fix import path to reuse _ctx.build even on direct run ────────────────────
FILE_DIR = Path(__file__).resolve()
BASE_DIR = FILE_DIR.parents[2]
RAG_DIR = BASE_DIR / "scripts" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
try:
    from _ctx import build as build_context
except Exception as e:
    raise RuntimeError(f"Не могу импортировать _ctx.build из {RAG_DIR}: {e}")

# ── LLM backends ─────────────────────────────────────────────────────────────
def call_llm_openai(system_prompt, user_prompt, model_name, max_tokens=400, temperature=0.2):
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR calling OpenAI: {e}]"

def call_llm_gemini(system_prompt, user_prompt, model_name="gemini-1.5-flash",
                    max_output_tokens=400, temperature=0.2):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            ),
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"[ERROR calling Gemini: {e}]"

def call_llm_local_stub(system_prompt, user_prompt, **_):
    return (user_prompt[:1200] + "\n\n[local stub: replace with OpenAI/Gemini]").strip()


# ── Prompts ──────────────────────────────────────────────────────────────────
PROMPT_SYS = (
    "Ты — эксперт по нефтегазовой отрасли Казахстана. Отвечай кратко и по делу, "
    "используя ТОЛЬКО предоставленный контекст. Если данных недостаточно — укажи это. "
    "Сохраняй язык вопроса. В конце (в обычном режиме) укажи источники и узлы графа."
)

# В RAW-режиме просим НИЧЕГО лишнего не добавлять вообще
PROMPT_SYS_RAW = (
    "Ты — эксперт по нефтегазовой отрасли Казахстана. Отвечай кратко и по делу, "
    "используя ТОЛЬКО предоставленный контекст. Если данных недостаточно — укажи это. "
    "Сохраняй язык вопроса. Ответь ОДНИМ коротким абзацем. "
    "НЕ добавляй никаких разделов, пунктов списка, заголовков, ссылок, источников, узлов графа, сносок или пояснений."
)

PROMPT_USER_TMPL = (
    "Вопрос:\n{q}\n\nКонтекст:\n{ctx}\n\nТребования:\n"
    "- Используй факты только из контекста; без источника — не утверждай.\n"
    "- Краткость; при нехватке данных так и скажи.\n"
)


GRAPH_FACT_RX = re.compile(r"^Связь:\s*(.+?)\s*—\s*(.+?)\s*\(вес", re.IGNORECASE)

def extract_graph_nodes_from_context(ctx: str, limit: int = 8):
    nodes, in_graph = [], False
    for line in ctx.splitlines():
        if "Граф-факты:" in line:
            in_graph = True; continue
        if in_graph:
            m = GRAPH_FACT_RX.search(line)
            if m:
                nodes += [m.group(1).strip(), m.group(2).strip()]
            if len(nodes) >= limit: break
    seen, uniq = set(), []
    for n in nodes:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq[:limit]

def collect_sources(items):
    srcs, chunk_ids, seen = [], [], set()
    for it in items:
        s = it.get("source", "")
        if s and s not in seen:
            seen.add(s)
            srcs.append(f"{it.get('category','?')}/{it.get('language','?')}:{s}")
        chunk_ids.append(it.get("chunk_id",""))
    return srcs, chunk_ids

def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser(description="RAG + Graph k-hop pipeline (CLI)")
    ap.add_argument("--q", "--query", dest="query", required=True, help="Вопрос пользователя")
    ap.add_argument("--config", default=str(BASE_DIR / "config.yaml"), help="Путь к config.yaml")
    ap.add_argument("--provider", choices=["auto","openai","gemini","local"], default="auto",
                    help="Перекрыть провайдера модели (иначе из config.yaml)")
    ap.add_argument("--raw", action="store_true", help="Печатать только текст ответа (для eval)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)

    provider_cfg = (cfg.get("model", {}).get("provider") or "local").lower()
    provider = provider_cfg if args.provider == "auto" else args.provider
    model_name = cfg.get("model", {}).get("name", "gpt-4o-mini")
    max_tokens = int(cfg.get("model", {}).get("max_tokens", 400))
    temperature = float(cfg.get("model", {}).get("temperature", 0.2))

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        provider = "local"
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        provider = "local"

    # Retrieval + Graph
    ctx, items = build_context(args.query, cfg)
    sources, _chunk_ids = collect_sources(items)
    graph_nodes = extract_graph_nodes_from_context(ctx)

    # LLM select
    prompt_sys = PROMPT_SYS_RAW if args.raw else PROMPT_SYS
    if provider == "openai":
        llm = lambda q,c: call_llm_openai(prompt_sys, PROMPT_USER_TMPL.format(q=q, ctx=c),
                                          model_name, max_tokens, temperature)
    elif provider == "gemini":
        llm = lambda q,c: call_llm_gemini(prompt_sys, PROMPT_USER_TMPL.format(q=q, ctx=c),
                                          model_name, max_tokens, temperature)
    else:
        llm = lambda q,c: call_llm_local_stub(prompt_sys, PROMPT_USER_TMPL.format(q=q, ctx=c))

    answer = llm(args.query, ctx)

    # ── RAW mode for evaluators ──
    if args.raw:
        print((answer or "").strip())
        return

    # Pretty output for manual runs
    print("\n===== CONTEXT (preview) =====\n")
    preview = ctx[:1000] + ("…" if len(ctx) > 1000 else "")
    print(preview)

    print("\n===== ANSWER =====\n")
    print(answer)

    print("\n===== CITATIONS =====")
    if sources:
        print("Источники:")
        for s in sources[:12]:
            print("*", s)
    else:
        print("— нет источников")

    if graph_nodes:
        print("\nУзлы графа:")
        print("* " + "; ".join(graph_nodes[:12]))
    else:
        print("\nУзлы графа: — нет")

    print("\nMeta:")
    print("* chunks:", len(items))

if __name__ == "__main__":
    main()
