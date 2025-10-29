import argparse, csv, os, yaml
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm


# общий билдер контекста (тонкая обёртка над _ctx_impl.build_context)
from _ctx import build as build_context


# =========================
#  LLM backends
# =========================

def call_llm_openai(system_prompt: str,
                    user_prompt: str,
                    model_name: str,
                    max_tokens: int = 400,
                    temperature: float = 0.2) -> str:
    """Вызов модели OpenAI"""
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR calling OpenAI: {e}]"


def call_llm_gemini(system_prompt: str,
                    user_prompt: str,
                    model_name: str = "gemini-1.5-flash",
                    max_output_tokens: int = 400,
                    temperature: float = 0.2) -> str:
    """Вызов Google Gemini API (новый v1)"""
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



def call_llm_local_stub(system_prompt: str,
                        user_prompt: str,
                        **_) -> str:
    """Оффлайн-заглушка"""
    return (user_prompt[:1200] + "\n\n[local stub: replace with OpenAI or Gemini]").strip()


# =========================
#  Prompts
# =========================

PROMPT_SYS = (
    "Ты — эксперт по нефтегазовой отрасли Казахстана. "
    "Отвечай кратко, по делу, используя только предоставленный контекст. "
    "Если данных недостаточно — честно укажи это. Сохраняй язык вопроса."
)

PROMPT_USER_TMPL = (
    "Вопрос:\n{q}\n\n"
    "Контекст:\n{ctx}\n\n"
    "Требование:\n"
    "- Используй факты только из контекста.\n"
    "- Если вопрос на русском — отвечай по-русски; если на английском — по-английски.\n"
    "- По возможности укажи источник(и) в конце кратко.\n"
)


# =========================
#  IO helpers
# =========================

def detect_question_column(fieldnames: List[str]) -> str:
    """Определяем колонку с вопросом"""
    lower = [c.lower() for c in fieldnames]
    if "question" in lower:
        return fieldnames[lower.index("question")]
    if "вопрос" in lower:
        return fieldnames[lower.index("вопрос")]
    return fieldnames[0]


def collect_sources(items: List[Dict]) -> Tuple[str, str]:
    """Собираем уникальные источники и chunk_id"""
    srcs = []
    chunk_ids = []
    seen = set()
    for it in items:
        s = it.get("source", "")
        if s and s not in seen:
            seen.add(s)
            srcs.append(s)
        chunk_ids.append(it.get("chunk_id", ""))
    return " | ".join(srcs), " | ".join(chunk_ids)


# =========================
#  Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parents[2]
    ap.add_argument("--input", default=base_dir / "submission_template.csv", help="Путь к CSV с вопросами")
    ap.add_argument("--output", required=True, help="Куда сохранить CSV с ответами")
    ap.add_argument("--config",  default=base_dir / "config.yaml", help="Путь к config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    provider = (cfg.get("model", {}).get("provider") or "local").lower()
    model_name = cfg.get("model", {}).get("name", "gpt-4o-mini")
    max_tokens = int(cfg.get("model", {}).get("max_tokens", 400))
    temperature = float(cfg.get("model", {}).get("temperature", 0.2))

    # =========================
    #  Проверка ключей
    # =========================
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path)

    if provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            print("🔑 OpenAI API key loaded successfully.")
        else:
            print("⚠️  OPENAI_API_KEY не найден. Перехожу на локальную заглушку.")
            provider = "local"

    elif provider == "gemini":
        if os.getenv("GEMINI_API_KEY"):
            print("🔑 Gemini API key loaded successfully.")
        else:
            print("⚠️  GEMINI_API_KEY не найден. Перехожу на локальную заглушку.")
            provider = "local"

    else:
        print("ℹ️  Provider set to 'local' — используем оффлайн режим.")

    # =========================
    #  Выбор модели
    # =========================
    if provider == "gemini":
        llm = lambda q, c: call_llm_gemini(PROMPT_SYS, PROMPT_USER_TMPL.format(q=q, ctx=c),
                                           model_name, max_tokens, temperature)
    elif provider == "openai":
        llm = lambda q, c: call_llm_openai(PROMPT_SYS, PROMPT_USER_TMPL.format(q=q, ctx=c),
                                           model_name, max_tokens, temperature)
    else:
        llm = lambda q, c: call_llm_local_stub(PROMPT_SYS, PROMPT_USER_TMPL.format(q=q, ctx=c))

    # =========================
    #  Чтение CSV и прогресс
    # =========================
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        qcol = detect_question_column(reader.fieldnames)

        out_rows = []
        rows_iter = list(reader)
        for r in tqdm(rows_iter, desc="🧩 Processing questions", ncols=100):
            q = (r.get(qcol) or "").strip()
            if not q:
                continue

            ctx, items = build_context(q, args.config)
            ans = llm(q, ctx)
            sources_str, chunk_ids_str = collect_sources(items)

            out_rows.append({
                "question": q,
                "answer": ans,
                "sources": sources_str,
                "chunk_ids": chunk_ids_str
            })

    # =========================
    #  Запись результата
    # =========================
    fieldnames = ["question", "answer", "sources", "chunk_ids"]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"✅ Saved: {args.output}  (rows: {len(out_rows)})")


if __name__ == "__main__":
    main()
