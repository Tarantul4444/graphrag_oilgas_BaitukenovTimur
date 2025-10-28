import argparse, json, re
from pathlib import Path

from langdetect import detect, DetectorFactory
from PyPDF2 import PdfReader

# Для стабилизации langdetect (детерминированные результаты)
DetectorFactory.seed = 42

# Настройка размеров чанков по категориям
CHUNK_SIZES = {
    "theory": 900,
    "analytics": 1000,
    "law": 500,
    "patents": 700,
    "standards": 500,
}

CATEGORIES = ["theory", "analytics", "law", "patents", "standards"]
LANGS = ["ru", "en"]  # ожидаемые подкаталоги с языком

# --- OCR (опционально) ---
def try_import_ocr_modules():
    """Ленивая загрузка OCR-зависимостей, если включен флаг --use-ocr."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        return convert_from_path, pytesseract
    except Exception as e:
        print(f"⚠️  OCR modules not available: {e}")
        return None, None


def clean_text(text: str) -> str:
    # Удаляем лишние пробелы/переводы строк
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800):
    """Режем по словам, чтобы не ломать предложения грубо.
    chunk_size — в словах.
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return
    for i in range(0, n, chunk_size):
        yield " ".join(words[i : i + chunk_size])


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Быстрое извлечение текста из PDF через PyPDF2."""
    text_parts = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                # Может вернуть None на сканах
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name} with PyPDF2: {e}")
    return clean_text(" ".join(text_parts))


def extract_text_with_ocr(pdf_path: Path, ocr_lang: str = "rus+eng") -> str:
    """Медленное, но универсальное извлечение текста через OCR."""
    convert_from_path, pytesseract = try_import_ocr_modules()
    if convert_from_path is None or pytesseract is None:
        return ""
    try:
        pages = convert_from_path(str(pdf_path))
        text_parts = []
        for img in pages:
            txt = pytesseract.image_to_string(img, lang=ocr_lang) or ""
            if txt:
                text_parts.append(txt)
        return clean_text(" ".join(text_parts))
    except Exception as e:
        print(f"❌ OCR failed for {pdf_path.name}: {e}")
        return ""


def safe_detect_language(sample: str, fallback: str) -> str:
    """Определяем язык урывком текста; при ошибке — фоллбек на язык папки."""
    try:
        sample = sample[:1000] if sample else ""
        if not sample or len(sample) < 20:
            return fallback
        code = detect(sample)
        # normalize: langdetect возвращает 'ru'/'en' и пр.
        return "ru" if code.startswith("ru") else ("en" if code.startswith("en") else fallback)
    except Exception:
        return fallback


def process_pdf_file(
    file_path: Path,
    category: str,
    lang_folder: str,
    out_all_handle,
    per_category_handle,
    min_text_len: int = 120,
    use_ocr: bool = False,
):
    """Обработка одного PDF: извлечение текста, язык, чанкинг и запись записей в JSONL."""
    rel_source = str(file_path)
    base_stem = file_path.stem

    # 1) PyPDF2
    text = extract_text_from_pdf(file_path)

    # 2) Если текста мало — опционально OCR
    if use_ocr and len(text) < min_text_len:
        print(f"⚠️  {file_path.name}: no/low text ({len(text)} chars). Trying OCR…")
        text = extract_text_with_ocr(file_path, ocr_lang="rus+eng")

    # 3) Если всё ещё пусто — пропускаем
    if len(text) < min_text_len:
        print(f"🚫 Skipping {file_path.name}: not enough extractable text ({len(text)} chars).")
        return 0

    # 4) Язык
    detected_lang = safe_detect_language(text, fallback=lang_folder)

    # 5) Размер чанка по категории
    chunk_size = CHUNK_SIZES.get(category, 800)

    # 6) Пишем чанки стримингово
    count = 0
    for i, chunk in enumerate(chunk_text(text, chunk_size=chunk_size)):
        record = {
            "chunk_id": f"{category}_{detected_lang}_{base_stem}_{i}",
            "category": category,
            "language": detected_lang,
            "source": rel_source,
            "text": chunk,
        }
        # в общий
        out_all_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # и в category-файл
        per_category_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

    print(f"✅ {file_path.name}: {count} chunks ({category}/{lang_folder}, size={chunk_size})")
    return count


def walk_category(
    base_dir: Path,
    category: str,
    out_dir: Path,
    out_all_handle,
    min_text_len: int,
    use_ocr: bool,
) -> int:
    """Обрабатывает все PDF в категории и пишет в <category>_chunks.jsonl + all_chunks.jsonl."""
    per_cat_path = out_dir / f"{category}_chunks.jsonl"
    total = 0
    with per_cat_path.open("w", encoding="utf-8") as per_cat_f:
        for lang in LANGS:
            folder = base_dir / category / lang
            if not folder.exists():
                continue
            for pdf in sorted(folder.glob("*.pdf")):
                total += process_pdf_file(
                    file_path=pdf,
                    category=category,
                    lang_folder=lang,
                    out_all_handle=out_all_handle,
                    per_category_handle=per_cat_f,
                    min_text_len=min_text_len,
                    use_ocr=use_ocr,
                )
    print(f"📦 Saved {total} chunks → {per_cat_path.name}")
    return total


def parse_args():
    ap = argparse.ArgumentParser(description="Parse & chunk PDFs into JSONL.")
    ap.add_argument("--base-dir", type=str, default="data", help="Base data dir with category/lang subfolders.")
    ap.add_argument("--out-dir", type=str, default="data/chunks", help="Output directory for JSONL chunks.")
    ap.add_argument("--min-text-len", type=int, default=120, help="Min chars to accept PDF text (skip otherwise).")
    ap.add_argument("--use-ocr", type=int, default=0, help="Use OCR fallback for scanned PDFs (0/1).")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "all_chunks.jsonl"
    total_all = 0

    print("🚀 Starting parse & chunk")
    print(f"   Base dir: {base_dir.resolve()}")
    print(f"   Out dir : {out_dir.resolve()}")
    print(f"   Use OCR : {bool(args.use_ocr)}")
    print(f"   Min len : {args.min_text_len}")

    # Открываем общий файл один раз и пишем стримингово
    with all_path.open("w", encoding="utf-8") as out_all_f:
        for category in CATEGORIES:
            total_all += walk_category(
                base_dir=base_dir,
                category=category,
                out_dir=out_dir,
                out_all_handle=out_all_f,
                min_text_len=args.min_text_len,
                use_ocr=bool(args.use_ocr),
            )

    print(f"🎯 Done. Total chunks written: {total_all}")
    print(f"📄 Combined file: {all_path}")


if __name__ == "__main__":
    main()