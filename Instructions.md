# ⚙️ Инструкция по запуску проекта GraphRAG Oil & Gas (Kazakhstan)

## 0. Требования к окружению
- **OS:** Windows 10/11, macOS или Linux  
- **Python:** 3.10–3.12 (рекомендуется 3.11)  
- **Git:** установлен  
- **Свободное место:** ≥ 3 ГБ  

---

## 1. Установка и активация виртуальной среды
```bash
git clone <URL-репозитория> graphrag_oilgas
cd graphrag_oilgas

# Windows
python -m venv .venv
. .venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Установка зависимостей Python
Создайте файл `requirements.txt` со следующим содержимым:

```text
# --- Base ---
pandas~=2.3.3
numpy~=2.3.4
langchain
faiss-cpu~=1.12.0
networkx~=3.5
sentence-transformers~=5.1.2
openai
rouge-score
rouge
scikit-learn
nltk~=3.9.2
pyarrow
fastparquet
rank_bm25~=0.2.2
pyyaml~=6.0.3
dotenv~=0.9.9
tqdm~=4.67.1
google-genai
requests

# --- PDF and text processing ---
PyPDF2~=3.0.1
langdetect~=1.0.9
pytesseract
pdf2image
pillow

# --- Jupyter and ipynb ---
jupyter
notebook
ipykernel
nbformat
nbconvert

# --- Optional utilities ---
tika
beautifulsoup4
pdfkit
python-dotenv~=1.2.1
```

Установите зависимости:
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Дополнительно для NLTK:
```bash
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
PY
```

## 3. Установка системных инструментов (OCR и PDF)

### Windows (через Chocolatey)
```powershell
choco install -y tesseract poppler
```
Проверьте:
```powershell
tesseract --version
pdfinfo -v
```

Если не находятся — добавьте в PATH:
```
C:\Program Files\Tesseract-OCR\
C:\Program Files\poppler-*\Library\bin\
```

### macOS
```bash
brew install tesseract poppler
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-rus poppler-utils
```

## 4. Настройка .env
Создайте файл `.env` в корне проекта:
```env
GEMINI_API_KEY=ваш_ключ_от_Gemini
# OPENAI_API_KEY=ваш_ключ_от_OpenAI
```

## 5. Структура данных
```kotlin
data/
  theory/
  analytics/
  law/
  patents/
  standards/
```
Внутри каждой папки — PDF или TXT файлы на русском и английском.

## 6. Пошаговый запуск пайплайна

### 6.1 Парсинг и разбиение документов
```bash
python scripts/ingest/01_parse_and_chunk.py --base-dir data --out-dir data/chunks --use-ocr 1
```

### 6.2 Создание эмбеддингов и индекса FAISS
```bash
python scripts/ingest/02_build_embeddings.py
```

### 6.3 BM25 индекс и проверка (опционально)
```bash
python scripts/retrieval/03_build_bm25.py
python scripts/retrieval/04_search_bm25.py --q "тенгиз добыча нефти"
```

### 6.4 Извлечение сущностей и построение графа
```bash
python scripts/graph/01_ner_extract.py
python scripts/graph/02_build_graph.py
```

### 6.5 Запрос через RAG + Graph
```bash
python scripts/retrieval/rag_pipeline.py --q "Каковы запасы нефти в Казахстане?" --provider gemini
```

## 7. Генерация ответов в CSV
```bash
python scripts/rag/02_generate_answers.py --input "./russian_submission_template.csv" --output "./submission_ru.csv"
```

## 8. Оценка качества
```bash
python scripts/eval/01_run_batch_eval.py
```
Результаты сохраняются в `eval/results.csv`.

## 9. Полезные команды
```bash
# Проверка BM25
python scripts/retrieval/04_search_bm25.py --q "тенгиз нефть"

# Гибридный поиск
python scripts/retrieval/05_hybrid_retrieve.py --q "стандарты проб нефти"

# Пайплайн с источниками и узлами
python scripts/retrieval/rag_pipeline.py --q "экспорт нефти Казахстана" --provider gemini
```
