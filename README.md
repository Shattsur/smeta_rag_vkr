# smeta_rag_project

Репозиторий с **gen_rag** — гибридный retrieval (BM25 + вектор) с RRF-fusion и дообученным реранкером (NVIDIA Nemotron 1B, 4-bit LoRA); параметры подбираются NSGA-II. Достигнутый общий score в эксперименте: **2.9736 / 3.0**.

Общий код парсинга источников и нормализации метаданных чанков вынесен в **`shared/`** и подключается из `gen_rag`.

## Структура

- `.env.example` — шаблон переменных окружения.
- `.gitignore`, `pyproject.toml`, `requirements.txt` — конфигурация репозитория.
- `smeta_rag_env/` — виртуальное окружение (локально; в `.gitignore`).
- `shared/` — библиотека для монорепо:
  - `shared/chunking/chunk_metadata.py` — метаданные чанков (используется `gen_rag`).
  - `shared/prepare_db/` — парсеры PDF, ФСНБ XML и QA; дефолты путей относительны CWD — **запускайте скрипты из каталога `gen_rag/`**.
  - `shared/io/`, `shared/llm/` — заготовки под утилиты.
- `gen_rag/` — основной пакет (см. `gen_rag/README.md`):
  - `.env`, `.env.example`, `requirements.txt`.
  - `data/`, `chroma_db/`, `models/`, `output/` — артефакты (в `.gitignore`).
  - `scripts/` — пайплайн RAG и обучение/оптимизация.
- `archive/` — исторические версии скриптов.

## Установка

С активированным окружением (например `smeta_rag_env`):

```bash
pip install -r requirements.txt
pip install -r gen_rag/requirements.txt
pip install -e .
```

Без `pip install -e .` импорты `shared` и `gen_rag` из скриптов всё равно работают за счёт `sys.path` в точках входа; editable-установка удобнее для REPL и тестов.

## Запуск RAG

Из каталога `gen_rag/`, чтобы совпали относительные пути и `.env`:

```bash
cd gen_rag
python scripts/rag_gen.py --ask "Ваш вопрос"
```

Парсеры (`shared/prepare_db/`, обёртки в `gen_rag/scripts/prepare_db/`) рассчитаны на **текущую рабочую директорию `gen_rag/`** (пути вида `./data/raw/...`, `./data/chunks/...`).

## Ответственности

| Ресурс | Расположение |
|--------|----------------|
| Git, `pyproject.toml`, корневой `requirements.txt` | корень |
| Виртуальное окружение | корень (`smeta_rag_env/` — локально) |
| Парсинг источников, метаданные чанков | `shared/` |
| Данные, Chroma, модели реранкера, скрипты RAG | `gen_rag/` |

Подробнее по сценариям подготовки данных и обучения — в **`gen_rag/README.md`**.
