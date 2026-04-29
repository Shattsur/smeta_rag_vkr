# gen_rag

Рабочая реализация RAG для сметной нормативной документации:
- гибридный retrieval (BM25 + векторный поиск)
- RRF fusion
- нейросетевой реранкер (в т.ч. дообученный Nemotron 1B 4-bit)
- оптимизация параметров NSGA-II

## Структура

```text
gen_rag/
├── .env / .env.example
├── README.md
├── requirements.txt
├── data/                      # локальные данные (gitignored)
├── chroma_db/                 # локальная векторная БД (gitignored)
├── models/                    # локальные модели/адаптеры (gitignored)
├── output/                    # локальные артефакты (gitignored)
└── scripts/
    ├── rag_gen.py
    ├── fit_params_optuna_nsgaii_v3.py
    ├── load_to_chroma_base.py
    ├── reranker_train_nemotron.py
    ├── quantize_nemotron_reranker.py
    ├── rerank_create_dataset_v2.py
    ├── rerank_create_qa_minstroy.py
    ├── push_adapter_to_hf.py
    └── prepare_db/            # совместимые обёртки
```

Парсеры источников вынесены в `shared/prepare_db/` на уровне репозитория; из `gen_rag` вызываются через тонкие обёртки в `gen_rag/scripts/prepare_db/`.

## Установка

Из корня монорепо:

```bash
pip install -r requirements.txt
pip install -r gen_rag/requirements.txt
pip install -e .
```

## Запуск

Запускайте из `gen_rag/`:

```bash
cd gen_rag
python scripts/rag_gen.py --ask "Ваш вопрос"
```

## Подготовка базы знаний

Запускайте из `gen_rag/`:

```bash
# PDF -> chunks
python scripts/prepare_db/parse_pdf.py --input ./data/raw/pdfs --output ./data/chunks

# XML ФСНБ -> chunks
python scripts/prepare_db/parse_fsnb_xml.py --input ./data/raw/fsnb.xml --output ./data/chunks/fsnb_chunks.jsonl

# Объединение
python scripts/prepare_db/merge_and_prepare_chunks.py --out ./data/chunks/all_chunks.jsonl

# Загрузка в Chroma
python scripts/load_to_chroma_base.py --chunks ./data/chunks/all_chunks.jsonl --chroma-path ./chroma_db
```

## Обучение / оптимизация

```bash
python scripts/rerank_create_qa_minstroy.py
python scripts/rerank_create_dataset_v2.py
python scripts/reranker_train_nemotron.py
python scripts/quantize_nemotron_reranker.py
python scripts/fit_params_optuna_nsgaii_v3.py --output ./output/ga_optimization
```

## Программное использование

```python
from scripts.rag_gen import SmetaRAGApp, Settings

settings = Settings(chroma_dir="./chroma_db")
app = SmetaRAGApp(settings)
result = app.query("Как рассчитать накладные расходы?")
print(result["answer"])
```