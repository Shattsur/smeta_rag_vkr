# graph_rag

Каркас для будущей реализации Graph RAG.

## Что уже подготовлено

- отдельный подпроект в монорепо
- своя зона артефактов: `data/`, `graph_db/`, `models/`, `output/`
- свой шаблон переменных окружения: `.env.example`
- общий слой парсинга источников: `shared/prepare_db/`

## Рекомендуемая структура

```text
graph_rag/
├── .env.example
├── requirements.txt
├── README.md
├── data/
├── graph_db/
├── models/
├── output/
└── scripts/
    ├── build_graph.py
    ├── load_to_graph_db.py
    ├── rag_graph.py
    └── eval_graph.py
```

## Следующие шаги

1. Выбрать графовое хранилище (Neo4j / NetworkX / другое).
2. Реализовать extraction pipeline (entity/relation) в `scripts/build_graph.py`.
3. Реализовать загрузку графа в `scripts/load_to_graph_db.py`.
4. Реализовать retrieval + answer generation в `scripts/rag_graph.py`.
5. Добавить quality-eval в `scripts/eval_graph.py`.

## Запуск (когда будет реализовано)

```bash
cd graph_rag
python scripts/rag_graph.py --query "Ваш вопрос"
```