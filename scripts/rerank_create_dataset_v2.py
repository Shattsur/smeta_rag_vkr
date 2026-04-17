import os
import json
import random
import re
import time
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from openai import OpenAI
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# ===================== ПАРАМЕТРЫ =====================
DATA_PATHS = [
    r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\raw\gge_qa.jsonl",
    r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\raw\smetnoedelo_qa.cleaned.jsonl",
    r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\rerank\qa_minstroy.jsonl",
]

FSNB_CHUNKS_PATH = r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\raw\fsnb_chunks.jsonl"

TRAIN_PATH = r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\rerank\train_pairs.jsonl"
EVAL_PATH  = r"C:\Files\AI\SmetaGPT\smeta_rag_project\data\rerank\eval_pairs.jsonl"

# ===================== ГИБРИДНЫЕ ПАРАМЕТРЫ MINING =====================
NUM_HARD_NEGATIVES = 7
RANGE_MIN = 10
RANGE_MAX = 150
MAX_SCORE = 0.80
BATCH_SIZE = 16
SEED = 42

RANDOM_NEGATIVES_TRAIN = 3
RANDOM_NEGATIVES_EVAL  = 2

USE_FSNB_CHUNKS_IN_CORPUS = False
USE_FSNB_CHUNKS_IN_RANDOM = False
FSNB_SAMPLE_FRACTION = 0.2

# Синтетика через LM Studio
USE_SYNTHETIC_NEGATIVES = True
SYNTHETIC_FRACTION = 0.25
SYNTHETIC_NEG_PER_POS = 2

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "openai/gpt-oss-20b"   
LM_STUDIO_TEMPERATURE = 0.75                # чуть выше для разнообразия
LM_STUDIO_MAX_TOKENS = 900
LM_STUDIO_API_KEY = "any"


FILE_SAMPLE_RATES = {
    "gge_qa.jsonl": 1.0,
    "smetnoedelo_qa.cleaned.jsonl": 1.0,
    "qa_minstroy.jsonl": 0.15,
}

os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)

# ===================== ЗАГРУЗКА ДАННЫХ (без изменений) =====================
def load_unique_pairs():
    pairs_set = set()
    for path in DATA_PATHS:
        if not os.path.exists(path):
            continue
        filename = os.path.basename(path)
        sample_rate = FILE_SAMPLE_RATES.get(filename, 1.0)

        file_pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q = item.get("question") or ""
                a = item.get("answer") or ""

                if "text" in item and "Вопрос:" in item["text"] and "Ответ:" in item["text"]:
                    parts = item["text"].split("Ответ:", 1)
                    if len(parts) == 2:
                        q = parts[0].replace("Вопрос:", "").strip()
                        a = parts[1].strip()

                if q and a and len(q) >= 15 and len(a) >= 30:
                    file_pairs.append((q.strip(), a.strip()))

        if sample_rate < 1.0 and file_pairs:
            random.seed(SEED + hash(filename) % 1000)
            n_samples = max(1, int(len(file_pairs) * sample_rate))
            file_pairs = random.sample(file_pairs, n_samples)
            print(f"📄 {filename}: {len(file_pairs)} пар ({sample_rate*100:.1f}%)")

        for q, a in file_pairs:
            pairs_set.add((q, a))

    if not pairs_set:
        raise ValueError("❌ Не найдено ни одной валидной QA-пары!")

    q_list, a_list = zip(*pairs_set)
    return Dataset.from_dict({"question": list(q_list), "answer": list(a_list)})

def load_fsnb_chunks(sample_fraction=1.0):
    if not os.path.exists(FSNB_CHUNKS_PATH): return []
    chunks = [json.loads(line).get("text", "").strip() 
              for line in open(FSNB_CHUNKS_PATH, "r", encoding="utf-8") 
              if line.strip() and len(json.loads(line).get("text", "").strip()) >= 30]
    if sample_fraction < 1.0:
        random.seed(SEED)
        n = int(len(chunks) * sample_fraction)
        chunks = random.sample(chunks, min(n, len(chunks)))
    return chunks

# ===================== LM STUDIO (оптимизировано под Qwen3.5) =====================
client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

def generate_synthetic_hard_negative(query: str, positive_answer: str) -> list[str]:
    prompt = f"""Ты — ведущий эксперт по сметному делу и государственной экспертизе проектов.
Вопрос: "{query}"
Правильный ответ: "{positive_answer}"

Сгенерируй ровно {SYNTHETIC_NEG_PER_POS} профессионально звучащих, но НЕПРАВИЛЬНЫХ ответов.
Каждый ответ — на отдельной строке, без номеров и пояснений.
Используй правильную терминологию, но допусти критическую ошибку или уйди от сути вопроса."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LM_STUDIO_TEMPERATURE,
                max_tokens=LM_STUDIO_MAX_TOKENS,
            )
            text = response.choices[0].message.content.strip()
            lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 60]
            return lines[:SYNTHETIC_NEG_PER_POS]
        except Exception as e:
            print(f"⚠️ LLM попытка {attempt+1}/3 ошибка: {e}")
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []

# ===================== ОСТАЛЬНЫЕ ФУНКЦИИ (без изменений) =====================
def mine_bm25_hard_negatives(dataset: Dataset, corpus: list[str], num_negatives: int = 7, range_min: int = 10, range_max: int = 150):
    print("⛏️  Майним BM25 hard negatives...")
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    new_rows = []
    for row in tqdm(dataset, desc="BM25 mining"):
        q = row["query"]
        pos_doc = row["document"]
        scores = bm25.get_scores(q.split())
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hard_neg_idx = [i for i in sorted_idx[range_min:range_max] if corpus[i] != pos_doc][:num_negatives]
        for idx in hard_neg_idx:
            new_rows.append({"query": q, "document": corpus[idx], "label": 0})
    return Dataset.from_list(new_rows)

def merge_all(pos_ds, dense_ds, bm25_ds):
    combined = list(pos_ds) + list(dense_ds) + list(bm25_ds)
    seen = set()
    unique = []
    for row in combined:
        key = (row["query"], row["document"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return Dataset.from_list(unique)

def add_synthetic_negatives(dataset: Dataset, fraction: float):
    if not USE_SYNTHETIC_NEGATIVES: return dataset
    print(f"🧪 Генерируем синтетические hard negatives через LM Studio ({fraction*100:.0f}% позитивов)...")
    new_rows = []
    pos_rows = [row for row in dataset if row["label"] == 1]
    num_to_augment = int(len(pos_rows) * fraction)
    random.seed(SEED)
    selected_pos = random.sample(pos_rows, num_to_augment)

    for row in tqdm(selected_pos, desc="LM Studio синтетика"):
        q = row["query"]
        pos_a = row["document"]
        neg_docs = generate_synthetic_hard_negative(q, pos_a)
        for neg_doc in neg_docs:
            if len(neg_doc.strip()) > 60:
                new_rows.append({"query": q, "document": neg_doc.strip(), "label": 0})

    print(f"✅ Сгенерировано {len(new_rows)} синтетических hard негативов")
    return Dataset.from_list(list(dataset) + new_rows)

def add_random_negatives(dataset, corpus, num_random):
    random.seed(SEED)
    existing = {(row["query"], row["document"]) for row in dataset}
    new_rows = []
    pos_rows = [row for row in dataset if row["label"] == 1]
    for row in pos_rows:
        q = row["query"]
        pos_doc = row["document"]
        candidates = [d for d in corpus if d != pos_doc and (q, d) not in existing]
        if candidates:
            chosen = random.sample(candidates, min(num_random, len(candidates)))
            for doc in chosen:
                new_rows.append({"query": q, "document": doc, "label": 0})
                existing.add((q, doc))
    return Dataset.from_list(list(dataset) + new_rows)

# ===================== ОСНОВНОЙ PIPELINE =====================
print("🚀 Загружаем данные...")
full_dataset = load_unique_pairs()
print(f"✅ Всего уникальных QA-пар: {len(full_dataset)}")

corpus_qa = list(set(ex["answer"] for ex in full_dataset))
corpus = corpus_qa.copy()

full_dataset = full_dataset.shuffle(SEED)
split_idx = int(0.9 * len(full_dataset))
train_ds = full_dataset.select(range(split_idx))
eval_ds = full_dataset.select(range(split_idx, len(full_dataset)))

train_ds = train_ds.rename_columns({"question": "query", "answer": "document"})
eval_ds = eval_ds.rename_columns({"question": "query", "answer": "document"})

print(f"📚 Размер корпуса для mining: {len(corpus):,}")

print("🔥 Загружаем bi-encoder...")
bi_encoder = SentenceTransformer("deepvk/USER2-base", device="cuda")

print("⛏️  Dense mining (bi-encoder) для train...")
train_hard_dense = mine_hard_negatives(dataset=train_ds, model=bi_encoder, corpus=corpus,
    num_negatives=NUM_HARD_NEGATIVES, range_min=RANGE_MIN, range_max=RANGE_MAX,
    max_score=MAX_SCORE, sampling_strategy="top", batch_size=BATCH_SIZE,
    output_format="labeled-pair", use_faiss=True, include_positives=False)

print("⛏️  Dense mining (bi-encoder) для eval...")
eval_hard_dense = mine_hard_negatives(dataset=eval_ds, model=bi_encoder, corpus=corpus,
    num_negatives=NUM_HARD_NEGATIVES, range_min=RANGE_MIN, range_max=RANGE_MAX,
    max_score=MAX_SCORE, sampling_strategy="top", batch_size=BATCH_SIZE,
    output_format="labeled-pair", use_faiss=True, include_positives=False)

train_hard_bm25 = mine_bm25_hard_negatives(train_ds, corpus, NUM_HARD_NEGATIVES, RANGE_MIN, RANGE_MAX)
eval_hard_bm25  = mine_bm25_hard_negatives(eval_ds, corpus, NUM_HARD_NEGATIVES, RANGE_MIN, RANGE_MAX)

train_pos = Dataset.from_list([{"query": row["query"], "document": row["document"], "label": 1} for row in train_ds])
eval_pos  = Dataset.from_list([{"query": row["query"], "document": row["document"], "label": 1} for row in eval_ds])

train_hard = merge_all(train_pos, train_hard_dense, train_hard_bm25)
eval_hard  = merge_all(eval_pos, eval_hard_dense, eval_hard_bm25)
print(f"✅ Hybrid mining готов: Train = {len(train_hard)} пар, Eval = {len(eval_hard)} пар")

train_hard = add_synthetic_negatives(train_hard, SYNTHETIC_FRACTION)
eval_hard  = add_synthetic_negatives(eval_hard, SYNTHETIC_FRACTION * 0.6)

print("🎲 Добавляем случайные негативы...")
random_corpus = corpus if USE_FSNB_CHUNKS_IN_RANDOM else corpus_qa
train_dataset = add_random_negatives(train_hard, random_corpus, RANDOM_NEGATIVES_TRAIN)
eval_dataset  = add_random_negatives(eval_hard, random_corpus, RANDOM_NEGATIVES_EVAL)

train_dataset.to_json(TRAIN_PATH, force_ascii=False)
eval_dataset.to_json(EVAL_PATH, force_ascii=False)

pos_train = sum(1 for ex in train_dataset if ex["label"] == 1)
neg_train = len(train_dataset) - pos_train
pos_eval = sum(1 for ex in eval_dataset if ex["label"] == 1)
neg_eval = len(eval_dataset) - pos_eval

print(f"\n✅ ПРОФЕССИОНАЛЬНЫЙ HYBRID ДАТАСЕТ ГОТОВ!")
print(f"Train: {len(train_dataset):,} пар (позитивов: {pos_train}, негативов: {neg_train}) → {TRAIN_PATH}")
print(f"Eval:  {len(eval_dataset):,} пар (позитивов: {pos_eval}, негативов: {neg_eval}) → {EVAL_PATH}")
print("\nГотово к обучению reranker’а!")