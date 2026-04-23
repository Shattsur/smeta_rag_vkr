import json
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ===================== НАСТРОЙКИ =====================
# Пути относительно корня подпроекта gen_rag
_BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH  = _BASE_DIR / "data" / "raw" / "official_docs.jsonl"
OUTPUT_PATH = _BASE_DIR / "data" / "rerank" / "qa_minstroy.jsonl"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

MODEL_NAME = "saiga_yandexgpt_8b_gguf"   

PERCENT = 1.0                        # 25% чанков (можно 0.01 для теста)

# ===================== ПРОМПТ =====================
SYSTEM_PROMPT = (
    "Ты — эксперт-сметчик. Сформулируй один естественный и конкретный вопрос "
    "по тексту нормативного документа Минстроя. Вопрос должен быть таким, "
    "чтобы на него можно было ответить, используя данный текст. "
    "Не добавляй пояснений, не пиши 'Вопрос:' в начале. "
    "Вопрос должен начинаться с заглавной буквы и заканчиваться вопросительным знаком."
)

# ===================== ЗАГРУЗКА =====================
print("Загружаем official_docs.jsonl...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

print(f"Всего чанков: {len(chunks)}")

random.seed(42)
selected_chunks = random.sample(chunks, int(len(chunks) * PERCENT))
print(f"Выбрано {PERCENT*100}% чанков: {len(selected_chunks)}")

data = []

for chunk in tqdm(selected_chunks, desc="Генерируем вопросы"):
    text = chunk.get("text", "").strip()
    chunk_id = chunk.get("chunk_id", "unknown")

    if len(text) < 80:
        continue

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Текст:\n{text}"}
            ],
            temperature=0.1,
            max_tokens=150
        )
        question = resp.choices[0].message.content.strip()
        if not question.endswith("?"):
            question += "?"

        data.append({
            "question": question,
            "answer": text,
            "source": f"official_docs_{chunk_id}",
            "type": "qa_minstroy",
            "qa_id": f"minstroy_{chunk_id}"
        })

    except Exception as e:
        print(f"Ошибка на чанке {chunk_id}: {e}")

# Сохраняем
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ ГОТОВО!")
print(f"Создано {len(data)} вопросов-ответов (только positive)")
print(f"Файл сохранён: {OUTPUT_PATH}")
print("Формат: question / answer — готов для обучения reranker’а")