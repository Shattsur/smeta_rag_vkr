# quantize_nemotron_reranker.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import time

MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"
_BASE_DIR = Path(__file__).resolve().parents[1]
SAVE_DIR = _BASE_DIR / "models" / "Nemotron-Rerank-1B-4bit"

print("🔄 Начинаем 4-битное квантование NVIDIA Nemotron Rerank 1B v2...")

t0 = time.time()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="left"          # важно для Nemotron
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Немного фиксим токенизатор (Nemotron требует pad_token = eos_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

t1 = time.time()
print(f"✅ Квантование Nemotron завершено за {t1-t0:.1f} секунд")
print(f"✅ Модель сохранена в: {SAVE_DIR.resolve()}")