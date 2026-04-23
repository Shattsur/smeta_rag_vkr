# push_adapter_to_hf.py — БЕЗ README.md
"""
Загрузка LoRA-адаптера на HF Hub без README.md (чтобы избежать валидации метаданных).
"""
import json
from pathlib import Path
from huggingface_hub import login, create_repo, create_commit, CommitOperationAdd, delete_file
import os

# ===================== НАСТРОЙКИ =====================
# Пути относительно корня подпроекта gen_rag
_BASE_DIR = Path(__file__).resolve().parents[1]
ADAPTER_PATH = _BASE_DIR / "models" / "Nemotron-Rerank-1B-4bit-finetuned" / "ndcg_best"
HF_REPO_ID = "Shattsur/nemotron-smeta-4bit-adapter"
BASE_MODEL_HF = "nvidia/llama-nemotron-rerank-1b-v2"  # ← HF ID базы
PRIVATE = True

# ===================== 1. АВТОРИЗАЦИЯ =====================
print("🔐 Авторизация на Hugging Face...")
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    hf_token = input("Введите HF_TOKEN: ").strip()
login(token=hf_token)

# ===================== 2. СОЗДАНИЕ/ОЧИСТКА РЕПО =====================
print(f"📦 Подготовка репо: {HF_REPO_ID} (private={PRIVATE})...")
create_repo(repo_id=HF_REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

# 🔥 Удаляем README.md если он есть (чтобы не валидировался)
try:
    from huggingface_hub import delete_file
    delete_file(path_in_repo="README.md", repo_id=HF_REPO_ID, repo_type="model", token=hf_token)
    print("🗑️ README.md удалён из репо")
except:
    pass  # Файла нет — ок

# ===================== 3. ИСПРАВЛЕНИЕ adapter_config.json =====================
print("🔧 Исправление adapter_config.json...")
config_path = ADAPTER_PATH / "adapter_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 🔥 КРИТИЧНО: Заменяем локальный путь на HF model_id
config["base_model_name_or_path"] = BASE_MODEL_HF
config["task_type"] = "SEQ_CLS"
config["problem_type"] = "regression"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"✅ base_model_name_or_path → {BASE_MODEL_HF}")

# ===================== 4. УДАЛЕНИЕ ЛОКАЛЬНОГО README.md =====================
readme_local = ADAPTER_PATH / "README.md"
if readme_local.exists():
    readme_local.unlink()
    print("🗑️ Локальный README.md удалён")

# ===================== 5. ПОДГОТОВКА ФАЙЛОВ =====================
print("📋 Подготовка файлов к загрузке...")
files_to_upload = []
for file_path in ADAPTER_PATH.rglob("*"):
    if not file_path.is_file():
        continue
    # Пропускаем служебные файлы и README
    if file_path.name in [".git", ".gitignore", "README.md", "README"] or file_path.suffix in [".pyc"]:
        continue
    if "__pycache__" in str(file_path):
        continue
    
    path_in_repo = file_path.relative_to(ADAPTER_PATH).as_posix()
    files_to_upload.append(
        CommitOperationAdd(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo
        )
    )
    size_mb = file_path.stat().st_size / 1024 / 1024
    print(f"   📄 {path_in_repo} ({size_mb:.2f} MB)")

print(f"\n✅ Всего файлов: {len(files_to_upload)}")

# ===================== 6. ЗАГРУЗКА ЧЕРЕЗ create_commit =====================
print(f"\n🚀 Загрузка на HF: {HF_REPO_ID}...")
try:
    create_commit(
        repo_id=HF_REPO_ID,
        repo_type="model",
        operations=files_to_upload,
        commit_message="Upload LoRA adapter for Smeta domain (4-bit, no README to avoid validation)",
        token=hf_token,
        create_pr=False
    )
    print("✅ Загрузка завершена!")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    # Фоллбэк: пофайловая загрузка
    print("🔄 Попытка пофайловой загрузки...")
    for op in files_to_upload:
        try:
            create_commit(
                repo_id=HF_REPO_ID,
                repo_type="model",
                operations=[op],
                commit_message=f"Upload {op.path_in_repo}",
                token=hf_token
            )
            print(f"   ✅ {op.path_in_repo}")
        except Exception as e2:
            print(f"   ❌ {op.path_in_repo}: {e2}")

# ===================== 7. ПРОВЕРКА =====================
print(f"\n🔍 Проверка...")
from huggingface_hub import list_repo_files
try:
    uploaded = list_repo_files(repo_id=HF_REPO_ID, repo_type="model", token=hf_token)
    print(f"   Загружено: {len(uploaded)} файлов")
    for f in sorted(uploaded):
        print(f"   ✓ {f}")
except Exception as e:
    print(f"   ⚠ Не удалось проверить: {e}")

print(f"\n🌐 Репозиторий: https://huggingface.co/{HF_REPO_ID}")
print(f"\n📝 Теперь в fit_params_optuna_nsgaii_v3.py используйте:")
print(f'   "{HF_REPO_ID}"')