# -*- coding: utf-8 -*-
"""
scripts/extract_pdf.py - ВЕРСИЯ 7.2 (GRAPH‑BASED CHUNKING)
Исправления:
- Сохранение переводов строк при удалении маркеров страниц
- Поддержка цифровых подпунктов (1), 2) ...) в дополнение к буквенным
- Улучшено выделение границ пунктов (сохраняются переводы строк)
- Уникальные идентификаторы для повторяющихся номеров пунктов
- Слияние мелких чанков внутри одного логического блока
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

# ===================== CONFIG =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
OUT_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = OUT_DIR / "official_docs.jsonl"

# ЛИМИТЫ
HARD_MAX_CHARS = 2000
TARGET_CHARS = 1500          # увеличен для уменьшения дробления
MIN_CHUNK_CHARS = 50
OVERLAP_CHARS = 100
MAX_ITERATIONS = 10000

# OCR
OCR_DPI = 150
OCR_PSM = 6
OCR_LANG = "rus+eng"
OCR_TIMEOUT_SEC = 45
OCR_MAX_PAGES = 10

# ===================== REGEX =====================
FGIS_RE = re.compile(r"сведения\s+сформированы\с+фгис\s+цс", re.IGNORECASE)
FGIS_URL_RE = re.compile(r"fgiscs\.minstroyrf\.ru", re.IGNORECASE)

FOOTER_PATTERNS = [
    r"Страница\s*\d+\s*(?:из\s*\d+)?",
    r"документ\s+сохранен\s+с\s+портала",
    r"docs\.cntd\.ru",
    r"электронного\s+фонда\s+из\s+более\s+\d+",
    r"нормативно-[а-я]+\s+документов",
    r"приказ\s+министерства\s+строительства",
    r"минстрой\s+россии",
    r"зарегистрировано\s+в\s+минюсте",
    r"№\s*\d+/пр",
    r"Собрание\s+законодательства",
    r"©.*?CNTD",
    r"Об утверждении Методики определения стоимости работ по подготовке проектной документации \(с изменениями на \d+ \w+ \d+ года\)",
    r"и жилищно-коммунального хозяйства Российской Федерации от \d+ \w+ \d+ г\.",
    r"— \d+ \d+ нормативно-правовых и",
]
FOOTER_RE = re.compile("|".join(FOOTER_PATTERNS), re.IGNORECASE)

ARTIFACTS_RE = re.compile(r"(?i)(©б|©a)")

# Пункты: цифра(ы) с точкой в начале строки или после перевода строки
CLAUSE_RE = re.compile(r"(?:^|\n)\s*(?P<num>\d+(?:\.\d+){0,3})\.\s+")
# Таблицы как отдельные блоки
TABLE_RE = re.compile(r"(?:^|\n)\s*Таблица\s+(?P<num>\d+(?:\.\d+){0,3})\s*", re.IGNORECASE)
# Подпункты: буква или цифра с закрывающей скобкой в начале строки
SUBCLAUSE_RE = re.compile(r"(?:^|\n)\s*(?P<num>\d+|[а-яё])\)\s+", re.IGNORECASE)

EDITORIAL_RE = re.compile(
    r"\s*\([^)]*?(?:в редакции|изм\.|ред\.|См\. предыдущую).*?\)\s*",
    re.IGNORECASE
)

LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
DIGIT_RE = re.compile(r"\d")

PAGE_MARK_PREFIX = "<<<PAGE:"
PAGE_MARK_SUFFIX = ">>>"

# Множество русских сокращений для разбиения на предложения
RUSSIAN_ABBREVIATIONS = {
    'см', 'т.е', 'и т.д', 'и т.п', 'т.п', 'т.д',
    'г', 'стр', 'рис', 'табл', 'п', 'ч', 'разд',
    'прим', 'ред', 'мин', 'тыс', 'млн', 'млрд',
    'им', 'обл', 'ул', 'пер', 'пр', 'тел', 'факс',
    'e-mail', 'http', 'https', 'ftp', 'www', 'со',
    'т.е.', 'т.д.', 'т.п.'
}

# ===================== HELPERS =====================
def extract_pages_from_text(text: str) -> Tuple[int, int]:
    marks = re.findall(rf"{re.escape(PAGE_MARK_PREFIX)}(\d+){re.escape(PAGE_MARK_SUFFIX)}", text)
    if not marks:
        return 0, 0
    nums = list(map(int, marks))
    return min(nums), max(nums)


def strip_page_marks(text: str) -> str:
    """Удаляет маркеры страниц, но сохраняет последующие пробелы/переводы строк."""
    if not text:
        return ""
    # Просто удаляем маркер, не трогая пробельные символы после него
    return re.sub(rf"{re.escape(PAGE_MARK_PREFIX)}\d+{re.escape(PAGE_MARK_SUFFIX)}", "", text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\u00a0", " ")
    t = ARTIFACTS_RE.sub("Об", t)
    t = FOOTER_RE.sub("", t)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def clean_editorial_notes(text: str) -> str:
    if not text:
        return ""
    t = EDITORIAL_RE.sub("", text)
    return clean_text(t)


def drop_stamp_lines(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    kept = [ln for ln in lines if ln.strip() and not (FGIS_RE.search(ln) or FGIS_URL_RE.search(ln))]
    return "\n".join(kept).strip()


def text_quality_metrics(text: str) -> Dict[str, Any]:
    n = len(text)
    if n == 0:
        return {"len": 0, "letter_ratio": 0, "noisy": True}
    letters = len(LETTER_RE.findall(text))
    digits = len(DIGIT_RE.findall(text))
    return {
        "len": n,
        "letter_ratio": round(letters / n, 4),
        "digit_ratio": round(digits / n, 4),
        "noisy": letters / n < 0.3 and digits / n > 0.3
    }


# ===================== OCR =====================
_TESSERACT_EXE_CACHE = None
_ocr_pages_count = 0

def find_tesseract_exe() -> Optional[Path]:
    global _TESSERACT_EXE_CACHE
    if _TESSERACT_EXE_CACHE is not None:
        return _TESSERACT_EXE_CACHE
    env = os.getenv("TESSERACT_EXE")
    if env and Path(env).exists():
        _TESSERACT_EXE_CACHE = Path(env)
        return _TESSERACT_EXE_CACHE
    win_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if win_default.exists():
        _TESSERACT_EXE_CACHE = win_default
        return _TESSERACT_EXE_CACHE
    from shutil import which
    w = which("tesseract")
    if w and Path(w).exists():
        _TESSERACT_EXE_CACHE = Path(w)
        return _TESSERACT_EXE_CACHE
    return None


def render_page_to_png(doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def ocr_page_png(png_bytes: bytes, tesseract_exe: Path, lang: str, psm: int) -> str:
    global _ocr_pages_count
    if _ocr_pages_count >= OCR_MAX_PAGES:
        return ""
    
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "page.png"
        img_path.write_bytes(png_bytes)
        cmd = [str(tesseract_exe), str(img_path), "stdout", "-l", lang, "--psm", str(psm), "--oem", "1"]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=False, timeout=OCR_TIMEOUT_SEC, check=False)
            _ocr_pages_count += 1
            return (cp.stdout or b"").decode("utf-8", errors="replace").strip()
        except:
            return ""


@dataclass
class PageExtract:
    page: int
    text: str
    mode: str
    quality: Dict[str, Any]


def extract_page_text(doc: fitz.Document, page_index: int, tesseract_exe: Optional[Path], ocr_lang: str) -> PageExtract:
    try:
        page = doc.load_page(page_index)
        raw = page.get_text("text") or ""
    except Exception as e:
        print(f"            [WARN] Страница {page_index}: ошибка get_text: {e}")
        return PageExtract(page_index + 1, "", "error", {"len": 0, "noisy": True})
    
    if len(raw.strip()) < 20:
        if tesseract_exe:
            try:
                png = render_page_to_png(doc, page_index, OCR_DPI)
                ocr_raw = ocr_page_png(png, tesseract_exe, ocr_lang, OCR_PSM)
                if len(ocr_raw) > 20:
                    return PageExtract(page_index + 1, clean_text(ocr_raw), "ocr", text_quality_metrics(ocr_raw))
            except:
                pass
        return PageExtract(page_index + 1, "", "empty", {"len": 0, "noisy": True})

    t_clean = clean_text(raw)
    t_clean = drop_stamp_lines(t_clean)
    t_clean = clean_editorial_notes(t_clean)
    q = text_quality_metrics(t_clean)

    if (q["len"] < 100 or q["noisy"]) and tesseract_exe:
        try:
            png = render_page_to_png(doc, page_index, OCR_DPI)
            ocr_raw = ocr_page_png(png, tesseract_exe, ocr_lang, OCR_PSM)
            ocr_clean = clean_text(clean_editorial_notes(ocr_raw))
            if len(ocr_clean) > q["len"] * 1.2:
                return PageExtract(page_index + 1, ocr_clean, "ocr", text_quality_metrics(ocr_clean))
        except:
            pass

    return PageExtract(page_index + 1, t_clean, "text", q)


# ===================== GRAPH CHUNKING =====================
def find_clause_boundaries(text: str) -> List[Tuple[int, str]]:
    boundaries = []
    for m in CLAUSE_RE.finditer(text):
        num = m.group("num")
        start = m.start()
        # Простая проверка, чтобы не захватывать номера внутри слов (например, "п.5")
        if start > 0 and text[start-1].isalpha():
            continue
        boundaries.append((start, num))
    
    for m in TABLE_RE.finditer(text):
        num = m.group("num")
        start = m.start()
        boundaries.append((start, f"table:{num}"))
    
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def find_subclause_boundaries(text: str) -> List[Tuple[int, str]]:
    boundaries = []
    for m in SUBCLAUSE_RE.finditer(text):
        boundaries.append((m.start(), m.group("num")))
    return boundaries


def is_abbreviation(text: str, position: int) -> bool:
    """Проверяет, является ли слово, заканчивающееся на позиции position, сокращением."""
    if position <= 0:
        return False
    
    start = position
    while start > 0 and text[start-1].isalpha():
        start -= 1
    
    word = text[start:position+1].lower()
    
    if word in RUSSIAN_ABBREVIATIONS:
        return True
    
    if '.' in word and len(word.split('.')) <= 3:
        return True
    
    return False


def split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения с учётом сокращений."""
    if not text:
        return []
    
    text = re.sub(r'\s+', ' ', text)
    sentences = []
    current = []
    i = 0
    length = len(text)
    
    while i < length:
        current.append(text[i])
        
        if text[i] in '.!?':
            if text[i] == '.' and is_abbreviation(text, i):
                i += 1
                continue
            
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
        i += 1
    
    if current:
        sentence = ''.join(current).strip()
        if sentence:
            sentences.append(sentence)
    
    return sentences


def force_split_by_sentences(text: str, max_size: int = TARGET_CHARS) -> List[str]:
    if len(text) <= max_size:
        return [text]
    
    sentences = split_sentences(text)
    
    if len(sentences) <= 1:
        words = text.split()
        chunks = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > max_size and current:
                chunks.append(current.strip())
                current = w
            else:
                current = current + " " + w if current else w
        if current:
            chunks.append(current.strip())
        return chunks
    
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current:
        chunks.append(current.strip())
    
    final_chunks = []
    for ch in chunks:
        if len(ch) > max_size:
            for i in range(0, len(ch), max_size - OVERLAP_CHARS):
                part = ch[i:i+max_size]
                if part:
                    final_chunks.append(part.strip())
        else:
            final_chunks.append(ch)
    return final_chunks


def merge_small_chunks(chunks: List[Dict], max_size: int) -> List[Dict]:
    if len(chunks) <= 1:
        return chunks
    merged = []
    current = chunks[0].copy()
    for nxt in chunks[1:]:
        if (current.get("parent_id") == nxt.get("parent_id") and
            current.get("subclause") == nxt.get("subclause") and
            len(current["text"]) + len(nxt["text"]) <= max_size):
            current["text"] += " " + nxt["text"]
            current["page_end"] = max(current.get("page_end", 0), nxt.get("page_end", 0))
        else:
            merged.append(current)
            current = nxt.copy()
    merged.append(current)
    return merged


def split_into_clauses_and_subclauses(full_text: str, doc_id: str) -> List[Dict[str, Any]]:
    text = (full_text or "").strip()
    if not text:
        return []
    
    global_p_start, global_p_end = extract_pages_from_text(text)
    clause_boundaries = find_clause_boundaries(text)
    clause_counter = {}
    
    all_chunks = []
    
    if not clause_boundaries:
        clean = clean_text(strip_page_marks(text))
        if len(clean) >= MIN_CHUNK_CHARS:
            sub_chunks = force_split_by_sentences(clean, TARGET_CHARS)
            for i, sub in enumerate(sub_chunks):
                all_chunks.append({
                    "chunk_id": f"{doc_id}__0.{i+1}" if len(sub_chunks) > 1 else f"{doc_id}__0",
                    "text": sub,
                    "level": 1,
                    "parent_id": None,
                    "clause": "0",
                    "clause_title": "Общий текст",
                    "subclause": None,
                    "has_children": False,
                    "page_start": global_p_start,
                    "page_end": global_p_end,
                })
        return all_chunks
    
    # Введение (до первого пункта)
    first_pos = clause_boundaries[0][0]
    if first_pos > 0:
        intro = text[:first_pos].strip()
        p_start, p_end = extract_pages_from_text(intro)
        clean_intro = clean_text(strip_page_marks(intro))
        if len(clean_intro) >= MIN_CHUNK_CHARS:
            intro_chunks = force_split_by_sentences(clean_intro, TARGET_CHARS)
            for i, sub in enumerate(intro_chunks):
                all_chunks.append({
                    "chunk_id": f"{doc_id}__0.{i+1}" if len(intro_chunks) > 1 else f"{doc_id}__0",
                    "text": sub,
                    "level": 1,
                    "parent_id": None,
                    "clause": "0",
                    "clause_title": "Введение",
                    "subclause": None,
                    "has_children": False,
                    "page_start": p_start if p_start > 0 else global_p_start,
                    "page_end": p_end if p_end > 0 else global_p_end,
                })
    
    # Основные пункты
    for i, (pos, num_full) in enumerate(clause_boundaries):
        is_table = num_full.startswith("table:")
        num = num_full.replace("table:", "")
        
        clause_counter.setdefault(num, 0)
        clause_counter[num] += 1
        if clause_counter[num] > 1:
            clause_instance = f"{num}_{clause_counter[num]}"
        else:
            clause_instance = num
        parent_id = f"{doc_id}__{clause_instance}"
        
        end_pos = clause_boundaries[i + 1][0] if i + 1 < len(clause_boundaries) else len(text)
        block_raw = text[pos:end_pos].strip()
        p_start, p_end = extract_pages_from_text(block_raw)
        
        # Извлечение заголовка пункта
        if is_table:
            title_match = re.search(r"Таблица\s+\d+(?:\.\d+){0,3}\s*(.+)", block_raw, re.IGNORECASE)
            title = title_match.group(1).strip()[:150] if title_match else f"Таблица {num}"
        else:
            title_match = re.search(r"\d+(?:\.\d+){0,3}\.\s*(.+?)(?:\n|$)", block_raw)
            title = title_match.group(1).strip()[:150] if title_match else f"Пункт {num}"
        
        # Поиск подпунктов
        sub_boundaries = find_subclause_boundaries(block_raw)
        # Фильтрация подпунктов (по порядку, для буквенных сохраняем порядок, для цифровых просто берём все)
        valid_subs = []
        if sub_boundaries:
            # Для буквенных проверяем алфавитный порядок
            expected_letters = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
            last_letter_idx = -1
            for sub_pos, sub_num in sub_boundaries:
                if sub_num.isalpha() and sub_num in expected_letters:
                    idx = expected_letters.index(sub_num)
                    if idx > last_letter_idx - 2:
                        valid_subs.append((sub_pos, sub_num))
                        last_letter_idx = idx
                else:
                    # Цифровые подпункты принимаем все подряд
                    valid_subs.append((sub_pos, sub_num))
            # Сортируем по позиции (на случай если порядок нарушен)
            valid_subs.sort(key=lambda x: x[0])
        
        if not valid_subs:
            # Нет подпунктов
            clean_block = clean_text(strip_page_marks(block_raw))
            clean_block = clean_editorial_notes(clean_block)
            if len(clean_block) >= MIN_CHUNK_CHARS:
                split_size = HARD_MAX_CHARS if is_table else TARGET_CHARS
                sub_chunks = force_split_by_sentences(clean_block, split_size)
                chunk_dicts = []
                for txt in sub_chunks:
                    chunk_dicts.append({
                        "text": txt,
                        "parent_id": parent_id,
                        "subclause": None,
                        "page_start": p_start if p_start > 0 else global_p_start,
                        "page_end": p_end if p_end > 0 else global_p_end,
                    })
                merged = merge_small_chunks(chunk_dicts, split_size)
                for j, cd in enumerate(merged):
                    all_chunks.append({
                        "chunk_id": f"{parent_id}.{j+1}" if len(merged) > 1 else parent_id,
                        "text": cd["text"],
                        "level": 1,
                        "parent_id": None,
                        "clause": clause_instance,
                        "clause_title": title,
                        "subclause": None,
                        "has_children": False,
                        "page_start": cd["page_start"],
                        "page_end": cd["page_end"],
                    })
        else:
            # Есть подпункты – сначала текст до первого подпункта (введение пункта)
            first_sub_pos = valid_subs[0][0]
            intro_raw = block_raw[:first_sub_pos].strip()
            clean_intro = clean_text(strip_page_marks(intro_raw))
            clean_intro = clean_editorial_notes(clean_intro)
            
            if clean_intro and len(clean_intro) >= 30:
                intro_chunks = force_split_by_sentences(clean_intro, TARGET_CHARS)
                chunk_dicts = []
                for txt in intro_chunks:
                    chunk_dicts.append({
                        "text": txt,
                        "parent_id": parent_id,
                        "subclause": None,
                        "page_start": p_start if p_start > 0 else global_p_start,
                        "page_end": p_end if p_end > 0 else global_p_end,
                    })
                merged = merge_small_chunks(chunk_dicts, TARGET_CHARS)
                for j, cd in enumerate(merged):
                    all_chunks.append({
                        "chunk_id": f"{parent_id}.{j+1}" if len(merged) > 1 else parent_id,
                        "text": cd["text"],
                        "level": 1,
                        "parent_id": None,
                        "clause": clause_instance,
                        "clause_title": title,
                        "subclause": None,
                        "has_children": True,
                        "page_start": cd["page_start"],
                        "page_end": cd["page_end"],
                    })
            
            # Затем каждый подпункт
            for j, (sub_pos, sub_num) in enumerate(valid_subs):
                sub_start = sub_pos
                sub_end = valid_subs[j + 1][0] if j + 1 < len(valid_subs) else len(block_raw)
                sub_raw = block_raw[sub_start:sub_end].strip()
                clean_sub = clean_text(strip_page_marks(sub_raw))
                clean_sub = clean_editorial_notes(clean_sub)
                
                if len(clean_sub) < MIN_CHUNK_CHARS:
                    continue
                
                sub_chunks = force_split_by_sentences(clean_sub, TARGET_CHARS)
                chunk_dicts = []
                for txt in sub_chunks:
                    chunk_dicts.append({
                        "text": txt,
                        "parent_id": parent_id,
                        "subclause": sub_num,
                        "page_start": p_start if p_start > 0 else global_p_start,
                        "page_end": p_end if p_end > 0 else global_p_end,
                    })
                merged = merge_small_chunks(chunk_dicts, TARGET_CHARS)
                for k, cd in enumerate(merged):
                    all_chunks.append({
                        "chunk_id": f"{parent_id}__{sub_num}.{k+1}" if len(merged) > 1 else f"{parent_id}__{sub_num}",
                        "text": cd["text"],
                        "level": 2,
                        "parent_id": parent_id,
                        "clause": clause_instance,
                        "clause_title": title,
                        "subclause": sub_num,
                        "has_children": False,
                        "page_start": cd["page_start"],
                        "page_end": cd["page_end"],
                    })
    
    return all_chunks


def get_document_title(doc: fitz.Document, doc_id: str) -> str:
    try:
        text = doc.load_page(0).get_text("text")[:1000]
        for line in text.splitlines():
            line = line.strip()
            if 20 < len(line) < 200 and not FOOTER_RE.search(line):
                return line[:180]
    except:
        pass
    return doc_id.replace("_", " ").replace("-", " ").strip()


# ===================== MAIN =====================
def process_pdf(pdf_path: Path, out_f, tesseract_exe: Optional[Path], ocr_lang: str) -> Tuple[int, str, int]:
    global _ocr_pages_count
    _ocr_pages_count = 0
    
    doc_id = pdf_path.stem.strip()
    start_time = time.time()

    try:
        with fitz.open(pdf_path) as doc:
            print(f"            [INFO] Страниц: {doc.page_count}")
            
            title = get_document_title(doc, doc_id)
            print(f"            [INFO] Заголовок: {title[:80]}...")
            
            pages = [extract_page_text(doc, i, tesseract_exe, ocr_lang) for i in range(doc.page_count)]
            
            total_raw_chars = sum(len(p.text) for p in pages)
            non_empty_pages = sum(1 for p in pages if p.text)
            print(f"            [INFO] Извлечено: {total_raw_chars} симв. | Страниц с текстом: {non_empty_pages}/{doc.page_count}")
            
            extraction_mode = "mixed" if len({p.mode for p in pages}) > 1 else (pages[0].mode if pages else "text")
            
            full_parts = [f"{PAGE_MARK_PREFIX}{p.page}{PAGE_MARK_SUFFIX}\n{p.text}\n" for p in pages if p.text]
            full_text = "\n".join(full_parts).strip()

            if not full_text:
                print(f"      ⚠️ Пустой документ (нет текста после извлечения)")
                return 0, extraction_mode, 0

            chunks = split_into_clauses_and_subclauses(full_text, doc_id)
            written = 0
            too_large = 0

            for chunk_data in chunks:
                if len(chunk_data["text"]) < MIN_CHUNK_CHARS:
                    continue
                
                if len(chunk_data["text"]) > HARD_MAX_CHARS:
                    too_large += 1

                output_chunk = {
                    "chunk_id": chunk_data["chunk_id"],
                    "text": chunk_data["text"],
                    "source": pdf_path.name,
                    "metadata": {
                        "type": "official_normative",
                        "doc_id": doc_id,
                        "title": title,
                        "clause": chunk_data["clause"],
                        "clause_title": chunk_data["clause_title"],
                        "subclause": chunk_data["subclause"],
                        "level": chunk_data["level"],
                        "parent_id": chunk_data["parent_id"],
                        "has_children": chunk_data["has_children"],
                        "page_start": chunk_data["page_start"],
                        "page_end": chunk_data["page_end"],
                        "extraction": extraction_mode,
                        "char_count": len(chunk_data["text"]),
                    },
                }
                out_f.write(json.dumps(output_chunk, ensure_ascii=False) + "\n")
                written += 1
            
            del chunks, full_text, full_parts
            gc.collect()

            duration = time.time() - start_time
            ocr_count = sum(1 for p in pages if p.mode == "ocr")
            
            status = f"⚠️ {too_large} больших" if too_large > 0 else "✅ все в лимите"
            print(f"      → чанков: {written} | {status} | режим: {extraction_mode} | OCR: {ocr_count} | {duration:.2f}с")
            return written, extraction_mode, ocr_count

    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 0, "error", 0


def main() -> None:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] Нет PDF в {PDF_DIR}")
        print(f"[DEBUG] PDF_DIR = {PDF_DIR}")
        print(f"[DEBUG] PDF_DIR.exists() = {PDF_DIR.exists()}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tesseract_exe = find_tesseract_exe()
    ocr_lang = OCR_LANG if tesseract_exe else "eng"

    print(f"[INFO] PDF: {len(pdf_files)} | Tesseract: {'✅' if tesseract_exe else '❌'}")
    print(f"[INFO] Режим: ГРАФ v7.2 (поддержка цифровых подпунктов, сохранение переводов строк)")
    print(f"[INFO] PDF_DIR: {PDF_DIR}")

    total = 0
    start = time.time()
    
    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
            print(f"      [DEBUG] Размер файла: {pdf_path.stat().st_size:,} байт")
            
            chunks, mode, ocr_p = process_pdf(pdf_path, out_f, tesseract_exe, ocr_lang)
            total += chunks

    print(f"\n🎉 ГОТОВО! Чанков: {total} | Время: {time.time()-start:.1f}с")
    print(f"Файл: {OUT_FILE}")


if __name__ == "__main__":
    main()