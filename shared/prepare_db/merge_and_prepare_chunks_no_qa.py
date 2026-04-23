# scripts/merge_and_prepare_chunks_no_qa.py
# -*- coding: utf-8 -*-
"""
Merge ONLY official_docs.jsonl + fsnb_chunks.jsonl
into data/chunks/all_chunks.jsonl

Без QA.
Улучшения:
- Для official_docs: группируем чанки по clause (пункту) и объединяем текст в один большой чанк.
  Это решает проблему "разорванных" пунктов (например, п. 170 на 3 части → один полный чанк).
- Если в official несколько частей (__pN), объединяем их в один чанк с полным текстом пункта.
- Для FSNB: оставляем как есть (расценки обычно атомарны).
- Chunk_size/overlap теперь не используются для official (объединяем полностью).
- Добавлен --parent-mode для управления (по умолчанию объединяем official по clause).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


def project_root() -> Path:
    # Возвращает CWD (см. другие скрипты shared/prepare_db/): запускайте
    # из корня подпроекта (gen_rag/ или graph_rag/), либо передавайте
    # явные пути через argparse.
    return Path.cwd()


def jload_line(line: str) -> Optional[Dict[str, Any]]:
    line = (line or "").strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def _md(rec: Dict[str, Any]) -> Dict[str, Any]:
    md = rec.get("metadata")
    return md if isinstance(md, dict) else {}


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def main() -> int:
    pr = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=str, default=str(pr / "data" / "raw" / "official_docs.jsonl"))
    parser.add_argument("--fsnb", type=str, default=str(pr / "data" / "raw" / "fsnb_chunks.jsonl"))
    parser.add_argument("--out", type=str, default=str(pr / "data" / "chunks" / "all_chunks.jsonl"))
    parser.add_argument("--min-chars", type=int, default=50, help="Минимальная длина чанка")
    args = parser.parse_args()

    official_path = Path(args.official)
    fsnb_path = Path(args.fsnb)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not official_path.exists():
        raise FileNotFoundError(f"official_docs.jsonl not found: {official_path}")
    if not fsnb_path.exists():
        raise FileNotFoundError(f"fsnb_chunks.jsonl not found: {fsnb_path}")

    total = 0
    official_chunks = 0
    fsnb_chunks = 0
    skipped_official = skipped_fsnb = 0

    print("[INFO] Merging chunks (без QA, с объединением пунктов official по clause)...")
    print(f"[INFO] official : {official_path}")
    print(f"[INFO] fsnb     : {fsnb_path}")
    print(f"[INFO] out      : {out_path}")

    with out_path.open("w", encoding="utf-8") as out_f:
        # 1) Official docs: группируем по clause и объединяем текст
        clause_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        with official_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                rec = jload_line(line)
                if rec is None:
                    skipped_official += 1
                    continue

                text = clean_text(rec.get("text") or "")
                if not text:
                    skipped_official += 1
                    continue

                md_in = _md(rec)
                clause = clean_text(md_in.get("clause") or "")
                doc_id = clean_text(md_in.get("doc_id") or "")

                # Ключ группы: doc_id + clause (чтобы не смешивать пункты из разных документов)
                group_key = f"{doc_id}__{clause}" if clause else f"official_line{line_num}"

                clause_groups[group_key].append({
                    "text_part": text,
                    "source": clean_text(rec.get("source") or md_in.get("source") or md_in.get("source_file") or ""),
                    "md_base": md_in,
                    "orig_chunk_id": clean_text(rec.get("chunk_id") or md_in.get("chunk_id") or f"official_line{line_num}"),
                    "part_number": _as_int(md_in.get("part_number"), 1),
                })

        # Объединяем группы в один чанк на пункт
        for group_key, parts in clause_groups.items():
            if not parts:
                continue

            # Сортируем части по part_number (если есть)
            parts.sort(key=lambda x: x["part_number"])

            full_text = "\n\n".join(p["text_part"] for p in parts)
            if len(full_text) < args.min_chars:
                skipped_official += 1
                continue

            # Базовые метаданные берём из первой части
            base_md = parts[0]["md_base"]
            source = parts[0]["source"]

            doc_type = clean_text(base_md.get("type") or base_md.get("doc_type") or "official_normative")
            doc_id = clean_text(base_md.get("doc_id") or "")
            title = clean_text(base_md.get("title") or "")
            clause = clean_text(base_md.get("clause") or "")
            clause_title = clean_text(base_md.get("clause_title") or "")
            page_start = min(_as_int(p["md_base"].get("page_start", 0)) for p in parts)
            page_end = max(_as_int(p["md_base"].get("page_end", 0)) for p in parts)
            extraction = clean_text(base_md.get("extraction") or "")

            chunk_id = f"official_clause_{group_key}"

            out_chunk = {
                "chunk_id": chunk_id,
                "text": full_text,
                "source": source,
                "metadata": {
                    "type": doc_type,
                    "doc_id": doc_id,
                    "title": title,
                    "clause": clause,
                    "clause_title": clause_title,
                    "page_start": page_start,
                    "page_end": page_end,
                    "extraction": extraction,
                    "orig_parts": len(parts),
                    "group_key": group_key,
                },
            }

            out_f.write(json.dumps(out_chunk, ensure_ascii=False) + "\n")
            total += 1
            official_chunks += 1

        # 2) FSNB chunks — как есть
        with fsnb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                ch = jload_line(line)
                if ch is None:
                    skipped_fsnb += 1
                    continue

                text = clean_text(ch.get("text") or "")
                if not text or len(text) < args.min_chars:
                    skipped_fsnb += 1
                    continue

                md_in = _md(ch)

                chunk_id = clean_text(ch.get("chunk_id") or md_in.get("chunk_id") or f"fsnb_{line_num}")
                source = clean_text(
                    ch.get("source")
                    or ch.get("source_file")
                    or md_in.get("source")
                    or md_in.get("source_file")
                )
                title = clean_text(ch.get("title") or md_in.get("title"))
                work_code = clean_text(ch.get("work_code") or md_in.get("work_code"))

                out_chunk = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "metadata": {
                        "type": "fsnb_normative",
                        "title": title,
                        "work_code": work_code,
                        "measure": clean_text(md_in.get("measure")),
                        "table_code": clean_text(md_in.get("table_code")),
                        "base_name": clean_text(md_in.get("base_name")),
                    },
                }

                out_f.write(json.dumps(out_chunk, ensure_ascii=False) + "\n")
                total += 1
                fsnb_chunks += 1

    print("[INFO] Done.")
    print(f"[INFO] Total chunks      : {total}")
    print(f"[INFO] Official chunks   : {official_chunks} (skipped: {skipped_official})")
    print(f"[INFO] FSNB chunks       : {fsnb_chunks} (skipped: {skipped_fsnb})")
    print(f"[INFO] Wrote             : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())