# scripts/merge_and_prepare_chunks.py
# -*- coding: utf-8 -*-
"""
Merge official_docs.jsonl + fsnb_chunks.jsonl + gge_qa.jsonl + smetnoedelo_qa.cleaned.jsonl
into data/chunks/all_chunks.jsonl

ИСПРАВЛЕНИЯ:
✅ Сохранение subclause и clause_hierarchy из extract_pdf.py v2.0
✅ Параметры: chunk_size=1500, chunk_overlap=300, min_chars=100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter


def project_root() -> Path:
    return Path(__file__).resolve().parents[2] 


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


def extract_gge_qa_id(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 4 and parts[-3] == "questions":
        return f"{parts[-2]}_{parts[-1]}"
    return ""


def extract_slug_id(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    path = urlparse(url).path.strip("/")
    if not path:
        return ""
    seg = path.split("/")[-1]
    return seg or ""


def iter_qa_records(qa_path: Path, qa_type: str) -> Iterator[Tuple[str, str, str, str, int]]:
    with qa_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            rec = jload_line(line)
            if not rec:
                yield ("", "", "", "", line_num)
                continue

            q = clean_text(rec.get("question") or "")
            a = clean_text(rec.get("answer") or "")
            url = (rec.get("source") or rec.get("url") or "").strip()

            if not q or not a:
                yield ("", "", "", "", line_num)
                continue

            qa_id = (rec.get("qa_id") or "").strip()
            if not qa_id:
                if qa_type == "qa_gge":
                    qa_id = extract_gge_qa_id(url) or str(line_num)
                else:
                    qa_id = extract_slug_id(url) or str(line_num)

            yield (q, a, url, qa_id, line_num)


def _split_text(splitter: RecursiveCharacterTextSplitter, text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [clean_text(x) for x in splitter.split_text(text) if clean_text(x)]


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
    parser.add_argument("--qa-gge", type=str, default=str(pr / "data" / "raw" / "gge_qa.jsonl"))
    parser.add_argument("--qa-smetnoedelo", type=str, default=str(pr / "data" / "raw" / "smetnoedelo_qa.cleaned.jsonl"))
    parser.add_argument("--out", type=str, default=str(pr / "data" / "chunks" / "all_chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=1500)      # ✅ Увеличено
    parser.add_argument("--chunk-overlap", type=int, default=300)    # ✅ Увеличено
    parser.add_argument("--min-chars", type=int, default=100)        # ✅ Увеличено
    args = parser.parse_args()

    official_path = Path(args.official)
    fsnb_path = Path(args.fsnb)
    gge_path = Path(args.qa_gge)
    smetnoedelo_path = Path(args.qa_smetnoedelo)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not official_path.exists():
        raise FileNotFoundError(f"official_docs.jsonl not found: {official_path}")
    if not fsnb_path.exists():
        raise FileNotFoundError(f"fsnb_chunks.jsonl not found: {fsnb_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total = 0
    official_chunks = 0
    fsnb_chunks = 0
    gge_chunks = 0
    smetnoedelo_chunks = 0
    skipped_official_empty = 0
    skipped_official_bad = 0
    skipped_fsnb_empty = 0
    skipped_fsnb_bad = 0
    skipped_gge_rows = 0
    skipped_smetnoedelo_rows = 0

    print("[INFO] Merging chunks...")
    print(f"[INFO] official       : {official_path}")
    print(f"[INFO] fsnb           : {fsnb_path}")
    print(f"[INFO] qa_gge         : {gge_path} ({'OK' if gge_path.exists() else 'NOT FOUND - will skip'})")
    print(f"[INFO] qa_smetnoedelo : {smetnoedelo_path} ({'OK' if smetnoedelo_path.exists() else 'NOT FOUND - will skip'})")
    print(f"[INFO] out            : {out_path}")

    with out_path.open("w", encoding="utf-8") as out:
        # 1) Official docs
        with official_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                rec = jload_line(line)
                if rec is None:
                    skipped_official_bad += 1
                    continue

                text = clean_text(rec.get("text") or "")
                if not text:
                    skipped_official_empty += 1
                    continue

                md_in = _md(rec)

                base_chunk_id = clean_text(rec.get("chunk_id") or md_in.get("chunk_id") or f"official_line{line_num}")
                source = clean_text(rec.get("source") or md_in.get("source") or md_in.get("source_file") or "")

                doc_type = clean_text(md_in.get("type") or md_in.get("doc_type") or "official_normative")
                doc_id = clean_text(md_in.get("doc_id") or "")
                title = clean_text(md_in.get("title") or "")
                clause = clean_text(md_in.get("clause") or "")
                
                # ✅ НОВОЕ: Сохраняем subclause и clause_hierarchy
                subclause = clean_text(md_in.get("subclause") or "")
                clause_hierarchy = clean_text(md_in.get("clause_hierarchy") or clause)
                
                clause_title = clean_text(md_in.get("clause_title") or "")
                page_start = _as_int(md_in.get("page_start"), 0)
                page_end = _as_int(md_in.get("page_end"), 0)
                extraction = clean_text(md_in.get("extraction") or "")

                parts = [text]
                if len(text) > int(args.chunk_size) * 2:
                    parts = _split_text(splitter, text) or [text]

                for part_idx, part in enumerate(parts, 1):
                    if len(part) < int(args.min_chars):
                        continue

                    chunk_id = base_chunk_id if len(parts) == 1 else f"{base_chunk_id}__p{part_idx}"

                    out_chunk = {
                        "chunk_id": chunk_id,
                        "text": part,
                        "source": source,
                        "metadata": {
                            "type": doc_type or "official_normative",
                            "doc_id": doc_id,
                            "title": title,
                            "clause": clause,
                            "subclause": subclause if subclause else None,  # ✅ СОХРАНЯЕМ
                            "clause_hierarchy": clause_hierarchy,            # ✅ СОХРАНЯЕМ
                            "clause_title": clause_title,
                            "page_start": page_start,
                            "page_end": page_end,
                            "extraction": extraction,
                            "line_num": line_num,
                            "part_number": part_idx,
                            "total_parts": len(parts),
                            "orig_chunk_id": base_chunk_id,
                        },
                    }

                    out.write(json.dumps(out_chunk, ensure_ascii=False) + "\n")
                    total += 1
                    official_chunks += 1

        # 2) FSNB chunks
        with fsnb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                ch = jload_line(line)
                if ch is None:
                    skipped_fsnb_bad += 1
                    continue

                text = clean_text(ch.get("text") or "")
                if not text:
                    skipped_fsnb_empty += 1
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

                out.write(json.dumps(out_chunk, ensure_ascii=False) + "\n")
                total += 1
                fsnb_chunks += 1

        # 3) GGE QA
        if gge_path.exists():
            for q, a, url, qa_id, line_num in iter_qa_records(gge_path, qa_type="qa_gge"):
                if not q or not a:
                    skipped_gge_rows += 1
                    continue

                qa_text_full = f"Вопрос: {q}\n\nОтвет: {a}"
                parts = _split_text(splitter, qa_text_full)
                if not parts:
                    skipped_gge_rows += 1
                    continue

                title = (q[:200] + "…") if len(q) > 200 else q

                for part_idx, part in enumerate(parts, 1):
                    if len(part) < int(args.min_chars):
                        continue

                    chunk = {
                        "chunk_id": f"qa_gge_{qa_id}_{part_idx}",
                        "text": part,
                        "source": url,
                        "question": q,
                        "answer": a,
                        "metadata": {
                            "type": "qa_gge",
                            "title": title,
                            "qa_id": qa_id,
                            "url": url,
                            "line_num": line_num,
                            "part_number": part_idx,
                            "total_parts": len(parts),
                            "question": q,
                            "answer": a,
                        },
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1
                    gge_chunks += 1
        else:
            print(f"[INFO] QA GGE file not found, skipped: {gge_path}")

        # 4) Smetnoedelo QA cleaned
        if smetnoedelo_path.exists():
            for q, a, url, qa_id, line_num in iter_qa_records(smetnoedelo_path, qa_type="qa_smetnoedelo"):
                if not q or not a:
                    skipped_smetnoedelo_rows += 1
                    continue

                qa_text_full = f"Вопрос: {q}\n\nОтвет: {a}"
                parts = _split_text(splitter, qa_text_full)
                if not parts:
                    skipped_smetnoedelo_rows += 1
                    continue

                title = (q[:200] + "…") if len(q) > 200 else q

                for part_idx, part in enumerate(parts, 1):
                    if len(part) < int(args.min_chars):
                        continue

                    chunk = {
                        "chunk_id": f"qa_smetnoedelo_{qa_id}_{part_idx}",
                        "text": part,
                        "source": url,
                        "question": q,
                        "answer": a,
                        "metadata": {
                            "type": "qa_smetnoedelo",
                            "title": title,
                            "qa_id": qa_id,
                            "url": url,
                            "line_num": line_num,
                            "part_number": part_idx,
                            "total_parts": len(parts),
                            "question": q,
                            "answer": a,
                        },
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1
                    smetnoedelo_chunks += 1
        else:
            print(f"[INFO] QA Smetnoedelo file not found, skipped: {smetnoedelo_path}")

    print("[INFO] Done.")
    print(f"[INFO] Total chunks           : {total}")
    print(
        f"[INFO] Official chunks        : {official_chunks} "
        f"(skipped empty: {skipped_official_empty}, bad_json: {skipped_official_bad})"
    )
    print(f"[INFO] FSNB chunks            : {fsnb_chunks} (skipped empty: {skipped_fsnb_empty}, bad_json: {skipped_fsnb_bad})")
    print(f"[INFO] QA GGE chunks          : {gge_chunks} (skipped rows: {skipped_gge_rows})")
    print(f"[INFO] QA Smetnoedelo chunks  : {smetnoedelo_chunks} (skipped rows: {skipped_smetnoedelo_rows})")
    print(f"[INFO] Wrote                  : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())