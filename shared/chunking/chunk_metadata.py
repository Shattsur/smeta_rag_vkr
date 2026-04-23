# -*- coding: utf-8 -*-
"""Нормализация метаданных чанков для Chroma (общая для rag_gen и load_to_chroma_base)."""
import json
from typing import Any, Dict, Optional

_CHROMA_ALLOWED = (str, int, float, bool)


def st(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def _coerce_metadata_value(v: Any) -> Optional[Any]:
    if v is None:
        return None
    if isinstance(v, _CHROMA_ALLOWED):
        return v
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s if s else None
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def normalize_chunk_metadata(obj: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    raw_md: Dict[str, Any] = obj.get("metadata", {})
    text: str = obj.get("text", "")
    md: Dict[str, Any] = {}
    md["chunk_id"] = chunk_id
    md["text_length"] = len(text)
    source = st(obj.get("source") or raw_md.get("source") or raw_md.get("source_file") or "")
    if source:
        md["source"] = source
    doc_type = st(raw_md.get("doc_type") or raw_md.get("type") or "unknown")
    md["doc_type"] = doc_type
    _optional_str_fields = [
        ("work_code", ["work_code"]),
        ("question", ["question"]),
        ("answer", ["answer"]),
        ("title", ["title"]),
        ("clause", ["clause"]),
        ("subclause", ["subclause"]),
        ("clause_hierarchy", ["clause_hierarchy"]),
        ("clause_title", ["clause_title"]),
        ("extraction", ["extraction"]),
        ("section", ["section"]),
        ("subsection", ["subsection"]),
    ]
    for target_key, source_keys in _optional_str_fields:
        for sk in source_keys:
            val = st(raw_md.get(sk) or obj.get(sk) or "")
            if val:
                md[target_key] = val
                break
    for key in ("page_start", "page_end", "chunk_index", "total_chunks"):
        raw_val = raw_md.get(key)
        if raw_val is not None:
            try:
                md[key] = int(raw_val)
            except (ValueError, TypeError):
                pass
    clean_md: Dict[str, Any] = {}
    for k, v in md.items():
        coerced = _coerce_metadata_value(v)
        if coerced is None:
            continue
        if isinstance(coerced, str) and not coerced:
            continue
        clean_md[k] = coerced
    return clean_md
