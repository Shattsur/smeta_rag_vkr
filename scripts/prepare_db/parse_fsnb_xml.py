# scripts/parse_fsnb_xml.py
# Парсинг XML ФСНБ-2022 (ГЭСН/ГЭСНм/...) и сохранение нормативов в JSONL
# Запуск: python scripts/parse_fsnb_xml.py

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    work_code: str
    title: str
    metadata: dict


def project_root() -> Path:
    # .../smeta_rag_project/scripts/parse_fsnb_xml.py -> .../smeta_rag_project
    return Path(__file__).resolve().parents[1]


def iter_normative_chunks(xml_path: Path, start_index: int = 0) -> tuple[int, list[Chunk]]:
    """
    Возвращает (next_index, chunks) для одного файла.
    Делает обычный parse (ElementTree) — для твоих размеров может быть ок,
    но при желании можно будет заменить на iterparse.
    """
    chunks: list[Chunk] = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Parse failed: {xml_path.name}: {e}", file=sys.stderr)
        return start_index, chunks

    # 1) Определяем тип XML по root-тегу
    # У тебя встречается root="base" (нормативы) и root="NewDataSet" (ТГ/прочее). :contentReference[oaicite:3]{index=3}
    root_tag = (root.tag or "").split("}")[-1]  # на случай namespace
    if root_tag.lower() != "base":
        # Это не файл с Work/Section/NameGroup; пропускаем.
        return start_index, chunks

    base_name = root.get("BaseName", xml_path.name)

    # 2) Достаём нормативы: Section[@Type='Таблица'] -> NameGroup -> Work
    # В отчёте структуры для base_root это реально присутствует. :contentReference[oaicite:4]{index=4}
    for table in root.findall(".//Section[@Type='Таблица']"):
        table_code = table.get("Code", "") or ""
        table_name = table.get("Name", "") or ""

        for name_group in table.findall(".//NameGroup"):
            begin_name = name_group.get("BeginName", "") or ""

            for work in name_group.findall("Work"):
                work_code = work.get("Code", "") or ""
                end_name = work.get("EndName", "") or ""
                measure = work.get("MeasureUnit", "") or ""

                full_title = f"{begin_name} {end_name}".strip() or table_name or work_code

                # Content / Item
                content_items = []
                for item in work.findall(".//Content/Item"):
                    txt = item.get("Text", "") or ""
                    if txt.strip():
                        content_items.append(txt.strip())
                content = "\n".join(content_items)

                # Resources / Resource
                res_lines = []
                for res in work.findall(".//Resources/Resource"):
                    res_code = res.get("Code", "") or ""
                    res_name = res.get("EndName", "") or ""
                    res_qty = res.get("Quantity", "") or ""
                    # Не лепим лишние пробелы, если полей нет
                    left = " ".join([p for p in [res_code, res_name] if p]).strip()
                    if left or res_qty:
                        res_lines.append(f"{left} — {res_qty}".strip(" —"))
                resources_text = "; ".join(res_lines)

                # NrSp / ReasonItem
                nr_sp = []
                for reason in work.findall(".//NrSp/ReasonItem"):
                    nr = reason.get("Nr", "") or ""
                    sp = reason.get("Sp", "") or ""
                    if nr or sp:
                        nr_sp.append(f"Норма: {nr}, Спецификация: {sp}".strip().strip(","))
                nr_text = " | ".join(nr_sp)

                chunk_text = "\n".join([
                    f"НОРМАТИВ: {work_code} {full_title}".strip(),
                    f"Единица измерения: {measure}".strip(),
                    f"Сборник: {base_name} | Таблица {table_code} {table_name}".strip(),
                    "",
                    "СОДЕРЖАНИЕ РАБОТЫ:",
                    content if content else "Не указано",
                    "",
                    "РЕСУРСЫ:",
                    resources_text if resources_text else "Не указано",
                    "",
                    "НОРМАТИВНЫЕ ССЫЛКИ:",
                    nr_text if nr_text else "Не указано",
                ]).strip()

                chunk_id = f"fsnb_{start_index}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=xml_path.name,
                    work_code=work_code,
                    title=f"{work_code} {full_title}".strip(),
                    metadata={
                        "type": "fsnb_normative",
                        "table_code": table_code,
                        "measure": measure,
                        "base_name": base_name,
                    },
                ))
                start_index += 1

    return start_index, chunks


def main() -> int:
    pr = project_root()

    parser = argparse.ArgumentParser(description="Parse FSNB XML (base-root) to JSONL chunks.")
    parser.add_argument("--xml-dir", type=str, default=str(pr / "data" / "fsnb_2022_izm16_xml"))
    parser.add_argument("--out", type=str, default=str(pr / "data" / "raw" / "fsnb_chunks.jsonl"))
    parser.add_argument("--glob", type=str, default="*.xml")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    out_path = Path(args.out)

    if not xml_dir.exists():
        print(f"[ERROR] XML dir not found: {xml_dir}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(xml_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No XML files found in {xml_dir} with glob={args.glob}")
        return 0

    total = 0
    skipped = 0
    idx = 0

    print(f"[INFO] XML dir: {xml_dir}")
    print(f"[INFO] Output : {out_path}")
    print(f"[INFO] Files  : {len(files)}")

    with out_path.open("w", encoding="utf-8") as f:
        for xml_path in files:
            print(f"[INFO] Scan: {xml_path.name}")

            idx_before = idx
            idx, chunks = iter_normative_chunks(xml_path, start_index=idx)

            # root != base (например NewDataSet) -> chunks пустые, считаем как skipped
            if not chunks:
                # если idx не изменился — значит ничего не извлекли
                if idx == idx_before:
                    skipped += 1
                continue

            for ch in chunks:
                f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")
                total += 1

    print(f"[INFO] Done. Extracted chunks: {total}")
    print(f"[INFO] Skipped files (non-normative structure or empty): {skipped}")
    print(f"[INFO] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
