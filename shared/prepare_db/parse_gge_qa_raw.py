# -*- coding: utf-8 -*-
"""
prepare_gge_qa_raw.py

Pipeline в одном файле:

1) SCRAPE:
   Скачивает все Q/A с https://gge.ru/services/questions/ (пагинация ?PAGEN_1=)
   -> пишет data/qa/gge_qa.csv (Question, Answer, URL)

2) CONVERT:
   Конвертирует data/qa/gge_qa.csv -> data/raw/gge_qa.jsonl
   - чистит текст (NBSP, лишние пробелы, пустые строки)
   - дедупит (сначала по URL, затем по (Question+Answer))
   - извлекает qa_id из URL вида /services/questions/<num>/<num>/
   - пишет JSONL, 1 запись = 1 Q/A (без чанкинга)

Запуск:
  python scripts/prepare_gge_qa_raw.py              # scrape + convert
  python scripts/prepare_gge_qa_raw.py --skip-scrape # только convert (если csv уже есть)
  python scripts/prepare_gge_qa_raw.py --skip-convert # только scrape (если jsonl не нужен)

Параметры:
  --list-url   (по умолчанию https://gge.ru/services/questions/)
  --out-csv    (по умолчанию data/qa/gge_qa.csv)
  --out-jsonl  (по умолчанию data/raw/gge_qa.jsonl)
  --sleep-list пауза между страницами листинга
  --sleep-item пауза между карточками
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    base: str = "https://gge.ru"
    list_url: str = "https://gge.ru/services/questions/"
    out_csv: Path = Path("data/qa/gge_qa.csv")
    out_jsonl: Path = Path("data/raw/gge_qa.jsonl")
    encoding: str = "utf-8"
    timeout: int = 30
    sleep_list: float = 0.15
    sleep_item: float = 0.20
    min_q_len: int = 5
    min_a_len: int = 10


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36"
)

# /services/questions/<num>/<num>/
Q_PATH_RE = re.compile(r"^/services/questions/\d+/\d+/?$")
# <num>/<num>/ (относительно /services/questions/)
Q_REL_RE = re.compile(r"^\d+/\d+/?$")

# /services/questions/2616/95860/
QA_ID_RE = re.compile(r"^/services/questions/(\d+)/(\d+)/?$")

QA_BLOCK_RE = re.compile(
    r"Вопрос\s*:\s*(?P<body>.*?)(?:\n\s*Распечатать\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# -----------------------------
# Shared helpers
# -----------------------------
def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def fetch(session: requests.Session, url: str, timeout: int) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.encoding:
        r.encoding = "utf-8"
    return r.text


def soup_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text("\n", strip=True).replace("\u00a0", " ")
    return re.sub(r"\n{3,}", "\n\n", txt).strip()


def extract_max_page(list_html: str) -> int:
    soup = BeautifulSoup(list_html, "html.parser")
    nums: List[int] = []
    for a in soup.select('a[href*="PAGEN_1="]'):
        href = a.get("href", "")
        m = re.search(r"PAGEN_1=(\d+)", href)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 1


def normalize_question_path(href: str) -> str | None:
    """
    Приводит href к виду /services/questions/<num>/<num>/
    Возвращает None если это не карточка вопроса.
    """
    href = (href or "").strip()
    if not href:
        return None

    if "://" in href:
        path = urlparse(href).path
    else:
        path = href

    if Q_PATH_RE.match(path):
        return path if path.endswith("/") else path + "/"

    if Q_REL_RE.match(path):
        path = "/services/questions/" + path.lstrip("/")
        return path if path.endswith("/") else path + "/"

    return None


def extract_question_links(list_html: str) -> List[str]:
    soup = BeautifulSoup(list_html, "html.parser")
    out: List[str] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        qpath = normalize_question_path(href)
        if qpath and qpath not in seen:
            seen.add(qpath)
            out.append(qpath)
    return out


def parse_qa_from_page(html: str, min_q_len: int, min_a_len: int) -> Tuple[str, str]:
    txt = soup_text(html)
    m = QA_BLOCK_RE.search(txt)
    if not m:
        return "", ""

    block = (m.group("body") or "").strip()
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if not lines:
        return "", ""

    # Вопрос — обычно первая строка, но иногда вопрос разнесён на несколько строк
    q_lines = [lines[0]]
    k = 1
    while k < len(lines) and lines[k].endswith("?"):
        q_lines.append(lines[k])
        k += 1

    question = " ".join(q_lines).strip()
    answer = "\n".join(lines[k:]).strip()

    question = norm_text(question)
    answer = norm_text(answer)

    if len(question) < min_q_len or len(answer) < min_a_len:
        return "", ""

    return question, answer


def extract_qa_id(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    path = urlparse(url).path
    m = QA_ID_RE.match(path)
    if not m:
        return ""
    return f"{m.group(1)}_{m.group(2)}"


# -----------------------------
# Stage 1: scrape -> CSV
# -----------------------------
def scrape_to_csv(cfg: Config) -> None:
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": UA,
            "Accept-Language": "ru,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )

    first_html = fetch(session, cfg.list_url, cfg.timeout)
    max_page = extract_max_page(first_html)
    print("[INFO] Max page:", max_page)

    all_links: List[str] = []
    for p in range(1, max_page + 1):
        url = cfg.list_url if p == 1 else f"{cfg.list_url}?PAGEN_1={p}"
        html = fetch(session, url, cfg.timeout)
        links = extract_question_links(html)

        if p == 1 and len(links) == 0:
            soup = BeautifulSoup(html, "html.parser")
            sample = [a.get("href", "") for a in soup.select("a[href]")][:30]
            print("[DEBUG] sample hrefs:", sample)

        print(f"[INFO] Page {p}: links={len(links)}")
        all_links.extend(links)
        time.sleep(cfg.sleep_list)

    # unique, order-preserving
    seen: set[str] = set()
    all_links = [h for h in all_links if not (h in seen or seen.add(h))]
    print("[INFO] Total links:", len(all_links))

    rows: List[Tuple[str, str, str]] = []
    fails = 0

    for i, rel_path in enumerate(all_links, 1):
        url = urljoin(cfg.base, rel_path)
        try:
            html = fetch(session, url, cfg.timeout)
            q, a = parse_qa_from_page(html, cfg.min_q_len, cfg.min_a_len)
            if q and a:
                rows.append((q, a, url))
            else:
                fails += 1
                if fails <= 5:
                    t = soup_text(html)
                    snippet = t[t.find("Вопрос"): t.find("Вопрос") + 200] if "Вопрос" in t else t[:200]
                    print("[WARN] FAIL parse:", url)
                    print("[WARN] Preview:", snippet)
        except Exception as e:
            fails += 1
            if fails <= 5:
                print("[WARN] FAIL fetch:", url, "->", e)

        time.sleep(cfg.sleep_item)

    with cfg.out_csv.open("w", encoding=cfg.encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Answer", "URL"])
        w.writerows(rows)

    print(f"[OK] Scraped: {len(rows)} rows -> {cfg.out_csv.as_posix()} (fails={fails})")


# -----------------------------
# Stage 2: CSV -> JSONL
# -----------------------------
def iter_rows(csv_path: Path, encoding: str) -> Iterable[dict]:
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            q = norm_text(row.get("Question", ""))
            a = norm_text(row.get("Answer", ""))
            url = (row.get("URL") or "").strip()

            if not q or not a:
                continue

            qa_id = extract_qa_id(url) or str(i)

            yield {
                "source": url,
                "type": "qa_gge",
                "qa_id": qa_id,
                "question": q,
                "answer": a,
            }


def convert_csv_to_jsonl(cfg: Config) -> None:
    if not cfg.out_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.out_csv}")

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 1) дедуп по URL (если URL есть)
    by_url: Dict[str, dict] = {}
    no_url: List[dict] = []

    total_in = 0
    for rec in iter_rows(cfg.out_csv, cfg.encoding):
        total_in += 1
        url = (rec.get("source") or "").strip()
        if url:
            if url not in by_url:
                by_url[url] = rec
        else:
            no_url.append(rec)

    recs = list(by_url.values()) + no_url
    kept_after_url = len(recs)

    # 2) дедуп по содержимому (Question+Answer)
    seen_pair: set[Tuple[str, str]] = set()
    final: List[dict] = []
    for rec in recs:
        key = (rec["question"], rec["answer"])
        if key in seen_pair:
            continue
        seen_pair.add(key)
        final.append(rec)

    with cfg.out_jsonl.open("w", encoding=cfg.encoding) as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Read rows (non-empty Q/A): {total_in}")
    print(f"[OK] After URL dedup          : {kept_after_url}")
    print(f"[OK] After content dedup      : {len(final)}")
    print(f"[OK] Wrote                    : {cfg.out_jsonl.as_posix()}")


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--list-url", type=str, default=cfg.list_url)
    p.add_argument("--out-csv", type=str, default=str(cfg.out_csv))
    p.add_argument("--out-jsonl", type=str, default=str(cfg.out_jsonl))
    p.add_argument("--timeout", type=int, default=cfg.timeout)
    p.add_argument("--sleep-list", type=float, default=cfg.sleep_list)
    p.add_argument("--sleep-item", type=float, default=cfg.sleep_item)
    p.add_argument("--skip-scrape", action="store_true", help="skip scraping stage")
    p.add_argument("--skip-convert", action="store_true", help="skip conversion stage")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        list_url=args.list_url,
        out_csv=Path(args.out_csv),
        out_jsonl=Path(args.out_jsonl),
        timeout=int(args.timeout),
        sleep_list=float(args.sleep_list),
        sleep_item=float(args.sleep_item),
    )

    if not args.skip_scrape:
        scrape_to_csv(cfg)

    if not args.skip_convert:
        convert_csv_to_jsonl(cfg)


if __name__ == "__main__":
    main()
