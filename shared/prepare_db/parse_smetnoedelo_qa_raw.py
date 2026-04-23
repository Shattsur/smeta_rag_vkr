# -*- coding: utf-8 -*-
"""
prepare_smetnoedelo_qa_raw.py

Надёжный парсер Q/A со smetnoedelo.ru/vopros-otvet/
- собирает ссылки по PAGEN_1
- нормализует .html/ -> .html (исправляет ваш 404-кейс)
- retry + exponential backoff + jitter для 429/503/504 и похожих
- потоковая запись JSONL (можно продолжать после остановки)
- дедуп по URL и по (question+answer)

Примеры:
  python scripts/prepare_smetnoedelo_qa_raw.py
  python scripts/prepare_smetnoedelo_qa_raw.py --start 1 --max-pages 95
  python scripts/prepare_smetnoedelo_qa_raw.py --start 69 --max-pages 95   # если до 68 дошёл и дальше 503
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class Config:
    base: str = "https://smetnoedelo.ru"
    list_url: str = "https://smetnoedelo.ru/vopros-otvet/"
    out_jsonl: Path = Path("data/raw/smetnoedelo_qa.jsonl")
    timeout: int = 30

    # rate limiting
    sleep_list: float = 0.6
    sleep_item: float = 0.8
    jitter: float = 0.35  # добавка случайной паузы

    # retries / backoff
    retries: int = 6
    backoff_base: float = 1.4   # экспоненциальный рост
    backoff_cap: float = 60.0   # максимум паузы между ретраями

    # min lengths
    min_q_len: int = 5
    min_a_len: int = 10

    encoding: str = "utf-8"


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36"
)

# допустим, карточки выглядят как /vopros-otvet/<slug>.html  (иногда с лишним /)
ITEM_RE = re.compile(r"^/vopros-otvet/[^/]+\.html/?$")


def norm_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def normalize_item_url(url: str) -> str:
    """
    Фикс .html/ -> .html (у вас именно этот кейс).
    Также убираем фрагменты и лишние хвостовые / (кроме корня).
    """
    url = (url or "").strip()
    if not url:
        return url

    p = urlparse(url)
    path = p.path or ""

    if path.endswith(".html/"):
        path = path[:-1]  # убрать последний '/'

    # на всякий случай: двойные //
    path = re.sub(r"/{2,}", "/", path)

    # убрать fragment
    p2 = p._replace(path=path, fragment="")
    return urlunparse(p2)


def sleep_polite(base: float, jitter: float) -> None:
    time.sleep(base + random.random() * jitter)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": UA,
            "Accept-Language": "ru,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return s


def fetch(session: requests.Session, url: str, cfg: Config) -> Optional[str]:
    """
    Вежливый fetch с ретраями на типичных временных/лимитных кодах.
    """
    url = normalize_item_url(url)
    last_err: Optional[Exception] = None

    for attempt in range(cfg.retries + 1):
        try:
            r = session.get(url, timeout=cfg.timeout, allow_redirects=True)
            # иногда сайт может отдавать 503/429 и т.п.
            if r.status_code in (429, 500, 502, 503, 504, 520, 521, 522, 524):
                raise requests.HTTPError(f"{r.status_code} for {url}", response=r)

            r.raise_for_status()
            if not r.encoding:
                r.encoding = cfg.encoding
            return r.text

        except Exception as e:
            last_err = e
            if attempt >= cfg.retries:
                break

            # backoff + jitter
            backoff = min(cfg.backoff_cap, (cfg.backoff_base ** attempt))
            # если есть Retry-After — уважаем
            retry_after = None
            if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                ra = e.response.headers.get("Retry-After")
                if ra and ra.isdigit():
                    retry_after = float(ra)

            wait_s = (retry_after if retry_after is not None else backoff)
            wait_s += random.random() * cfg.jitter
            time.sleep(wait_s)

    # все ретраи исчерпаны
    return None


def extract_max_page(list_html: str) -> int:
    soup = BeautifulSoup(list_html, "html.parser")
    nums: List[int] = []
    for a in soup.select('a[href*="PAGEN_1="]'):
        href = a.get("href", "")
        m = re.search(r"PAGEN_1=(\d+)", href)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 1


def extract_item_links(list_html: str, base: str) -> List[str]:
    soup = BeautifulSoup(list_html, "html.parser")
    out: List[str] = []
    seen: Set[str] = set()

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        # абсолютный/относительный -> абсолютный
        abs_url = urljoin(base, href)
        abs_url = normalize_item_url(abs_url)

        path = urlparse(abs_url).path
        if not ITEM_RE.match(path):
            continue

        if abs_url not in seen:
            seen.add(abs_url)
            out.append(abs_url)

    return out


def parse_qa(html: str, cfg: Config) -> Tuple[str, str]:
    """
    Без знания точной разметки делаем аккуратно:
    - берем заголовок h1 как вопрос (часто так)
    - ответ: основной текст статьи (article / .content / main), иначе весь текст
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) вопрос
    h1 = soup.select_one("h1")
    question = norm_text(h1.get_text(" ", strip=True)) if h1 else ""

    # 2) ответ
    candidates = [
        soup.select_one("article"),
        soup.select_one("main"),
        soup.select_one(".content"),
        soup.select_one(".post-content"),
        soup.select_one(".entry-content"),
    ]
    block = next((c for c in candidates if c is not None), None)
    if block is None:
        text = soup.get_text("\n", strip=True)
    else:
        text = block.get_text("\n", strip=True)

    text = norm_text(text)

    # часто h1 дублируется в тексте — уберём первое вхождение
    if question and text.lower().startswith(question.lower()):
        text = norm_text(text[len(question):])

    answer = text

    if len(question) < cfg.min_q_len or len(answer) < cfg.min_a_len:
        return "", ""
    return question, answer


def load_seen_urls(out_jsonl: Path) -> Set[str]:
    seen: Set[str] = set()
    if not out_jsonl.exists():
        return seen

    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                url = (rec.get("source") or "").strip()
                if url:
                    seen.add(url)
            except Exception:
                continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=0, help="0 = auto detect")
    parser.add_argument("--out", type=str, default=str(Config().out_jsonl))
    parser.add_argument("--sleep-list", type=float, default=Config().sleep_list)
    parser.add_argument("--sleep-item", type=float, default=Config().sleep_item)
    parser.add_argument("--retries", type=int, default=Config().retries)
    args = parser.parse_args()

    cfg = Config(
        out_jsonl=Path(args.out),
        sleep_list=float(args.sleep_list),
        sleep_item=float(args.sleep_item),
        retries=int(args.retries),
    )
    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    session = make_session()

    first_html = fetch(session, cfg.list_url, cfg)
    if not first_html:
        raise RuntimeError(f"Cannot fetch list url: {cfg.list_url}")

    detected_max = extract_max_page(first_html)
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else detected_max
    start = max(1, args.start)

    print(f"[INFO] Max page detected: {detected_max}; will parse: {start}..{max_pages}")

    # собираем ссылки
    links_all: List[str] = []
    list_fail = 0
    for p in range(start, max_pages + 1):
        list_url = cfg.list_url if p == 1 else f"{cfg.list_url}?PAGEN_1={p}"
        html = fetch(session, list_url, cfg)
        if not html:
            list_fail += 1
            print(f"[WARN] list page failed: {list_url}")
            # если подряд много фейлов — вероятно, бан/лимит, лучше остановиться
            if list_fail >= 5:
                print("[WARN] Too many list failures in a row; stopping list crawl.")
                break
            continue

        list_fail = 0
        links = extract_item_links(html, cfg.base)
        print(f"[INFO] Page {p}: links={len(links)}")
        links_all.extend(links)
        sleep_polite(cfg.sleep_list, cfg.jitter)

    # unique
    seen_tmp: Set[str] = set()
    links_all = [u for u in links_all if not (u in seen_tmp or seen_tmp.add(u))]
    print(f"[INFO] Total unique question links: {len(links_all)}")

    # resume support
    seen_urls = load_seen_urls(cfg.out_jsonl)
    if seen_urls:
        print(f"[INFO] Resume: already have {len(seen_urls)} urls in {cfg.out_jsonl.name}")

    # дедуп по контенту (в рамках текущего запуска)
    seen_pair: Set[Tuple[str, str]] = set()

    ok = 0
    fail = 0

    with cfg.out_jsonl.open("a", encoding=cfg.encoding) as out:
        for idx, url in enumerate(links_all, 1):
            url = normalize_item_url(url)
            if url in seen_urls:
                continue

            html = fetch(session, url, cfg)
            if not html:
                fail += 1
                print(f"[WARN] item failed: {url}")
                sleep_polite(cfg.sleep_item, cfg.jitter)
                continue

            q, a = parse_qa(html, cfg)
            if not q or not a:
                fail += 1
                print(f"[WARN] parse failed: {url}")
                sleep_polite(cfg.sleep_item, cfg.jitter)
                continue

            key = (q, a)
            if key in seen_pair:
                # локальный дедуп; URL всё равно отметим как “seen”
                seen_urls.add(url)
                continue
            seen_pair.add(key)

            rec = {
                "source": url,
                "type": "qa_smetnoedelo",
                "question": q,
                "answer": a,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

            ok += 1
            seen_urls.add(url)

            if ok % 50 == 0:
                print(f"[INFO] progress: ok={ok} fail={fail} last={url}")

            sleep_polite(cfg.sleep_item, cfg.jitter)

    print(f"[OK] Done. ok={ok} fail={fail} -> {cfg.out_jsonl.as_posix()}")


if __name__ == "__main__":
    main()
