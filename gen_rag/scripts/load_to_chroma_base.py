# -*- coding: utf-8 -*-
"""
scripts/load_to_chroma_base.py — v4.2
Загрузка/обновление чанков в Chroma (idempotent: upsert по chunk_id).
По умолчанию: без полного сброса; при большом числе чанков — «умный кэш» (пропуск), см. --force-load.
Метаданные — ``normalize_chunk_metadata`` из ``chunk_metadata`` (согласовано с rag_gen).
"""

import os
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PR = Path(__file__).resolve().parents[1]
_REPO_ROOT = PR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chromadb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from shared.chunking.chunk_metadata import normalize_chunk_metadata

DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "deepvk/USER2-base")
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "smeta_collection")
DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_PATH", str(PR / "chroma_db_base"))
DEFAULT_CHUNKS_JSONL = os.environ.get("CHUNKS_JSONL", str(PR / "data" / "chunks" / "all_chunks.jsonl"))
DEFAULT_RESET = os.environ.get("CHROMA_RESET", "false").lower() == "true"
DEFAULT_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "512"))
SMART_CACHE_THRESHOLD = int(os.environ.get("CHROMA_CACHE_SKIP_MIN", "500"))
CHROMA_FORCE_LOAD = os.environ.get("CHROMA_FORCE_LOAD", "").lower() in ("1", "true", "yes")


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def st(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def safe_json_load(line: str, line_num: int) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        eprint(f"[WARN] Line {line_num}: Bad JSON, skipping.")
        return None


def choose_device(cli_device: str) -> str:
    if cli_device in ("cuda", "cpu"):
        return cli_device
    return "cuda" if torch.cuda.is_available() and os.environ.get("USE_GPU", "true").lower() == "true" else "cpu"


def infer_dtype(device: str, cli_dtype: str) -> torch.dtype:
    if cli_dtype != "auto":
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cli_dtype]
    return torch.float16 if device == "cuda" else torch.float32


class USER2Embedder:
    def __init__(self, model_name_or_path: str, device: str, max_length: int,
                 batch_size: int, dtype: torch.dtype, truncate_dim: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype).eval().to(device)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            bt = texts[i:i + self.batch_size]
            batch = self.tokenizer(bt, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            emb = self.mean_pooling(outputs, batch["attention_mask"])
            if self.truncate_dim:
                emb = emb[:, :self.truncate_dim]
            emb = F.normalize(emb, p=2, dim=1)
            out.extend(emb.float().cpu().tolist())
        return out

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {doc}" for doc in docs]
        return self._encode(prefixed)

    # ←←← ИСПРАВЛЕНИЕ: добавлен метод embed_query
    def embed_query(self, query: str) -> List[float]:
        prefixed = f"search_query: {query}"
        return self._encode([prefixed])[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Загрузка чанков в Chroma")
    ap.add_argument("--jsonl", default=DEFAULT_CHUNKS_JSONL)
    ap.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reset", action="store_true", help="Полный сброс коллекции")
    g.add_argument("--no-reset", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--truncate-dim", type=int, default=None)
    ap.add_argument(
        "--sanity-query", type=str, default="Устройство подвесных потолков",
        help="Пустая строка или --skip-sanity — не выполнять проверочный запрос",
    )
    ap.add_argument("--sanity-topk", type=int, default=5)
    ap.add_argument("--skip-sanity", action="store_true", help="Пропустить sanity query")
    ap.add_argument(
        "--force-load", action="store_true",
        help="Игнорировать smart-cache (всегда прогнать jsonl/upsert). Env: CHROMA_FORCE_LOAD=1",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    chroma_path = Path(args.chroma_path)

    if not jsonl_path.exists():
        eprint(f"[ERROR] JSONL not found: {jsonl_path}")
        return 1

    reset = args.reset or (DEFAULT_RESET and not args.no_reset)
    device = choose_device(args.device)
    dtype = infer_dtype(device, args.dtype)

    print("=" * 70)
    print("Chroma Loader v4.2 (upsert, shared chunk metadata)")
    print(f"- reset: {reset}")
    print("=" * 70)

    embedder = USER2Embedder(args.model, device, args.max_length, args.embed_batch_size, dtype, args.truncate_dim)
    print("✅ Embedder OK")

    client = chromadb.PersistentClient(path=str(chroma_path))
    if reset:
        try:
            client.delete_collection(args.collection)
            print("🗑️ Коллекция сброшена")
        except Exception as e:
            eprint(f"[WARN] delete_collection: {e}")

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model}
    )
    print("✅ Chroma OK")

    existing_count = collection.count()
    force_load = args.force_load or CHROMA_FORCE_LOAD
    if not reset and not force_load and existing_count > SMART_CACHE_THRESHOLD:
        print(
            f"✅ Smart Cache: уже {existing_count} документов (>{SMART_CACHE_THRESHOLD}) — "
            f"пропускаем загрузку. Для повторной индексации: --force-load или CHROMA_FORCE_LOAD=1"
        )
    else:
        print(f"📦 Загрузка чанков (было в коллекции: {existing_count}) — upsert по chunk_id...")

        batch_docs, batch_ids, batch_mds = [], [], []
        seen_ids = set()
        duplicate_count = bad_rows = loaded_count = 0

        def flush_batch():
            nonlocal loaded_count
            if not batch_docs:
                return
            try:
                embs = embedder.embed_documents(batch_docs)
                collection.upsert(
                    documents=batch_docs, ids=batch_ids, metadatas=batch_mds, embeddings=embs
                )
                loaded_count += len(batch_docs)
            except Exception as e:
                eprint(f"[WARN] upsert() failed: {e}")

        it = enumerate(jsonl_path.open("r", encoding="utf-8"), 1)
        if not args.no_progress:
            it = tqdm(it, desc="Загрузка", unit="chunk")

        for line_num, line in it:
            if args.max_lines and line_num > args.max_lines:
                break
            obj = safe_json_load(line, line_num)
            if not obj:
                bad_rows += 1
                continue
            chunk_id = st(obj.get("chunk_id", ""))
            text = st(obj.get("text", ""))
            if not chunk_id or not text:
                bad_rows += 1
                continue
            if chunk_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(chunk_id)
            md = normalize_chunk_metadata(obj, chunk_id)
            batch_docs.append(text)
            batch_ids.append(chunk_id)
            batch_mds.append(md)
            if len(batch_docs) >= args.batch_size:
                flush_batch()
                batch_docs, batch_ids, batch_mds = [], [], []

        flush_batch()
        print(f"\n✅ Загрузка завершена — всего: {collection.count()}")

    # Sanity query
    do_sanity = st(args.sanity_query) and not args.skip_sanity
    if do_sanity:
        print("\nSanity query:", args.sanity_query)
        q_emb = embedder.embed_query(args.sanity_query)
        res = collection.query(query_embeddings=[q_emb], n_results=args.sanity_topk,
                               include=["documents", "metadatas", "distances"])
        for rank, (dist, id_, md, doc) in enumerate(zip(res["distances"][0], res["ids"][0], res["metadatas"][0], res["documents"][0])):
            cluster_id = md.get('cluster_id', 'N/A') if md else 'N/A'
            work_code = md.get('work_code', 'N/A') if md else 'N/A'
            print(f"{rank+1:02d}) dist={dist:.4f} id={id_} work_code={work_code} cluster={cluster_id}")
            print(f"    {(doc[:200]).replace('\n', ' ')}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())