"""入库层：解析 -> 分块 -> BGE-m3 向量化 -> 写入 Chroma。

用法：python -m rag.cli build
- 每次执行全量重建（幂等）。
- 向量化结果会缓存到 rag/cache/，内容未变时跳过向量化（加速后续重建）。
- chunk 主键使用「相对路径哈希 + 文件名 + 序号」，避免重名文件冲突。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    RAG_DIR,
    STORE_DIR,
    SUPPORTED_EXTS,
)
from .chunk import chunk_text
from .extract import extract_file, file_category

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
CACHE_DIR = RAG_DIR / "cache"


def _collect_files() -> list[Path]:
    files = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.name != ".DS_Store":
            files.append(p)
    return files


def _signature(files: list[Path]) -> str:
    key = "\n".join(sorted(str(f.relative_to(DATA_DIR)) for f in files))
    key += f"\n{EMBED_MODEL}\n{CHUNK_SIZE}\n{CHUNK_OVERLAP}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _parse(files: list[Path]) -> list[tuple[str, str, dict]]:
    rows: list[tuple[str, str, dict]] = []
    for f in files:
        try:
            rec = extract_file(f)
        except Exception as e:  # 单个文件解析失败不影响整体（如 .ppt）
            logging.warning("跳过 %s：%s", f.name, e)
            continue
        rel = str(f.relative_to(DATA_DIR))
        uid = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
        category = file_category(f, DATA_DIR)
        chunks = chunk_text(rec["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, (text, section) in enumerate(chunks):
            rows.append(
                (
                    f"{uid}::{f.stem}::chunk{i}",
                    text,
                    {
                        "source": rel,
                        "filename": f.name,
                        "category": category,
                        "section": section or "",
                        "ext": rec["ext"],
                    },
                )
            )
        logging.info("解析 %s -> %d chunks（%s）", f.name, len(chunks), category)
    return rows


def _save_cache(sig: str, rows, embeddings: np.ndarray) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(CACHE_DIR / f"{sig}.emb.npy", embeddings)
    with open(CACHE_DIR / f"{sig}.rows.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)


def _load_cache(sig: str):
    emb_path = CACHE_DIR / f"{sig}.emb.npy"
    rows_path = CACHE_DIR / f"{sig}.rows.json"
    if emb_path.exists() and rows_path.exists():
        embeddings = np.load(emb_path)
        with open(rows_path, encoding="utf-8") as fh:
            rows = json.load(fh)
        return rows, embeddings
    return None


def build() -> None:
    files = _collect_files()
    sig = _signature(files)

    cached = _load_cache(sig)
    if cached:
        rows, embeddings = cached
        print(f"命中向量缓存，跳过向量化（{len(rows)} chunks）")
    else:
        rows = _parse(files)
        if not rows:
            print("未解析到任何内容，请检查 nlp_ 目录。")
            return
        print(f"加载 embedding 模型 {EMBED_MODEL} …")
        model = SentenceTransformer(EMBED_MODEL)
        docs = [r[1] for r in rows]
        print(f"向量化 {len(rows)} 个 chunk …")
        embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
        _save_cache(sig, rows, embeddings)

    print("写入 Chroma …")
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        metadatas=[r[2] for r in rows],
        embeddings=[e.tolist() for e in embeddings],
    )
    print(f"完成：已入库 {len(rows)} 个 chunk，向量库位于 {STORE_DIR}")
