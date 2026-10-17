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

import numpy as np

from .config import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    PARENT_CHILD_ENABLED,
    PARENT_SIZE,
    STORE_DIR,
    SUPPORTED_EXTS,
)
from .chunk import chunk_parent_child, chunk_text
from .extract import extract_files, file_category, ocr_status, parser_status
from .retrieval import parents_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _collect_files() -> list[Path]:
    files = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.name != ".DS_Store":
            files.append(p)
    return files


def _is_broken_name(name: str) -> bool:
    """文件名里出现替换符/私用区字符 = 编码损坏（常见于 GBK 环境下解压的 zip）。"""
    return any(c == "\ufffd" or "\ue000" <= c <= "\uf8ff" for c in name)


def _check_broken_names(files: list[Path]) -> None:
    broken = [f for f in files if _is_broken_name(str(f.relative_to(DATA_DIR)))]
    if broken:
        for f in broken[:10]:
            logging.warning("文件名编码损坏：%s", ascii(str(f.relative_to(DATA_DIR))))
        raise SystemExit(
            f"检测到 {len(broken)} 个文件名编码损坏，入库后来源路径会是乱码。"
            "请从原始压缩包重新解压/拷贝资料后重试（不要用 GBK 环境下的解压工具）。"
        )


def _signature(files: list[Path]) -> str:
    key = "\n".join(sorted(str(f.relative_to(DATA_DIR)) for f in files))
    # 建库参数变了必须让向量缓存失效，否则会张冠李戴复用旧向量
    key += f"\n{EMBED_MODEL}\n{COLLECTION_NAME}\n{CHUNK_SIZE}\n{CHUNK_OVERLAP}"
    key += f"\nparent_child={PARENT_CHILD_ENABLED}\nparent_size={PARENT_SIZE}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _parse(files: list[Path]) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str]], dict]:
    """返回 (rows, skipped, parents)。parents 仅在父子分块模式下非空。

    skipped 包含两类文件：解析失败（缺后端/文件损坏）与解析成功但无文本
    （扫描版 PDF 未开 OCR、空白文档），避免它们静默丢失。
    """
    rows: list[tuple[str, str, dict]] = []
    skipped: list[tuple[str, str]] = []
    parents: dict[str, str] = {}
    for f, rec_or_err in extract_files(files):
        if isinstance(rec_or_err, Exception):  # 单个文件解析失败不影响整体（如 .ppt）
            logging.warning("跳过 %s：%s", f.name, rec_or_err)
            skipped.append((f.name, str(rec_or_err)))
            continue
        rec = rec_or_err
        if not rec["text"].strip():
            logging.warning("跳过 %s：解析结果为空（扫描版 PDF 可开 OCR_ENABLED，空白文档请检查）", f.name)
            skipped.append((f.name, "解析结果为空"))
            continue
        rel = str(f.relative_to(DATA_DIR))
        uid = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
        category = file_category(f, DATA_DIR)
        base = {
            "source": rel,
            "filename": f.name,
            "category": category,
            "ext": rec["ext"],
        }

        if PARENT_CHILD_ENABLED:
            children, file_parents = chunk_parent_child(
                rec["text"], uid, CHUNK_SIZE, CHUNK_OVERLAP, PARENT_SIZE
            )
            for p in file_parents:
                parents[p["parent_id"]] = p["text"]
            for i, ch in enumerate(children):
                meta = dict(base)
                meta.update(
                    {
                        "section": ch["section"] or "",
                        "parent_id": ch["parent_id"],
                        "parent_offset": ch["parent_offset"],
                        "parent_length": ch["parent_length"],
                    }
                )
                rows.append((f"{uid}::{f.stem}::chunk{i}", ch["text"], meta))
            n = len(children)
        else:
            chunks = chunk_text(rec["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, (text, section) in enumerate(chunks):
                meta = dict(base)
                meta["section"] = section or ""
                rows.append((f"{uid}::{f.stem}::chunk{i}", text, meta))
            n = len(chunks)
        logging.info("解析 %s -> %d chunks（%s）", f.name, n, category)
    return rows, skipped, parents


def _report_parsers() -> None:
    """打印各格式解析后端可用性，把「静默跳过」变成显式告警。"""
    status = parser_status()
    parts = [f"{ext}={state}" for ext, state in status.items()]
    logging.info("解析后端：%s", "  ".join(parts))
    degraded = [ext for ext, state in status.items() if state != "available"]
    if degraded:
        logging.warning(
            "以下格式当前不可用（相关文件将被跳过）：%s；"
            "安装 LibreOffice 可支持 .doc/.ppt，pip install python-docx python-pptx 支持 .docx/.pptx",
            " ".join(degraded),
        )
    ocr = ocr_status()
    logging.info("PDF OCR 兜底：%s（OCR_ENABLED=true + pip install rapidocr-onnxruntime 可开启）", ocr)


def _save_cache(sig: str, rows, embeddings: np.ndarray, skipped: list[tuple[str, str]]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(CACHE_DIR / f"{sig}.emb.npy", embeddings)
    _cache_rows_path(sig).write_text(
        json.dumps({"rows": rows, "skipped": skipped}, ensure_ascii=False), encoding="utf-8"
    )


def _cache_rows_path(sig: str) -> Path:
    return CACHE_DIR / f"{sig}.rows.json"


def _load_cache(sig: str):
    """返回 (rows, embeddings, skipped)；缓存缺失或格式过旧时返回 None。"""
    emb_path = CACHE_DIR / f"{sig}.emb.npy"
    rows_path = _cache_rows_path(sig)
    if not (emb_path.exists() and rows_path.exists()):
        return None
    embeddings = np.load(emb_path)
    try:
        payload = json.loads(rows_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "rows" not in payload:
        return None  # 旧格式缓存（仅 rows 列表）：重新解析一次以拿到跳过清单
    return (
        [tuple(r) for r in payload["rows"]],
        embeddings,
        [tuple(s) for s in payload.get("skipped", [])],
    )


def build() -> None:
    _report_parsers()
    files = _collect_files()
    _check_broken_names(files)
    sig = _signature(files)
    skipped: list[tuple[str, str]] = []
    parents: dict[str, str] = {}

    cached = _load_cache(sig)
    if cached:
        rows, embeddings, skipped = cached
        print(f"命中向量缓存，跳过向量化（{len(rows)} chunks）")
    else:
        rows, skipped, parents = _parse(files)
        if not rows:
            print("未解析到任何内容，请检查 nlp_ 目录。")
            return
        print(f"加载 embedding 模型 {EMBED_MODEL} …")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        docs = [r[1] for r in rows]
        print(f"向量化 {len(rows)} 个 chunk …")
        embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
        _save_cache(sig, rows, embeddings, skipped)

    if skipped:
        print(f"\n共跳过 {len(skipped)} 个文件（解析失败或文本为空）：")
        for name, reason in skipped:
            print(f"  - {name}：{reason}")

    # 父子分块：父块文本写入 sidecar，供查询时按需展开上下文
    if PARENT_CHILD_ENABLED:
        if not parents:  # 命中向量缓存时 parents 为空，从缓存目录回读（可能缺失）
            parents = _load_parents_sidecar()
        _save_parents_sidecar(parents)
        print(f"父子分块：{len(parents)} 个父块已写入 {parents_path(COLLECTION_NAME).name}")

    print("写入 Chroma …")
    import chromadb

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


def _save_parents_sidecar(parents: dict[str, str]) -> None:
    path = parents_path(COLLECTION_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(parents, fh, ensure_ascii=False)


def _load_parents_sidecar() -> dict[str, str]:
    path = parents_path(COLLECTION_NAME)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
