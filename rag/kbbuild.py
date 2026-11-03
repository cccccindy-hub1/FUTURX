"""判例库 / 名言库建向量库（在服务器上执行；本地只跑 mock 测试，不下载模型）。

安全约束（写进代码，不靠约定）：
- 只写**自己的**目录：rag/case_store、rag/quote_store（或 --store 指定）；目录必须不是论文库的 STORE_DIR，
  也不能在 STORE_DIR 里面或把 STORE_DIR 包在里面——命中任何一种直接拒绝；
- 目标集合已存在且非空时拒绝（除非显式 --rebuild，且只删除**这个**集合）；绝不触碰论文库 Chroma；
- 集合名不能等于论文库的 COLLECTION_NAME。

输入是第一阶段 `chunk` 命令产出的 chunks.jsonl（CASE_CACHE_DIR / QUOTE_CACHE_DIR）：
- 判例：每块元数据 case_id / seq / chapter / chapter_no / title / kind(card|body) / heading / 案号等；
- 名言：每块元数据 speech_id / year / title / kind，以及 paragraphs_json（原文段落，作答环节程序照录用）。
向量化用 BGE-m3（与论文库同一模型 EMBED_MODEL）；不需要 LLM，**调用次数 0**。

用法（项目根目录）：
  python -m rag.kbbuild cases  [--cache DIR] [--store DIR] [--rebuild] [--dry-run]
  python -m rag.kbbuild quotes [--cache DIR] [--store DIR] [--rebuild] [--dry-run]
--dry-run：只读 chunks.jsonl、校验路径与元数据并打印将要写入的块数，不加载模型、不写任何东西。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as base

BATCH = 32


class UnsafeStore(RuntimeError):
    pass


def check_store_safe(store: Path, collection: str, *, thesis_store: Path | None = None, thesis_collection: str | None = None) -> None:
    """拒绝任何可能碰到论文库的目标。"""
    t_store = Path(thesis_store if thesis_store is not None else base.STORE_DIR).resolve()
    t_coll = thesis_collection if thesis_collection is not None else base.COLLECTION_NAME
    s = Path(store).resolve()
    if s == t_store or t_store in s.parents or s in t_store.parents:
        raise UnsafeStore(f"目标目录 {s} 与论文库目录 {t_store} 相同或互相包含，拒绝写入（论文库 Chroma 不能被读写或覆盖）")
    if collection == t_coll:
        raise UnsafeStore(f"集合名 {collection!r} 与论文库集合名相同，拒绝")


def load_chunks(cache_dir: Path) -> list[dict]:
    path = Path(cache_dir) / "chunks.jsonl"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}；先运行 chunk 命令生成")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def case_rows(chunks: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    ids, docs, metas = [], [], []
    for c in chunks:
        m = c.get("meta") or {}
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append({"case_id": c["case_id"], "seq": int(c["seq"]), "chapter": c["chapter"], "chapter_no": int(c["chapter_no"]),
                      "title": c["title"], "kind": c["kind"], "part": int(c.get("part", 0)), "heading": c.get("heading", ""),
                      "case_number": m.get("case_number", ""), "court": m.get("court", ""),
                      "judgment_date": m.get("judgment_date", ""), "source_url": m.get("source_url", "")})
    return ids, docs, metas


def quote_rows(chunks: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    ids, docs, metas = [], [], []
    for c in chunks:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append({"speech_id": c["speech_id"], "year": int(c["year"]), "title": c["title"], "kind": c["kind"],
                      "paragraphs_json": json.dumps(c["paragraphs"], ensure_ascii=False)})
    return ids, docs, metas


def build(kind: str, cache_dir: Path, store: Path, collection: str, *, rebuild: bool = False, dry_run: bool = False,
          embed=None, client_factory=None) -> dict:
    """embed(list[str]) -> list[list[float]]；client_factory(path) -> chromadb 客户端。二者可注入（测试用 mock）。"""
    check_store_safe(store, collection)
    chunks = load_chunks(cache_dir)
    ids, docs, metas = (case_rows if kind == "cases" else quote_rows)(chunks)
    if len(set(ids)) != len(ids):
        raise SystemExit("chunks.jsonl 里有重复的 chunk_id")
    info = {"kind": kind, "store": str(Path(store).resolve()), "collection": collection, "chunks": len(ids), "dry_run": dry_run}
    if dry_run:
        return info
    if client_factory is None:
        import chromadb
        client_factory = lambda p: chromadb.PersistentClient(path=str(p))        # noqa: E731
    store_p = Path(store)
    if store_p.is_dir() and any(store_p.iterdir()) and not rebuild:
        raise SystemExit(f"目录 {store_p} 已存在且非空，拒绝写入；确认要重建请加 --rebuild（只会删除这个库自己的集合，不动目录里的其它东西）")
    store_p.mkdir(parents=True, exist_ok=True)
    client = client_factory(store)
    existing = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if collection in existing:
        col = client.get_collection(collection)
        if col.count() > 0 and not rebuild:
            raise SystemExit(f"集合 {collection!r} 已有 {col.count()} 条，拒绝覆盖；确认要重建请加 --rebuild（只会删除这个集合）")
        client.delete_collection(collection)                                      # 只删自己的集合
    col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
    if embed is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(base.EMBED_MODEL)
        embed = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()      # noqa: E731
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i + BATCH], documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH], embeddings=embed(docs[i:i + BATCH]))
    info["written"] = col.count()
    return info


def chunk_if_needed(kind: str, doc: Path | None, cache: Path) -> Path:
    """cache 里还没有 chunks.jsonl 时，用 doc（默认取 CASE_DOC / QUOTE_DOC）先切块；已有则直接用。返回 cache。"""
    if (cache / "chunks.jsonl").is_file():
        return cache
    if kind == "cases":
        from .cases import config as c
        from .cases.cli import export_cases as export
    else:
        from .quotes import config as c
        from .quotes.cli import export_quotes as export
    src = Path(doc) if doc else (c.CASE_DOC if kind == "cases" else c.QUOTE_DOC)
    if not src.is_file():
        raise SystemExit(f"找不到原始资料 {src}（在 .env 里设 {'CASE_DOC' if kind == 'cases' else 'QUOTE_DOC'}，"
                         f"或放到 rag/{'case' if kind == 'cases' else 'quote'}_data/ 下）")
    export(src, cache)
    return cache


def run_cli(kind: str, argv=None) -> int:
    """python -m rag.cases.build / rag.quotes.build 的共同入口。"""
    ap = argparse.ArgumentParser(prog=f"rag.{'cases' if kind == 'cases' else 'quotes'}.build",
                                 description=f"{'判例库' if kind == 'cases' else '名言库'}：切块（如缺）+ 建向量库；"
                                             "独立目录，不碰论文库 Chroma；调用大模型 0 次，只用本地 BGE-m3")
    ap.add_argument("--doc", default=None, help="原始 docx，默认取 .env 的 CASE_DOC / QUOTE_DOC（rag/case_data/、rag/quote_data/）")
    ap.add_argument("--cache", default=None, help="切块输出目录，默认 CASE_CACHE_DIR / QUOTE_CACHE_DIR")
    ap.add_argument("--store", default=None, help="向量库目录，默认 rag/case_store / rag/quote_store")
    ap.add_argument("--rebuild", action="store_true", help="目录已存在且非空时允许重建（只删除这个库自己的集合）")
    ap.add_argument("--dry-run", action="store_true", help="只切块 + 校验并打印将写入的块数，不加载模型、不写向量库")
    a = ap.parse_args(argv)
    if kind == "cases":
        from .cases import config as c
        cache, store, coll = Path(a.cache or c.CASE_CACHE_DIR), Path(a.store or c.CASE_STORE_DIR), c.CASE_COLLECTION
    else:
        from .quotes import config as c
        cache, store, coll = Path(a.cache or c.QUOTE_CACHE_DIR), Path(a.store or c.QUOTE_STORE_DIR), c.QUOTE_COLLECTION
    try:
        check_store_safe(store, coll)                                    # 先检查目标安全，再做任何事
        chunk_if_needed(kind, Path(a.doc) if a.doc else None, cache)
        info = build(kind, cache, store, coll, rebuild=a.rebuild, dry_run=a.dry_run)
    except UnsafeStore as e:
        raise SystemExit(f"拒绝：{e}")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag.kbbuild", description="判例库 / 名言库建向量库（独立目录，不碰论文库 Chroma）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cases", "quotes"):
        p = sub.add_parser(name)
        p.add_argument("--cache", default=None, help="chunk 命令的输出目录（含 chunks.jsonl）")
        p.add_argument("--store", default=None, help="向量库目录，默认 CASE_STORE_DIR / QUOTE_STORE_DIR")
        p.add_argument("--rebuild", action="store_true", help="集合已存在且非空时允许重建（只删除这个集合）")
        p.add_argument("--dry-run", action="store_true", help="只校验与统计，不加载模型、不写任何东西")
    a = ap.parse_args()
    if a.cmd == "cases":
        from .cases import config as c
        cache, store, coll = Path(a.cache or c.CASE_CACHE_DIR), Path(a.store or c.CASE_STORE_DIR), c.CASE_COLLECTION
    else:
        from .quotes import config as c
        cache, store, coll = Path(a.cache or c.QUOTE_CACHE_DIR), Path(a.store or c.QUOTE_STORE_DIR), c.QUOTE_COLLECTION
    try:
        info = build(a.cmd, cache, store, coll, rebuild=a.rebuild, dry_run=a.dry_run)
    except UnsafeStore as e:
        raise SystemExit(f"拒绝：{e}")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
