"""名言库命令行：合集按「一次讲话一块」切块（chunk）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.quotes.cli chunk --doc /path/习近平总书记关于数智的有关表述合集.docx [--out DIR]

输出到 QUOTE_CACHE_DIR（默认 rag/quote_cache）：chunks.jsonl、stats.json。
并打印统计：讲话条数、每条字数、去掉的重复段落。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .chunker import QuoteChunk, chunk_quotes
from .config import QUOTE_CACHE_DIR, QUOTE_DOC
from .parser import Speech, parse_quotes

EXPECTED_SPEECHES = 41      # 设计文档：约 41 次讲话 / 贺信 / 指示 / 文章


def quote_stats(speeches: list[Speech], chunks: list[QuoteChunk]) -> dict:
    sizes = [s.chars for s in speeches]
    return {
        "speeches": len(speeches), "expected_speeches": EXPECTED_SPEECHES,
        "years": sorted({s.year for s in speeches}),
        "per_year": dict(sorted(Counter(s.year for s in speeches).items())),
        "per_kind": dict(Counter(s.kind for s in speeches)),
        "total_chars": sum(sizes),
        "chars": {"min": min(sizes), "median": sorted(sizes)[len(sizes) // 2], "max": max(sizes)},
        "per_speech": [{"speech_id": s.speech_id, "kind": s.kind, "title": s.title, "paragraphs": len(s.paragraphs),
                        "chars": s.chars} for s in speeches],
        "removed_duplicates": [{"speech_id": s.speech_id, "title": s.title, "count": len(s.removed_duplicates),
                                "paragraph_starts": s.removed_duplicates} for s in speeches if s.removed_duplicates],
        "removed_duplicate_total": sum(len(s.removed_duplicates) for s in speeches),
        "same_title_in_one_year": [{"year": y, "title": t, "speech_ids": ids} for (y, t), ids in _same_titles(speeches)],
        "chunks": len(chunks),
    }


def _same_titles(speeches: list[Speech]):
    by: dict[tuple[int, str], list[str]] = {}
    for s in speeches:
        by.setdefault((s.year, s.title), []).append(s.speech_id)
    return [(k, v) for k, v in by.items() if len(v) > 1]


def export_quotes(doc: Path, out_dir: Path) -> dict:
    speeches = parse_quotes(doc)
    chunks = chunk_quotes(speeches)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for k in chunks:
            fh.write(json.dumps(k.to_dict(), ensure_ascii=False) + "\n")
    stats = quote_stats(speeches, chunks)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.quotes", description="名言库（习近平总书记关于数智的有关表述）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chunk", help="按「一次讲话一块」切块，输出 JSONL 与统计（不写向量库）")
    ch.add_argument("--doc", default=None, help="合集 docx 路径，默认取 QUOTE_DOC")
    ch.add_argument("--out", default=None, help="输出目录，默认 QUOTE_CACHE_DIR")
    args = parser.parse_args()
    if args.cmd == "chunk":
        doc = Path(args.doc) if args.doc else QUOTE_DOC
        if not doc.is_file():
            raise SystemExit(f"名言库文件不存在：{doc}（用 --doc 或 QUOTE_DOC 指定）")
        out = Path(args.out) if args.out else QUOTE_CACHE_DIR
        stats = export_quotes(doc, out)
        print(json.dumps({k: v for k, v in stats.items() if k != "per_speech"}, ensure_ascii=False, indent=2))
        print(f"\n每条字数（{len(stats['per_speech'])} 条）：")
        for r in stats["per_speech"]:
            print(f"  {r['speech_id']} {r['kind']} {r['paragraphs']:2d}段 {r['chars']:5d}字  {r['title'][:36]}")
        print(f"已写出：{out}")


if __name__ == "__main__":
    main()
