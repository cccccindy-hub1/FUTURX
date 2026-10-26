"""课程知识库命令行（M1：教材解析与分块，不建向量库）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.course.cli chunk --textbook /path/书稿.docx [--out DIR]

输出到 COURSE_CACHE_DIR/<版本>/：units.jsonl、parents.jsonl、chunks.jsonl、stats.json。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .chunker import ChunkSet, chunk_textbook
from .config import BOOK_VERSION, COURSE_CACHE_DIR, COURSE_TEXTBOOK, PARA_SPLIT
from .textbook import Textbook, parse_docx


def _pct(xs: list[int], q: float) -> int:
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def stats(book: Textbook, cs: ChunkSet) -> dict:
    lens = sorted(len(c.text) for c in cs.chunks)
    plens = sorted(len(p.text) for p in cs.parents)
    # 完整性：未超过 PARA_SPLIT 的段落都应完整落在某一个子块里
    by_anchor = {u.anchor: u for u in book.units}
    split_paras = {
        a for c in cs.chunks if c.split_part is not None and c.chunk_type != "table"
        for a in c.anchors
    }
    broken = [a for a in split_paras if len(by_anchor[a].text) <= PARA_SPLIT]
    return {
        "book_version": cs.book_version,
        "source_sha256": cs.source_sha256,
        "n_paragraphs": book.n_paragraphs,
        "units_by_zone": dict(Counter(u.zone for u in book.units)),
        "units_by_kind": dict(Counter(u.kind for u in book.units)),
        "chapters": sorted({u.chapter for u in book.units if u.chapter is not None}),
        "parents": len(cs.parents),
        "parent_len": {"median": _pct(plens, 0.5), "p90": _pct(plens, 0.9), "max": plens[-1] if plens else 0},
        "chunks": len(cs.chunks),
        "chunks_by_type": dict(Counter(c.chunk_type for c in cs.chunks)),
        "chunk_len": {
            "min": lens[0] if lens else 0, "median": _pct(lens, 0.5),
            "p90": _pct(lens, 0.9), "max": lens[-1] if lens else 0,
        },
        "paragraphs_split": len(split_paras),
        "short_paragraphs_split": len(broken),
        "duplicate_chunk_ids": len(cs.chunks) - len({c.chunk_id for c in cs.chunks}),
    }


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def export(textbook: Path, out_dir: Path, version: str = BOOK_VERSION) -> dict:
    book = parse_docx(textbook)
    cs = chunk_textbook(book, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "units.jsonl", (asdict(u) for u in book.units))
    _write_jsonl(out_dir / "parents.jsonl", (asdict(p) for p in cs.parents))
    _write_jsonl(out_dir / "chunks.jsonl", (c.to_dict() for c in cs.chunks))
    summary = stats(book, cs)
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.course", description="《数据要素》课程知识库")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chunk", help="解析教材并分块，输出 JSONL 与统计（不写向量库）")
    ch.add_argument("--textbook", default=None, help="教材 docx 路径，默认取 COURSE_TEXTBOOK")
    ch.add_argument("--out", default=None, help="输出目录，默认 COURSE_CACHE_DIR/<版本>")
    ch.add_argument("--version", default=BOOK_VERSION, help="教材版本号（写入 chunk_id）")
    args = parser.parse_args()

    if args.cmd == "chunk":
        textbook = Path(args.textbook) if args.textbook else COURSE_TEXTBOOK
        if not textbook.is_file():
            raise SystemExit(f"教材文件不存在：{textbook}（用 --textbook 或 COURSE_TEXTBOOK 指定）")
        out = Path(args.out) if args.out else COURSE_CACHE_DIR / args.version
        summary = export(textbook, out, args.version)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"已写出：{out}")


if __name__ == "__main__":
    main()
