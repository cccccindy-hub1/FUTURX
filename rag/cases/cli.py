"""判例库命令行：判例按「卡片 + 正文块」切块（chunk）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.cases.cli chunk --doc /path/数据法典型判例库（十章分类完整版）.docx [--out DIR]

输出到 CASE_CACHE_DIR（默认 rag/case_cache）：cases.jsonl（每案一行）、chunks.jsonl、stats.json。
并打印统计：判例总数（应为 47）、各章数量、没有要点的案子及其卡片取自哪里。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .chunker import chunk_cases
from .config import CASE_CACHE_DIR, CASE_DOC
from .parser import ParsedCase, parse_cases

EXPECTED_CASES = 47
EXPECTED_PER_CHAPTER = {1: 2, 2: 7, 3: 1, 4: 2, 5: 3, 6: 11, 7: 5, 8: 7, 9: 4, 10: 5}


def case_row(c: ParsedCase, chunks) -> dict:
    cards = [k for k in chunks if k.seq == c.seq and k.kind == "card"]
    return {"case_id": f"case:{c.seq:02d}", "seq": c.seq, "chapter": c.chapter, "chapter_no": c.chapter_no, "title": c.title,
            "meta": c.meta, "chars": c.body_chars, "n_chunks": sum(1 for k in chunks if k.seq == c.seq),
            "card_source": cards[0].card_source if cards else ""}


def case_stats(cases: list[ParsedCase], chunks) -> dict:
    per_chapter = Counter(c.chapter for c in cases)
    per_no = Counter(c.chapter_no for c in cases)
    cards = {k.seq: k for k in chunks if k.kind == "card"}
    by_src = Counter(k.card_source for k in cards.values())
    no_points = [{"seq": c.seq, "chapter": c.chapter, "title": c.title[:40], "card_source": cards[c.seq].card_source,
                  "card_chars": cards[c.seq].body_chars} for c in cases if cards[c.seq].card_source != "points"]
    bodies = [k.body_chars for k in chunks if k.kind == "body"]
    return {
        "cases": len(cases), "expected_cases": EXPECTED_CASES,
        "per_chapter": dict(sorted(per_chapter.items())),
        "per_chapter_matches_design": dict(per_no) == EXPECTED_PER_CHAPTER,
        "chunks": len(chunks), "cards": len(cards), "bodies": len(bodies),
        "card_source": dict(by_src),
        "without_points": no_points,
        "body_chars": {"min": min(bodies), "median": sorted(bodies)[len(bodies) // 2], "max": max(bodies),
                       "under_800": sum(b < 800 for b in bodies), "over_1200": sum(b > 1200 for b in bodies)},
        "tables_in_cases": sum(1 for c in cases for b in c.blocks if b.kind == "table"),
        "appendix_excluded": True,
    }


def export_cases(doc: Path, out_dir: Path) -> dict:
    cases = parse_cases(doc)
    chunks = chunk_cases(cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cases.jsonl", "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(case_row(c, chunks), ensure_ascii=False) + "\n")
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for k in chunks:
            fh.write(json.dumps(k.to_dict(), ensure_ascii=False) + "\n")
    stats = case_stats(cases, chunks)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.cases", description="判例库（数据法典型判例）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chunk", help="判例切成「卡片 + 正文块」，输出 JSONL 与统计（不写向量库）")
    ch.add_argument("--doc", default=None, help="判例库 docx 路径，默认取 CASE_DOC")
    ch.add_argument("--out", default=None, help="输出目录，默认 CASE_CACHE_DIR")
    args = parser.parse_args()
    if args.cmd == "chunk":
        doc = Path(args.doc) if args.doc else CASE_DOC
        if not doc.is_file():
            raise SystemExit(f"判例库文件不存在：{doc}（用 --doc 或 CASE_DOC 指定）")
        out = Path(args.out) if args.out else CASE_CACHE_DIR
        stats = export_cases(doc, out)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if stats["cases"] != EXPECTED_CASES:
            print(f"警告：判例数 {stats['cases']} 与设计文档的 {EXPECTED_CASES} 不一致")
        print(f"已写出：{out}")


if __name__ == "__main__":
    main()
