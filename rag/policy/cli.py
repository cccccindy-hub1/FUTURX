"""政策库命令行：数据政策汇编按条/事项切块（chunk）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.policy.cli chunk --doc /path/数据政策汇编.docx [--out DIR]

输出到 POLICY_CACHE_DIR（默认 rag/policy_cache）：docs.jsonl、chunks.jsonl、stats.json、review.xlsx（人工审阅用）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import build_policy_set, doc_row, policy_stats
from .config import POLICY_CACHE_DIR, POLICY_DOC


def _write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


_REVIEW_DOC_COLS = ["doc_no", "policy", "level", "doc_type", "status", "doc_number", "date", "chars",
                    "n_chunks", "n_indexed", "excluded", "duplicate_of", "superseded_by"]
_REVIEW_CHUNK_COLS = ["chunk_id", "doc_no", "policy", "doc_type", "kind", "path", "title", "part", "chars",
                      "indexed", "flags", "text"]


def _write_policy_review(path: Path, ps, summary: dict) -> None:
    """审阅表：统计 / 政策文件 / 切块 三个工作表。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "统计"
    for k, v in summary.items():
        ws.append([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 120

    ws = wb.create_sheet("政策文件")
    ws.append(_REVIEW_DOC_COLS)
    for d in ps.docs:
        row = doc_row(d)
        ws.append([row[k] for k in _REVIEW_DOC_COLS])
    ws.column_dimensions["B"].width = 50

    ws = wb.create_sheet("切块")
    ws.append(_REVIEW_CHUNK_COLS)
    for c in ps.chunks:
        ws.append([c.chunk_id, c.doc_no, c.policy, c.doc_type, c.kind, " › ".join(c.path), c.title, c.part,
                   len(c.text), c.indexed, ",".join(c.flags), c.text])
    for col, w in zip("ABCDEFGHIJKL", (26, 7, 30, 9, 9, 30, 24, 5, 6, 7, 16, 90)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=12, max_col=12):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    for sheet in wb.worksheets[1:]:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
    wb.save(path)


def export_policy(doc: Path, out_dir: Path) -> dict:
    ps = build_policy_set(doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "docs.jsonl", (doc_row(d) for d in ps.docs))
    _write_jsonl(out_dir / "chunks.jsonl", (c.to_dict() for c in ps.chunks))
    summary = policy_stats(ps)
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_policy_review(out_dir / "review.xlsx", ps, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.policy", description="政策库（数据政策汇编）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chunk", help="汇编按条/事项切块，输出 JSONL、统计与审阅表（不写向量库）")
    ch.add_argument("--doc", default=None, help="汇编 docx 路径，默认取 POLICY_DOC")
    ch.add_argument("--out", default=None, help="输出目录，默认 POLICY_CACHE_DIR")
    args = parser.parse_args()

    if args.cmd == "chunk":
        doc = Path(args.doc) if args.doc else POLICY_DOC
        if not doc.is_file():
            raise SystemExit(f"政策汇编文件不存在：{doc}（用 --doc 或 POLICY_DOC 指定）")
        out = Path(args.out) if args.out else POLICY_CACHE_DIR
        summary = export_policy(doc, out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"已写出：{out}")


if __name__ == "__main__":
    main()
