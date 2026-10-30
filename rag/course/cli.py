"""课程知识库命令行：教材解析与分块（chunk）、政策库切块（policy-chunk）、检索问答（ask）、HTTP 服务（serve）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.course.cli chunk --textbook /path/书稿.docx [--out DIR]
  python -m rag.course.cli policy-chunk --doc /path/数据政策汇编.docx [--out DIR]
  python -m rag.course.cli ask "什么是数据要素？" [--json]
  python -m rag.course.cli serve [--host 127.0.0.1] [--port 8100]   # HTTP 服务，见 rag.course.api

输出到 COURSE_CACHE_DIR/<版本>/：units.jsonl、parents.jsonl、chunks.jsonl、stats.json。
政策库输出到 COURSE_CACHE_DIR/policy/：docs.jsonl、chunks.jsonl、stats.json、review.xlsx（人工审阅用）。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .chunker import ChunkSet, chunk_textbook
from .config import BOOK_VERSION, COURSE_CACHE_DIR, COURSE_POLICY_DOC, COURSE_TEXTBOOK, PARA_SPLIT
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

    from .policy import doc_row
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
    from .policy import build_policy_set, doc_row, policy_stats

    ps = build_policy_set(doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "docs.jsonl", (doc_row(d) for d in ps.docs))
    _write_jsonl(out_dir / "chunks.jsonl", (c.to_dict() for c in ps.chunks))
    summary = policy_stats(ps)
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_policy_review(out_dir / "review.xlsx", ps, summary)
    return summary


def _run_ask(args) -> None:
    from .llm import LLMNotConfigured, OpenAICompatClient
    from .pipeline import CourseContext, ask, result_to_json

    ctx = CourseContext.load(Path(args.root) if args.root else None, Path(args.textbook) if args.textbook else None)
    select = []
    for s in args.select:
        cid, _, role = s.partition(":")
        select.append((cid, role or "question_concept"))
    llm = None
    if not args.dry_run:
        try:
            llm = OpenAICompatClient()
        except LLMNotConfigured as e:
            raise SystemExit(str(e))
    result = result_to_json(ask(args.question, ctx=ctx, llm=llm, active_concepts=args.active,
                                strategy=args.strategy, dry_run=args.dry_run, select=select))
    if args.json or args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if result["fallback"]:
        print(f"[insufficient_evidence] {result['fallback']['reason']}：{result['fallback']['detail']}")
        return
    print(result["answer"]["text"])
    print("\n引用：" + "；".join(f"{c['chunk_id']}（{' › '.join(c['path'])}）" for c in result["citations"]))
    print(f"调用统计：{json.dumps(result['trace']['totals'], ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.course", description="《数据要素》课程知识库")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("chunk", help="解析教材并分块，输出 JSONL 与统计（不写向量库）")
    ch.add_argument("--textbook", default=None, help="教材 docx 路径，默认取 COURSE_TEXTBOOK")
    ch.add_argument("--out", default=None, help="输出目录，默认 COURSE_CACHE_DIR/<版本>")
    ch.add_argument("--version", default=BOOK_VERSION, help="教材版本号（写入 chunk_id）")

    pc = sub.add_parser("policy-chunk", help="数据政策汇编按条/事项切块，输出 JSONL、统计与审阅表（不写向量库）")
    pc.add_argument("--doc", default=None, help="汇编 docx 路径，默认取 COURSE_POLICY_DOC")
    pc.add_argument("--out", default=None, help="输出目录，默认 COURSE_CACHE_DIR/policy")

    ask = sub.add_parser("ask", help="Agentic 检索问答（选词→召回→筛选→作答）")
    ask.add_argument("question", help="学生问题")
    ask.add_argument("--root", default=None, help="本体数据根目录（只读），默认 COURSE_ONTO_ROOT")
    ask.add_argument("--textbook", default=None, help="教材 docx，默认在数据根目录/教材 下查找")
    ask.add_argument("--active", nargs="*", default=[], help="对话中已激活的概念ID")
    ask.add_argument("--strategy", choices=["A", "B"], default="B",
                     help="mention 召回策略：B（默认）=仅 question_concept 做字面召回，related_concept 只召回边证据；"
                          "A=所有选中概念都做字面召回")
    ask.add_argument("--dry-run", action="store_true", help="不调用任何 LLM：只跑本地召回估算（需 --select）")
    ask.add_argument("--select", nargs="*", default=[],
                     help="dry-run 手动指定概念，形如 C01-0001 或 C01-0001:related_concept")
    ask.add_argument("--json", action="store_true", help="输出完整 JSON")

    serve = sub.add_parser("serve", help="HTTP 服务：教材与本体加载一次，POST /course/ask 提问")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    serve.add_argument("--port", type=int, default=8100, help="端口（默认 8100，与 rag.serve 的 8000 分开）")
    serve.add_argument("--root", default=None, help="本体数据根目录（只读），默认 COURSE_ONTO_ROOT")
    serve.add_argument("--textbook", default=None, help="教材 docx，默认在数据根目录/教材 下查找")
    args = parser.parse_args()

    if args.cmd == "ask":
        _run_ask(args)
        return
    if args.cmd == "serve":
        from .api import run_http
        run_http(args.host, args.port, root=Path(args.root) if args.root else None,
                 textbook=Path(args.textbook) if args.textbook else None)
        return
    if args.cmd == "policy-chunk":
        doc = Path(args.doc) if args.doc else COURSE_POLICY_DOC
        if not doc.is_file():
            raise SystemExit(f"政策汇编文件不存在：{doc}（用 --doc 或 COURSE_POLICY_DOC 指定）")
        out = Path(args.out) if args.out else COURSE_CACHE_DIR / "policy"
        summary = export_policy(doc, out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"已写出：{out}")
        return
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
