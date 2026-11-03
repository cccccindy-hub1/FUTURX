"""试卷库命令行。

用法（在项目根目录 FUTURX 下）：
  python -m rag.exam.cli build [--teacher 教师版.docx] [--student 学生版.docx] [--out DIR]
  python -m rag.exam.cli search "出3道数据资产的基础题" [--answer]

build 输出到 EXAM_CACHE_DIR（默认 rag/exam_cache，不进仓库）：
  questions.jsonl（每题一行，含挂接的知识点）、stats.json、review.xlsx（人工审阅：题目 / 知识点挂接）。
服务启动时直接解析 docx（约 2 秒），不依赖这里的导出；导出只供审阅与核对。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import config
from .store import ExamBank, format_items


def bank_stats(bank: ExamBank) -> dict:
    by_ch: dict[str, Counter] = {}
    for it in bank.items:
        c = by_ch.setdefault(f"ch{it.q.chapter:02d} {it.q.chapter_title}", Counter())
        c["questions"] += 1
        c["linked"] += bool(it.kp_ids)
        c[it.link_method.split("+")[0].split(":")[0]] += 1
    return {**bank.stats, "by_chapter": {k: dict(v) for k, v in by_ch.items()},
            "kp_covered": len({k for it in bank.items for k in it.kp_ids}),
            "concepts_covered": len(bank.by_concept)}


def export_bank(bank: ExamBank, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fh:
        for it in bank.items:
            row = it.q.to_dict()
            row.update(student_concepts=it.concepts, kp_ids=it.kp_ids, link_method=it.link_method)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = bank_stats(bank)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_review(out / "review.xlsx", bank)
    return stats


def _write_review(path: Path, bank: ExamBank) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "题目"
    ws.append(["question_id", "章", "章名", "题型", "题号", "基础", "题干", "答案", "题干概念", "挂接知识点（主体）",
               "挂接方式", "知识点数"])
    for it in bank.items:
        q = it.q
        ws.append([q.question_id, q.chapter, q.chapter_title, q.qtype_cn, q.number, "是" if q.basic else "",
                   q.stem[:500], q.answer[:200], "；".join(q.concepts), "；".join(it.concepts), it.link_method,
                   len(it.kp_ids)])
    ws2 = wb.create_sheet("知识点挂接")
    ws2.append(["question_id", "kp_id", "章", "主体", "关系", "客体"])
    for it in bank.items:
        for k in it.kp_ids:
            v = bank.kp_view(k)
            ws2.append([it.q.question_id, k, v.get("chapter"), v.get("subject"), v.get("relation"), v.get("object")])
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
    wb.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rag.exam.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="解析题库、挂接知识点并导出审阅文件")
    b.add_argument("--teacher", type=Path, default=config.EXAM_TEACHER_DOC)
    b.add_argument("--student", type=Path, default=config.EXAM_STUDENT_DOC)
    b.add_argument("--student-set", type=Path, default=config.EXAM_STUDENT_SET)
    b.add_argument("--out", type=Path, default=config.EXAM_CACHE_DIR)
    s = sub.add_parser("search", help="自然语言找题（不调用 LLM）")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=None)
    s.add_argument("--answer", action="store_true", help="附参考答案与解析")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        bank = ExamBank.build(args.teacher, args.student_set, args.student)
        stats = export_bank(bank, args.out)
        print(json.dumps({k: stats[k] for k in ("questions", "declared_total", "linked", "kp_covered",
                                                "count_mismatch", "flags", "student_check") if k in stats},
                         ensure_ascii=False, indent=2))
        print(f"已导出到 {args.out}")
    else:
        bank = ExamBank.build()
        r = bank.search(args.query, args.k)
        print(f"[{r['status']} / {r['strategy']}] {json.dumps(r['intent'], ensure_ascii=False)}")
        print(format_items(r["items"], include_answer=args.answer or r["intent"]["want_answer"]))


if __name__ == "__main__":
    main()
