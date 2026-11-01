"""M2 命令行：加载本体数据，输出数据质量报告与 Top3 候选方案示例。

    python -m rag.course.report --root /path/to/Ontology_KG [--textbook x.docx] [--out dir]

只读数据根目录；报告写到 COURSE_CACHE_DIR/m2（可用 --out 改）。不调用任何 LLM。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import config
from .aliases import build_alias_table
from .ontology import OntologyPaths, load_concepts, load_records, load_student_set
from .quality import build_report, collect_candidates
from .relevance import SCHEMES, compare_schemes, rank
from .evidence import AnchorIndex
from .textbook import parse_docx


def _label(by_id: dict, cid: str) -> str:
    c = by_id.get(cid)
    return f"{cid}《{c.name}》" if c else cid


def top3_examples(candidates, concepts, anchors, n: int = 6, seed: int = 7) -> str:
    by_id = concepts.by_id
    pool = sorted(cid for cid, cs in candidates.items() if len(cs) > 3 and cid in by_id)
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    lines = ["# Top3 候选方案示例（onto2，随机固定种子抽样，仅含候选数>3的概念）", ""]
    for cid in sorted(pick):
        cs = candidates[cid]
        lines.append(f"## {_label(by_id, cid)}  —— 候选 {len(cs)} 个")
        for name, fn in SCHEMES.items():
            lines.append(f"- 方案{name}：")
            for s in rank(cs, fn, anchors):
                preds = "/".join(sorted({e.predicate_id for e in s.candidate.edges}))
                lines.append(f"  - {_label(by_id, s.candidate.object_id)} [{preds}] 分={s.score} {s.detail}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=config.COURSE_ONTO_ROOT)
    ap.add_argument("--textbook", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=config.COURSE_CACHE_DIR / "m2")
    args = ap.parse_args(argv)

    paths = OntologyPaths(args.root)
    concepts = load_concepts(paths.concept_xlsx)
    records = [r for v in ("onto1", "onto2") for r in load_records(paths.onto_dir(v), v)]
    student = load_student_set(paths.student_xlsx) if paths.student_xlsx.exists() else None
    tb = args.textbook or paths.find_textbook() or (config.COURSE_TEXTBOOK if config.COURSE_TEXTBOOK.exists() else None)
    book = parse_docx(tb) if tb else None
    aliases = build_alias_table(concepts.concepts)

    rep = build_report(concepts, records, aliases, book=book, student_rows=student)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quality_report.json").write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "quality_report.md").write_text(rep.to_markdown(), encoding="utf-8")

    cands = collect_candidates(records, view="onto2", known=set(concepts.by_id))
    anchors = AnchorIndex(book) if book else None
    (args.out / "top3_examples.md").write_text(top3_examples(cands, concepts, anchors), encoding="utf-8")
    cmp = compare_schemes(cands, anchors)
    cmp.pop("tops")
    (args.out / "top3_scheme_compare.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(rep.to_markdown())
    print(f"\n输出目录：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
