"""离线召回评测（不调用模型）：每题标 1–3 个必须召回的教材块（用关键句判断命中），比较
  A. 仅本体召回（选中概念来自评测集 ontology_selection_v1，缺失的题记为「无记录」不计入分母）
  B. 本体 + BM25 兜底
  C. 仅 BM25
的命中率。关键句必须在教材里只落在一个子块（否则按「任一含关键句的块被召回」计，并在结果里标注）。

用法：python -m rag.course.recall_eval [--topk 8] [--out rag/eval_data/course_recall_eval_result.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .pipeline import CourseContext
from .recall import DEFAULT_STRATEGY

GOLD = Path(__file__).resolve().parents[1] / "eval_data" / "course_recall_gold.json"


def key_chunks(ctx: CourseContext, key: str) -> list[str]:
    return [c.chunk_id for c in ctx.chunkset.chunks if key in c.text]


def evaluate(ctx: CourseContext, gold: list[dict], topk: int) -> dict:
    rows, tot = [], {"ontology": [0, 0], "ontology+bm25": [0, 0], "bm25": [0, 0]}
    for g in gold:
        q = g["question"]
        targets = [{"key": m["key"], "chunks": key_chunks(ctx, m["key"])} for m in g["must_recall"]]
        bad = [t["key"] for t in targets if not t["chunks"]]
        if bad:
            raise ValueError(f"{g['id']} 关键句在教材中找不到：{bad}")
        bm = ctx.bm25.search(q, topk)
        bm_ids = {c.chunk_id for c, _ in bm}
        modes: dict[str, set[str] | None] = {"bm25": bm_ids}
        sel = g.get("ontology_selection_v1")
        if sel:
            rr = ctx.recaller.recall([(s["concept_id"], s["role"]) for s in sel], DEFAULT_STRATEGY)
            onto_ids = {i.chunk_id for i in rr.items}
            modes["ontology"] = onto_ids
            modes["ontology+bm25"] = onto_ids | bm_ids
            modes["_onto_count"] = len(onto_ids)
        else:
            modes["ontology"] = modes["ontology+bm25"] = None
        row = {"id": g["id"], "question": q, "n_targets": len(targets), "ontology_recalled": modes.get("_onto_count"),
               "bm25_topk": [{"chunk_id": c.chunk_id, "path": " › ".join(c.path[-2:]), "score": round(s, 2)} for c, s in bm],
               "targets": []}
        for t in targets:
            hit = {m: (None if ids is None else bool(set(t["chunks"]) & ids)) for m, ids in modes.items() if m != "_onto_count"}
            row["targets"].append({"key": t["key"], "n_chunks_with_key": len(t["chunks"]), "hit": hit})
            for m, h in hit.items():
                if h is not None:
                    tot[m][1] += 1
                    tot[m][0] += int(h)
        rows.append(row)
    # 同一批「有本体记录」的题上，三种方式的命中率才可比
    with_sel = [r for r in rows if r["ontology_recalled"] is not None]
    def rate(sub, m):
        hs = [t["hit"][m] for r in sub for t in r["targets"]]
        return {"hit": sum(hs), "total": len(hs), "rate": round(sum(hs) / len(hs), 3) if hs else None}
    return {"topk": topk, "rows": rows,
            "comparable_questions": [r["id"] for r in with_sel],
            "summary_comparable": {m: rate(with_sel, m) for m in ("ontology", "ontology+bm25", "bm25")},
            "summary_all_bm25_only": rate(rows, "bm25"), "questions_without_ontology_record": [r["id"] for r in rows if r["ontology_recalled"] is None]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=config.COURSE_BM25_TOPK)
    ap.add_argument("--out", default=str(GOLD.with_name("course_recall_eval_result.json")))
    a = ap.parse_args()
    res = evaluate(CourseContext.load(), json.load(open(GOLD, encoding="utf-8")), a.topk)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("topk", "comparable_questions", "summary_comparable", "summary_all_bm25_only",
                                           "questions_without_ontology_record")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
