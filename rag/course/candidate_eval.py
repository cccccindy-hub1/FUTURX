"""离线评估（不调用模型）：每个关键句所在教材块「对应的概念」是否进入 single 模式的候选清单。

「对应概念」= 本体里把该教材块当作证据的概念：用 Recaller 的边证据反查（哪些概念的召回结果包含该块）。
只统计 edge_evidence 来源（不含 mention，mention 太宽泛），并只看可选概念。
用法：python -m rag.course.candidate_eval [--limit 30] [--out rag/eval_data/course_candidate_eval_result.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .pipeline import CourseContext
from .recall import EDGE
from .recall_eval import GOLD, key_chunks
from .single_select import build_candidates


def concepts_for_chunks(ctx: CourseContext, chunk_ids: set[str]) -> dict[str, list[str]]:
    """chunk_id -> 以该块为边证据的概念 ID 列表。"""
    out: dict[str, set[str]] = {c: set() for c in chunk_ids}
    for cid in ctx.index.selectable:
        for it in ctx.recaller.recall([(cid, "related_concept")]).items:       # related：只走边证据，不做 mention
            if it.chunk_id in out and any(r.via == EDGE for r in it.reasons):
                out[it.chunk_id].add(cid)
    return {k: sorted(v) for k, v in out.items()}


def evaluate(ctx: CourseContext, gold: list[dict], limit: int) -> dict:
    all_chunks = {c for g in gold for m in g["must_recall"] for c in key_chunks(ctx, m["key"])}
    owners = concepts_for_chunks(ctx, all_chunks)
    rows, tot_k, tot_hit_any, tot_hit_all, tot_c, tot_c_hit = [], 0, 0, 0, 0, 0
    for g in gold:
        cands = build_candidates(g["question"], ctx.index, limit)
        cand_ids = {c.concept_id for c in cands}
        keys = []
        for m in g["must_recall"]:
            chunks = key_chunks(ctx, m["key"])
            cs = sorted({c for ch in chunks for c in owners.get(ch, [])})
            hit = sorted(set(cs) & cand_ids)
            keys.append({"key": m["key"], "chunks": chunks, "owner_concepts": cs, "in_candidates": hit,
                         "missing": [c for c in cs if c not in cand_ids],
                         "key_covered": bool(hit) if cs else None, "concept_coverage": round(len(hit) / len(cs), 3) if cs else None})
            tot_k += 1 if cs else 0
            tot_hit_any += int(bool(hit)) if cs else 0
            tot_hit_all += int(len(hit) == len(cs)) if cs else 0
            tot_c += len(cs)
            tot_c_hit += len(hit)
        rows.append({"id": g["id"], "question": g["question"], "n_candidates": len(cands),
                     "candidates": [c.to_dict() for c in cands], "keys": keys})
    rate = lambda a, b: round(a / b, 3) if b else None
    return {"limit": limit, "rows": rows,
            "summary": {"keys_with_owner_concepts": tot_k,
                        "key_covered_by_any_owner_concept": f"{tot_hit_any}/{tot_k}", "rate_any": rate(tot_hit_any, tot_k),
                        "key_covered_by_all_owner_concepts": f"{tot_hit_all}/{tot_k}", "rate_all": rate(tot_hit_all, tot_k),
                        "owner_concept_coverage": f"{tot_c_hit}/{tot_c}", "rate_concepts": rate(tot_c_hit, tot_c)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=config.COURSE_SELECT_CANDIDATES)
    ap.add_argument("--out", default=str(GOLD.with_name("course_candidate_eval_result.json")))
    a = ap.parse_args()
    res = evaluate(CourseContext.load(), json.load(open(GOLD, encoding="utf-8")), a.limit)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
