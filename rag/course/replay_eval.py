"""离线重放（不调用模型）：用上一次端到端评测记录的「模型选择」，按当前候选/展开/召回/截断逻辑重放。

模型当时选中的是旧候选清单里的代表 ID（记录在 select_stage.expanded 的 candidate 与未展开的 selected_concepts 里）；
这里取每条记录里 role=question/related 的「代表 ID」，在新候选清单里找到同一名称的候选，再按新逻辑展开。
若模型选的概念不在新清单里（例如被邻居配额挤出），记为 missing 并跳过——这本身就是候选变化的影响。
用法：python -m rag.course.replay_eval [--baseline rag/eval_data/course_e2e_eval_result.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .passage_filter import cap_for_filter
from .pipeline import CourseContext
from .recall import DEFAULT_STRATEGY
from .recall_eval import GOLD, key_chunks
from .single_select import _expand, build_candidates
from .agent_select import SelectResult


def replay_question(ctx: CourseContext, row: dict, gold: dict | None) -> dict:
    q = row["question"]
    old_sel = row["selected_concepts"]
    # 旧记录里每个「选中候选」的代表 ID：展开记录里的 candidate，其余为未合并候选的自身
    exp = {e["candidate"]: set(e["to"]) for e in row["select_stage"].get("expanded", [])}
    covered = set().union(*exp.values()) if exp else set()
    reps = [(e["candidate"], None) for e in row["select_stage"].get("expanded", [])]
    role_of = {s["concept_id"]: s["role"] for s in old_sel}
    picks = []
    for cand, _ in reps:
        picks.append((cand, role_of.get(cand, "related_concept")))
    for s in old_sel:
        if s["concept_id"] not in covered:
            picks.append((s["concept_id"], s["role"]))
    cands = build_candidates(q, ctx.index, config.COURSE_SELECT_CANDIDATES)
    by_rep = {c.concept_id: c for c in cands}
    member_to_rep = {m: c.concept_id for c in cands for m, _ in c.members}
    q_ids, r_ids, missing = [], [], []
    for cid, role in picks:
        if cid not in member_to_rep:
            missing.append(cid)
            continue
        (q_ids if role == "question_concept" else r_ids).append(cid)
    res = SelectResult()
    sel = _expand(q_ids, r_ids, by_rep, member_to_rep, res, q, ctx.index)
    rr = ctx.recaller.recall([(s.concept_id, s.role) for s in sel], DEFAULT_STRATEGY)
    if config.COURSE_BM25_TOPK > 0:
        ctx.recaller.merge_extra(rr, ctx.bm25.search(q, config.COURSE_BM25_TOPK))
    kept, st = cap_for_filter(rr.items, question=q, concept_names=[ctx.index.concepts[s.concept_id].name for s in sel])
    kept_ids = {i.chunk_id for i in kept}
    all_ids = {i.chunk_id for i in rr.items}
    out = {"id": row["id"], "old": {"concepts": len(old_sel), "recalled": len(row["recalled"]),
                                    "kept": (row["filter_cap"] or {}).get("kept"), "dropped": (row["filter_cap"] or {}).get("dropped")},
           "new": {"candidates": len(cands), "concepts": len(sel), "recalled": len(rr.items), "kept": len(kept), "dropped": st["dropped"]},
           "missing_old_picks": missing, "expanded": res.expanded, "gold": []}
    for m in (gold or {}).get("must_recall", []):
        ids = set(key_chunks(ctx, m["key"]))
        out["gold"].append({"key": m["key"], "recalled": bool(ids & all_ids), "in_top_kept": bool(ids & kept_ids),
                            "old_recalled": bool(ids & {x["chunk_id"] for x in row["recalled"]}),
                            "old_in_filter": bool(ids & {x["chunk_id"] for x in row["recalled"] if x["grade"] is not None})})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(GOLD.with_name("course_e2e_eval_result.json")))
    ap.add_argument("--out", default=str(GOLD.with_name("course_replay_result.json")))
    a = ap.parse_args()
    ctx = CourseContext.load()
    gold = {g["id"]: g for g in json.load(open(GOLD, encoding="utf-8"))}
    rows = [replay_question(ctx, r, gold.get(r["id"])) for r in json.load(open(a.baseline, encoding="utf-8"))["rows"] if not r["error"]]
    Path(a.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows={len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
