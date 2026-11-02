"""课程题完整评测（真实调用）：直接调用 pipeline.ask，记录每块来源（ontology/bm25/both）、筛选等级、引用。

规则：仅非高峰（北京时间高峰为 09–12、14–18 点）；每题 1 遍、不重试（SDK 自动重试已关）、失败记失败；
调用硬上限 max_calls，到上限立即停止。结果只含异常类型名，不含 key。
用法：python -m rag.course.eval_full [--max-calls 100] [--out rag/eval_data/course_full_eval_result.json]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from .. import config as rc
from ..eval_router import BJ, in_window
from . import config
from .eval_full_report import select_ids, summarize
from .llm import LLMError, LLMNotConfigured, OpenAICompatClient
from .pipeline import CourseContext, ask, result_to_json
from .recall_eval import GOLD, key_chunks


MAX_LLM_FAIL_STREAK = 2


class CapReached(RuntimeError):
    pass


class CappedLLM:
    def __init__(self, inner, shared: dict):
        self.inner, self.s = inner, shared

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, step, messages, **kw):
        if self.s["calls"] >= self.s["cap"]:
            raise CapReached("call cap reached")
        if not self.s.get("allow_peak") and not in_window(datetime.now(BJ)):
            raise CapReached("left off-peak window")
        self.s["calls"] += 1
        return self.inner.chat(step, messages, **kw)


def run_one(ctx, llm, g: dict) -> dict:
    t0 = time.perf_counter()
    row = {"id": g["id"], "question": g["question"], "error": ""}
    try:
        res = result_to_json(ask(g["question"], ctx=ctx, llm=llm))
        tr = res["trace"]
        calls = tr["calls"]
        grade = {v["chunk_id"]: v["grade"] for v in res["filtered"]}
        src = {r["chunk_id"]: r.get("source") for r in res["recalled"]}
        answer = res.get("answer") or {}
        row.update(
            fallback=res["fallback"], answer=answer.get("text"), citations=answer.get("citations", []),
            selected_concepts=res["selected_concepts"],
            select_calls=sum(c["step"] == "select" for c in calls),
            select_rounds=tr["stages"].get("select", {}).get("rounds", []),
            llm_calls=len(calls), prompt_tokens=tr["totals"]["prompt_tokens"],
            completion_tokens=tr["totals"]["completion_tokens"], reasoning_tokens=tr["totals"]["reasoning_tokens"],
            recalled=[{"chunk_id": r["chunk_id"], "source": r.get("source"), "grade": grade.get(r["chunk_id"]),
                       "path": " › ".join(r["path"][-3:])} for r in res["recalled"]],
            recall_stage={k: v for k, v in tr["stages"].get("recall", {}).items() if k in ("sources", "bm25", "chunks")},
            invalid_citations=answer.get("invalid_citations", []))
        _ = src
    except CapReached as e:
        row["error"] = f"CapReached:{e}"
    except Exception as e:  # noqa: BLE001 - 只记类型名，不记消息（消息可能含请求细节）
        row["error"] = type(e).__name__
        if isinstance(e, LLMError):
            m = re.match(r"\[[^\]]*\] LLM 调用失败：(\w+)", str(e))
            row["error_cause"] = m.group(1) if m else "unknown"
    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=100)
    ap.add_argument("--out", default=str(GOLD.with_name("course_full_eval_result.json")))
    ap.add_argument("--ids", default="", help="只跑指定题号，逗号分隔（如 r13,r14）；不给则跑全部，其余行为不变")
    a = ap.parse_args()
    if not in_window(datetime.now(BJ)):
        raise SystemExit("当前为北京时间高峰时段，不运行")
    try:
        llm = OpenAICompatClient()
    except LLMNotConfigured:
        raise SystemExit("COURSE_LLM_* 未配置")
    llm._client = llm._client.with_options(max_retries=0)
    shared = {"calls": 0, "cap": a.max_calls}
    capped = CappedLLM(llm, shared)
    ctx = CourseContext.load()
    gold = select_ids(json.load(open(GOLD, encoding="utf-8")), a.ids)
    for g in gold:                                      # 关键句 -> 子块
        for m in g["must_recall"]:
            m["chunks"] = key_chunks(ctx, m["key"])
    started = datetime.now(BJ).isoformat(timespec="seconds")      # 开始时间在循环前记录（此前误在结束后取）
    rows, stopped, llm_fail_streak = [], "", 0
    for g in gold:
        row = run_one(ctx, capped, g)
        rows.append(row)
        if row["error"].startswith("CapReached"):
            stopped = row["error"]
            break
        llm_fail_streak = llm_fail_streak + 1 if row["error"] == "LLMError" else 0
        if llm_fail_streak >= MAX_LLM_FAIL_STREAK:                # 连续失败多半是超时/断网/睡眠，继续跑只会白耗调用
            stopped = f"LLMError x{llm_fail_streak} in a row"
            break
    out = {"config": {"select_reasoning": config.COURSE_REASONING["select"], "max_steps": config.COURSE_AGENT_MAX_STEPS,
                      "fast_path": config.COURSE_SELECT_FAST_PATH, "bm25_topk": config.COURSE_BM25_TOPK,
                      "model": config.COURSE_LLM_MODEL},
           "started_bj": started, "finished_bj": datetime.now(BJ).isoformat(timespec="seconds"), "total_calls": shared["calls"],
           "cap": a.max_calls, "stopped": stopped, "rows": rows}
    out["summary"] = summarize(rows, gold)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: calls={shared['calls']}/{a.max_calls} rows={len(rows)} stopped={stopped!r}")


if __name__ == "__main__":
    main()
