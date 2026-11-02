"""端到端评测（真实调用）：r11–r20 经 UnifiedService.ask（含路由），记录端到端与分环节耗时、来源/等级/引用、warnings。

规则：默认仅非高峰（北京时间高峰为 09–12、14–18 点，价格翻倍；--allow-peak 显式放开，结果中记录）；每题 1 遍、不重试（SDK 自动重试已关）；失败记失败；
连续 2 题 LLMError 即停；每次调用前检查非高峰窗口；总调用硬上限 max_calls，到上限立即停止。结果只含异常类型名，不含 key。
用法：python -m rag.course.eval_e2e [--max-calls 45] [--out rag/eval_data/course_e2e_eval_result.json] [--allow-peak]
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
from ..router import build_router_client
from ..unified import UnifiedService
from . import config
from .eval_full import MAX_LLM_FAIL_STREAK, CapReached, CappedLLM
from .eval_full_report import summarize
from .llm import LLMError, LLMNotConfigured, OpenAICompatClient
from .pipeline import CourseContext
from .recall_eval import GOLD, key_chunks


def run_one(svc: UnifiedService, g: dict, shared: dict) -> dict:
    t0 = time.perf_counter()
    c0 = shared["calls"]
    row = {"id": g["id"], "question": g["question"], "error": ""}
    try:
        out = svc.ask(g["question"])
        tr = out["trace"]
        ctr = tr.get("course") or {}
        calls = ctr.get("calls", [])
        by = (ctr.get("totals") or {}).get("by_step", {})
        row.update(route=out["route"], status=out["status"], router=tr["router"], answer=out["answer"],
                   total_s=tr["total_s"], router_s=tr["router"]["latency_s"],
                   step_s={k: v["latency_s"] for k, v in by.items()},
                   step_calls={k: v["calls"] for k, v in by.items()},
                   course_calls=len(calls), prompt_tokens=sum(c["prompt_tokens"] for c in calls),
                   completion_tokens=sum(c["completion_tokens"] for c in calls),
                   selected_concepts=(out.get("course") or {}).get("selected_concepts"),
                   fallback=(out.get("course") or {}).get("fallback"),
                   select_stage={k: v for k, v in (ctr.get("stages", {}).get("select") or {}).items()
                                 if k in ("fallback", "end_reason", "dropped", "expanded", "reason", "candidates")},
                   filter_cap=(ctr.get("stages", {}).get("filter") or {}).get("cap"),
                   filter_skipped=(ctr.get("stages", {}).get("filter") or {}).get("skipped", False),
                   warnings=(ctr.get("stages", {}).get("answer") or {}).get("warnings", []),
                   recall_stage={k: v for k, v in (ctr.get("stages", {}).get("recall") or {}).items() if k in ("sources", "chunks")},
                   citations=[c.get("chunk_id") for c in out["sources"]],
                   recalled=(out.get("course") or {}).get("recalled", []))
    except CapReached as e:
        row["error"] = f"CapReached:{e}"
    except Exception as e:  # noqa: BLE001 - 只记类型名
        row["error"] = type(e).__name__
        if isinstance(e, LLMError):
            m = re.match(r"\[[^\]]*\] LLM 调用失败：(\w+)", str(e))
            row["error_cause"] = m.group(1) if m else "unknown"
    row["llm_calls_total"] = shared["calls"] - c0
    row["e2e_s"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=45)
    ap.add_argument("--out", default=str(GOLD.with_name("course_e2e_eval_result.json")))
    ap.add_argument("--allow-peak", action="store_true", help="允许在北京时间高峰时段运行（价格翻倍）")
    a = ap.parse_args()
    if not a.allow_peak and not in_window(datetime.now(BJ)):
        raise SystemExit("当前为北京时间高峰时段，不运行（确需运行加 --allow-peak）")
    try:
        cllm = OpenAICompatClient()
    except LLMNotConfigured:
        raise SystemExit("COURSE_LLM_* 未配置")
    rclient = build_router_client()
    if rclient is None:
        raise SystemExit("路由未配置")
    for c in (cllm, rclient):
        c._client = c._client.with_options(max_retries=0)
    shared = {"calls": 0, "cap": a.max_calls, "allow_peak": a.allow_peak}
    ctx = CourseContext.load()
    svc = UnifiedService(router_client=CappedLLM(rclient, shared), course_ctx=ctx, course_llm=CappedLLM(cllm, shared))
    gold = json.load(open(GOLD, encoding="utf-8"))
    for g in gold:
        for m in g["must_recall"]:
            m["chunks"] = key_chunks(ctx, m["key"])
    started = datetime.now(BJ).isoformat(timespec="seconds")
    rows, stopped, streak = [], "", 0
    for g in gold:
        row = run_one(svc, g, shared)
        rows.append(row)
        if row["error"].startswith("CapReached"):
            stopped = row["error"]
            break
        streak = streak + 1 if row["error"] == "LLMError" else 0
        if streak >= MAX_LLM_FAIL_STREAK:
            stopped = f"LLMError x{streak} in a row"
            break
    out = {"config": {"select_mode": config.COURSE_SELECT_MODE, "candidates": config.COURSE_SELECT_CANDIDATES,
                      "select_reasoning": config.COURSE_REASONING["select"], "bm25_topk": config.COURSE_BM25_TOPK,
                      "filter_reasoning": config.COURSE_REASONING["filter"], "answer_reasoning": config.COURSE_REASONING["answer"],
                      "filter_skip_max": config.COURSE_FILTER_SKIP_MAX, "book_version": config.BOOK_VERSION,
                      "allow_peak": a.allow_peak, "select_model": config.COURSE_SELECT_MODEL,
                      "filter_model": config.COURSE_FILTER_MODEL, "answer_model": config.COURSE_ANSWER_MODEL,
                      "answer_max_chunks": config.COURSE_ANSWER_MAX_CHUNKS,
                      "filter_max_chunks": config.COURSE_FILTER_MAX_CHUNKS, "model": config.COURSE_LLM_MODEL,
                      "router_model": rc.ROUTER_MODEL},
           "started_bj": started, "finished_bj": datetime.now(BJ).isoformat(timespec="seconds"),
           "total_calls": shared["calls"], "cap": a.max_calls, "stopped": stopped, "rows": rows}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: calls={shared['calls']}/{a.max_calls} rows={len(rows)} stopped={stopped!r}")


if __name__ == "__main__":
    main()
