"""课程题经 /auto/ask 完整作答的评测（真实调用）：取 router_eval.json 里前 N 道 course 题。

规则：仅非高峰（北京时间高峰为 09–12、14–18 点）；每题 1 遍、不重试（SDK 自动重试已关）、失败记失败；
调用硬上限 max_calls（路由 + 课程管线所有 LLM 调用合计），到上限立即停止，不再发起任何调用。
结果只含异常类型名，不含 key。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from . import config
from .eval_router import BJ, EVAL_PATH, in_window
from .router import build_router_client
from .unified import UnifiedService


class CapReached(RuntimeError):
    pass


class CountingLLM:
    """包装 LLMClient：统计调用次数与 token，到硬上限抛 CapReached（在发起请求之前）。"""

    def __init__(self, inner, shared: dict):
        self.inner, self.s = inner, shared

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, step, messages, **kw):
        if self.s["calls"] >= self.s["cap"]:
            raise CapReached("call cap reached")
        if not in_window(datetime.now(BJ)):
            raise CapReached("left off-peak window")
        self.s["calls"] += 1
        self.s["q_calls"] += 1
        res = self.inner.chat(step, messages, **kw)
        self.s["q_in"] += res.prompt_tokens
        self.s["q_out"] += res.completion_tokens
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ids", default="", help="逗号分隔的题号（如 r13）；给定时忽略 --n")
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--out", default=str(EVAL_PATH.with_name("course_eval_result.json")))
    a = ap.parse_args()
    if not in_window(datetime.now(BJ)):
        raise SystemExit("当前为北京时间高峰时段，不运行")
    from .serve import load_course_backend
    ctx, cllm = load_course_backend()
    rclient = build_router_client()
    if ctx is None or cllm is None or rclient is None:
        raise SystemExit("课程库或路由未就绪")
    for c in (cllm, rclient):
        c._client = c._client.with_options(max_retries=0)
    shared = {"calls": 0, "cap": a.max_calls, "q_calls": 0, "q_in": 0, "q_out": 0}
    svc = UnifiedService(router_client=CountingLLM(rclient, shared), course_ctx=ctx, course_llm=CountingLLM(cllm, shared))
    qs = [q for q in json.load(open(EVAL_PATH, encoding="utf-8")) if q["expected_route"] == "course"]
    qs = [q for q in qs if q["id"] in a.ids.split(",")] if a.ids else qs[: a.n]
    rows, stopped = [], ""
    for q in qs:
        shared.update(q_calls=0, q_in=0, q_out=0)
        t0 = time.perf_counter()
        row = {"id": q["id"], "question": q["question"], "error": ""}
        try:
            out = svc.ask(q["question"])
            course = out.get("course") or {}
            ctr = (out.get("trace") or {}).get("course") or {}
            row.update(route=out["route"], status=out["status"], answer=out["answer"],
                       citations=[c.get("chunk_id") for c in out["sources"]],
                       citation_paths=[c.get("path") for c in out["sources"]],
                       fallback=course.get("fallback"),
                       selected_concepts=course.get("selected_concepts"),
                       steps=[c["step"] for c in ctr.get("calls", [])],
                       select_rounds=((ctr.get("stages") or {}).get("select") or {}).get("rounds", []),
                       fast_path=((ctr.get("stages") or {}).get("select") or {}).get("fast_path", ""),
                       s_placeholder_in_answer="[S#]" in (out["answer"] or "") or "S#" in (out["answer"] or ""))
        except CapReached as e:
            row["error"], stopped = f"CapReached:{e}", str(e)
        except Exception as e:  # noqa: BLE001 - 只记类型名
            row["error"] = type(e).__name__
        row.update(llm_calls=shared["q_calls"], prompt_tokens=shared["q_in"], completion_tokens=shared["q_out"],
                   elapsed_s=round(time.perf_counter() - t0, 2))
        rows.append(row)
        if stopped:
            break
    from .course import config as cc
    res = {"select_reasoning_config": cc.COURSE_REASONING["select"], "fast_path": cc.COURSE_SELECT_FAST_PATH, "max_steps": cc.COURSE_AGENT_MAX_STEPS,
           "model": config.ROUTER_MODEL, "started_bj": datetime.now(BJ).isoformat(timespec="seconds"),
           "total_calls": shared["calls"], "cap": a.max_calls, "stopped": stopped, "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: calls={shared['calls']}/{a.max_calls} rows={len(rows)} stopped={stopped!r}")


if __name__ == "__main__":
    main()
