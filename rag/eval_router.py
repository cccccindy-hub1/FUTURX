"""路由评测：对 rag/eval_data/router_eval.json 每题只调一次路由模型，不重试、不追加、不走规则兜底。

用法：python -m rag.eval_router [--out rag/eval_data/router_eval_result.json] [--wait]
- 只在非高峰运行（北京时间高峰 09–12、14–18 点）；--wait 时在高峰期等待到非高峰，否则直接退出；
- 失败（超时/报错/JSON 或字段非法）的题记为失败，不重试；
- 结果与日志只记异常类型名，不含 key。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .course.llm import parse_json_object
from .router import ROUTER_SYSTEM, ROUTES, build_router_client, normalize_routes

EVAL_PATH = Path(__file__).parent / "eval_data" / "router_eval.json"
BJ = timezone(timedelta(hours=8))


def in_window(now: datetime) -> bool:
    """非高峰：北京时间高峰为 09–12、14–18 点（价格翻倍），其余时段为非高峰。"""
    h = now.astimezone(BJ).hour
    return not (9 <= h < 12 or 14 <= h < 18)


def single_call(client, question: str) -> dict:
    """一次调用，不重试、不兜底。返回 {pred, complexity, reason, latency_s, error}。"""
    t0 = time.perf_counter()
    try:
        res = client.chat("route", [{"role": "system", "content": ROUTER_SYSTEM},
                                    {"role": "user", "content": question}],
                          model=config.ROUTER_MODEL, json_mode=True)
        obj = parse_json_object(res.content)
        routes = normalize_routes(obj.get("routes"), fallback=str(obj.get("route", "")).strip() or None)
        route = routes[0]
        return {"pred": route, "pred_routes": routes, "complexity": str(obj.get("complexity", "")), "reason": str(obj.get("reason", ""))[:80],
                "latency_s": round(res.latency_s or time.perf_counter() - t0, 3), "error": ""}
    except Exception as e:  # noqa: BLE001 - 只记类型名
        return {"pred": None, "complexity": "", "reason": "", "latency_s": round(time.perf_counter() - t0, 3),
                "error": type(e).__name__}


def summarize(rows: list[dict]) -> dict:
    per, conf = {}, Counter()
    for r in ROUTES:
        sub = [x for x in rows if x["expected_route"] == r]
        ok = sum(x["pred"] == r for x in sub)
        per[r] = {"n": len(sub), "correct": ok, "failed": sum(x["error"] != "" for x in sub),
                  "accuracy": round(ok / len(sub), 3) if sub else None}
    for x in rows:
        conf[(x["expected_route"], x["pred"] or "FAILED")] += 1
    ok = sum(x["pred"] == x["expected_route"] for x in rows)
    th = [x for x in rows if x["expected_route"] == "thesis" and x["pred"] == "thesis"]
    lat = [x["latency_s"] for x in rows if not x["error"]]
    return {"overall_accuracy": round(ok / len(rows), 3) if rows else None, "per_class": per,
            "confusion": {f"{a}->{b}": n for (a, b), n in sorted(conf.items())},
            "thesis_complexity_accuracy": round(sum(x["complexity"] == x["expected_complexity"] for x in th) / len(th), 3)
            if th else None,
            "latency_mean_s": round(sum(lat) / len(lat), 3) if lat else None,
            "latency_max_s": max(lat) if lat else None,
            "calls": len(rows), "failures": sum(x["error"] != "" for x in rows)}


def run(client, questions: list[dict], now_fn=lambda: datetime.now(BJ)) -> dict:
    rows, skipped = [], 0
    for q in questions:
        if not in_window(now_fn()):          # 中途滑出窗口：剩余题目不再调用
            skipped += 1
            continue
        rows.append({**q, **single_call(client, q["question"])})
    return {"rows": rows, "skipped_outside_window": skipped, "summary": summarize(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EVAL_PATH.with_name("router_eval_result.json")))
    ap.add_argument("--wait", action="store_true", help="窗口外等待到 12–14 / 18 点以后再开始")
    a = ap.parse_args()
    while not in_window(datetime.now(BJ)):
        if not a.wait:
            raise SystemExit("当前为北京时间高峰时段，不运行（加 --wait 可等待）")
        time.sleep(30)
    client = build_router_client()
    if client is None:
        raise SystemExit("ROUTER_API_BASE / ROUTER_API_KEY 未配置")
    client._client = client._client.with_options(max_retries=0)      # SDK 默认会自动重试，这里禁用
    started = datetime.now(BJ).isoformat(timespec="seconds")
    out = run(client, json.load(open(EVAL_PATH, encoding="utf-8")))
    out.update(model=config.ROUTER_MODEL, reasoning=config.ROUTER_REASONING, started_bj=started)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
