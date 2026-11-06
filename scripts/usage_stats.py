"""调用日志统计（需求5）：读 logs/usage/YYYYMMDD.jsonl，输出日请求量、路由/状态分布、
按 route 的 p50/p95 延迟、平均字数、追问出现率。

用法：
    python scripts/usage_stats.py                 # 今天
    python scripts/usage_stats.py --day 20261106
    python scripts/usage_stats.py --days 7        # 最近 7 天汇总
    python scripts/usage_stats.py --dir /path/logs/usage
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.usage_log import log_dir  # noqa: E402


def _pct(values: list[float], q: float) -> float:
    """线性插值分位数（q 取 0–100）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            continue
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    routes = Counter(r.get("route") or "unknown" for r in rows)
    statuses = Counter(r.get("status") or "unknown" for r in rows)
    lat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ms = r.get("duration_ms")
        if isinstance(ms, (int, float)):
            lat[r.get("route") or "unknown"].append(float(ms))
    chars = [r.get("answer_chars") or len(r.get("answer") or "") for r in rows]
    follow = sum(1 for r in rows if r.get("followups"))
    with_fu_gate = sum(1 for r in rows if r.get("status") == "ok")
    return {
        "total": total,
        "routes": routes.most_common(),
        "statuses": statuses.most_common(),
        "latency": {k: {"n": len(v), "p50": round(_pct(v, 50)), "p95": round(_pct(v, 95))}
                    for k, v in sorted(lat.items())},
        "avg_chars": round(sum(chars) / total) if total else 0,
        "median_chars": round(statistics.median(chars)) if chars else 0,
        "followup_rate": (follow / with_fu_gate) if with_fu_gate else 0.0,
        "errors": Counter(r.get("error") for r in rows if r.get("error")).most_common(),
        "key_ids": Counter(r.get("key_id") or "none" for r in rows).most_common(),
    }


def fmt_ms(v: float) -> str:
    return f"{v:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def format_report(s: dict, label: str) -> str:
    out = [f"=== {label}  共 {s['total']} 条"]
    if not s["total"]:
        return out[0] + "（无记录）"
    out.append("路由分布：" + "，".join(f"{k} {v}" for k, v in s["routes"]))
    out.append("状态分布：" + "，".join(f"{k} {v}" for k, v in s["statuses"]))
    out.append("端到端延迟（含 LLM 生成）：")
    for route, v in s["latency"].items():
        out.append(f"  {route:<10} n={v['n']:<4} p50={fmt_ms(v['p50']):<8} p95={fmt_ms(v['p95'])}")
    out.append(f"回答字数：平均 {s['avg_chars']}，中位 {s['median_chars']}")
    out.append(f"追问出现率：{s['followup_rate'] * 100:.1f}%（分母为 status=ok 的请求）")
    if s["errors"]:
        out.append("错误类型：" + "，".join(f"{k}×{v}" for k, v in s["errors"]))
    out.append("key 分布：" + "，".join(f"{k}×{v}" for k, v in s["key_ids"]))
    return "\n".join(out)


def pick_files(day: str | None, days: int, base: Path) -> list[Path]:
    if day:
        return [base / f"{day}.jsonl"]
    if days <= 1:
        return [base / f"{time.strftime('%Y%m%d')}.jsonl"]
    out = []
    now = time.time()
    for i in range(days):
        out.append(base / f"{time.strftime('%Y%m%d', time.localtime(now - i * 86400))}.jsonl")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="调用日志统计")
    ap.add_argument("--day", default="", help="指定日期 YYYYMMDD（默认今天）")
    ap.add_argument("--days", type=int, default=1, help="最近 N 天（含今天）")
    ap.add_argument("--dir", default="", help="日志目录（默认 RAG_LOG_DIR 或 logs/usage）")
    a = ap.parse_args(argv)
    base = Path(a.dir) if a.dir else log_dir()
    files = pick_files(a.day or None, a.days, base)
    rows = load(files)
    label = a.day or (f"最近 {a.days} 天" if a.days > 1 else time.strftime("%Y-%m-%d"))
    print(format_report(summarize(rows), f"{label}  {base}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
