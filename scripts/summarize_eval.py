"""10 题评测汇总：格式（复用 check_answer_format）+ 来自调用日志的分阶段延迟与 TTFT。

用法：
    python scripts/summarize_eval.py rag/eval_data/course_e2e_stage3_p1.json [更多评测文件]
    python scripts/summarize_eval.py <eval.json> --log-dir logs/usage --day 20261106

延迟只统计 origin=eval 的调用日志记录，按评测文件里的题面匹配；
日志缺失（例如评测不在本机跑）时延迟表标注“无日志”。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.usage_log import log_dir  # noqa: E402

STAGES = ("router", "select", "filter", "answer", "followup")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_logs(base: Path, day: str | None) -> list[dict]:
    files = [base / f"{day}.jsonl"] if day else sorted(base.glob("*.jsonl"))
    rows: list[dict] = []
    for p in files:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except OSError:
            continue
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测汇总（格式 + 分阶段延迟）")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--day", default="")
    a = ap.parse_args(argv)

    rows: list[dict] = []
    for f in a.files:
        p = Path(f)
        if not p.is_file():
            print(f"[skip] {p} 不存在")
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        got = doc.get("rows") if isinstance(doc, dict) else doc
        rows.extend(got or [])

    print(f"=== 题量 {len(rows)}，来源 {', '.join(Path(x).name for x in a.files)}")
    print(f"{'id':5}{'route':8}{'status':9}{'e2e_s':>7}{'select':>8}{'filter':>8}{'answer':>8}{'字数':>6}")
    for r in rows:
        s = r.get("step_s") or {}
        print(f"{r.get('id', ''):5}{r.get('route', ''):8}{r.get('status', ''):9}{r.get('total_s', 0):>7.1f}"
              f"{s.get('select', 0):>8.1f}{s.get('filter', 0):>8.1f}{[v for k, v in s.items() if k == 'answer'][0] if 'answer' in s else 0:>8.1f}"
              f"{len(r.get('answer') or ''):>6}")

    logs = [x for x in load_logs(Path(a.log_dir) if a.log_dir else log_dir(), a.day or None)
            if x.get("origin") == "eval"]
    print(f"\n调用日志（origin=eval）：{len(logs)} 条")
    if logs:
        print(f"{'阶段':8}{'n':>4}{'p50':>9}{'p95':>9}")
        for st in STAGES:
            v = [x["stages"][st] for x in logs if (x.get("stages") or {}).get(st)]
            if v:
                print(f"{st:8}{len(v):>4}{pct(v, 50):>8.0f}ms{pct(v, 95):>7.0f}ms")
        tt = [x["time_to_first_delta_ms"] for x in logs if x.get("time_to_first_delta_ms")]
        if tt:
            print(f"{'TTFT':8}{len(tt):>4}{pct(tt, 50):>8.0f}ms{pct(tt, 95):>7.0f}ms")
        else:
            print("TTFT    —（本批全为非流式 /auto/ask，无首字数据）")
        dur = [x["duration_ms"] for x in logs if x.get("duration_ms")]
        if dur:
            print(f"{'e2e':8}{len(dur):>4}{pct(dur, 50):>8.0f}ms{pct(dur, 95):>7.0f}ms")
    else:
        print("无日志（评测不写本机 logs/，或日期不对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
