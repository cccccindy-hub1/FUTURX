"""两级压测（需求6）。

Tier 1（服务器侧，0 成本）：从试卷库取 N 道原题，用「判断这道题是否正确：<题干>」形式打 /auto/ask，
命中题库预匹配 → 0 次模型调用，只压信号量、线程池、JSON 序列化与 SSE/HTTP 栈。

Tier 2（供应商侧，近 0 成本）：直接对上游 /v1/chat/completions 发 5-token 微请求，并发阶梯观察 429 与延迟坍塌。

用法：
    python scripts/load_test.py tier1 --n 100 --concurrency 100
    python scripts/load_test.py tier2 --levels 10,30,50,100 --per 10

Tier 1 需要题库（EXAM_TEACHER_DOC + 学生知识集合）可用；Tier 2 需要 COURSE_LLM_* 或 LLM_* 配置。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mem_current() -> str:
    try:
        import subprocess

        out = subprocess.run(["systemctl", "show", "futurx", "-p", "MemoryCurrent", "--value"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        v = int(out)
        return f"{v / 1024 / 1024:.0f}MiB" if v > 0 else out
    except Exception:  # noqa: BLE001
        return "?"


def journal_events() -> int:
    """cgroup memory.events 里 low 的累计触发次数（读不到返回 -1）。"""
    try:
        return _events_total(Path("/sys/fs/cgroup/system.slice/futurx.service/memory.events"))
    except Exception:  # noqa: BLE001
        return -1


def _events_total(path: Path) -> int:
    total = 0
    for line in path.read_text().splitlines():
        key, _, val = line.partition(" ")
        if key == "low":
            total += int(val or 0)
    return total


def tier1(n: int, concurrency: int, host: str, key: str) -> int:
    from rag.exam.store import ExamBank

    bank = ExamBank.build()
    stems = [it.q.stem for it in bank.items if len(it.q.stem) >= 20][:n]
    if len(stems) < n:
        print(f"[warn] 题库只有 {len(stems)} 道够长的题干，按此数量跑")
    questions = [f"判断这道题是否正确：{s}" for s in stems]

    import httpx

    before_mem, before_ev = mem_current(), journal_events()
    lat: list[float] = []
    codes: dict[int, int] = {}
    types: dict[str, int] = {}
    t_all = time.perf_counter()

    def one(q: str):
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{host}/auto/ask", json={"question": q, "include_trace": True},
                           headers={"Authorization": f"Bearer {key}"}, timeout=120)
            dt = (time.perf_counter() - t0) * 1000
            body = {}
            try:
                body = r.json()
            except ValueError:
                pass
            return r.status_code, dt, (body.get("route"), (body.get("trace") or {}).get("prematch", {}).get("hit") is not None)
        except Exception as e:  # noqa: BLE001
            return -1, (time.perf_counter() - t0) * 1000, (type(e).__name__, False)

    peak_mem = before_mem
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, q) for q in questions]
        for i, f in enumerate(futs):
            code, dt, meta = f.result()
            codes[code] = codes.get(code, 0) + 1
            key_ = str(meta[0])
            types[key_] = types.get(key_, 0) + 1
            if code == 200:
                lat.append(dt)
            if i % 20 == 0:
                peak_mem = mem_current()

    total_s = time.perf_counter() - t_all
    after_mem, after_ev = mem_current(), journal_events()
    print(f"\n=== Tier 1  题目 {len(questions)}  并发 {concurrency}")
    print(f"总耗时 {total_s:.1f}s，吞吐 {len(questions) / total_s:.1f} req/s")
    print("状态码：" + "，".join(f"{k}×{v}" for k, v in sorted(codes.items())))
    print("路由分布：" + "，".join(f"{k}×{v}" for k, v in sorted(types.items())))
    if lat:
        print(f"延迟：p50 {pct(lat, 50):.0f}ms  p95 {pct(lat, 95):.0f}ms  max {max(lat):.0f}ms")
    print(f"MemoryCurrent：{before_mem} → {after_mem}（采样峰值 {peak_mem}）")
    print(f"memory.events(low 累计)：{before_ev} → {after_ev}")
    return 0


def _micro(host: str, key: str, model: str, timeout: float = 60):
    import httpx

    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{host}/v1/chat/completions",
                       json={"model": model, "messages": [{"role": "user", "content": "只回答：好"}],
                             "max_tokens": 5, "stream": False},
                       headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        return r.status_code, (time.perf_counter() - t0) * 1000, r.text[:200]
    except Exception as e:  # noqa: BLE001
        return -1, (time.perf_counter() - t0) * 1000, type(e).__name__


def tier2(levels: list[int], per: int, base_url: str, api_key: str, model: str) -> int:
    """直接打上游供应方。每级并发 level，请求数 per。"""
    from concurrent.futures import ThreadPoolExecutor as Pool

    print(f"\n=== Tier 2  供应商侧微请求（max_tokens=5）  {base_url}  model={model}")
    print(f"{'并发':>4}{'请求':>5}{'成功':>5}{'429':>5}{'其它':>5}{'p50':>9}{'p95':>9}{'max':>9}")
    for level in levels:
        import httpx

        def one(_):
            t0 = time.perf_counter()
            try:
                r = httpx.post(f"{base_url}/chat/completions",
                               json={"model": model, "messages": [{"role": "user", "content": "只回答：好"}],
                                     "max_tokens": 5},
                               headers={"Authorization": f"Bearer {api_key}"}, timeout=90)
                return r.status_code, (time.perf_counter() - t0) * 1000
            except Exception as e:  # noqa: BLE001
                return -1, (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with Pool(max_workers=level) as pool:
            res = list(pool.map(one, range(per)))
        lat = [d for c, d in res if c == 200]
        ok = sum(1 for c, _ in res if c == 200)
        r429 = sum(1 for c, _ in res if c == 429)
        other = len(res) - ok - r429
        print(f"{level:>4}{per:>5}{ok:>5}{r429:>5}{other:>5}"
              f"{pct(lat, 50):>8.0f}ms{pct(lat, 95):>7.0f}ms{(max(lat) if lat else 0):>7.0f}ms")
        _ = time.perf_counter() - t0
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="两级压测")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t1 = sub.add_parser("tier1")
    t1.add_argument("--n", type=int, default=100)
    t1.add_argument("--concurrency", type=int, default=100)
    t1.add_argument("--host", default="http://127.0.0.1:8001")
    t1.add_argument("--key", default="")

    t2 = sub.add_parser("tier2")
    t2.add_argument("--levels", default="10,30,50,100")
    t2.add_argument("--per", type=int, default=10)
    t2.add_argument("--base-url", default="")
    t2.add_argument("--api-key", default="")
    t2.add_argument("--model", default="")

    a = ap.parse_args(argv)
    if a.cmd == "tier1":
        return tier1(a.n, a.concurrency, a.host, a.key)
    from rag import config as rc
    from rag.course import config as cc

    return tier2([int(x) for x in a.levels.split(",") if x.strip()], a.per,
                 (a.base_url or rc.LLM_API_BASE).rstrip("/"), a.api_key or rc.LLM_API_KEY,
                 a.model or cc.COURSE_FILTER_MODEL or rc.LLM_MODEL)


if __name__ == "__main__":
    sys.exit(main())
