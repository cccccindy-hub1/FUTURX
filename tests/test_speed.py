"""响应速度测试：初始化服务 -> 随机抽取 20 条 prompt -> 逐条计时 -> 汇总均值。

指标说明（单位：秒）：
- server_init_time  : 服务初始化耗时（加载 SentenceTransformer + Chroma + OpenAI 客户端）
- rag_time          : 检索耗时（embedding + 向量检索 + 上下文/prompt 组装）
- transmission_time : 首字节传输耗时（发起 LLM 请求到收到首个 token；含网络往返 + 上游 prefill）
- llm_generation    : 流式生成耗时（首个 token -> 最后一个 token）
- total_time        : 单条端到端耗时（= rag_time + transmission_time + llm_generation）

报告：<项目根>/speed_test_report.txt
      逐条追加写入（中断后已测数据仍保留），全部测完后把均值汇总置顶重写。

运行方式（任选）：
    python tests/test_speed.py
    python -m tests.test_speed
    RUN_SPEED_TEST=1 pytest tests/test_speed.py -s

注意：本测试会加载真实模型并访问真实 LLM，请勿纳入日常离线 CI。
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

# 允许直接 python tests/test_speed.py 运行：把项目根目录加入 sys.path
_SYS_PATH_ROOT = Path(__file__).resolve().parents[1]
if str(_SYS_PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYS_PATH_ROOT))

import pytest

from rag.benchmark import build_items, load_raw
from rag.config import BENCHMARK_SEED, ROOT
from rag.query import RAGService

SPEED_N_PROMPTS = int(os.environ.get("SPEED_N_PROMPTS", "20"))
SPEED_SEED = int(os.environ.get("SPEED_SEED", str(BENCHMARK_SEED)))
SPEED_REPORT_PATH = Path(os.environ.get("SPEED_REPORT_PATH", ROOT / "speed_test_report.txt"))

_COLS = ("rag", "transmission", "generation", "total")
_HEADER_LINE = f"{'#':>3} {'问题':<42} {'rag':>7} {'传输':>7} {'生成':>7} {'总耗时':>7}"


def _pick_prompts(n: int, seed: int) -> list:
    """从 data_benchmark 清洗后的样本中随机抽取 n 条。"""
    items = build_items(load_raw())
    if not items:
        return []
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


def _clip(text: str, width: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _f(v):
    return f"{v:7.3f}" if v is not None else f"{'-':>7}"


def _render_row(i: int, q: str, r: dict) -> str:
    return (
        f"{i:>3} {_clip(q):<42} {_f(r['rag'])} {_f(r['transmission'])} "
        f"{_f(r['generation'])} {_f(r['total'])}"
        f"\n    Answer (full):\n{r['answer']}"
    )


def _summarize(rows: list[dict]) -> dict:
    summary: dict = {}
    for key in _COLS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = (sum(vals) / len(vals)) if vals else None
    return summary


def run_speed_test(
    n: int = SPEED_N_PROMPTS,
    seed: int = SPEED_SEED,
    report_path: Path = SPEED_REPORT_PATH,
) -> dict:
    # 1) 服务初始化（服务启动开销）
    t0 = time.perf_counter()
    svc = RAGService()
    server_init_time = time.perf_counter() - t0

    items = _pick_prompts(n, seed)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # 2) 逐条测试，边测边写入报告（中断后已测数据仍保留）
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("FUTURX RAG 响应速度测试\n")
        fh.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"样本数：{len(items)} | 随机种子：{seed}\n")
        fh.write(f"服务初始化耗时：{server_init_time:.3f}s\n")
        fh.write("=" * 100 + "\n\n")
        fh.write("（逐条统计，测试进行中……）\n")
        fh.write(_HEADER_LINE + "\n")
        fh.flush()

        for i, item in enumerate(items, 1):
            row: dict = {
                "question": item.question,
                "rag": None,
                "transmission": None,
                "generation": None,
                "total": None,
                "error": None,
                "answer": "",
            }
            try:
                t = time.perf_counter()
                answer, _, _ = svc.answer(item.question)
                total = time.perf_counter() - t
                timing = getattr(svc, "timing", {}) or {}
                row["rag"] = timing.get("retrieval")
                row["transmission"] = timing.get("llm_ttft")
                row["generation"] = timing.get("llm_generation")
                row["total"] = total
                row["answer"] = answer or ""
            except Exception as e:  # 单条失败不影响整体
                row["error"] = f"{type(e).__name__}: {e}"

            rows.append(row)
            rendered = _render_row(i, item.question, row)
            fh.write(rendered + "\n")
            print(rendered, flush=True)
            if row["error"]:
                fh.write(f"    [错误] {row['error']}\n")
                print(f"    [错误] {row['error']}", flush=True)
            fh.flush()

    # 3) 计算均值
    summary = _summarize(rows)

    # 4) 重写报告：把均值汇总置顶
    labels = {
        "rag": "RAG 检索耗时均值",
        "transmission": "传输（首字节）耗时均值",
        "generation": "流式生成耗时均值",
        "total": "端到端总耗时均值",
    }
    lines = [
        "FUTURX RAG 响应速度测试报告",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"样本数：{len(rows)} | 随机种子：{seed}",
        "=" * 100,
        "",
        "【均值汇总】（单位：秒）",
        f"  服务初始化时间 (server_init_time) : {server_init_time:.3f}",
    ]
    for key in _COLS:
        v = summary[key]
        lines.append(f"  {labels[key]:<28} : {v:.3f}" if v is not None else f"  {labels[key]:<28} : -")
    lines += ["", "=" * 100, "", "【逐条统计】（单位：秒）", _HEADER_LINE]

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        for i, r in enumerate(rows, 1):
            fh.write(_render_row(i, r["question"], r) + "\n")
            if r["error"]:
                fh.write(f"    [错误] {r['error']}\n")

    result = {"server_init_time": server_init_time, "n": len(rows), **summary}
    return result


def main() -> None:
    result = run_speed_test()
    print("=" * 60)
    print("服务初始化时间：%.3f s" % result["server_init_time"])
    for key in _COLS:
        v = result[key]
        print(f"{key:<14}: {v:.3f} s" if v is not None else f"{key:<14}: -")
    print(f"报告已写入：{SPEED_REPORT_PATH}")


@pytest.mark.skipif(
    os.environ.get("RUN_SPEED_TEST") != "1",
    reason="会加载真实模型并访问真实 LLM，设置 RUN_SPEED_TEST=1 才会运行",
)
def test_speed():
    result = run_speed_test()
    assert result["n"] > 0


if __name__ == "__main__":
    main()
