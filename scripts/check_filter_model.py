"""核对筛选模型（COURSE_FILTER_MODEL）能否按生产方式工作：发 1 次真实请求，检查 JSON 模式与关闭思考。

在项目根目录运行（.env 由 rag.config 在导入时按项目现有方式加载）：
    python scripts/check_filter_model.py

做什么：
- 读取 COURSE_FILTER_MODEL，没设就报错退出（退出码 2）；
- 用现有 OpenAICompatClient 和筛选环节的真实提示词（prompts.FILTER_SYSTEM / FILTER_USER / FILTER_ITEM），
  喂 3 个假教材块和 1 个问题，response_format=json_object，reasoning 按 COURSE_FILTER_REASONING（默认 disabled）；
- 只发 1 次请求，不重试；
- 输出：模型名、请求是否成功、返回是否为合法 JSON 且含 direct 和 background 两个键、reasoning 长度、
  prompt/completion tokens、耗时；全部通过最后一行打印 PASS（退出码 0），否则打印 FAIL 和原因（退出码 1）。
出错只输出异常类型名，不输出 key、请求头和异常正文。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:               # 允许 `python scripts/check_filter_model.py` 直接运行
    sys.path.insert(0, str(ROOT))

from rag.course import config, prompts  # noqa: E402  (导入 rag.config 时会加载 .env)
from rag.course.llm import LLMError, LLMNotConfigured, OpenAICompatClient, parse_json_value  # noqa: E402

QUESTION = "什么是数据要素？"
# 3 个假教材块：S1 直接回答、S2 是前提概念、S3 无关。编号与格式和线上筛选一致（[S#] 章节路径 + 正文）。
FAKE_CHUNKS = [
    ("S1", ["第一章 数据要素", "第一节 数据要素的概念界定"],
     "数据要素是指根据特定生产需求，经过汇聚、整理、加工等环节形成的，能够投入到生产经营活动中、参与价值创造的数据资源。"),
    ("S2", ["第一章 数据要素", "第一节 数据要素的概念界定", "（二）相近概念辨析"],
     "数据资源是指具有价值创造潜力的数据的总称，是数据资产与数据要素的共同来源。"),
    ("S3", ["第八章 数据安全", "第二节 数据安全风险"],
     "数据泄露事件的处置流程包括发现、研判、通报和整改四个阶段。"),
]

EXIT_OK, EXIT_FAIL, EXIT_NOT_SET = 0, 1, 2


def build_messages() -> list[dict]:
    """与 passage_filter._judge_batch 在原生 JSON 模式下构造的消息完全相同。"""
    items = "\n\n".join(prompts.FILTER_ITEM.format(ref=ref, path=" › ".join(path), text=text)
                        for ref, path, text in FAKE_CHUNKS)
    return [{"role": "system", "content": prompts.FILTER_SYSTEM},
            {"role": "user", "content": prompts.FILTER_USER.format(question=QUESTION, items=items)}]


def run_check(client, model: str) -> dict:
    """发 1 次请求并核对返回。永不抛异常；error 只含异常类型名。"""
    rep = {"model": model, "reasoning_setting": None, "request_ok": False, "error": "", "json_valid": False,
           "has_direct": False, "has_background": False, "reasoning_chars": None, "reasoning_tokens": None,
           "prompt_tokens": None, "completion_tokens": None, "latency_s": None, "problems": []}
    fn = getattr(client, "reasoning_for", None)
    rep["reasoning_setting"] = (fn("filter") if fn else None) or "(未设置，不发送 reasoning 参数)"
    try:
        res = client.chat("filter", build_messages(), model=model, json_mode=True)
    except LLMError as e:                     # 消息已由客户端脱敏，这里仍只取底层异常类型名
        m = re.match(r"\[[^\]]*\] LLM 调用失败：(\w+)", str(e))
        rep["error"] = m.group(1) if m else "LLMError"
        rep["problems"].append(f"请求失败（{rep['error']}）")
        return rep
    except Exception as e:  # noqa: BLE001 - 只记类型名
        rep["error"] = type(e).__name__
        rep["problems"].append(f"请求失败（{rep['error']}）")
        return rep
    rep.update(request_ok=True, reasoning_chars=res.reasoning_chars, reasoning_tokens=res.reasoning_tokens,
               prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, latency_s=round(res.latency_s, 3))
    try:
        obj = parse_json_value(res.content)
    except ValueError:
        rep["problems"].append("返回不是合法 JSON")
        return rep
    if not isinstance(obj, dict):
        rep["problems"].append("返回的 JSON 顶层不是对象")
        return rep
    rep["json_valid"] = True
    rep["has_direct"], rep["has_background"] = "direct" in obj, "background" in obj
    for k in ("direct", "background"):
        if k not in obj:
            rep["problems"].append(f"缺少键 {k}")
        elif not isinstance(obj[k], list):
            rep["problems"].append(f"{k} 不是数组")
    if config.COURSE_REASONING.get("filter", "").strip().lower() == "disabled" and (rep["reasoning_chars"] or rep["reasoning_tokens"]):
        rep["problems"].append(f"已设置 reasoning=disabled，但返回仍有思考内容（chars={rep['reasoning_chars']}, tokens={rep['reasoning_tokens']}）")
    return rep


def format_report(rep: dict) -> list[str]:
    ok = lambda b: "是" if b else "否"
    lines = [f"模型名: {rep['model']}",
             f"reasoning 设置: {rep['reasoning_setting']}",
             f"请求成功: {ok(rep['request_ok'])}" + (f"（异常类型: {rep['error']}）" if rep["error"] else ""),
             f"返回为合法 JSON: {ok(rep['json_valid'])}",
             f"含 direct 键: {ok(rep['has_direct'])}  含 background 键: {ok(rep['has_background'])}",
             f"reasoning 长度: chars={rep['reasoning_chars']} tokens={rep['reasoning_tokens']}",
             f"tokens: prompt={rep['prompt_tokens']} completion={rep['completion_tokens']}",
             f"耗时: {rep['latency_s']}s"]
    lines.append("PASS" if not rep["problems"] else "FAIL: " + "；".join(rep["problems"]))
    return lines


def main(argv=None, *, client_factory=None) -> int:
    model = (config.COURSE_FILTER_MODEL_ENV or "").strip()
    if not model:
        print("FAIL: 未设置 COURSE_FILTER_MODEL（请在 .env 或环境变量中设置，例如 deepseek/deepseek-v4-flash）")
        return EXIT_NOT_SET
    try:
        client = (client_factory or OpenAICompatClient)()
    except LLMNotConfigured as e:
        print(f"FAIL: 课程 LLM 未配置（{type(e).__name__}）：请设置 COURSE_LLM_API_BASE / COURSE_LLM_API_KEY")
        return EXIT_FAIL
    except Exception as e:  # noqa: BLE001 - 只输出类型名
        print(f"FAIL: 无法创建客户端（{type(e).__name__}）")
        return EXIT_FAIL
    rep = run_check(client, model)
    lines = format_report(rep)
    print("\n".join(lines))
    return EXIT_OK if lines[-1] == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
