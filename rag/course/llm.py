"""OpenAI 兼容客户端封装（胜算云 DeepSeek V4 Pro）+ trace。

- openai 包只在 OpenAICompatClient 构造时才 import；测试用 ScriptedLLM 完全不依赖它；
- key 只存在客户端私有属性里：不进 repr、trace、异常文本（异常里出现的 key 会被替换为 ***）；
- temperature 固定 0；reasoning_content 只记长度，不返回给调用方、不进入下一步输入；
- API_BASE 不设默认值，未配置时抛 LLMNotConfigured（只提变量名，不含值）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from . import config


class LLMNotConfigured(RuntimeError):
    """缺少 COURSE_LLM_API_BASE / COURSE_LLM_API_KEY，或未安装 openai。"""


class LLMError(RuntimeError):
    """调用失败（消息已脱敏）。"""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_chars: int = 0
    latency_s: float = 0.0


class LLMClient(Protocol):
    def chat(self, step: str, messages: list[dict], *, model: str,
             tools: list[dict] | None = None, json_mode: bool = False) -> LLMResult: ...


class OpenAICompatClient:
    def __init__(self, base: str | None = None, key: str | None = None, *, timeout: float = 120.0,
                 reasoning: dict[str, str] | None = None):
        self._base = (config.COURSE_LLM_API_BASE if base is None else base).strip()
        self._key = (config.COURSE_LLM_API_KEY if key is None else key).strip()
        if not self._base:
            raise LLMNotConfigured(
                "COURSE_LLM_API_BASE 未配置：请在 FUTURX/.env 中填写胜算云的 OpenAI 兼容接口地址。"
            )
        if not self._key:
            raise LLMNotConfigured("COURSE_LLM_API_KEY 未配置：请在 FUTURX/.env 中填写。")
        try:
            import openai
        except ImportError:
            raise LLMNotConfigured("未安装 openai 包（pip install openai）。") from None
        # reasoning：覆盖 step -> 思考模式（供路由等非课程步骤使用）；缺省按 config.COURSE_REASONING
        self._reasoning = reasoning
        self._client = openai.OpenAI(base_url=self._base, api_key=self._key, timeout=timeout)

    def __repr__(self) -> str:
        return "OpenAICompatClient(configured=True)"

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def reasoning_for(self, step: str) -> str:
        if self._reasoning is not None and step in self._reasoning:
            return self._reasoning[step]
        return config.COURSE_REASONING.get(step, "")

    def chat(self, step: str, messages: list[dict], *, model: str,
             tools: list[dict] | None = None, json_mode: bool = False) -> LLMResult:
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        reasoning = reasoning_param(self.reasoning_for(step))
        if reasoning:
            kwargs["extra_body"] = {"reasoning": reasoning}
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - 统一脱敏后重抛
            raise LLMError(f"[{step}] LLM 调用失败：{type(e).__name__}: {self._redact(str(e))}") from None
        res = parse_response(resp, time.perf_counter() - t0)
        _report(step, model, res)
        return res

    def chat_stream(self, step: str, messages: list[dict], *, model: str, on_delta) -> LLMResult:
        """流式生成（只用于作答）：content 片段逐段回调 on_delta；reasoning 片段只计长度。返回汇总后的 LLMResult。"""
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0, "stream": True,
                        "stream_options": {"include_usage": True}}
        reasoning = reasoning_param(self.reasoning_for(step))
        if reasoning:
            kwargs["extra_body"] = {"reasoning": reasoning}
        t0 = time.perf_counter()
        parts: list[str] = []
        r_chars = 0
        usage = None
        try:
            for chunk in self._client.chat.completions.create(**kwargs):
                usage = getattr(chunk, "usage", None) or usage
                for ch in getattr(chunk, "choices", None) or []:
                    delta = getattr(ch, "delta", None)
                    r_chars += len(getattr(delta, "reasoning_content", None) or "")
                    text = getattr(delta, "content", None) or ""
                    if text:
                        parts.append(text)
                        on_delta(text)
        except Exception as e:  # noqa: BLE001 - 统一脱敏后重抛
            raise LLMError(f"[{step}] LLM 调用失败：{type(e).__name__}: {self._redact(str(e))}") from None
        details = getattr(usage, "completion_tokens_details", None)
        res = LLMResult(content="".join(parts), prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0, reasoning_chars=r_chars,
                        latency_s=time.perf_counter() - t0)
        _report(step, model, res)
        return res


def _report(step: str, model: str, res: LLMResult) -> None:
    """把一次调用上报给调用日志（rag.usage_log）；无当前请求或日志出错都不影响本调用。"""
    try:
        from .. import usage_log
        usage_log.record_llm(step, model, res.prompt_tokens, res.completion_tokens, res.latency_s)
    except Exception:  # noqa: BLE001 - 日志失败绝不影响主流程
        pass


_EFFORTS = ("low", "medium", "high")


def reasoning_param(mode: str) -> dict | None:
    """COURSE_*_REASONING -> 胜算云 reasoning 对象；空值返回 None（不发送，用模型默认）。"""
    mode = (mode or "").strip().lower()
    if not mode:
        return None
    if mode == "disabled":
        return {"type": "disabled"}
    if mode in _EFFORTS:
        return {"type": "enabled", "effort": mode}
    raise ValueError(f"无效的 reasoning 设置：{mode}（可选 disabled / low / medium / high）")


def parse_response(resp, latency_s: float = 0.0) -> LLMResult:
    """OpenAI 兼容响应 -> LLMResult。reasoning_content 只记长度。"""
    msg = resp.choices[0].message
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        fn = tc.function
        try:
            args = json.loads(fn.arguments or "{}")
        except (TypeError, ValueError):
            args = {"_parse_error": True}
        if not isinstance(args, dict):
            args = {"_parse_error": True}
        tool_calls.append(ToolCall(getattr(tc, "id", None) or f"call_{i}", fn.name, args))
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return LLMResult(
        content=getattr(msg, "content", None) or "",
        tool_calls=tool_calls,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        reasoning_chars=len(getattr(msg, "reasoning_content", None) or ""),
        latency_s=latency_s,
    )


_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")


def parse_json_value(text: str):
    """容错解析：去掉 Markdown 围栏，必要时截取首个 {…} 或 […]。失败抛 ValueError。"""
    s = _FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                continue
    raise ValueError("回复中没有可解析的 JSON")


def parse_json_object(text: str) -> dict:
    obj = parse_json_value(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层不是对象")
    return obj


# ---------------------------------------------------------------- trace

@dataclass
class Trace:
    calls: list[dict] = field(default_factory=list)
    stages: dict[str, dict] = field(default_factory=dict)

    def totals(self) -> dict:
        keys = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "reasoning_chars", "latency_s")
        out = {k: sum(c.get(k, 0) for c in self.calls) for k in keys}
        out["latency_s"] = round(out["latency_s"], 3)
        out["calls"] = len(self.calls)
        steps: dict[str, dict] = {}
        for c in self.calls:
            st = steps.setdefault(c["step"], {"calls": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0})
            st["calls"] += 1
            st["latency_s"] = round(st["latency_s"] + c.get("latency_s", 0), 3)
            st["prompt_tokens"] += c.get("prompt_tokens", 0)
            st["completion_tokens"] += c.get("completion_tokens", 0)
        out["by_step"] = steps
        return out

    def to_dict(self) -> dict:
        return {"calls": self.calls, "totals": self.totals(), "stages": self.stages}


class TracingLLM:
    """包装任意 LLMClient，把每次调用的模型、步骤、token 与耗时记入 Trace（不记 prompt 正文）。"""

    def __init__(self, inner: LLMClient, trace: Trace):
        self.inner, self.trace = inner, trace

    def reasoning_for(self, step: str) -> str | None:
        """透传内层客户端的思考模式配置（内层没有则为 None），供选词轮次记录使用。"""
        fn = getattr(self.inner, "reasoning_for", None)
        return fn(step) if fn else None

    def chat(self, step: str, messages: list[dict], *, model: str,
             tools: list[dict] | None = None, json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        res = self.inner.chat(step, messages, model=model, tools=tools, json_mode=json_mode)
        return self._record(step, model, bool(tools), json_mode, res, t0)

    def __getattr__(self, name):
        if name == "chat_stream" and hasattr(self.inner, "chat_stream"):   # 内层支持流式时才对外暴露
            def chat_stream(step, messages, *, model, on_delta):
                t0 = time.perf_counter()
                res = self.inner.chat_stream(step, messages, model=model, on_delta=on_delta)
                return self._record(step, model, False, False, res, t0, stream=True)
            return chat_stream
        raise AttributeError(name)

    def _record(self, step, model, tools, json_mode, res, t0, stream=False) -> LLMResult:
        reasoning_for = getattr(self.inner, "reasoning_for", None)
        _report(step, model, res)                     # 流式/非流式都经这里，统一上报调用日志
        self.trace.calls.append({
            "step": step, "model": model, "tools": bool(tools), "json_mode": json_mode, "stream": stream,
            "reasoning": reasoning_for(step) if reasoning_for else None,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "reasoning_tokens": res.reasoning_tokens, "reasoning_chars": res.reasoning_chars,
            "latency_s": round(res.latency_s or (time.perf_counter() - t0), 3),
            "tool_calls": [tc.name for tc in res.tool_calls],
        })
        return res
