"""scripts/check_filter_model.py 的离线测试：全部 mock，不发真实请求，不读 .env。"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from rag.course import config, passage_filter, prompts
from rag.course.llm import LLMError, LLMNotConfigured, LLMResult

SENTINEL_KEY = "kb-SENTINEL-KEY-DO-NOT-LEAK-0123456789"
_SPEC = importlib.util.spec_from_file_location("check_filter_model", Path(__file__).resolve().parent.parent / "scripts" / "check_filter_model.py")
cfm = importlib.util.module_from_spec(_SPEC)
sys.modules["check_filter_model"] = cfm
_SPEC.loader.exec_module(cfm)


class FakeClient:
    """记录调用，返回预设结果或抛预设异常。"""

    def __init__(self, content=None, exc=None, reasoning_chars=0, reasoning_tokens=0, mode="disabled"):
        self.content, self.exc, self.mode = content, exc, mode
        self.calls = []
        self.res = dict(reasoning_chars=reasoning_chars, reasoning_tokens=reasoning_tokens)

    def reasoning_for(self, step):
        return self.mode

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "model": model, "json_mode": json_mode})
        if self.exc:
            raise self.exc
        return LLMResult(content=self.content, prompt_tokens=321, completion_tokens=12, latency_s=0.75, **self.res)


GOOD = json.dumps({"direct": ["S1"], "background": ["S2"]})


@pytest.fixture(autouse=True)
def filter_model_set(monkeypatch):
    monkeypatch.setattr(config, "COURSE_FILTER_MODEL_ENV", "deepseek/deepseek-v4-flash")
    monkeypatch.setattr(config, "COURSE_FILTER_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setitem(config.COURSE_REASONING, "filter", "disabled")


def _run(client, capsys):
    code = cfm.main(client_factory=lambda: client)
    out = capsys.readouterr().out
    return code, out, out.strip().splitlines()


# ---------------------------------------------------------------- 四种情况

def test_success_prints_report_and_pass(capsys):
    c = FakeClient(content=GOOD)
    code, out, lines = _run(c, capsys)
    assert code == 0 and lines[-1] == "PASS"
    assert "模型名: deepseek/deepseek-v4-flash" in out and "请求成功: 是" in out and "返回为合法 JSON: 是" in out
    assert "含 direct 键: 是  含 background 键: 是" in out
    assert "reasoning 长度: chars=0 tokens=0" in out and "prompt=321 completion=12" in out and "耗时: 0.75s" in out
    assert "reasoning 设置: disabled" in out


def test_invalid_json_prints_fail_with_reason(capsys):
    code, out, lines = _run(FakeClient(content="这不是 JSON，是一段话"), capsys)
    assert code == 1 and lines[-1].startswith("FAIL") and "不是合法 JSON" in lines[-1]
    assert "请求成功: 是" in out and "返回为合法 JSON: 否" in out and "PASS" not in out


@pytest.mark.parametrize("body,missing", [({"direct": ["S1"]}, "background"), ({"background": ["S2"]}, "direct"), ({}, "direct")])
def test_missing_key_prints_fail(capsys, body, missing):
    code, out, lines = _run(FakeClient(content=json.dumps(body)), capsys)
    assert code == 1 and lines[-1].startswith("FAIL") and f"缺少键 {missing}" in lines[-1]
    assert "返回为合法 JSON: 是" in out


def test_non_list_values_and_non_object_fail(capsys):
    code, _, lines = _run(FakeClient(content=json.dumps({"direct": "S1", "background": []})), capsys)
    assert code == 1 and "direct 不是数组" in lines[-1]
    code, _, lines = _run(FakeClient(content=json.dumps(["S1"])), capsys)
    assert code == 1 and "顶层不是对象" in lines[-1]


def test_filter_model_not_set_exits_nonzero_without_any_request(monkeypatch, capsys):
    monkeypatch.setattr(config, "COURSE_FILTER_MODEL_ENV", "")
    monkeypatch.setattr(config, "COURSE_FILTER_MODEL", "deepseek/deepseek-v4-pro")        # 回退值存在，但不算「设了」
    c = FakeClient(content=GOOD)
    code = cfm.main(client_factory=lambda: c)
    out = capsys.readouterr().out
    assert code != 0 and code == cfm.EXIT_NOT_SET and "未设置 COURSE_FILTER_MODEL" in out and "PASS" not in out
    assert c.calls == []                                                                     # 没设就不发请求


# ---------------------------------------------------------------- 请求内容：与线上筛选一致，只发 1 次

def test_sends_exactly_one_request_with_real_filter_prompts_and_json_mode(capsys):
    c = FakeClient(content=GOOD)
    _run(c, capsys)
    assert len(c.calls) == 1
    call = c.calls[0]
    assert call["step"] == "filter" and call["json_mode"] is True and call["model"] == "deepseek/deepseek-v4-flash"
    sysm, user = (m["content"] for m in call["messages"])
    assert sysm == prompts.FILTER_SYSTEM                                                     # 筛选环节的真实系统提示（原生 JSON 模式，不加回退后缀）
    assert user.startswith("问题：" + cfm.QUESTION) and len(re.findall(r"^\[S\d\] 章节路径：", user, flags=re.M)) == 3


def test_request_messages_identical_to_pipeline_judge_batch(ctx_free=None):
    """脚本的消息必须与 passage_filter._judge_batch 对同样的块构造出的完全一致，防止两边提示词悄悄分叉。"""
    from rag.course.recall import Recalled
    items = [Recalled(f"id{n}", "p", path, text, "body", ["P1"], []) for n, (ref, path, text) in enumerate(cfm.FAKE_CHUNKS, 1)]
    ref_of = {it.chunk_id: ref for it, (ref, _, _) in zip(items, cfm.FAKE_CHUNKS)}
    seen = {}

    class Spy:
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            seen["messages"] = messages
            return LLMResult(content=GOOD)
    passage_filter._judge_batch(cfm.QUESTION, items, ref_of, Spy(), "m", True, {"parse_failures": 0})
    assert seen["messages"] == cfm.build_messages()


def test_real_client_receives_reasoning_param_from_course_filter_reasoning(monkeypatch):
    """用真实的 OpenAICompatClient（假的 openai 包）核对：json_object 与 reasoning 参数确实随请求发出。"""
    from types import SimpleNamespace as NS
    from rag.course import llm
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            def create(**kw):
                seen.clear(); seen.update(kw)
                msg = NS(content=GOOD, tool_calls=None, reasoning_content=None)
                return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=5, completion_tokens=3, completion_tokens_details=NS(reasoning_tokens=0)))
            self.chat = NS(completions=NS(create=create))
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=FakeOpenAI))
    for setting, expected in (("disabled", {"type": "disabled"}), ("low", {"type": "enabled", "effort": "low"})):
        monkeypatch.setitem(config.COURSE_REASONING, "filter", setting)
        code = cfm.main(client_factory=lambda: llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY))
        assert seen["response_format"] == {"type": "json_object"} and seen["extra_body"] == {"reasoning": expected}
        assert seen["model"] == "deepseek/deepseek-v4-flash" and seen["temperature"] == 0 and code == 0


# ---------------------------------------------------------------- reasoning 与 token 的呈现

def test_reasoning_present_while_disabled_is_a_failure(capsys):
    code, out, lines = _run(FakeClient(content=GOOD, reasoning_chars=88, reasoning_tokens=20), capsys)
    assert code == 1 and lines[-1].startswith("FAIL") and "reasoning=disabled" in lines[-1]
    assert "reasoning 长度: chars=88 tokens=20" in out


def test_reasoning_allowed_when_not_disabled(monkeypatch, capsys):
    monkeypatch.setitem(config.COURSE_REASONING, "filter", "low")
    code, out, lines = _run(FakeClient(content=GOOD, reasoning_chars=88, reasoning_tokens=20, mode="low"), capsys)
    assert code == 0 and lines[-1] == "PASS" and "reasoning 设置: low" in out


def test_reasoning_unset_is_reported(monkeypatch, capsys):
    code, out, _ = _run(FakeClient(content=GOOD, mode=""), capsys)
    assert "不发送 reasoning 参数" in out


# ---------------------------------------------------------------- 出错只输出异常类型名，不泄露 key / 请求头 / 异常正文

@pytest.mark.parametrize("exc", [
    LLMError(f"[filter] LLM 调用失败：APITimeoutError: Request timed out. Authorization: Bearer {SENTINEL_KEY} key={SENTINEL_KEY}"),
    RuntimeError(f"401 Authorization: Bearer {SENTINEL_KEY} headers={{'x-api-key': '{SENTINEL_KEY}'}} body=secret-detail"),
])
def test_errors_print_only_exception_type_name(capsys, exc):
    code, out, lines = _run(FakeClient(exc=exc), capsys)
    assert code == 1 and lines[-1].startswith("FAIL") and "请求成功: 否" in out
    assert SENTINEL_KEY not in out and "Bearer" not in out and "Authorization" not in out and "secret-detail" not in out and "timed out" not in out
    name = "APITimeoutError" if isinstance(exc, LLMError) else "RuntimeError"
    assert f"异常类型: {name}" in out


def test_client_creation_errors_print_only_type_name(capsys):
    def boom():
        raise LLMNotConfigured(f"COURSE_LLM_API_KEY 未配置 {SENTINEL_KEY}")
    code = cfm.main(client_factory=boom)
    out = capsys.readouterr().out
    assert code == 1 and "LLMNotConfigured" in out and SENTINEL_KEY not in out and "FAIL" in out

    def boom2():
        raise ValueError(f"bad {SENTINEL_KEY}")
    code = cfm.main(client_factory=boom2)
    out = capsys.readouterr().out
    assert code == 1 and "ValueError" in out and SENTINEL_KEY not in out


def test_script_source_never_references_env_file_or_key_values():
    src = Path(cfm.__file__).read_text(encoding="utf-8")
    assert "API_KEY" not in src.replace("COURSE_LLM_API_KEY / COURSE_LLM_API_BASE", "").replace("COURSE_LLM_API_BASE / COURSE_LLM_API_KEY", "")
    assert ".env" in src and "open(" not in src                                              # 只通过 rag.config 加载 .env，不自己读文件


def test_filter_model_env_marker_does_not_change_existing_fallback_behavior():
    """COURSE_FILTER_MODEL 仍在未设置时回退到 COURSE_LLM_MODEL（脚本另用 _ENV 判断是否显式设置）。"""
    import os
    if "COURSE_FILTER_MODEL" not in os.environ:
        import importlib
        import rag.course.config as c
        assert c.COURSE_FILTER_MODEL == (c.COURSE_FILTER_MODEL_ENV or c.COURSE_LLM_MODEL)
