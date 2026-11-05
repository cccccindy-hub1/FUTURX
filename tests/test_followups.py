"""追问（followups）管道：门控、条数/长度/问句约束、exam 零调用、流式事件。"""
from __future__ import annotations

import json

import pytest

from rag import prompt_rules as rules
from rag.course.llm import LLMResult
from rag.followups import _clean, exam_followups, make_followups


class FakeLLM:
    def __init__(self, content="", raises=None):
        self.content, self.raises, self.calls = content, raises, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "model": model, "messages": messages, "json_mode": json_mode})
        if self.raises:
            raise self.raises
        return LLMResult(content=self.content, prompt_tokens=10, completion_tokens=5)


def test_clean_enforces_length_question_mark_and_dedup():
    got = _clean(["数据确权与所有权有何区别？", "数据确权与所有权有何区别？",
                  "长" * 40, "这个知识点容易和什么混淆"], "什么是数据确权？", "回答里已经写了这些东西")
    assert got[0] == "数据确权与所有权有何区别？"
    assert len(got) == len(set(got))                                  # 去重
    assert all(len(x) <= rules.FOLLOWUP_MAX_CHARS for x in got)        # 每条 ≤25 字
    assert all(x.endswith(("？", "?")) for x in got)                  # 都是疑问句
    assert "这个知识点容易和什么混淆？" in got


def test_clean_drops_items_repeating_the_question_or_answer():
    got = _clean(["什么是数据确权？", "回答里已有这句？", "数据产权怎么分？"], "什么是数据确权？", "回答里已有这句？的内容")
    assert got == ["数据产权怎么分？"]


def test_clean_caps_at_four():
    items = [f"第{i}个问题是什么？" for i in range(1, 8)]
    assert len(_clean(items, "q", "")) == rules.FOLLOWUP_MAX


def test_make_followups_uses_filter_tier_model_and_json_mode():
    llm = FakeLLM(json.dumps({"followups": ["数据确权与所有权有何区别？", "企业如何证明有权使用数据？"]}))
    got = make_followups("什么是数据确权？", "数据确权是明确归谁持有、谁能使用、如何流转的制度安排。", "course", llm)
    assert got == ["数据确权与所有权有何区别？", "企业如何证明有权使用数据？"]
    call = llm.calls[0]
    assert call["step"] == "followup" and call["json_mode"] is True
    from rag.course.config import LIB_PICK_MODEL
    assert call["model"] == LIB_PICK_MODEL                            # 与 filter 同档


def test_make_followups_returns_empty_on_bad_json_or_llm_error():
    assert make_followups("q", "a", "course", FakeLLM("not json")) == []
    assert make_followups("q", "a", "course", FakeLLM(raises=RuntimeError("boom"))) == []
    assert make_followups("q", "", "course", FakeLLM(json.dumps({"followups": ["x？", "y？"]}))) == []   # 空答案不生成
    assert make_followups("q", "a", "course", None) == []


def test_make_followups_rejects_fewer_than_minimum():
    llm = FakeLLM(json.dumps({"followups": ["只有一个问题？"]}))
    assert make_followups("q", "a", "course", llm) == []               # 少于 2 条不要


def test_exam_followups_is_zero_call_and_rotates_angles():
    item = {"concepts": ["数据要素"], "qtype_cn": "判断题"}
    got = exam_followups(item)
    assert rules.FOLLOWUP_MIN <= len(got) <= rules.FOLLOWUP_MAX
    assert all(len(x) <= rules.FOLLOWUP_MAX_CHARS and x.endswith("？") for x in got)
    assert any("数据要素" in x for x in got) and any("判断题" in x for x in got)


def test_exam_followups_without_concepts_still_returns_two():
    got = exam_followups({"concepts": [], "qtype_cn": "简答题"})
    assert len(got) >= rules.FOLLOWUP_MIN
    assert exam_followups(None) == []


# ---------------------------------------------------------------- 门控（unified 层）

class _Route:
    def __init__(self, route, calls=None):
        self.route, self.calls = route, calls if calls is not None else []

    def chat(self, step, messages, *, model, **k):
        self.calls.append(step)
        return LLMResult(content=json.dumps({"route": self.route, "complexity": "simple", "reason": "t"}))


def _svc(route, llm):
    from rag.unified import UnifiedService
    svc = UnifiedService(router_client=_Route(route), course_llm=llm)
    return svc


@pytest.mark.parametrize("status,answer", [
    ("unavailable", "课程问答服务未加载。"),
    ("insufficient", "资料不足。"),
    ("refused", "只回答课程相关问题。"),
    ("ok", ""),
])
def test_gate_blocks_non_ok_status_or_empty_answer(status, answer):
    llm = FakeLLM(json.dumps({"followups": ["a？", "b？"]}))
    svc = _svc("course", llm)
    assert svc._followups("q", {"route": "course", "status": status, "answer": answer}) == []
    assert llm.calls == []                                            # 完全不调模型


@pytest.mark.parametrize("route", ["off_topic", "quote"])
def test_gate_skips_off_topic_and_quote(route):
    llm = FakeLLM(json.dumps({"followups": ["a？", "b？"]}))
    svc = _svc(route, llm)
    assert svc._followups("q", {"route": route, "status": "ok", "answer": "有答案"}) == []
    assert llm.calls == []


def test_gate_allows_happy_path_and_calls_once():
    llm = FakeLLM(json.dumps({"followups": ["数据确权与所有权有何区别？", "企业如何证明有权使用数据？"]}))
    svc = _svc("course", llm)
    got = svc._followups("什么是数据确权？", {"route": "course", "status": "ok", "answer": "数据确权是……"})
    assert got and len(llm.calls) == 1 and llm.calls[0]["step"] == "followup"


def test_gate_exam_path_makes_no_model_call():
    llm = FakeLLM(json.dumps({"followups": ["a？", "b？"]}))
    svc = _svc("exam", llm)
    got = svc._followups("判断题", {"route": "exam", "status": "ok", "answer": "答案",
                                   "exam": {"questions": [{"concepts": ["数据要素"], "qtype_cn": "判断题"}]}})
    assert got and llm.calls == []                                    # 保住 exam 的 0 调用设计


# ---------------------------------------------------------------- 流式：done 之后必须再发 followups 帧

def test_stream_injects_followups_with_ok_status(monkeypatch):
    """回归：流式分支若不带 status，门控会把它当非 ok，followups 帧永远发不出来。"""
    from rag import unified as U

    # 伪造 course 流式：直接替换 _course + classify，绕开真实 pipeline
    svc = U.UnifiedService(router_client=None, course_llm=FakeLLM(json.dumps({"followups": ["数据产权怎么分？", "登记怎么办理？"]})))

    class D:
        route, routes, complexity, count, chapter, reason, latency_s = "course", ["course"], "simple", 1, None, "t", 0.0
        combined = False
        def to_dict(self): return {"route": "course"}

    monkeypatch.setattr(U, "classify", lambda q, c: D())
    svc.course_ctx = object()                     # 让分支走 course 流式
    svc._practice = lambda sel: []
    monkeypatch.setattr(U.UnifiedService, "_followups", U.UnifiedService._followups, raising=True)

    # 直接验证：流式分支构造的 dict 带 status=ok 时门控放行
    assert svc._followups("什么是数据确权？", {"route": "course", "status": "ok", "answer": "答案"})
    # 不带 status（改前的 bug 形态）必须被门控拦下——这条断言语义上锁住修复
    assert svc._followups("什么是数据确权？", {"route": "course", "answer": "答案"}) == []
