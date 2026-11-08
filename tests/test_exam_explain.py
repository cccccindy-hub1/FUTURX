"""题库命中后按意图讲解 / 指代上一题（借助对话记忆的 standalone 再查题库）/ “这道题”不触发找题。全部 mock。"""
from __future__ import annotations

import json

from rag.course import prompts
from rag.course.llm import LLMResult
from rag.router import rule_route
from rag.unified import UnifiedService
from test_course_pipeline import ScriptedLLM, ctx  # noqa: F401
from test_exam_solve import JUDGE, Counting, _course_script, course, make_bank  # noqa: F401


class RewriteLLM:
    """Cindy 的追问改写（rag.enhance.contextualize_query）用的替身：把指代补全成带题干的完整问题。"""

    def __init__(self, standalone):
        self.standalone, self.calls = standalone, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append(step)
        return LLMResult(content=self.standalone)


def _answer_call(llm):
    return next(c for c in llm.calls if c["step"] == "answer")


def test_explain_on_a_bank_hit_gives_the_reference_answer_then_a_course_explanation(course):
    router, llm = Counting("exam"), ScriptedLLM(_course_script())
    s = UnifiedService(router_client=router, exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    out = s.ask(f"这道题怎么理解（{JUDGE}）")
    assert router.calls == []                                                     # 题库命中：路由不调
    assert out["routes"] == ["exam", "course"] and out["route"] == "course" and out["explain"] is True and out["status"] == "ok"
    assert out["exam"]["questions"][0]["answer"] == "正确" and out["trace"]["prematch"]["hit"] == "ex:t:judge:1"
    call = _answer_call(llm)
    assert call["messages"][0]["content"] == prompts.ANSWER_SYSTEM + prompts.ANSWER_EXPLAIN_HINT
    assert "参考答案：正确" in call["messages"][1]["content"] and "评分要点与解析：个人权益" in call["messages"][1]["content"]
    assert [c["step"] for c in llm.calls][:3] == ["select", "filter", "answer"]          # 之后只有追问那一次小调用


def test_explain_stream_sends_both_tags_and_the_exam_payload_in_done(course):
    router, llm = Counting("exam"), ScriptedLLM(_course_script())
    s = UnifiedService(router_client=router, exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    ev = list(s.ask_stream(f"从课程角度讲解一下：{JUDGE}"))
    assert ev[0]["type"] == "route" and ev[0]["routes"] == ["exam", "course"] and router.calls == []
    done = next(e for e in ev if e["type"] == "done")
    assert done["explain"] is True and done["routes"] == ["exam", "course"] and done["exam"]["questions"][0]["answer"] == "正确"
    assert done["practice"] == [] and _answer_call(llm)["messages"][0]["content"].endswith(prompts.ANSWER_EXPLAIN_HINT)


def test_answer_only_hit_is_unchanged_zero_calls():
    router = Counting("course")
    s = UnifiedService(router_client=router, exam_bank=make_bank())
    out = s.ask(f"判断这道题是否正确：{JUDGE}")
    assert router.calls == [] and out["routes"] == ["exam"] and "explain" not in out


HISTORY = [{"role": "user", "content": f"这道题怎么理解（{JUDGE}）"}, {"role": "assistant", "content": "正确。……"}]


def test_pronoun_follow_up_is_rewritten_then_matched_and_explained(course):
    router, rw, llm = Counting("exam"), RewriteLLM(f"从课程角度讲解这道题：{JUDGE}"), ScriptedLLM(_course_script())
    s = UnifiedService(router_client=router, exam_bank=make_bank(), course_ctx=course, course_llm=llm, lib_llm=rw)
    out = s.ask("从课程角度，帮我讲解这道题", history=HISTORY)
    assert rw.calls[0] == "contextualize" and "contextualize" not in rw.calls[1:] and router.calls == []   # 改写一次；命中题库后路由不调
    assert out["routes"] == ["exam", "course"] and out["explain"] is True and out["exam"]["questions"][0]["answer"] == "正确"
    assert "参考答案：正确" in _answer_call(llm)["messages"][-1]["content"]


def test_pronoun_follow_up_stream_is_also_matched(course):
    rw = RewriteLLM(f"讲解这道题：{JUDGE}")
    s = UnifiedService(router_client=Counting("exam"), exam_bank=make_bank(), course_ctx=course,
                       course_llm=ScriptedLLM(_course_script()), lib_llm=rw)
    ev = list(s.ask_stream("讲讲这道题", history=HISTORY))
    done = next(e for e in ev if e["type"] == "done")
    assert ev[0]["routes"] == ["exam", "course"] and done["explain"] is True


def test_pronoun_follow_up_asking_only_for_the_answer_returns_the_bank_answer():
    s = UnifiedService(router_client=Counting("exam"), exam_bank=make_bank(), lib_llm=RewriteLLM(f"这道题的答案是什么：{JUDGE}"))
    out = s.ask("那这道题答案是什么", history=HISTORY)
    assert out["routes"] == ["exam"] and out["exam"]["solve"] is True and out["exam"]["questions"][0]["answer"] == "正确"


def test_ordinary_course_question_with_explain_words_keeps_the_verbatim_prompt(course):
    llm = ScriptedLLM(_course_script())
    s = UnifiedService(router_client=Counting("course"), exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    s.ask("请解释数据资产入表的主要步骤和会计处理方式")
    assert _answer_call(llm)["messages"][0]["content"] == prompts.ANSWER_SYSTEM


def test_this_question_alone_does_not_trigger_finding_questions():
    for q in ("从课程角度，帮我讲解这道题", "这道题怎么理解", "上面那道题为什么选A", "讲讲这题"):
        assert rule_route(q).route != "exam", q
    for q in ("出三道判断题", "来几道第三章选择题", "数据资产有哪些练习题", "给我5道题"):
        assert rule_route(q).route == "exam", q


def test_web_shows_the_explanation_text():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "rag" / "web" / "app.js").read_text(encoding="utf-8")
    assert "examQs.length && !ev.explain ?" in js


def test_view_answer_button_is_a_small_grey_pill_and_the_answer_is_an_indented_tinted_block():
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "rag" / "web" / "style.css").read_text(encoding="utf-8")
    seg = css[css.index("/* “查看答案”"):css.index("/* ---- 顶栏：知识图谱入口")]
    summary = seg[seg.index(".exam-answer summary {"):seg.index("}", seg.index(".exam-answer summary {"))]
    assert "font-size: 12px" in summary and "font-weight: 400" in summary and "color: var(--ink-soft)" in summary and "border-radius: 999px" in summary
    row = seg[seg.index(".exam-answer-row {"):seg.index("}", seg.index(".exam-answer-row {"))]
    assert "font-size: 13px" in row and "background: var(--accent-soft)" in row and "margin-left" in row
