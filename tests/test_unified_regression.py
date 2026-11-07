"""回归：判例库 / 名言库 / 组合问答接入后，纯课程、试卷、论文、拒答四类问题的行为不能变。

做法：同一个 mock 路由 + 同一份脚本，分别用「没有判例 / 名言服务」和「挂着判例 / 名言服务」两种配置跑 /auto/ask 与 /auto/ask/stream，
逐字比较输出（去掉耗时字段）并比较各环节 LLM 调用次数。判例 / 名言服务一旦被误触发（检索、作答、挑段落），假服务会直接报错。
全部 mock，不调用真实模型。
"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag import config
from rag.course import config as cc
from rag.course.llm import LLMResult
from rag.unified import UnifiedService, router as unified_router
from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: F401  (ctx 为课程夹具)


@pytest.fixture(autouse=True)
def single_mode(request, monkeypatch):
    if "ctx" in request.fixturenames:
        request.getfixturevalue("ctx")
    monkeypatch.setattr(cc, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(cc, "COURSE_SELECT_FAST_PATH", False)
    monkeypatch.setattr(cc, "COURSE_FILTER_SKIP_MAX", 0)


class Boom:
    """判例 / 名言索引的替身：被碰到就失败——纯课程 / 试卷 / 论文 / 拒答题不应触发它们（检索、选案、取材都不行）。"""
    refs = []
    by_id = {}

    def __getattr__(self, name):
        raise AssertionError(f"这类问题不应访问判例 / 名言库（访问了 {name}）")


def route_client(routes, count=1, chapter=None):
    calls = []

    class R:
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            calls.append(step)
            return LLMResult(content=json.dumps({"routes": routes, "complexity": "simple", "count": count, "chapter": chapter, "reason": "t"}), latency_s=0.1)
    r = R()
    r.calls = calls
    return r


def course_script():
    def filt(step, msgs):
        refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
        return jtxt({"direct": refs[:1], "background": refs[1:]})
    return [jtxt({"question_concepts": ["C01-0001"], "related_concepts": ["C01-0003"], "reason": "r"}), filt,
            LLMResult(content="数据要素是参与价值创造的数据资源[S1]。", prompt_tokens=10, completion_tokens=5)]


class Streaming(ScriptedLLM):
    def chat_stream(self, step, messages, *, model, on_delta):
        res = self.chat(step, messages, model=model)
        for i in range(0, len(res.content), 7):
            on_delta(res.content[i:i + 7])
        return res


class FakeExam:
    """最小的试卷库替身：与线上 ExamBank 的接口一致（search / by_concepts），不依赖题库文件。"""

    def by_concepts(self, names, k, *, chapter=None, qtypes=(), basic_only=False, exclude_ids=()):      # 与 ExamBank.by_concepts 同签名
        return []

    def search(self, q):
        return {"status": "no_match", "strategy": "none", "intent": {"want_answer": False}, "items": []}


class Adaptive:
    def run(self, q, top_k, complexity, **kwargs):
        return {"status": "ok", "answer": "论文答案", "sources": [{"source": "a.pdf"}], "contexts": ["c"], "trace": {"path": "simple"}}


def strip_times(x):
    """去掉所有耗时字段，剩下的必须逐字一致。"""
    if isinstance(x, dict):
        # conversation_id 是每次请求新建的会话号，不属于「行为是否一致」的比较范围
        return {k: strip_times(v) for k, v in x.items()
                if not (k.endswith("_s") or k in ("latency_s", "total_s", "calls", "totals", "stages",
                                                  "timings", "conversation_id"))}
    if isinstance(x, list):
        return [strip_times(v) for v in x]
    return x


def make(ctx_, with_new_libs, route, course_llm=None, exam=None):
    kw = dict(router_client=route, course_ctx=ctx_, course_llm=course_llm, exam_bank=exam)
    if with_new_libs:
        kw.update(case_index=Boom(), quote_index=Boom())
    svc = UnifiedService(**kw)
    svc.adaptive = Adaptive()
    if with_new_libs:                                    # 构造时 CaseRetriever / QuoteRetriever 已建好；它们的检索入口同样不许被调用
        for r in (svc.case_retriever, svc.quote_retriever):
            if r is not None:
                r.retrieve = lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应检索"))
    return svc


def sse(resp):
    return [json.loads(b[6:]) for b in resp.text.split("\n\n") if b.startswith("data: ")]


def app_for(svc):
    app = FastAPI()
    app.state.unified = svc
    app.include_router(unified_router)
    return TestClient(app)


# ---------------------------------------------------------------- 纯课程题

def test_pure_course_question_identical_with_and_without_case_quote_services(ctx):
    outs, steps = [], []
    for with_new in (False, True):
        llm = ScriptedLLM(course_script())
        rc = route_client(["course"])
        out = make(ctx, with_new, rc, llm).ask("什么是数据要素")
        outs.append(strip_times(out))
        steps.append((rc.calls, [c["step"] for c in llm.calls if c["step"] != "followup"]))   # 追问不计入作答管道
    assert outs[0] == outs[1] and steps[0] == steps[1] == (["route"], ["select", "filter", "answer"])         # 输出逐字一致；路由 1 + 选概念 1 + 筛选 1 + 作答 1 = 4 次
    assert outs[0]["route"] == "course" and outs[0]["routes"] == ["course"] and outs[0]["status"] == "ok" and outs[0]["answer"] == "数据要素是参与价值创造的数据资源[1]。"
    assert outs[0]["sources"][0]["n"] == 1 and "kind" not in outs[0]["sources"][0] or outs[0]["sources"][0].get("kind") in (None, "course")


def test_pure_course_stream_identical_events_with_and_without_new_libs(ctx):
    runs = []
    for with_new in (False, True):
        svc = make(ctx, with_new, route_client(["course"]), Streaming(course_script()))
        ev = sse(app_for(svc).post("/auto/ask/stream", json={"question": "什么是数据要素"}))
        runs.append([strip_times(e) for e in ev])
    assert runs[0] == runs[1]
    kinds = [e["type"] for e in runs[0]]
    assert kinds[0] == "route" and kinds[-1] == "done" and {"stage", "delta", "materials"} <= set(kinds)
    assert "practice" in runs[0][-1] and runs[0][-1]["answer"] == "数据要素是参与价值创造的数据资源[1]。"           # 线上的 practice 字段仍在
    assert all(e.get("kind") in (None, "course") for e in runs[0] if e["type"] == "materials")


def test_course_stream_done_keeps_practice_when_exam_bank_loaded(ctx):
    svc = make(ctx, True, route_client(["course"]), Streaming(course_script()), exam=FakeExam())
    done = sse(app_for(svc).post("/auto/ask/stream", json={"question": "什么是数据要素"}))[-1]
    assert done["type"] == "done" and done["practice"] == [] and done["status"] == "ok"


# ---------------------------------------------------------------- 试卷题

def test_exam_question_goes_to_exam_bank_and_never_touches_case_quote_or_models(ctx):
    outs = []
    for with_new in (False, True):
        llm = ScriptedLLM([])
        rc = route_client(["exam"])
        out = make(ctx, with_new, rc, llm, exam=FakeExam()).ask("出三道数据资产的练习题")
        outs.append(strip_times(out))
        assert llm.calls == [] and rc.calls == ["route"]                           # 试卷题只花 1 次路由调用
    assert outs[0] == outs[1] and outs[0]["route"] == "exam" and outs[0]["status"] == "no_match" and "exam" in outs[0]


def test_exam_stream_done_keeps_exam_field(ctx):
    for with_new in (False, True):
        svc = make(ctx, with_new, route_client(["exam"]), ScriptedLLM([]), exam=FakeExam())
        ev = sse(app_for(svc).post("/auto/ask/stream", json={"question": "出三道数据资产的练习题"}))
        assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["route"] == "exam" and "exam" in ev[-1] and ev[-1]["status"] == "no_match"


def test_exam_unavailable_when_no_bank_is_unchanged(ctx):
    out = make(ctx, True, route_client(["exam"]), ScriptedLLM([])).ask("出三道题")
    assert out["status"] == "unavailable" and "试卷库未加载" in out["answer"]


# ---------------------------------------------------------------- 论文题

def test_thesis_question_identical_and_never_touches_new_libs(ctx):
    outs = []
    for with_new in (False, True):
        llm = ScriptedLLM([])
        rc = route_client(["thesis"])
        out = make(ctx, with_new, rc, llm).ask("怎么写开题报告")
        outs.append(strip_times(out))
        # 论文库本身不调这个 LLM；追问（followup）是答案之后的可选小调用，允许发生
        assert [c["step"] for c in llm.calls] == ["followup"] or llm.calls == []
    assert outs[0] == outs[1] and outs[0]["route"] == "thesis" and outs[0]["answer"] == "论文答案" and outs[0]["sources"] == [{"source": "a.pdf"}]


def test_thesis_stream_puts_full_result_in_done(ctx):
    for with_new in (False, True):
        ev = sse(app_for(make(ctx, with_new, route_client(["thesis"]), ScriptedLLM([]))).post("/auto/ask/stream", json={"question": "怎么写开题报告"}))
        assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["answer"] == "论文答案" and ev[-1]["route"] == "thesis"


# ---------------------------------------------------------------- 拒答题

def test_off_topic_refusal_identical_costs_one_call_and_touches_nothing(ctx):
    outs = []
    for with_new in (False, True):
        llm = ScriptedLLM([])
        rc = route_client(["off_topic"])
        out = make(ctx, with_new, rc, llm).ask("今天天气怎么样")
        outs.append(strip_times(out))
        assert llm.calls == [] and rc.calls == ["route"]
    assert outs[0] == outs[1] and outs[0]["status"] == "refused" and outs[0]["answer"] == config.ROUTER_REFUSAL_TEXT and outs[0]["sources"] == []


def test_off_topic_stream_is_route_then_done(ctx):
    ev = sse(app_for(make(ctx, True, route_client(["off_topic"]), ScriptedLLM([]))).post("/auto/ask/stream", json={"question": "今天天气怎么样"}))
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["status"] == "refused"


def test_policy_route_without_the_policy_library_is_unavailable_and_touches_no_other_library(ctx):
    llm = ScriptedLLM([])
    out = make(ctx, True, route_client(["policy"]), llm).ask("数据二十条第几条规定")
    assert out["status"] == "unavailable" and "政策库未加载" in out["answer"] and llm.calls == []


# ---------------------------------------------------------------- 路由本身：旧格式、旧字段、旧 40 题

def test_old_style_router_json_with_single_route_still_dispatches_the_same_way(ctx):
    class Old:
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            return LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"}))
    llm = ScriptedLLM(course_script())
    out = make(ctx, True, Old(), llm).ask("什么是数据要素")
    assert out["route"] == "course" and out["routes"] == ["course"] and out["status"] == "ok" and [c["step"] for c in llm.calls if c["step"] != "followup"] == ["select", "filter", "answer"]


def test_old_style_router_json_is_honored_as_the_models_decision_not_the_rule_fallback():
    """旧格式只给 route：必须被当作模型的判断（source=llm）。用一个规则兜底会判成别的类别的问题来区分两者。"""
    from rag.router import classify, rule_route

    class Old:
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            return LLMResult(content=json.dumps({"route": "thesis", "complexity": "complex", "reason": "t"}))
    q = "什么是数据要素"                                    # 规则兜底会判 course
    assert rule_route(q).route == "course"
    d = classify(q, Old())
    assert (d.source, d.routes, d.route, d.complexity, d.llm_error) == ("llm", ["thesis"], "thesis", "complex", "")


def test_unavailable_new_libs_do_not_affect_old_routes_and_return_unavailable_for_their_own(ctx):
    svc = UnifiedService(router_client=route_client(["case"]), course_ctx=ctx, course_llm=ScriptedLLM([]))        # 没挂判例服务
    out = svc.ask("举三个判例")
    assert out["status"] == "unavailable" and "判例" in out["answer"]
    svc = UnifiedService(router_client=route_client(["quote"]), course_ctx=ctx, course_llm=ScriptedLLM([]))
    assert svc.ask("总书记怎么说")["status"] == "unavailable"


def test_rule_fallback_for_old_categories_unchanged_when_router_llm_fails(ctx):
    class Down:
        def chat(self, *a, **k):
            raise TimeoutError("t")
    for q, route in (("什么是数据要素", "course"), ("怎么写开题报告", "thesis"), ("出几道题", "exam"), ("今天天气怎么样", "off_topic")):
        out = UnifiedService(router_client=Down(), course_ctx=ctx, course_llm=ScriptedLLM([])).ask(q) if route == "off_topic" else None
        from rag.router import classify
        assert classify(q, Down()).route == route
        if out:
            assert out["status"] == "refused"
