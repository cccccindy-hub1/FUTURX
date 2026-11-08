"""全局路由 + 自适应 RAG + 统一入口：全部离线（mock 路由 LLM / 检索 / 生成），不读取 .env 里的真实 key。"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag import config
from test_course_pipeline import ctx  # noqa: F401  (课程夹具)
from rag.adaptive import AdaptiveRAG
from rag.course.llm import LLMResult, OpenAICompatClient
from rag.query import RAGService
from rag.retrieval import RetrievalResult
from rag.router import classify, rule_route
from rag.unified import UnifiedService, router as unified_router

SECRET = "sk-SECRET-KEY-123456"


class RouterLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "model": model, "json_mode": json_mode})
        if self.exc:
            raise self.exc
        return LLMResult(content=self.reply if isinstance(self.reply, str) else json.dumps(self.reply))


def route_llm(route, cx="simple"):
    return RouterLLM({"route": route, "complexity": cx, "reason": "t"})


# ---------------------------------------------------------------- 假检索 / 假生成

class FakeChat:
    """记录调用；按 system 内容区分 改写 / 拆分 / 作答。"""

    def __init__(self, subs=None, rewrite="改写后的查询"):
        self.subs, self.rewrite, self.log = subs, rewrite, []
        self.completions = self
        self.chat = self

    def create(self, **kw):
        sysmsg = kw["messages"][0]["content"]
        if "检索规划器" in sysmsg:
            kind, out = "decompose", json.dumps({"sub_questions": self.subs or []}, ensure_ascii=False)
        elif "查询改写器" in sysmsg:
            kind, out = "rewrite", self.rewrite
        else:
            kind, out = "answer", "最终回答[来源1]"
        self.log.append(kind)
        self.kwargs = getattr(self, "kwargs", []) + [kw]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


class FakeRAG:
    """retrieve_result 按 scripted 返回；mode / llm 与 RAGService 同名属性。"""
    prepare_from = RAGService.prepare_from

    def __init__(self, results, mode="dense", llm=None):
        self.results, self.mode, self.llm, self.queries = list(results), mode, llm, []

    def retrieve_result(self, query, top_k=4):
        self.queries.append(query)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def res(score=None, dist=0.2, doc="扎根理论资料", rid="d1"):
    dbg = {"rerank_scores": {rid: score}, "rerank_s": 0.01} if score is not None else {}
    return RetrievalResult(ids=[rid], documents=[doc], metadatas=[{"source": "a.pdf", "section": "s"}],
                           distances=[dist], scores=[score or 0.0], debug=dbg)


@pytest.fixture(autouse=True)
def _thresholds(monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.0, raising=False)


# ---------------------------------------------------------------- 路由

@pytest.mark.parametrize("route,cx", [("thesis", "simple"), ("thesis", "complex"), ("course", "simple"),
                                      ("policy", "simple"), ("off_topic", "simple")])
def test_llm_routes_all_four_categories(route, cx):
    llm = route_llm(route, cx)
    d = classify("任意问题", llm)
    assert (d.route, d.complexity, d.source) == (route, cx, "llm")
    assert llm.calls == [{"step": "route", "model": config.ROUTER_MODEL, "json_mode": True}]


def test_complexity_forced_simple_for_non_thesis():
    assert classify("q", route_llm("course", "complex")).complexity == "simple"


@pytest.mark.parametrize("bad", [RouterLLM(exc=TimeoutError("boom")), RouterLLM(reply="not json"),
                                 RouterLLM(reply={"route": "nonsense"}), None])
def test_failure_falls_back_to_rules(bad):
    d = classify("扎根理论怎么写进论文？", bad)
    assert d.source == "rules" and d.route == "thesis" and d.llm_error


def test_rules_cannot_decide_means_refuse():
    d = classify("今天吃什么好呢", RouterLLM(exc=RuntimeError("x")))
    assert d.route == "off_topic" and d.source == "rules_refuse"
    assert rule_route("数据安全法的合规要求").route == "policy"
    assert rule_route("什么是数据要素").route == "course"


def test_rule_policy_chapter_questions_go_to_course_but_clause_lookups_stay_policy():
    assert rule_route("数据政策的演进脉络分为哪几个阶段").route == "course"
    assert rule_route("如何对数据政策进行评估").route == "course"
    assert rule_route("数据政策的传导路径有哪些").route == "course"
    assert rule_route("数据安全法第几条规定了数据分类分级").route == "policy"


def test_rule_generic_words_alone_do_not_make_policy():
    assert rule_route("学校的规定通知我几点下课").route == "off_topic"


# ---------------------------------------------------------------- 统一入口：拒答 / 占位 / 分发

class Boom:
    def __getattr__(self, n):
        raise AssertionError("拒答/占位不应触碰检索或作答")


def test_off_topic_answers_with_the_model_when_configured_without_retrieval():
    """超出知识库：不检索，直接用大模型自身知识回答（不再拒答）。"""
    class Gen:
        reports_usage = True
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            assert step == "general" and messages[0]["role"] == "system" and messages[-1]["content"] == "今天天气怎么样"
            return LLMResult(content="今天多云，气温 12–20 度。")
    svc = UnifiedService(rag=Boom(), router_client=route_llm("off_topic"))
    svc._general_llm = Gen()
    out = svc.ask("今天天气怎么样")
    assert out["route"] == "off_topic" and out["status"] == "ok" and out["general"] is True
    assert out["answer"] == "今天多云，气温 12–20 度。" and out["sources"] == []     # 不编造出处
    assert out["trace"]["general"]["llm_calls"] == 1


def test_off_topic_without_a_general_model_still_refuses():
    """模型未配置：退回原拒答文案，行为与改动前一致。"""
    svc = UnifiedService(rag=Boom(), router_client=route_llm("off_topic"))
    svc._general_llm = None
    out = svc.ask("今天天气怎么样")
    assert out["route"] == "off_topic" and out["status"] == "refused"
    assert out["answer"] == config.ROUTER_REFUSAL_TEXT and out["sources"] == []


def test_general_answer_prompt_forbids_fake_citations():
    """通用作答的系统提示词必须禁止编造出处，否则库外回答会被误当成有据可查。"""
    assert "不编造" in config.GENERAL_SYSTEM or "不声称内容来自本知识库" in config.GENERAL_SYSTEM
    assert "页码" in config.GENERAL_SYSTEM and "条文号" in config.GENERAL_SYSTEM


def test_router_failure_on_irrelevant_question_still_refuses():
    svc = UnifiedService(rag=Boom(), router_client=RouterLLM(exc=TimeoutError("t")))
    out = svc.ask("给我讲个笑话")
    assert out["status"] == "refused" and out["trace"]["router"]["source"] == "rules_refuse"


def test_policy_without_the_library_loaded_is_unavailable_not_a_placeholder():
    svc = UnifiedService(rag=Boom(), router_client=route_llm("policy"))
    out = svc.ask("数据二十条讲了什么")
    assert out["route"] == "policy" and out["status"] == "unavailable" and "政策库未加载" in out["answer"] and "尚未上线" not in out["answer"]
    from rag import config as rc
    assert not hasattr(rc, "ROUTER_POLICY_TEXT")                                   # 占位文案已删除


def test_course_unavailable_without_context():
    out = UnifiedService(router_client=route_llm("course")).ask("什么是数据要素")
    assert out["route"] == "course" and out["status"] == "unavailable"


def test_course_dispatches_to_course_pipeline(monkeypatch):
    import rag.course.pipeline as pl
    seen = {}

    def fake_ask(q, *, ctx, llm, **kw):
        seen["q"] = q
        return {"answer": {"text": "课程答案"}, "citations": [{"chunk_id": "c1"}], "selected_concepts": [],
                "fallback": None, "trace": {"calls": []}}
    monkeypatch.setattr(pl, "ask", fake_ask)
    out = UnifiedService(router_client=route_llm("course"), course_ctx=object(), course_llm=object()).ask("什么是数据要素")
    assert out["answer"] == "课程答案" and seen["q"] == "什么是数据要素" and "course" in out["trace"]


# ---------------------------------------------------------------- 自适应：simple / 纠错

def test_simple_reliable_one_retrieval_and_generation():
    rag = FakeRAG([res(score=0.9)], llm=FakeChat())
    out = AdaptiveRAG(rag, min_rerank_score=0.3).run("扎根理论是什么")
    assert out["status"] == "ok" and rag.llm.log == ["answer"] and len(rag.queries) == 1
    t = out["trace"]["timings"]
    assert {"retrieve_s", "rerank_s", "generate_s", "total_s"} <= set(t) and t["rerank_s"] == 0.01


def test_adaptive_answer_uses_service_llm_kwargs_without_stream():
    """合并 main 后：作答沿用 RAGService._llm_kwargs（含 reasoning_effort 关闭思考），只是不流式。"""
    rag = FakeRAG([res(score=0.9)], llm=FakeChat())
    rag._llm_kwargs = lambda messages: {"model": "m", "messages": messages, "temperature": 0.2, "stream": True,
                                        "reasoning_effort": "none"}
    AdaptiveRAG(rag, min_rerank_score=0.3).run("扎根理论是什么")
    kw = rag.llm.kwargs[-1]
    assert kw["reasoning_effort"] == "none" and "stream" not in kw and kw["model"] == "m"


def test_auto_ask_is_bounded_by_concurrency_middleware():
    from rag.concurrency import RequestConcurrencyMiddleware
    assert "/auto/ask" in RequestConcurrencyMiddleware._PATHS


def test_low_rerank_score_rewrites_and_retries_once_then_answers():
    rag = FakeRAG([res(score=0.05), res(score=0.8)], llm=FakeChat())
    out = AdaptiveRAG(rag, min_rerank_score=0.3).run("模糊问题")
    assert out["status"] == "ok" and rag.queries == ["模糊问题", "改写后的查询"]
    assert rag.llm.log == ["rewrite", "answer"]


def test_still_low_after_retry_says_insufficient_without_generating():
    rag = FakeRAG([res(score=0.05)], llm=FakeChat())
    out = AdaptiveRAG(rag, min_rerank_score=0.3).run("无关问题")
    assert out["status"] == "insufficient" and out["answer"] == config.ADAPTIVE_INSUFFICIENT_TEXT
    assert "answer" not in rag.llm.log and len(rag.queries) == 2     # 只重查一次


def test_dense_without_rerank_uses_distance():
    rag = FakeRAG([res(dist=0.9)], mode="dense", llm=FakeChat())
    assert AdaptiveRAG(rag, max_distance=0.6).run("q")["status"] == "insufficient"
    rag = FakeRAG([res(dist=0.3)], mode="dense", llm=FakeChat())
    assert AdaptiveRAG(rag, max_distance=0.6).run("q")["status"] == "ok"


def test_hybrid_without_rerank_skips_correction():
    rag = FakeRAG([res(dist=0.99)], mode="hybrid", llm=FakeChat())
    out = AdaptiveRAG(rag).run("q")
    assert out["status"] == "ok" and out["trace"]["retrievals"][0]["kind"] == "skipped" and len(rag.queries) == 1


# ---------------------------------------------------------------- 自适应：complex

def test_complex_decomposes_and_merges_into_single_answer():
    chat = FakeChat(subs=["A是什么", "B是什么", "A和B区别", "第四个"])
    rag = FakeRAG([res(0.9, rid="a"), res(0.9, rid="b"), res(0.9, rid="c")], llm=chat)
    out = AdaptiveRAG(rag, max_subquestions=3).run("比较A和B", complexity="complex")
    tr = out["trace"]
    assert tr["path"] == "complex" and len(tr["sub_questions"]) == 3          # 子问题上限 3
    assert rag.queries == ["A是什么", "B是什么", "A和B区别"] and chat.log == ["decompose", "answer"]
    assert len(out["contexts"]) == 3


def test_complex_round_cap_is_hard():
    chat = FakeChat(subs=["x1", "x2"])
    chat_rewrites = iter(range(100))
    chat.rewrite = None
    orig = chat.create

    def create(**kw):                       # 每次改写都给新文本，逼出补查
        r = orig(**kw)
        if chat.log[-1] == "rewrite":
            r.choices[0].message.content = f"改写{next(chat_rewrites)}"
        return r
    chat.create = create
    rag = FakeRAG([res(score=0.01)], llm=chat)
    out = AdaptiveRAG(rag, min_rerank_score=0.3, max_rounds=2).run("复杂", complexity="complex")
    assert out["trace"]["rounds"] == 2 and len(rag.queries) == 2 + 2        # 第1轮2次 + 第2轮2次，不再多
    assert out["status"] == "insufficient"


def test_complex_partial_answers_flag_missing_subquestion():
    chat = FakeChat(subs=["好的子问题", "查不到的子问题"], rewrite="查不到的子问题")   # 改写无变化 -> 放弃
    rag = FakeRAG([res(0.9, rid="a"), res(0.01, rid="b")], llm=chat)
    out = AdaptiveRAG(rag, min_rerank_score=0.3).run("两部分", complexity="complex")
    assert out["status"] == "ok" and out["trace"]["unanswered_sub_questions"] == ["查不到的子问题"]


def test_complex_with_single_subquestion_degrades_to_simple():
    rag = FakeRAG([res(0.9)], llm=FakeChat(subs=["只有一个"]))
    assert AdaptiveRAG(rag).run("q", complexity="complex")["trace"]["path"] == "complex->simple"


# ---------------------------------------------------------------- HTTP + 保密

def _app(svc):
    app = FastAPI()
    app.state.unified = svc
    app.include_router(unified_router)
    return TestClient(app)


def test_http_auto_ask_returns_route_and_trace_and_omits_trace_on_request():
    c = _app(UnifiedService(router_client=route_llm("off_topic")))
    body = c.post("/auto/ask", json={"question": "你好"}).json()
    assert body["route"] == "off_topic" and body["trace"]["router"]["latency_s"] >= 0
    assert "trace" not in c.post("/auto/ask", json={"question": "你好", "include_trace": False}).json()


def test_http_downstream_error_is_502_without_secret():
    class Bad:
        def ask(self, *a):
            raise RuntimeError(f"401 key={SECRET}")
    r = _app(Bad()).post("/auto/ask", json={"question": "q"})
    assert r.status_code == 502 and SECRET not in r.text


def test_key_never_in_logs_trace_or_errors(caplog, monkeypatch):
    class Leaky(RouterLLM):
        def chat(self, *a, **k):
            raise RuntimeError(f"auth failed {SECRET}")      # 真实客户端已脱敏；这里验证路由层不回显异常正文
    with caplog.at_level(logging.DEBUG):
        d = classify("扎根理论怎么做", Leaky())
    assert SECRET not in json.dumps(d.to_dict()) and SECRET not in caplog.text and d.llm_error == "RuntimeError"
    pytest.importorskip("openai")
    c = OpenAICompatClient("https://x.invalid/v1", SECRET)
    assert SECRET not in repr(c) and SECRET not in c._redact(f"bad {SECRET}")


def test_eval_set_shape():
    rows = json.load(open("rag/eval_data/router_eval.json", encoding="utf-8"))
    from collections import Counter
    assert len(rows) == 40 and Counter(r["expected_route"] for r in rows) == {k: 10 for k in
                                                                              ("thesis", "course", "policy", "off_topic")}


# ---------------------------------------------------------------- serve 接线课程库

def test_load_course_backend_unconfigured_returns_none(monkeypatch):
    import rag.course.llm as cl
    from rag import serve

    def no_cfg(*a, **k):
        raise cl.LLMNotConfigured("x")
    monkeypatch.setattr(cl, "OpenAICompatClient", no_cfg)
    assert serve.load_course_backend() == (None, None)


def test_load_course_backend_missing_data_degrades_without_leaking(monkeypatch, caplog):
    import rag.course.llm as cl
    import rag.course.pipeline as pl
    from rag import serve

    monkeypatch.setattr(cl, "OpenAICompatClient", lambda *a, **k: object())

    def boom(*a, **k):
        raise FileNotFoundError(f"/secret/path {SECRET}")
    monkeypatch.setattr(pl.CourseContext, "load", classmethod(lambda cls, *a, **k: boom()))
    with caplog.at_level(logging.DEBUG):
        assert serve.load_course_backend() == (None, None)
    assert SECRET not in caplog.text and "FileNotFoundError" in caplog.text


def test_build_app_injects_course_and_survives_degradation(monkeypatch):
    from rag import serve

    class FakeSvc:
        llm = None
        mode = "dense"
    monkeypatch.setattr(serve, "RAGService", lambda **kw: FakeSvc())
    ctx, llm = object(), object()
    monkeypatch.setattr(serve, "load_course_backend", lambda: (ctx, llm))
    u = serve.build_app().state.unified
    assert u.course_ctx is ctx and u.course_llm is llm and u.rag is not None
    monkeypatch.setattr(serve, "load_course_backend", lambda: (None, None))
    app = serve.build_app()                      # 论文 RAG 照常启动
    assert app.state.unified.course_ctx is None
    assert TestClient(app).get("/health").json() == {"status": "ok"}
    out = app.state.unified.__class__(router_client=route_llm("course"), course_ctx=None, course_llm=None).ask("什么是数据要素")
    assert out["status"] == "unavailable"


# ---------------------------------------------------------------- 路由评测脚本（mock）

def test_eval_router_single_call_no_retry_and_summary():
    from datetime import datetime
    from rag import eval_router as er

    qs = [{"id": "1", "question": "a", "expected_route": "course", "expected_complexity": "simple"},
          {"id": "2", "question": "b", "expected_route": "thesis", "expected_complexity": "simple"},
          {"id": "3", "question": "c", "expected_route": "off_topic", "expected_complexity": "simple"}]
    replies = iter([{"route": "course", "complexity": "simple", "reason": "r"}, "not json",
                    {"route": "policy", "complexity": "simple", "reason": "x"}])

    class C:
        calls = 0

        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            C.calls += 1
            r = next(replies)
            return LLMResult(content=r if isinstance(r, str) else json.dumps(r), latency_s=0.5)
    ok_time = lambda: datetime(2026, 10, 31, 19, 0, tzinfo=er.BJ)
    out = er.run(C(), qs, ok_time)
    assert C.calls == 3                                     # 每题 1 次，失败也不重试
    s = out["summary"]
    assert s["failures"] == 1 and s["per_class"]["thesis"]["failed"] == 1
    assert s["confusion"]["off_topic->policy"] == 1 and s["overall_accuracy"] == round(1 / 3, 3)
    # 窗口外不调用
    C.calls = 0
    assert er.run(C(), qs, lambda: datetime(2026, 10, 31, 15, 0, tzinfo=er.BJ))["skipped_outside_window"] == 3 and C.calls == 0
    assert er.in_window(datetime(2026, 10, 31, 12, 30, tzinfo=er.BJ)) and not er.in_window(datetime(2026, 10, 31, 14, 0, tzinfo=er.BJ))
    # 高峰 09–12、14–18；其余（含凌晨 0–9 点）均为非高峰
    hours = {h: er.in_window(datetime(2026, 10, 31, h, 30, tzinfo=er.BJ)) for h in range(24)}
    assert [h for h, ok in hours.items() if not ok] == [9, 10, 11, 14, 15, 16, 17]
    assert hours[0] and hours[8] and hours[12] and hours[13] and hours[18] and hours[23]


def test_course_eval_counting_llm_hard_cap_blocks_before_calling(monkeypatch):
    from datetime import datetime
    from rag import eval_course as ec

    monkeypatch.setattr(ec, "in_window", lambda now: True)
    inner = RouterLLM({"route": "course"})
    shared = {"calls": 0, "cap": 2, "q_calls": 0, "q_in": 0, "q_out": 0}
    c = ec.CountingLLM(inner, shared)
    c.chat("a", [], model="m"); c.chat("a", [], model="m")
    with pytest.raises(ec.CapReached):
        c.chat("a", [], model="m")
    assert len(inner.calls) == 2 and shared["calls"] == 2
    monkeypatch.setattr(ec, "in_window", lambda now: False)
    shared["cap"] = 9
    with pytest.raises(ec.CapReached):
        c.chat("a", [], model="m")
    assert len(inner.calls) == 2


# ---------------------------------------------------------------- 作答占位符 [S#]

def test_sanitize_strips_literal_placeholder_and_keeps_valid_refs():
    from rag.course import prompts
    from rag.course.answer import sanitize_citations

    text, valid, invalid, _ = sanitize_citations("依据均为教材原文 [S#]，见[S1]。", {"S1": "de:x"})
    assert "[S#]" not in text and "S#" not in text and valid == ["de:x"] and invalid == []
    assert "[1]" in text and "de:x" not in text                        # 正文是序号，不是 chunk_id
    assert "占位符 “[S#]”" in prompts.ANSWER_SYSTEM                          # 提示词明确说明 [S#] 只是编号写法，不要原样写出占位符
    assert "如 [S2]" in prompts.ANSWER_SYSTEM


# ---------------------------------------------------------------- 选词快速路径 + 每轮 trace

def test_fast_path_unique_alias_skips_llm(ctx, monkeypatch):
    from rag.course import agent_select, config as cc
    monkeypatch.setattr(cc, "COURSE_SELECT_FAST_PATH", True)

    class NoLLM:
        def chat(self, *a, **k):
            raise AssertionError("快速路径不应调用 LLM")
    r = agent_select.select_concepts("数据要素的定义是什么", [], ctx.index, NoLLM(), native_tools=True)
    assert r.fast_path and r.end_reason == "fast_path" and r.rounds == []
    assert (r.selected[0].concept_id, r.selected[0].role) == ("C01-0001", "question_concept")
    rel = [s for s in r.selected[1:]]
    assert rel and all(s.role == "related_concept" and s.concept_id != "C01-0001" for s in rel)
    assert len(rel) <= cc.COURSE_FAST_RELATED_MAX and len({s.concept_id for s in r.selected}) == len(r.selected)


@pytest.mark.parametrize("q", ["数据要素和数据权属有什么区别", "什么是数据要素以及它的特征、分类、价值、来源、治理和流通的全部内容"])
def test_fast_path_declines_multi_concept_or_long_questions(ctx, q):
    from rag.course import agent_select
    assert agent_select.fast_select(q, ctx.index) is None


def test_fast_path_declines_ambiguous_alias_and_unknown(ctx):
    from rag.course import agent_select
    assert agent_select.fast_select("今天天气", ctx.index) is None
    for a in [a for a in ctx.index.aliases.active() if a.ambiguous]:     # 同形多概念的别名一律不走快速路径
        assert agent_select.fast_select(f"{a.surface}是什么", ctx.index) is None


def test_fast_path_declines_when_same_name_across_chapters(ctx, monkeypatch):
    from rag.course import agent_select
    monkeypatch.setitem(ctx.index._name_groups, "数据要素", ["C01-0001", "C09-0001"])
    assert agent_select.fast_select("数据要素的定义是什么", ctx.index) is None


def test_select_logs_each_round_latency_and_tokens(ctx):
    from rag.course import agent_select
    from test_course_pipeline import ScriptedLLM, finish_native, tc
    llm_ = ScriptedLLM([tc("search_vocab", query="数据要素"), finish_native(("C01-0001", "question_concept"))])
    r = agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, llm_, native_tools=True, fast_path=False)
    assert [x["round"] for x in r.rounds] == [1, 2] and r.rounds[0]["tools"] == ["search_vocab"]
    assert {"latency_s", "prompt_tokens", "completion_tokens"} <= set(r.rounds[0])
    assert r.to_dict()["rounds"] == r.rounds and r.to_dict()["fast_path"] == ""


def test_default_max_steps_is_seven_and_select_reasoning_disabled():
    import os
    from rag.course import config as cc
    if "COURSE_AGENT_MAX_STEPS" not in os.environ:
        assert cc.COURSE_AGENT_MAX_STEPS == 7                      # 6 次探索 + 1 次强制 finish
    if "COURSE_SELECT_REASONING" not in os.environ:
        assert cc.COURSE_REASONING["select"] == "disabled"         # 选词默认关闭思考


# ---------------------------------------------------------------- 快速路径关联概念 / 默认关闭 / 压缩 / 上限 / 本体为空

def test_fast_path_default_off_and_related_limit(ctx, monkeypatch):
    from rag.course import agent_select, config as cc
    import os
    if "COURSE_SELECT_FAST_PATH" not in os.environ:
        assert cc.COURSE_SELECT_FAST_PATH is False
    monkeypatch.setattr(cc, "COURSE_FAST_RELATED_MAX", 0)
    r = agent_select.fast_select("数据要素的定义是什么", ctx.index)
    assert [s.concept_id for s in r.selected] == ["C01-0001"]
    monkeypatch.setattr(cc, "COURSE_FAST_RELATED_MAX", 1)
    assert len(agent_select.fast_select("数据要素的定义是什么", ctx.index).selected) <= 2


def test_compress_tool_output_keeps_only_id_and_name():
    from rag.course.agent_select import compress_tool_output as c
    big = {"results": [{"concept_id": "C1", "name": "甲", "chapter": 1, "snippet": "很长的释义", "via": "bm25", "score": 1}]}
    assert c("search_vocab", big) == {"results": [{"concept_id": "C1", "name": "甲"}]}
    assert c("get_concept", {"concept_id": "C1", "name": "甲", "context": "长", "aliases": ["a"]}) == {"concept_id": "C1", "name": "甲"}
    nb = {"total": 2, "truncated": False, "neighbors": [{"concept_id": "C2", "name": "乙", "predicate_id": "p", "view": "onto1"},
                                                          {"concept_id": "C2", "name": "乙", "predicate_id": "q", "view": "onto2"}]}
    assert c("get_neighbors", nb)["neighbors"] == [{"concept_id": "C2", "name": "乙"}]
    assert c("search_vocab", {"error": "x"}) == {"error": "x"}


def test_older_tool_results_compressed_latest_kept_full(ctx):
    from rag.course import agent_select
    from test_course_pipeline import ScriptedLLM, finish_native, tc
    llm_ = ScriptedLLM([tc("search_vocab", query="数据要素"), tc("get_concept", concept_id="C01-0001"),
                        finish_native(("C01-0001", "question_concept"))])
    agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, llm_, max_steps=4, native_tools=True, fast_path=False)
    tools2 = [m for m in llm_.calls[1]["messages"] if m["role"] == "tool"]
    assert "snippet" in tools2[0]["content"]                       # 刚返回的结果完整保留
    tools3 = [m for m in llm_.calls[2]["messages"] if m["role"] == "tool"]
    assert "snippet" not in tools3[0]["content"] and "C01-0001" in tools3[0]["content"]   # 早先的已压缩，仍有 ID 与名称
    assert "same_name_ids" in tools3[1]["content"]                 # 第 3 次调用时 get_concept 是最新一轮结果，保持完整


def test_max_steps_four_means_at_most_four_select_calls(ctx):
    from rag.course import agent_select, prompts
    from test_course_pipeline import ScriptedLLM, tc
    llm_ = ScriptedLLM([tc("search_vocab", query="数据要素")] * 10)
    r = agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, llm_, max_steps=4, native_tools=True, fast_path=False)
    assert len(llm_.calls) == 4 and r.truncated and r.end_reason == "truncated" and len(r.rounds) == 4
    assert [t["function"]["name"] for t in llm_.calls[3]["tools"]] == ["finish"]
    assert len(llm_.calls[0]["tools"]) > 1                         # 前几次调用提供全部工具
    # 上限 1：唯一一次调用就是强制 finish
    llm1 = ScriptedLLM([tc("search_vocab", query="x")])
    r1 = agent_select.select_concepts("q", [], ctx.index, llm1, max_steps=1, native_tools=True, fast_path=False)
    assert len(llm1.calls) == 1 and [t["function"]["name"] for t in llm1.calls[0]["tools"]] == ["finish"]


def test_forced_finish_on_last_call_is_accepted(ctx):
    from rag.course import agent_select
    from test_course_pipeline import ScriptedLLM, finish_native, tc
    llm_ = ScriptedLLM([tc("search_vocab", query="a"), tc("search_vocab", query="b"), tc("search_vocab", query="c"),
                        finish_native(("C01-0001", "question_concept"))])
    r = agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, llm_, max_steps=4, native_tools=True, fast_path=False)
    assert len(llm_.calls) == 4 and r.finished and [s.concept_id for s in r.selected] == ["C01-0001"]


def test_answer_omits_ontology_section_when_no_relations(ctx):
    from rag.course import answer
    from test_course_pipeline import ScriptedLLM, _items
    kept = _items(ctx, "P0005")[:1]
    llm_ = ScriptedLLM([LLMResult(content="回答[S1]")])
    answer.answer_question("q", kept, [], llm_)
    sysm, user = (m["content"] for m in llm_.calls[0]["messages"])
    assert "本体" not in user and "（无）" not in user and "【教材原文】" in user
    assert "本体" in sysm and "若提供了【本体结构信息】" in sysm                  # 本体段为空时用户消息里不出现，系统提示只做条件说明
    rel = ctx.index.relations_among(["C01-0001", "C01-0003"])
    llm2 = ScriptedLLM([LLMResult(content="回答[S1]")])
    answer.answer_question("q", kept, rel, llm2)
    assert "【本体结构信息】" in llm2.calls[0]["messages"][1]["content"]


# ---------------------------------------------------------------- 选词轮次记录 reasoning

class _ReasoningLLM:
    """带 reasoning_for 的脚本化客户端：返回带 reasoning_tokens / reasoning_chars 的结果。"""
    def __init__(self, script, mode):
        self.script, self.mode = list(script), mode

    def reasoning_for(self, step):
        return self.mode

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        r = self.script.pop(0)
        r.reasoning_tokens, r.reasoning_chars, r.latency_s = 321, 654, 1.5
        return r


@pytest.mark.parametrize("mode,sent", [("", None), ("disabled", {"type": "disabled"}),
                                        ("low", {"type": "enabled", "effort": "low"})])
def test_select_rounds_record_reasoning_tokens_chars_and_sent_mode(ctx, mode, sent):
    from rag.course import agent_select
    from rag.course.llm import Trace, TracingLLM
    from test_course_pipeline import finish_native, tc
    inner = _ReasoningLLM([tc("search_vocab", query="数据要素"), finish_native(("C01-0001", "question_concept"))], mode)
    r = agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, TracingLLM(inner, Trace()),
                                     native_tools=True, fast_path=False)
    x = r.rounds[0]
    assert (x["reasoning_tokens"], x["reasoning_chars"], x["reasoning_mode"], x["reasoning_sent"]) == (321, 654, mode, sent)
    assert x["latency_s"] == 1.5 and r.to_dict()["rounds"][1]["reasoning_tokens"] == 321


def test_select_rounds_without_reasoning_for_records_none(ctx):
    from rag.course import agent_select
    from test_course_pipeline import ScriptedLLM, finish_native
    r = agent_select.select_concepts("请讲讲数据要素和权属", [], ctx.index, ScriptedLLM([finish_native(("C01-0001", "question_concept"))]),
                                     native_tools=True, fast_path=False)
    assert r.rounds[0]["reasoning_mode"] is None and r.rounds[0]["reasoning_sent"] is None


# ---------------------------------------------------------------- BM25 兜底召回

def test_chunk_bm25_finds_by_question_and_drops_stopwords(ctx):
    from rag.course.fallback import ChunkBM25
    b = ChunkBM25(ctx.chunkset, use_jieba=False)
    assert "什么" not in b.query_tokens("什么是数据要素") and b.search("数据要素", 0) == []
    hits = b.search("数据要素", 3)
    assert hits and all(s > 0 for _, s in hits) and hits == sorted(hits, key=lambda x: -x[1])


def test_merge_extra_dedups_and_marks_source(ctx):
    from rag.course.recall import BM25_VIA
    rr = ctx.recaller.recall([("C01-0001", "question_concept")])
    onto = {i.chunk_id for i in rr.items}
    extra = [c for c in ctx.chunkset.chunks if c.chunk_id not in onto][:2]
    both = ctx.chunkset.chunks[[c.chunk_id for c in ctx.chunkset.chunks].index(next(iter(onto)))]
    st = ctx.recaller.merge_extra(rr, [(both, 3.0)] + [(c, 1.0) for c in extra])
    ids = [i.chunk_id for i in rr.items]
    assert len(ids) == len(set(ids))                                      # 去重
    by = {i.chunk_id: i for i in rr.items}
    assert by[both.chunk_id].source == "both" and all(by[c.chunk_id].source == "bm25" for c in extra)
    assert all(i.source == "ontology" for i in rr.items if i.chunk_id in onto and i.chunk_id != both.chunk_id)
    assert st["overlap"] == 1 and st["added"] == len(extra) and st["hits"][0]["also_ontology"] is True
    assert by[extra[0].chunk_id].tier == 3 and any(r.via == BM25_VIA for r in by[extra[0].chunk_id].reasons)
    assert [i.tier for i in rr.items] == sorted(i.tier for i in rr.items)  # 仍按 tier 升序，BM25 命中不插队
    assert "source" in rr.items[0].to_dict() and rr.items[0].to_dict()["source"] in ("ontology", "both", "bm25")


def test_pipeline_bm25_rescues_missed_concept(ctx, monkeypatch):
    from rag.course import config as cc, pipeline
    from test_course_pipeline import ScriptedLLM, finish_native
    sel = [("C01-0003", "question_concept")]                               # 选词漏选了 C01-0001
    onto = {i.chunk_id for i in ctx.recaller.recall(sel).items}
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 5)
    only_bm25 = [c for c, _ in ctx.bm25.search("什么是数据要素", 5) if c.chunk_id not in onto]
    assert only_bm25, "夹具里应有只被 BM25 召回的块"
    llm_ = ScriptedLLM([finish_native(*sel), lambda s, m: (_ for _ in ()).throw(RuntimeError("stop"))])
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.ask("什么是数据要素", ctx=ctx, llm=llm_)
    filter_prompt = llm_.calls[1]["messages"][1]["content"]
    assert all(c.text in filter_prompt for c in only_bm25)                 # 漏选概念对应的块靠 BM25 进入了筛选
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 0)
    llm2 = ScriptedLLM([finish_native(*sel), lambda s, m: (_ for _ in ()).throw(RuntimeError("stop"))])
    with pytest.raises(RuntimeError):
        pipeline.ask("什么是数据要素", ctx=ctx, llm=llm2)
    assert not any(c.text in llm2.calls[1]["messages"][1]["content"] for c in only_bm25)   # 关闭后不会出现


def test_pipeline_trace_marks_sources_and_allows_empty_selection(ctx, monkeypatch):
    from rag.course import config as cc, pipeline
    from test_course_pipeline import ScriptedLLM, finish_native
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 5)
    llm_ = ScriptedLLM([finish_native(), lambda s, m: (_ for _ in ()).throw(RuntimeError("stop"))])
    with pytest.raises(RuntimeError, match="stop"):                       # 选词一个概念都没选，也能靠 BM25 进入筛选
        pipeline.ask("数据要素", ctx=ctx, llm=llm_)
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 0)
    r = pipeline.ask("q", ctx=ctx, llm=ScriptedLLM([finish_native()]))
    assert r["fallback"]["reason"] == "no_valid_concepts"                  # 关闭兜底时行为不变


def test_bm25_topk_zero_disables_and_default_is_six():
    import os
    from rag.course import config as cc
    if "COURSE_BM25_TOPK" not in os.environ:
        assert cc.COURSE_BM25_TOPK == 6


def test_recall_eval_runs_offline_and_counts_hits(ctx):
    from rag.course import recall_eval
    gold = [{"id": "g1", "question": "数据要素", "must_recall": [{"key": "数据要素"}],
             "ontology_selection_v1": [{"concept_id": "C01-0001", "role": "question_concept"}]},
            {"id": "g2", "question": "数据要素", "must_recall": [{"key": "数据要素"}], "ontology_selection_v1": None}]
    res = recall_eval.evaluate(ctx, gold, 3)
    assert res["comparable_questions"] == ["g1"] and res["questions_without_ontology_record"] == ["g2"]
    assert res["summary_comparable"]["ontology+bm25"]["total"] == 1
    with pytest.raises(ValueError):
        recall_eval.evaluate(ctx, [{"id": "x", "question": "q", "must_recall": [{"key": "不存在的关键句XYZ"}]}], 3)


def test_recall_gold_file_shape():
    g = json.load(open("rag/eval_data/course_recall_gold.json", encoding="utf-8"))
    assert [x["id"] for x in g] == [f"r{i}" for i in range(11, 21)]
    assert all(1 <= len(x["must_recall"]) <= 3 for x in g)


# ---------------------------------------------------------------- 完整评测汇总（离线）

def test_eval_full_summarize_counts_hits_grades_and_citations():
    from rag.course.eval_full_report import summarize
    gold = [{"id": "r11", "must_recall": [{"key": "a", "chunks": ["c1"]}, {"key": "b", "chunks": ["c2"]}]},
            {"id": "r14", "must_recall": [{"key": "c", "chunks": ["c9"]}]}]
    rows = [{"id": "r11", "error": "", "fallback": None, "citations": ["c1", "c3"],
             "recalled": [{"chunk_id": "c1", "source": "ontology", "grade": "direct"},
                          {"chunk_id": "c2", "source": "bm25", "grade": "background"},
                          {"chunk_id": "c3", "source": "bm25", "grade": "direct"},
                          {"chunk_id": "c4", "source": "bm25", "grade": "irrelevant"}]},
            {"id": "r14", "error": "", "fallback": None, "citations": [],
             "recalled": [{"chunk_id": "c9", "source": "both", "grade": "direct"}]},
            {"id": "r15", "error": "CapReached:call cap reached"}]
    s = summarize(rows, gold)
    assert s["hit_rate"]["r11-r13"]["ontology_only"] == "1/2" and s["hit_rate"]["r11-r13"]["ontology_plus_bm25"] == "2/2"
    assert s["hit_rate"]["r14-r20"]["ontology_only"] == "1/1"            # both 算作本体召回到
    assert s["bm25_only_chunk_grades"] == {"direct": 1, "background": 1, "irrelevant": 1, "ungraded": 0}
    assert s["bm25_only_chunks_cited_in_answers"] == 1                   # c3 是纯 BM25 块且被引用；c1 来自本体
    assert s["per_question"][2] == {"id": "r15", "error": "CapReached:call cap reached"}


def test_eval_full_capped_llm_stops_before_calling(monkeypatch):
    from rag.course import eval_full as ef
    monkeypatch.setattr(ef, "in_window", lambda now: True)
    inner = RouterLLM({"route": "course"})
    sh = {"calls": 0, "cap": 1}
    c = ef.CappedLLM(inner, sh)
    c.chat("a", [], model="m")
    with pytest.raises(ef.CapReached):
        c.chat("a", [], model="m")
    assert len(inner.calls) == 1


# ---------------------------------------------------------------- 完整评测：连续失败即停 / 开始时间 / 窗口

def _run_full(monkeypatch, tmp_path, errors):
    """用假的 run_one 驱动 main()：errors 为每题的 error 值列表。"""
    import sys
    from datetime import datetime
    from rag.course import eval_full as ef
    seq = iter(errors)
    monkeypatch.setattr(ef, "in_window", lambda now: True)
    monkeypatch.setattr(ef, "OpenAICompatClient", lambda: type("C", (), {"_client": type("X", (), {"with_options": lambda s, **k: s})()})())
    monkeypatch.setattr(ef, "CourseContext", type("CC", (), {"load": staticmethod(lambda: type("K", (), {"chunkset": type("S", (), {"chunks": []})()})())}))
    monkeypatch.setattr(ef, "key_chunks", lambda ctx, key: [])
    monkeypatch.setattr(ef, "run_one", lambda ctx, llm, g: {"id": g["id"], "question": "", "error": next(seq), "elapsed_s": 0})
    monkeypatch.setattr(ef, "summarize", lambda rows, gold: {})
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    ef.main()
    return json.load(open(out, encoding="utf-8"))


def test_full_eval_stops_after_two_consecutive_llm_errors(monkeypatch, tmp_path):
    d = _run_full(monkeypatch, tmp_path, ["", "LLMError", "LLMError"] + [""] * 20)
    assert [r["error"] for r in d["rows"]] == ["", "LLMError", "LLMError"] and "LLMError x2" in d["stopped"]


def test_full_eval_non_consecutive_llm_errors_do_not_stop(monkeypatch, tmp_path):
    d = _run_full(monkeypatch, tmp_path, ["LLMError", "", "LLMError", "", "LLMError"] + [""] * 20)
    assert len(d["rows"]) == 10 and d["stopped"] == ""


def test_full_eval_records_start_before_finish(monkeypatch, tmp_path):
    d = _run_full(monkeypatch, tmp_path, [""] * 10)
    assert d["started_bj"] <= d["finished_bj"] and "started_bj" in d


def test_capped_llm_checks_window_before_every_call(monkeypatch):
    from rag.course import eval_full as ef
    state = {"ok": True}
    monkeypatch.setattr(ef, "in_window", lambda now: state["ok"])
    inner = RouterLLM({"route": "course"})
    c = ef.CappedLLM(inner, {"calls": 0, "cap": 99})
    c.chat("a", [], model="m")
    state["ok"] = False                                                    # 进入高峰：下一次调用必须被挡住，且不触达内层
    with pytest.raises(ef.CapReached, match="off-peak"):
        c.chat("a", [], model="m")
    assert len(inner.calls) == 1


def test_llm_error_cause_extracted_without_message(monkeypatch):
    from rag.course import eval_full as ef
    from rag.course.llm import LLMError

    class Boom:
        def chat(self, *a, **k):
            raise LLMError(f"[select] LLM 调用失败：APITimeoutError: Request timed out. {SECRET}")
    monkeypatch.setattr(ef, "ask", lambda *a, **k: Boom().chat())
    row = ef.run_one(None, None, {"id": "r1", "question": "q"})
    assert row["error"] == "LLMError" and row["error_cause"] == "APITimeoutError" and SECRET not in json.dumps(row)


# ---------------------------------------------------------------- 全角括号引用

@pytest.mark.parametrize("raw", ["见［S1］。", "见【S1】。", "见〔S1〕。", "见[S1]。"])
def test_sanitize_normalizes_fullwidth_brackets(raw):
    from rag.course.answer import sanitize_citations
    text, valid, invalid, _ = sanitize_citations(raw, {"S1": "de:x"})
    assert text == "见[1]。" and valid == ["de:x"] and invalid == []          # 全角也能识别；正文是序号，chunk_id 只在 valid 里


def test_sanitize_fullwidth_invalid_placeholder_and_ontology_tags():
    from rag.course.answer import sanitize_citations
    text, valid, invalid, onto = sanitize_citations("甲［S9］乙【S#】丙【本体】丁［de:fake］", {"S1": "de:x"})
    assert "S9" not in text and "S#" not in text and "de:fake" not in text and valid == []
    assert set(invalid) == {"S9", "de:fake"} and onto and "[本体]" not in text and "本体" not in text.replace("丙", "")   # 本体标记被去掉，但记录 uses_ontology


# ---------------------------------------------------------------- 判例 / 名言路由：routes、count、chapter、旧 route 兼容

def routes_llm(routes, count=1, chapter=None, cx="simple"):
    return RouterLLM({"routes": routes, "complexity": cx, "count": count, "chapter": chapter, "reason": "t"})


@pytest.mark.parametrize("routes", [["case"], ["quote"], ["course", "case"], ["course", "quote"], ["case", "quote"]])
def test_llm_returns_routes_list_and_old_route_field_is_first(routes):
    d = classify("任意问题", routes_llm(routes))
    assert d.routes == routes and d.route == routes[0] and d.combined == (len(routes) > 1) and d.source == "llm"
    assert d.to_dict()["route"] == routes[0] and d.to_dict()["routes"] == routes


def test_old_single_route_json_still_works_and_gives_one_element_routes():
    d = classify("q", RouterLLM({"route": "course", "complexity": "simple", "reason": "t"}))
    assert (d.route, d.routes, d.count, d.chapter) == ("course", ["course"], 1, None)


@pytest.mark.parametrize("raw,expected", [
    (["course", "case", "quote"], ["course", "case"]),           # 最多 2 个
    (["course", "course"], ["course"]),                           # 去重
    (["thesis", "case"], ["thesis"]),                             # thesis 不参与组合
    (["policy", "course"], ["policy", "course"]), (["course", "policy", "case"], ["course", "policy"]),      # 政策参与组合，最多 2 个
    (["off_topic", "case"], ["off_topic"]), (["exam", "policy"], ["exam"]), (["policy", "thesis"], ["policy"]),
    (["case", "thesis"], ["case"]),                               # 非组合类混进来：不组合，保留可组合的
    (["nonsense", "case"], ["case"]), ("quote", ["quote"])])
def test_normalize_routes_enforces_combination_rules(raw, expected):
    from rag.router import normalize_routes
    assert normalize_routes(raw) == expected


def test_normalize_routes_rejects_garbage():
    from rag.router import normalize_routes
    with pytest.raises(ValueError):
        normalize_routes(["nonsense"])
    with pytest.raises(ValueError):
        normalize_routes([])


@pytest.mark.parametrize("raw,expected", [(None, 1), ("abc", 1), (0, 1), (-3, 1), (3, 3), ("4", 4), (9, 5), (5.9, 5)])
def test_count_is_clamped_to_1_5(raw, expected):
    d = classify("q", routes_llm(["case"], count=raw))
    assert d.count == expected


@pytest.mark.parametrize("raw,expected", [(6, 6), ("9", 9), (0, None), (11, None), (None, None), ("x", None)])
def test_chapter_is_1_to_10_or_none_and_only_kept_for_case(raw, expected):
    assert classify("q", routes_llm(["case"], chapter=raw)).chapter == expected
    assert classify("q", routes_llm(["course", "case"], chapter=raw)).chapter == expected
    assert classify("q", routes_llm(["course"], chapter=raw)).chapter is None             # 课程不带 chapter
    assert classify("q", routes_llm(["quote"], count=3, chapter=raw)).chapter is None


def test_count_ignored_for_non_case_quote_routes():
    assert classify("q", routes_llm(["course"], count=4)).count == 1


def test_combined_route_failure_falls_back_to_rules_with_same_shape():
    d = classify("结合判例说明数据确权为什么难", RouterLLM(exc=TimeoutError("t")))
    assert d.source == "rules" and d.routes == ["course", "case"] and d.route == "course" and d.llm_error == "TimeoutError"


def test_router_prompt_lists_new_categories_and_output_format():
    from rag.router import ROUTER_SYSTEM, CHAPTER_NAMES
    for w in ("case", "quote", "policy", "routes", "count", "chapter", "最多 2 个", "组合", "off_topic、thesis、exam 只能单独出现", "给我一句名人名言"):
        assert w in ROUTER_SYSTEM
    assert all(f"{n} {name}" in ROUTER_SYSTEM for n, name in CHAPTER_NAMES.items())     # 提示词里给出 10 个章名


# ---------------------------------------------------------------- 规则兜底：case / quote / 组合 / count / chapter / 列表

@pytest.mark.parametrize("q", ["你好，你的知识库目前具备哪些内容呢", "你好，介绍一下你自己", "你能做什么", "你有哪些功能",
                                "你是谁", "怎么用你", "你的资料来源是什么", "这个系统能回答什么"])
def test_meta_questions_are_recognized_by_rules(q):
    """问助手自身能力 / 知识库 -> meta（不拒答、不检索）。"""
    d = rule_route(q)
    assert d.route == "meta" and d.routes == ["meta"] and d.source == "rules"


def test_ability_question_with_self_reference_is_meta():
    """「你的能力体现在哪」这种换词问法也要认出来（只是不能把正文里的「能力」也算进来）。"""
    assert rule_route("你的能力体现在哪").route == "meta"
    assert rule_route("你帮我把这段论文的能力要求改一下").route == "thesis"


@pytest.mark.parametrize("q", ["今天天气怎么样", "数据能做什么", "帮我写一首关于数据的诗",
                               "你帮我把这段论文的能力要求改一下", "你能帮我把这篇论文的摘要润色一下吗"])
def test_unrelated_or_content_questions_are_not_meta(q):
    """与助手无关的实质问题不能被判成 meta：放宽边界时不能把正文问题也算进来。"""
    assert rule_route(q).route != "meta"


@pytest.mark.parametrize("q", ["你好", "您好", "在吗？", "你好呀，在吗？", "hello", "早上好呀", "嗨"])
def test_greetings_are_meta_not_refusal(q):
    """打招呼必须回欢迎语：拒答一句「你好」像是系统坏了（线上实际踩到的）。"""
    d = rule_route(q)
    assert d.route == "meta" and d.source == "rules"


@pytest.mark.parametrize("q", ["我之前都问过你哪些内容呢", "我之前问过你哪些问题", "我们刚才聊了什么",
                               "咱俩聊过什么", "我的提问记录", "你记得我问过什么吗"])
def test_asking_what_i_asked_before_is_meta(q):
    """问自己之前问过什么也归 meta：线上真的这么问了，却拿到一整篇能力清单（答非所问）。"""
    d = rule_route(q)
    assert d.route == "meta" and d.source == "rules"


@pytest.mark.parametrize("q", ["论文答辩老师会问哪些问题", "开题报告要回答哪些内容", "老师问我为什么选这个题目怎么答"])
def test_content_questions_about_being_asked_are_not_meta(q):
    """「问哪些」沾上答辩 / 开题就不能当 meta：主体得是「我 / 我们问过你」。"""
    assert rule_route(q).route != "meta"


@pytest.mark.parametrize("q,route", [("你好，我想问数据确权", "course"), ("你好，你的知识库有哪些内容", "meta"),
                                     ("你好我想了解一下数据资产", "course")])
def test_greeting_prefix_does_not_swallow_the_real_question(q, route):
    """「你好」后面跟着真问题时，按真问题走，不能整句当成打招呼。"""
    assert rule_route(q).route == route


@pytest.mark.parametrize("q,routes", [
    ("举三个关于数据爬取的判例", ["case"]), ("微博诉云智联案是怎么判的", ["case"]), ("指导性案例262号讲了什么", ["case"]),
    ("大数据搬家软件案的争议焦点是什么", ["case"]), ("（2021）浙8601民初309号案怎么判的", ["case"]),
    ("总书记关于数据安全是怎么说的", ["quote"]), ("给我两条习近平关于人工智能的原话", ["quote"]), ("引用一句总书记谈数字经济的话", ["quote"]),
    ("结合判例说明数据确权为什么难", ["course", "case"]), ("用总书记的讲话谈谈数据要素的作用", ["course", "quote"]),
    ("结合判例和总书记的讲话谈谈数据安全", ["case", "quote"]),
    ("论文里怎么引用判决书", ["thesis"]), ("这篇论文的讲话稿怎么写", ["thesis"]), ("案例分析法在论文里怎么用", ["thesis"]),
    ("什么是数据确权", ["course"]), ("今天天气怎么样", ["off_topic"])])
def test_rule_fallback_recognizes_case_quote_and_combinations(q, routes):
    d = rule_route(q)
    assert d.routes == routes and d.route == routes[0] and len(d.routes) <= 2 and d.source in ("rules", "rules_refuse")


@pytest.mark.parametrize("q,n", [("举三个判例", 3), ("给我两条总书记的话", 2), ("来五个案例", 5), ("举一个判例", 1), ("有哪些判例", 1),
                                  ("给我一条习近平关于人工智能的原话", 1), ("举两个关于隐私保护的案例", 2), ("要3个判例", 3)])
def test_extract_count(q, n):
    from rag.router import extract_count
    assert extract_count(q) == n


@pytest.mark.parametrize("q,ch", [("第六章有哪些判例", 6), ("第十章的判例", 10), ("第二章的案例", 2), ("第 3 章", 3), ("举个隐私保护的案例", 9),
                                   ("数据爬取的判例", 6), ("数据确权的案例", 2), ("数据要素和数据安全的案例", None), ("随便举个案例", None)])
def test_extract_chapter(q, ch):
    from rag.router import extract_chapter
    assert extract_chapter(q) == ch


def test_case_list_question_detection():
    from rag.router import is_case_list_question
    assert is_case_list_question("第六章有哪些判例？") and is_case_list_question("第二章都有哪些案例") and is_case_list_question("判例列表")
    assert not is_case_list_question("举三个第六章的判例") and not is_case_list_question("微博诉云智联案怎么判")


def test_rule_fallback_count_and_chapter_for_case():
    d = rule_route("举三个关于数据爬取的判例")
    assert (d.routes, d.count, d.chapter) == (["case"], 3, 6)
    d = rule_route("给我两条习近平关于人工智能的原话")
    assert (d.routes, d.count, d.chapter) == (["quote"], 2, None)


def test_old_forty_question_set_routes_unchanged_by_new_categories():
    """新增 case / quote 不能改变既有 40 题的规则兜底结果。

    原本就判错的 r27、r36 保持不变。r31（「你好呀，在吗？」）从此也多出一条差异：
    它的标注还是改版前的 off_topic，而现在归 meta（有意的行为修正）。
    这里**不改标注**——改了标注等于让评测迁就现状，看不出真实偏差。
    """
    rows = json.load(open("rag/eval_data/router_eval.json", encoding="utf-8"))
    bad = sorted(r["id"] for r in rows if rule_route(r["question"]).route != r["expected_route"])
    assert bad == ["r27", "r31", "r36"]
    r31 = next(r for r in rows if r["id"] == "r31")
    assert r31["expected_route"] == "off_topic" and rule_route(r31["question"]).route == "meta"


def test_new_router_eval_set_shape_and_rule_fallback_accuracy():
    rows = json.load(open("rag/eval_data/router_eval_cases_quotes.json", encoding="utf-8"))
    from collections import Counter
    assert len(rows) == 25 and len({r["id"] for r in rows}) == 25
    kinds = Counter("+".join(r["expected_routes"]) for r in rows)
    assert kinds["case"] == 8 and kinds["quote"] == 6 and sum(v for k, v in kinds.items() if "+" in k) == 7 and kinds["thesis"] == 2
    assert all(r["expected_route"] == r["expected_routes"][0] and len(r["expected_routes"]) <= 2 for r in rows)
    wrong = [r["id"] for r in rows if rule_route(r["question"]).routes != r["expected_routes"]]
    assert wrong == []                                                                  # 规则兜底在这 25 题上全对（不调用模型）
    assert all(rule_route(r["question"]).count == r["expected_count"] for r in rows if {"case", "quote"} & set(r["expected_routes"]))


def test_eval_router_summary_handles_new_classes_with_no_rows_and_combined_expectations():
    from rag import eval_router as er
    rows = [{"id": "1", "expected_route": "course", "expected_complexity": "simple", "pred": "course", "complexity": "simple", "error": "", "latency_s": 1.0},
            {"id": "2", "expected_route": "case", "expected_complexity": "simple", "pred": "course", "complexity": "simple", "error": "", "latency_s": 2.0}]
    s = er.summarize(rows)
    assert s["per_class"]["quote"] == {"n": 0, "correct": 0, "failed": 0, "accuracy": None}          # 没有该类题目：不除零
    assert s["per_class"]["case"]["accuracy"] == 0.0 and s["confusion"]["case->course"] == 1


def test_eval_router_single_call_accepts_routes_and_old_route_format():
    from rag import eval_router as er
    new = er.single_call(RouterLLM({"routes": ["course", "case"], "complexity": "simple", "reason": "t"}), "q")
    old = er.single_call(RouterLLM({"route": "thesis", "complexity": "simple", "reason": "t"}), "q")
    assert (new["pred"], new["pred_routes"], new["error"]) == ("course", ["course", "case"], "")
    assert (old["pred"], old["pred_routes"]) == ("thesis", ["thesis"])
    assert er.single_call(RouterLLM({"routes": ["nonsense"]}), "q")["error"] == "ValueError"


# ---------------------------------------------------------------- 政策库上线后的路由边界

def test_refusal_text_lists_the_six_scopes():
    from rag import config as rc
    for w in ("学位论文写作与指导", "《数据要素》课程知识点", "课程试题", "数据政策法规", "数据法律判例", "总书记关于数字领域的讲话名言"):
        assert w in rc.ROUTER_REFUSAL_TEXT


def test_policy_boundary_eval_set_rule_fallback_all_correct():
    rows = json.load(open("rag/eval_data/router_eval_policy.json", encoding="utf-8"))
    assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
    assert [r["id"] for r in rows if rule_route(r["question"]).routes != r["expected_routes"]] == []


@pytest.mark.parametrize("q,routes", [
    ("数据相关政策法规给一个实例", ["policy"]), ("数据爬虫有什么判例", ["case"]), ("数据安全法第二十一条规定了什么", ["policy"]),
    ("数据政策的演进分几个阶段", ["course"]), ("给我一句名人名言", ["quote"]), ("举一个政策法规的例子", ["policy"]),
    ("上海有哪些公共数据开放的规定", ["policy"]), ("数据二十条讲了什么", ["policy"]), ("论文里怎么引用数据安全法", ["thesis"]),
    ("结合政策法规说明数据确权为什么难", ["course", "policy"]), ("结合数据安全法的规定举一个判例", ["case", "policy"]),
    ("数据政策的功能和传导路径是什么", ["course"])])
def test_rule_fallback_policy_case_course_quote_boundaries(q, routes):
    assert rule_route(q).routes == routes


def test_quote_type_questions_never_fall_to_refusal_in_rules():
    for q in ("给我一句名人名言", "来一句金句", "总书记有什么原话", "引用一句习近平的话"):
        assert rule_route(q).routes == ["quote"], q


def test_llm_policy_combinations_are_accepted():
    for routes in (["course", "policy"], ["case", "policy"], ["quote", "policy"]):
        d = classify("q", routes_llm(routes))
        assert d.routes == routes and d.combined


# ---------------------------------------------------------------- meta 路由：零检索、零模型调用

from rag.unified import META_GREETING  # noqa: E402


def _meta_app(svc=None):
    from rag.unified import UnifiedService, router as unified_router
    app = FastAPI()
    app.state.unified = svc or UnifiedService(router_client=route_llm("meta"))
    app.include_router(unified_router)
    return TestClient(app)


def test_meta_answers_capabilities_without_any_llm_or_retrieval():
    """「你能做什么」不该拒答，也不该检索或调用模型（路由调用本身除外）。"""
    class Boom:
        def chat(self, *a, **k):
            raise AssertionError("meta 不该调用作答模型")
    svc = UnifiedService(router_client=route_llm("meta"))
    svc.course_ctx, svc.course_llm = object(), Boom()
    svc.exam, svc.adaptive = object(), object()
    svc.case_retriever = svc.quote_retriever = svc.policy_retriever = object()
    out = svc.ask("你好，你的知识库目前具备哪些内容呢")
    assert out["route"] == "meta" and out["status"] == "ok" and out["sources"] == []
    assert out["trace"]["meta"]["llm_calls"] == 0 and len(out["trace"]["meta"]["scopes"]) == 6
    for w in ("学位论文", "《数据要素》", "课程试题", "政策法规", "判例", "讲话"):
        assert w in out["answer"]
    assert "抱歉" not in out["answer"]


def test_meta_lists_only_loaded_libraries():
    svc = UnifiedService(router_client=route_llm("meta"))
    svc.course_ctx, svc.course_llm = object(), object()
    svc.exam, svc.adaptive = None, object()          # 试卷库未加载
    svc.case_retriever, svc.quote_retriever, svc.policy_retriever = object(), object(), None
    out = svc.ask("你能做什么")
    assert out["trace"]["meta"]["scopes"] == ["论文", "课程", "判例", "名言"]
    assert "课程试题" not in out["answer"] and "数据政策法规" not in out["answer"]


def test_meta_with_nothing_loaded_says_unavailable():
    out = UnifiedService(router_client=route_llm("meta")).ask("你能做什么")
    assert out["status"] == "unavailable" and out["answer"] == "知识库尚未加载，暂时无法作答。"


def test_greeting_gets_a_short_welcome_not_the_full_list():
    """打招呼只回一句欢迎语（附几个例子），不该甩一整篇能力清单。"""
    svc = UnifiedService(router_client=route_llm("meta"))
    svc.course_ctx, svc.course_llm = object(), object()
    svc.exam, svc.adaptive = object(), object()
    svc.case_retriever = svc.quote_retriever = svc.policy_retriever = object()
    out = svc.ask("你好")
    assert out["route"] == "meta" and out["status"] == "ok"
    assert out["trace"]["meta"]["greeting"] is True and out["trace"]["meta"]["llm_calls"] == 0
    assert out["answer"] == META_GREETING and "我能回答以下范围内的内容" not in out["answer"]
    full = svc.ask("你能做什么")["answer"]
    assert "我能回答以下范围内的内容" in full and len(full) > len(out["answer"])


def test_meta_stream_puts_the_answer_in_done_event():
    svc = UnifiedService(router_client=route_llm("meta"))
    svc.course_ctx, svc.course_llm = object(), object()
    svc.exam, svc.adaptive = object(), object()
    svc.case_retriever = svc.quote_retriever = svc.policy_retriever = object()
    ev = list(svc.ask_stream("你有哪些功能"))
    assert ev[0]["type"] == "route" and ev[0]["route"] == "meta"
    done = [e for e in ev if e.get("type") == "done"][0]
    assert done["status"] == "ok" and "学位论文" in done["answer"] and done["sources"] == []
    assert not [e for e in ev if e.get("type") == "followups"]      # 拒答类不生成追问，meta 同处理


def test_meta_route_is_never_combined():
    from rag.router import normalize_routes
    assert normalize_routes(["meta", "course"]) == ["meta"]
    assert normalize_routes(["course", "meta"]) == ["course"]


# ---------------------------------------------------------------- 「我之前问过什么」：按问答记录作答

def _recall_service(tmp_path, monkeypatch):
    from rag import qa_store
    from rag.unified import UnifiedService

    db = tmp_path / "qa.sqlite"
    monkeypatch.setenv("RAG_QA_DB", str(db))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    qa_store._inited.discard(str(db))
    svc = UnifiedService(router_client=route_llm("meta"))
    svc.course_ctx, svc.course_llm = object(), object()
    return svc, db


def test_recall_lists_my_own_questions_without_llm(tmp_path, monkeypatch):
    """「我之前都问过你哪些内容」列的是这个人问过的问题，不是能力清单；零检索、零模型调用。"""
    svc, db = _recall_service(tmp_path, monkeypatch)
    from rag import qa_store
    for q in ("什么是数据确权", "数据安全法第二十一条规定了什么"):
        qa_store.write({"request_id": "r", "question": q, "answer": "a", "status": "ok"}, user_id="stu-1")
    qa_store.write({"request_id": "r", "question": "别人的问题", "answer": "a", "status": "ok"}, user_id="stu-2")
    out = svc.ask("我之前都问过你哪些内容呢", user_id="stu-1")
    assert out["route"] == "meta" and out["status"] == "ok" and out["sources"] == []
    assert "什么是数据确权" in out["answer"] and "数据安全法第二十一条" in out["answer"]
    assert "别人的问题" not in out["answer"] and "我能回答以下范围内的内容" not in out["answer"]
    assert out["trace"]["meta"]["recall"]["llm_calls"] == 0
    assert out["followups"] == []                      # 回顾类不再顺带生成追问（会跑偏成别的知识点）


def test_recall_with_no_records_says_so_plainly(tmp_path, monkeypatch):
    svc, _ = _recall_service(tmp_path, monkeypatch)
    out = svc.ask("我之前问过你哪些问题", user_id="stu-new")
    assert out["route"] == "meta" and out["status"] == "ok"
    assert "还没有提问记录" in out["answer"]


def test_recall_scoped_to_current_conversation(tmp_path, monkeypatch):
    """带 X-Session-Id / conversation 时只列本次对话（「我们刚才聊了什么」）。"""
    svc, _ = _recall_service(tmp_path, monkeypatch)
    from rag import qa_store
    qa_store.write({"request_id": "r", "question": "这次会话里的问题", "answer": "a", "status": "ok"},
                   user_id="stu-1", session_id="conv-a")
    qa_store.write({"request_id": "r", "question": "另一次会话的问题", "answer": "a", "status": "ok"},
                   user_id="stu-1", session_id="conv-b")
    out = svc.ask("我们刚才聊了什么", user_id="stu-1", session_id="conv-a")
    assert "这次会话里的问题" in out["answer"] and "另一次会话的问题" not in out["answer"]


def test_recall_stream_puts_answer_in_done(tmp_path, monkeypatch):
    svc, _ = _recall_service(tmp_path, monkeypatch)
    from rag import qa_store
    qa_store.write({"request_id": "r", "question": "什么是数据资产", "answer": "a", "status": "ok"}, user_id="stu-1")
    ev = list(svc.ask_stream("我之前问过什么", user_id="stu-1"))
    assert ev[0]["type"] == "route" and ev[0]["route"] == "meta"
    done = [e for e in ev if e.get("type") == "done"][0]
    assert done["status"] == "ok" and "什么是数据资产" in done["answer"]
