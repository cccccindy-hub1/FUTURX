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


def test_rule_generic_words_alone_do_not_make_policy():
    assert rule_route("学校的规定通知我几点下课").route == "off_topic"


# ---------------------------------------------------------------- 统一入口：拒答 / 占位 / 分发

class Boom:
    def __getattr__(self, n):
        raise AssertionError("拒答/占位不应触碰检索或作答")


def test_off_topic_refuses_without_retrieval_or_generation():
    svc = UnifiedService(rag=Boom(), router_client=route_llm("off_topic"))
    out = svc.ask("今天天气怎么样")
    assert out["route"] == "off_topic" and out["status"] == "refused"
    assert out["answer"] == config.ROUTER_REFUSAL_TEXT and out["sources"] == []
    assert svc.adaptive is not None and out["trace"]["router"]["source"] == "llm"


def test_router_failure_on_irrelevant_question_still_refuses():
    svc = UnifiedService(rag=Boom(), router_client=RouterLLM(exc=TimeoutError("t")))
    out = svc.ask("给我讲个笑话")
    assert out["status"] == "refused" and out["trace"]["router"]["source"] == "rules_refuse"


def test_policy_returns_placeholder():
    svc = UnifiedService(rag=Boom(), router_client=route_llm("policy"))
    out = svc.ask("数据二十条讲了什么")
    assert out["route"] == "policy" and out["status"] == "not_available" and "尚未上线" in out["answer"]


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
    assert "[de:x]" in text
    assert "[S#]" not in prompts.ANSWER_SYSTEM.replace("“[S#]”", "")      # 提示词不再把 [S#] 当示例


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
    assert "不要说明本体材料缺失或未提供" in sysm
    rel = ctx.index.relations_among(["C01-0001", "C01-0003"])
    llm2 = ScriptedLLM([LLMResult(content="回答[S1]")])
    answer.answer_question("q", kept, rel, llm2)
    assert "【本体结构信息，非教材原文】" in llm2.calls[0]["messages"][1]["content"]


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
    assert text == "见[de:x]。" and valid == ["de:x"] and invalid == []


def test_sanitize_fullwidth_invalid_placeholder_and_ontology_tags():
    from rag.course.answer import sanitize_citations
    text, valid, invalid, onto = sanitize_citations("甲［S9］乙【S#】丙【本体】丁［de:fake］", {"S1": "de:x"})
    assert "S9" not in text and "S#" not in text and "de:fake" not in text and valid == []
    assert set(invalid) == {"S9", "de:fake"} and onto and "[本体]" in text
