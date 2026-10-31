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
