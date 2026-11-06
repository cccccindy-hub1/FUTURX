"""作答提速改动的离线测试：pick_for_answer 按相关分、chat_stream、TracingLLM 透传、answer 流式、
pipeline on_event、/auto/ask/stream 的 SSE、COURSE_FILTER_MODEL 只影响筛选。全部 mock，不发真实调用。"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag import concurrency
from rag.course import answer, config, llm, passage_filter, pipeline
from rag.course.llm import LLMResult, Trace, TracingLLM
from rag.course.passage_filter import FilterResult, Verdict
from rag.course.recall import BM25_VIA, Reason, Recalled
from rag.unified import UnifiedService, router as unified_router
from test_course_pipeline import SENTINEL_KEY, ScriptedLLM, ctx, jtxt  # noqa: F401  (ctx 为课程夹具)


@pytest.fixture(autouse=True)
def single_mode(request, monkeypatch):
    if "ctx" in request.fixturenames:
        request.getfixturevalue("ctx")                 # ctx 夹具把模式设为 agent；这里设回 single
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(config, "COURSE_SELECT_FAST_PATH", False)
    monkeypatch.setattr(config, "COURSE_FILTER_SKIP_MAX", 0)       # 本文件的管线测试按「总是筛选」写脚本


def pick() -> LLMResult:
    return jtxt({"question_concepts": ["C01-0001"], "related_concepts": ["C01-0003"], "reason": "r"})


def filt(step, msgs) -> LLMResult:
    refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
    return jtxt({"direct": refs[:1], "background": refs[1:]})


def route_llm(route="course"):
    return type("R", (), {"chat": lambda self, *a, **k: LLMResult(content=json.dumps(
        {"route": route, "complexity": "simple", "reason": "t"}), latency_s=0.1)})()


# ---------------------------------------------------------------- pick_for_answer：相关分排序与截断

def _it(cid, text, path=("章",)):
    return Recalled(cid, "p", list(path), text, "body", ["P1"],
                    [Reason(concept_id="C", via="edge_evidence", anchor="P1", evidence="verified", source_tier="expert")])


def _fr(grades: dict) -> FilterResult:
    return FilterResult(verdicts=[Verdict(c, g != "irrelevant", "", f"S{n}", g) for n, (c, g) in enumerate(grades.items(), 1)])


def test_pick_for_answer_orders_by_relevance_within_grade_and_truncates_least_relevant():
    q = "数据资产入表的基本思路"
    items = [_it("far1", "完全无关的话题甲"), _it("near", "数据资产入表的基本思路是纳入报表", ("入表",)),
             _it("far2", "另一个无关话题乙"), _it("mid", "数据资产的内涵", ("资产",)), _it("bg_near", "入表思路补充", ("入表",))]
    fr = _fr({"far1": "direct", "near": "direct", "far2": "direct", "mid": "direct", "bg_near": "background"})
    kept, capped = passage_filter.pick_for_answer(items, fr, 3, question=q)
    ids = [i.chunk_id for i in kept]
    assert ids[:2] == ["near", "mid"]                                    # 同为 direct：相关分高者在前
    assert ids[2] in ("far1", "far2") and len(kept) == 3                 # 第 3 名额给 direct（无关的两块之一），而不是 background
    assert set(capped) == ({"far1", "far2"} - {ids[2]}) | {"bg_near"} and len(capped) == 2     # 被截掉的是最不相关的 direct 与整个 background


def test_pick_for_answer_direct_before_background_even_if_background_more_relevant():
    items = [_it("d", "无关"), _it("b", "数据资产入表的基本思路", ("入表",))]
    kept, _ = passage_filter.pick_for_answer(items, _fr({"d": "direct", "b": "background"}), 5, question="数据资产入表")
    assert [i.chunk_id for i in kept] == ["d", "b"]


def test_pick_for_answer_without_question_keeps_recall_order_and_excludes_irrelevant():
    items = [_it("a", "x"), _it("b", "y"), _it("c", "z")]
    kept, capped = passage_filter.pick_for_answer(items, _fr({"a": "direct", "b": "irrelevant", "c": "direct"}), 5)
    assert [i.chunk_id for i in kept] == ["a", "c"] and capped == []


def test_pick_for_answer_is_deterministic_on_ties():
    items = [_it(f"x{i}", "同样的文本") for i in range(6)]
    fr = _fr({i.chunk_id: "direct" for i in items})
    a = passage_filter.pick_for_answer(items, fr, 3, question="同样")[0]
    b = passage_filter.pick_for_answer(items, fr, 3, question="同样")[0]
    assert [i.chunk_id for i in a] == [i.chunk_id for i in b] == ["x0", "x1", "x2"]    # 并列保持召回顺序


def test_answer_max_chunks_default_is_ten():
    import os
    if "COURSE_ANSWER_MAX_CHUNKS" not in os.environ:
        assert config.COURSE_ANSWER_MAX_CHUNKS == 10


# ---------------------------------------------------------------- chat_stream（假的分片流）

def _chunk(content=None, reasoning=None, usage=None, choices=True):
    delta = NS(content=content, reasoning_content=reasoning)
    return NS(choices=[NS(delta=delta)] if choices else [], usage=usage)


def _usage(p, c, r):
    return NS(prompt_tokens=p, completion_tokens=c, completion_tokens_details=NS(reasoning_tokens=r))


def _client(monkeypatch, stream_chunks, seen=None, raise_after=None):
    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            def create(**kw):
                if seen is not None:
                    seen.update(kw)
                def gen():
                    for n, ch in enumerate(stream_chunks):
                        if raise_after is not None and n == raise_after:
                            raise RuntimeError(f"401 bad key {SENTINEL_KEY}")
                        yield ch
                return gen()
            self.chat = NS(completions=NS(create=create))
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=FakeOpenAI))
    return llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY)


def test_chat_stream_concatenates_usage_and_counts_reasoning_length_only(monkeypatch):
    seen, got = {}, []
    c = _client(monkeypatch, [_chunk(reasoning="想一想"), _chunk(reasoning="再想"), _chunk(content="数据"),
                              _chunk(content="要素[S1]"), _chunk(usage=_usage(120, 30, 7), choices=False)], seen)
    res = c.chat_stream("answer", [{"role": "user", "content": "x"}], model="m", on_delta=got.append)
    assert got == ["数据", "要素[S1]"] and res.content == "数据要素[S1]"                 # 片段按序回调，汇总一致
    assert (res.prompt_tokens, res.completion_tokens, res.reasoning_tokens) == (120, 30, 7)
    assert res.reasoning_chars == len("想一想再想") and "想一想" not in res.content          # reasoning 只计长度，不进正文
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True} and seen["temperature"] == 0
    assert seen["extra_body"] == {"reasoning": {"type": "enabled", "effort": "low"}}      # 作答默认低强度思考
    assert res.latency_s >= 0


def test_chat_stream_error_is_redacted_and_wrapped(monkeypatch):
    c = _client(monkeypatch, [_chunk(content="a"), _chunk(content="b")], raise_after=1)
    got = []
    with pytest.raises(llm.LLMError) as e:
        c.chat_stream("answer", [{"role": "user", "content": "x"}], model="m", on_delta=got.append)
    assert SENTINEL_KEY not in str(e.value) and "***" in str(e.value) and got == ["a"]


def test_chat_stream_ignores_empty_and_usage_only_chunks(monkeypatch):
    c = _client(monkeypatch, [_chunk(content=""), _chunk(content=None), _chunk(content="x"), _chunk(usage=_usage(1, 2, 0), choices=False)])
    got = []
    res = c.chat_stream("answer", [], model="m", on_delta=got.append)
    assert got == ["x"] and res.content == "x" and res.completion_tokens == 2


# ---------------------------------------------------------------- TracingLLM 透传

class StreamingInner:
    def __init__(self):
        self.calls = []

    def reasoning_for(self, step):
        return "low"

    def chat(self, step, messages, **kw):
        self.calls.append(("chat", step))
        return LLMResult(content="full", prompt_tokens=1, completion_tokens=1)

    def chat_stream(self, step, messages, *, model, on_delta):
        self.calls.append(("stream", step))
        on_delta("he"); on_delta("llo")
        return LLMResult(content="hello", prompt_tokens=10, completion_tokens=5, reasoning_tokens=3, reasoning_chars=9, latency_s=0.5)


def test_tracing_llm_passes_through_chat_stream_and_records_stream_flag():
    t = Trace()
    tl = TracingLLM(StreamingInner(), t)
    got = []
    res = tl.chat_stream("answer", [], model="m", on_delta=got.append)
    assert got == ["he", "llo"] and res.content == "hello"
    c = t.calls[0]
    assert c["step"] == "answer" and c["stream"] is True and c["reasoning"] == "low" and c["reasoning_tokens"] == 3
    tl.chat("select", [], model="m")
    assert t.calls[1]["stream"] is False and t.totals()["calls"] == 2


def test_tracing_llm_hides_chat_stream_when_inner_has_none():
    tl = TracingLLM(ScriptedLLM([]), Trace())
    assert not hasattr(tl, "chat_stream")
    with pytest.raises(AttributeError):
        tl.chat_stream


# ---------------------------------------------------------------- answer：流式与非流式一致

def test_answer_streaming_matches_non_streaming(ctx):
    from test_course_pipeline import _items
    kept = _items(ctx, "P0005", "P0008")[:2]
    text = "要点甲[S1]，要点乙[S2]，结论[本体]。"

    class S:
        def chat(self, step, messages, **kw):
            return LLMResult(content=text, prompt_tokens=7, completion_tokens=3)

        def chat_stream(self, step, messages, *, model, on_delta):
            for ch in (text[:5], text[5:11], text[11:]):
                on_delta(ch)
            return LLMResult(content=text, prompt_tokens=7, completion_tokens=3)
    got = []
    a = answer.answer_question("q", kept, [], S(), on_delta=got.append)
    b = answer.answer_question("q", kept, [], S())
    assert "".join(got) == a.text == "要点甲[1]，要点乙[2]，结论。"                      # delta 已是学生可见文本：[n]，本体标记已去掉
    assert "S1" not in "".join(got) and kept[0].chunk_id not in "".join(got)
    assert a.to_dict() == b.to_dict() and "[S1]" not in a.text and kept[0].chunk_id not in a.text   # 流式与非流式结果一致


def test_answer_on_delta_falls_back_to_chat_when_client_has_no_stream(ctx):
    from test_course_pipeline import _items
    kept = _items(ctx, "P0005")[:1]
    got = []
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="整段[S1]")]), on_delta=got.append)
    assert got == ["整段[1]"] and "[1]" in ar.text and kept[0].chunk_id not in ar.text    # 整段内容一次性回调，且已是序号


# ---------------------------------------------------------------- pipeline on_event

class StreamScript(ScriptedLLM):
    """脚本化 LLM + chat_stream：作答那一步改走流式，分三段回调。"""

    def chat_stream(self, step, messages, *, model, on_delta):
        res = self.chat(step, messages, model=model)
        c = res.content
        for part in (c[:6], c[6:12], c[12:]):
            if part:
                on_delta(part)
        return res


def _course_script():
    return [pick(), filt, LLMResult(content="数据要素是参与价值创造的数据资源[S1]；其依赖数据权属[本体]。", prompt_tokens=5, completion_tokens=5)]


def test_pipeline_on_event_order_and_content(ctx):
    events = []
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=StreamScript(_course_script()), on_event=events.append)
    kinds = [(e["type"], e.get("stage")) for e in events]
    assert kinds[:2] == [("stage", "select"), ("stage", "filter")] and {k[0] for k in kinds[2:]} == {"delta", "materials"}
    sel, flt = events[0], events[1]
    assert sel["concepts"] and flt["kept"] >= 1 and flt["recalled"] >= flt["kept"]
    mats = [e for e in events if e["type"] == "materials"]
    assert [m["n"] for m in mats] == [1] and mats[0]["chunk_id"].startswith("de:") and mats[0]["path"]
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "[1]" in text and "[S1]" not in text and "de:" not in text         # delta 是学生可见文本：序号，没有 [S#] 和 chunk_id
    assert text == r["answer"]["text"]                                         # 流式拼起来与最终正文逐字一致
    # materials 在序号分配时发出：必须早于第一个带 "[1]" 的 delta，这样前端渲染 [1] 时来源已就绪
    first_with_marker = next(i for i, e in enumerate(events) if e["type"] == "delta" and "[1]" in e["text"])
    assert events.index(mats[0]) < first_with_marker
    assert r["trace"].calls[-1]["step"] == "answer" and r["trace"].calls[-1]["stream"] is True


def test_pipeline_without_on_event_is_unchanged_and_not_streaming(ctx):
    llm_ = StreamScript(_course_script())
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=llm_)
    assert r["answer"]["text"] and r["trace"].calls[-1]["stream"] is False


def test_pipeline_on_event_materials_match_answer_refs(ctx):
    events = []
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=StreamScript(_course_script()), on_event=events.append)
    mats = [e for e in events if e["type"] == "materials"]
    assert [m["chunk_id"] for m in mats] == r["answer"]["citations"] == [s["chunk_id"] for s in r["answer"]["sources"]]
    assert [m["n"] for m in mats] == [s["n"] for s in r["answer"]["sources"]] == [1]
    assert [m["path"] for m in mats] == [s["path"] for s in r["answer"]["sources"]]


# ---------------------------------------------------------------- /auto/ask/stream（SSE）

def _app(svc):
    app = FastAPI()
    app.state.unified = svc
    app.include_router(unified_router)
    return TestClient(app)


def _events(resp):
    out = []
    for blk in resp.text.split("\n\n"):
        if blk.startswith("data: "):
            out.append(json.loads(blk[6:]))
    return out


def test_sse_course_route_event_sequence(ctx):
    svc = UnifiedService(router_client=route_llm("course"), course_ctx=ctx, course_llm=StreamScript(_course_script()))
    r = _app(svc).post("/auto/ask/stream", json={"question": "什么是数据要素"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "route" and kinds[-1] == "done" and ev[0]["route"] == "course"
    first = {k: kinds.index(k) for k in ("route", "stage", "materials", "delta", "done")}
    assert first["route"] < first["stage"] < first["delta"] < first["done"] and first["materials"] < first["done"]
    marker_at = next(i for i, e in enumerate(ev) if e["type"] == "delta" and "[1]" in e["text"])
    assert first["materials"] < marker_at                                       # materials 先于第一个带 [1] 的 delta
    assert kinds.count("done") == 1 and "error" not in kinds
    done = ev[-1]
    text = "".join(e["text"] for e in ev if e["type"] == "delta")
    prefix = f"de:{config.BOOK_VERSION}:"                              # 教材版本随 COURSE_BOOK_VERSION 变化，不写死
    assert "[1]" in text and "[S1]" not in text and prefix not in text and prefix not in done["answer"]     # delta 与 done 的正文都只有序号
    assert text.strip() == done["answer"].strip()
    assert [s["n"] for s in done["sources"]] == [1] and done["sources"][0]["chunk_id"].startswith(prefix) and done["sources"][0]["path"]
    assert done["status"] == "ok" and done["sources"] and done["trace"]["course"]["totals"]["calls"] == 3


def test_sse_non_course_routes_are_not_streamed_but_complete_in_done():
    for route, status in (("off_topic", "refused"), ("policy", "unavailable")):
        r = _app(UnifiedService(router_client=route_llm(route))).post("/auto/ask/stream", json={"question": "q"})
        ev = _events(r)
        assert [e["type"] for e in ev] == ["route", "done"] and ev[0]["route"] == route
        assert ev[-1]["status"] == status and ev[-1]["answer"]


def test_sse_thesis_route_runs_adaptive_and_puts_result_in_done():
    class Adaptive:
        def run(self, q, top_k, complexity):
            return {"status": "ok", "answer": "论文答案", "sources": [{"source": "a"}], "contexts": ["c"], "trace": {"path": "simple"}}
    svc = UnifiedService(router_client=route_llm("thesis"))
    svc.adaptive = Adaptive()
    ev = _events(_app(svc).post("/auto/ask/stream", json={"question": "怎么写开题"}))
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["answer"] == "论文答案" and ev[-1]["sources"] == [{"source": "a"}]


def test_sse_mid_generation_error_emits_error_event_with_type_name_only(ctx):
    class Boom(StreamScript):
        def chat_stream(self, step, messages, *, model, on_delta):
            on_delta("已经生成了一半")
            raise RuntimeError(f"upstream 401 key={SENTINEL_KEY} secret detail")
    svc = UnifiedService(router_client=route_llm("course"), course_ctx=ctx, course_llm=Boom(_course_script()))
    r = _app(svc).post("/auto/ask/stream", json={"question": "什么是数据要素"})
    assert r.status_code == 200
    ev = _events(r)
    kinds = [e["type"] for e in ev]
    assert kinds[-1] == "error" and "done" not in kinds and "delta" in kinds
    assert ev[-1] == {"type": "error", "error": "RuntimeError"}
    assert SENTINEL_KEY not in r.text and "secret detail" not in r.text and "upstream" not in r.text


def test_sse_error_before_course_stage_also_error_event(ctx):
    class Bad(ScriptedLLM):
        def chat(self, *a, **k):
            raise ValueError("bad select")
    svc = UnifiedService(router_client=route_llm("course"), course_ctx=ctx, course_llm=Bad([]))
    ev = _events(_app(svc).post("/auto/ask/stream", json={"question": "什么是数据要素"}))
    assert ev[0]["type"] == "route" and ev[-1] == {"type": "error", "error": "ValueError"}


def test_sse_course_unavailable_returns_done_unavailable():
    ev = _events(_app(UnifiedService(router_client=route_llm("course"))).post("/auto/ask/stream", json={"question": "q"}))
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["status"] == "unavailable"


def test_sse_include_trace_false_omits_trace_from_done():
    ev = _events(_app(UnifiedService(router_client=route_llm("off_topic"))).post(
        "/auto/ask/stream", json={"question": "q", "include_trace": False}))
    assert "trace" not in ev[-1]


def test_sse_requires_api_key_when_configured(monkeypatch):
    import rag.api as api
    monkeypatch.setattr(api, "API_KEYS", ["k1"])
    c = _app(UnifiedService(router_client=route_llm("off_topic")))
    assert c.post("/auto/ask/stream", json={"question": "q"}).status_code == 401
    ok = c.post("/auto/ask/stream", json={"question": "q"}, headers={"Authorization": "Bearer k1"})
    assert ok.status_code == 200 and _events(ok)[-1]["type"] == "done"


def test_sse_generator_closed_early_cancels_pipeline_thread(ctx):
    started, release = threading.Event(), threading.Event()

    class Slow(StreamScript):
        def chat_stream(self, step, messages, *, model, on_delta):
            on_delta("a"); started.set()
            release.wait(3)
            on_delta("b")                                   # 此时已被取消：回调会抛 _Cancelled，线程静默结束
            return LLMResult(content="ab")
    svc = UnifiedService(router_client=route_llm("course"), course_ctx=ctx, course_llm=Slow(_course_script()))
    g = svc.ask_stream("什么是数据要素")
    for ev in g:
        if ev["type"] == "delta":
            break
    assert started.wait(3)
    g.close()                                               # 客户端断开
    release.set()
    deadline = time.time() + 3
    while time.time() < deadline and any(t.name == "auto-ask-stream" and t.is_alive() for t in threading.enumerate()):
        time.sleep(0.02)
    assert not any(t.name == "auto-ask-stream" and t.is_alive() for t in threading.enumerate())


# ---------------------------------------------------------------- 并发：流式名额持有到整个响应结束

def test_auto_ask_stream_is_in_concurrency_paths():
    assert "/auto/ask/stream" in concurrency.RequestConcurrencyMiddleware._PATHS and "/auto/ask" in concurrency.RequestConcurrencyMiddleware._PATHS


def test_stream_slot_held_until_response_finishes(ctx, monkeypatch):
    release, started = threading.Event(), threading.Event()
    monkeypatch.setenv("RAG_QUEUE_TIMEOUT_S", "0.5")          # 排队等待：超时才 503

    class Hold(StreamScript):
        def chat_stream(self, step, messages, *, model, on_delta):
            on_delta("x"); started.set()
            assert release.wait(30)
            return LLMResult(content="x[S1]")
    svc = UnifiedService(router_client=route_llm("course"), course_ctx=ctx, course_llm=Hold(_course_script()))
    app = FastAPI()
    app.state.unified = svc
    app.add_middleware(concurrency.RequestConcurrencyMiddleware, max_requests=1)
    app.include_router(unified_router)
    app.get("/health")(lambda: {"status": "ok"})
    box = {}

    def first():
        with TestClient(app) as c:
            box["r"] = c.post("/auto/ask/stream", json={"question": "什么是数据要素"})
    t = threading.Thread(target=first)
    t.start()
    assert started.wait(30)
    with TestClient(app) as c2:
        busy = c2.post("/auto/ask/stream", json={"question": "q"})
        # 满员时先排队，等不到名额才 503；等待期间流仍在进行 ⇒ 名额确实被持有到响应结束
        assert busy.status_code == 503 and busy.json()["error"]["code"] == "server_busy"
        assert c2.get("/health").status_code == 200
        busy2 = c2.post("/auto/ask", json={"question": "q"})
        assert busy2.status_code == 503                                                       # /auto/ask 与流式共用同一个名额池
    release.set()
    t.join(30)
    assert box["r"].status_code == 200 and _events(box["r"])[-1]["type"] == "done"
    with TestClient(app) as c3:                                                               # 流完整结束后名额已释放
        assert c3.post("/auto/ask/stream", json={"question": "q"}).status_code != 503


# ---------------------------------------------------------------- COURSE_FILTER_MODEL：只有 filter 步用它

class ModelRecorder(ScriptedLLM):
    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.models = getattr(self, "models", [])
        self.models.append((step, model))
        return super().chat(step, messages, model=model, tools=tools, json_mode=json_mode)


def test_filter_model_env_only_affects_filter_step(ctx, monkeypatch):
    monkeypatch.setattr(config, "COURSE_LLM_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setattr(config, "COURSE_SELECT_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setattr(config, "COURSE_ANSWER_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setattr(config, "COURSE_FILTER_MODEL", "deepseek/deepseek-v4-flash")
    course_llm = ModelRecorder(_course_script())
    rec = []
    router = type("R", (), {"chat": lambda self, step, messages, *, model, **k: (
        rec.append((step, model)), LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"})))[1]})()
    out = UnifiedService(router_client=router, course_ctx=ctx, course_llm=course_llm).ask("什么是数据要素")
    assert out["status"] == "ok"
    # 追问走 filter 同档模型（LIB_PICK_MODEL），不计入作答管道
    core = [(s, m) for s, m in course_llm.models if s != "followup"]
    assert dict(core) == {"select": "deepseek/deepseek-v4-pro", "filter": "deepseek/deepseek-v4-flash",
                          "answer": "deepseek/deepseek-v4-pro"}
    follow = [m for s, m in course_llm.models if s == "followup"]
    assert all(m == config.LIB_PICK_MODEL for m in follow)                                   # 追问用小模型
    assert rec == [("route", "deepseek/deepseek-v4-pro")]                                    # 路由仍是 pro
    assert len(core) + len(rec) == 4                                                          # 作答管道仍为 4


def test_filter_model_defaults_to_course_model_and_answer_reasoning_stays_low():
    import os
    if "COURSE_ANSWER_REASONING" not in os.environ:
        assert config.COURSE_REASONING["answer"] == "low"                                    # 作答保留低强度思考
    if "COURSE_FILTER_REASONING" not in os.environ:
        assert config.COURSE_REASONING["filter"] == "disabled"


def test_env_example_documents_flash_filter_without_changing_default():
    txt = open(".env.example", encoding="utf-8").read()
    assert "COURSE_FILTER_MODEL=deepseek/deepseek-v4-flash" in txt and "# COURSE_ANSWER_REASONING=low" in txt
    assert not any(l.startswith("COURSE_FILTER_MODEL=") for l in txt.splitlines())          # 只是注释示例，没有生效


def test_eval_e2e_config_records_three_models():
    src = open("rag/course/eval_e2e.py", encoding="utf-8").read()
    for k in ("select_model", "filter_model", "answer_model"):
        assert f'"{k}"' in src
