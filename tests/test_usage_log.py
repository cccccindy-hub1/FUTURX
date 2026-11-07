"""调用日志（需求5）测试：字段完整、并发写安全、异常隔离、保留期清理、key 指纹。"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from rag import usage_log


@pytest.fixture()
def logdir(tmp_path, monkeypatch):
    d = tmp_path / "usage"
    monkeypatch.setenv("RAG_LOG_DIR", str(d))
    monkeypatch.setenv("RAG_LOG_RETENTION_DAYS", "90")
    return d


def _read(d: Path) -> list[dict]:
    return [json.loads(line) for p in sorted(d.glob("*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_record_has_all_required_fields(logdir):
    from rag.course.llm import LLMResult

    rec = usage_log.record("什么是数据要素？", "http", "abcd1234")
    with rec:
        rec.record_llm("select", "deepseek/deepseek-v4-pro", 11158, 342, 2.59)
        rec.record_llm("answer", "deepseek/deepseek-v4-pro", 9000, 400, 5.1)
        rec.first_delta()
        rec.stage("followup", 1.2)
        rec.finish({
            "route": "course", "routes": ["course"], "complexity": "simple", "status": "ok",
            "answer": "数据要素是参与价值创造的数据资源[1]。", "followups": ["数据要素和数据资产区别？"],
            "sources": [{"n": 1, "chunk_id": "de:1102:ch01.1.1:000:a", "path": "第一章 › 第一节"}],
            "trace": {"router": {"latency_s": 2.0}, "prematch": {"ms": 12.5},
                      "course": {"totals": {"by_step": {"select": {"latency_s": 2.59},
                                                        "filter": {"latency_s": 2.6},
                                                        "answer": {"latency_s": 5.1}}}}},
        })
    rows = _read(logdir)
    assert len(rows) == 1
    r = rows[0]
    for k in ("request_id", "ts_start", "ts_end", "duration_ms", "key_id", "question", "route", "routes",
              "complexity", "status", "answer", "followups", "sources", "stages", "llm_calls",
              "time_to_first_delta_ms", "error", "origin"):
        assert k in r, k
    assert r["key_id"] == "abcd1234" and r["origin"] == "http"
    assert r["stages"]["router"] == 2000.0 and r["stages"]["prematch"] == 12.5
    assert r["stages"]["select"] == 2590.0 and r["stages"]["answer"] == 5100.0
    assert r["stages"]["followup"] == 1200.0                  # 显式 stage 覆盖 trace 推算
    assert r["time_to_first_delta_ms"] is not None and r["time_to_first_delta_ms"] >= 0
    assert [c["step"] for c in r["llm_calls"]] == ["select", "answer"]
    assert r["llm_calls"][0]["prompt_tokens"] == 11158
    assert r["llm_calls"][0]["latency_ms"] == 2590.0


def test_finish_is_idempotent(logdir):
    rec = usage_log.record("q", "eval")
    with rec:
        rec.finish({"route": "exam", "status": "ok", "answer": "a"})
        rec.finish({"route": "exam", "status": "ok", "answer": "b"})
    rows = _read(logdir)
    assert len(rows) == 1 and rows[0]["answer"] == "a"


def test_exception_in_body_records_error_without_raising(logdir):
    with pytest.raises(RuntimeError):
        with usage_log.record("q", "eval"):
            raise RuntimeError("boom")
    rows = _read(logdir)
    assert len(rows) == 1 and rows[0]["error"] == "RuntimeError"


def test_write_failure_never_propagates(monkeypatch, tmp_path):
    """目录不可写时只记警告，不影响调用方。"""
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(usage_log, "_append", lambda record: (_ for _ in ()).throw(OSError("disk full")))
    rec = usage_log.record("q", "http")
    assert rec.finish({"answer": "x"}) is not None            # 不抛异常
    usage_log.simple_record("q2")                              # 同样不抛


def test_concurrent_writes_are_not_interleaved(logdir):
    n = 64
    barrier = threading.Barrier(8)

    def work(i):
        barrier.wait()
        with usage_log.record(f"问题{i}", "eval") as rec:
            rec.finish({"route": "thesis", "status": "ok", "answer": f"答案{i}"})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = _read(logdir)
    assert len(rows) == n
    assert {r["question"] for r in rows} == {f"问题{i}" for i in range(n)}
    assert len({r["request_id"] for r in rows}) == n


def test_llm_report_without_current_record_is_noop():
    usage_log.record_llm("select", "m", 1, 2, 0.1)             # 不抛
    usage_log.stage("x", 1.0)
    usage_log.first_delta()


def test_retention_removes_old_files(logdir, monkeypatch):
    logdir.mkdir(parents=True)
    old = logdir / "19700101.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    recent = logdir / "29991231.jsonl"
    recent.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("RAG_LOG_RETENTION_DAYS", "90")
    monkeypatch.setattr(usage_log, "_last_cleanup_day", "")
    usage_log.write({"request_id": "x"})
    assert not old.exists() and recent.exists()


def test_key_id_is_stable_and_irreversible():
    import hashlib

    k = "sk-test-abcdef"
    assert usage_log.key_id(k) == hashlib.sha256(k.encode()).hexdigest()[:8]
    assert k not in usage_log.key_id(k)
    assert usage_log.key_id_from_header(f"Bearer {k}") == usage_log.key_id(k)
    assert usage_log.key_id_from_header("") is None
    assert usage_log.key_id_from_header("Basic xx") is None


def test_simple_record_shape(logdir):
    usage_log.simple_record("怎么写文献综述", key_id_value="deadbeef", route="thesis",
                            answer="先说结构。", stages={"answer": 900.0}, request_id="chatcmpl-1")
    r = _read(logdir)[0]
    assert r["request_id"] == "chatcmpl-1" and r["route"] == "thesis"
    assert r["stages"]["answer"] == 900.0 and r["stages"]["select"] is None
    assert r["key_id"] == "deadbeef" and r["origin"] == "http"


def test_stream_route_kept_when_client_disconnects(logdir):
    """流式：route 事件先到、done 不来（客户端断开）时，记录仍带 route 与 client_disconnect。"""
    from rag.course.llm import LLMResult
    from rag.unified import UnifiedService

    class Router:
        def chat(self, step, messages, **kw):
            import json as _j
            return LLMResult(content=_j.dumps({"route": "exam", "routes": ["exam"], "complexity": "simple"}),
                             latency_s=0.01, prompt_tokens=10, completion_tokens=5)

    class SlowLLM:
        def chat(self, step, messages, **kw):
            raise RuntimeError("should not be called in this test")

    class Bank:
        def search(self, q):
            return {"status": "no_match", "items": [], "strategy": "bm25", "intent": "find"}

    svc = UnifiedService(router_client=Router(), lib_llm=SlowLLM(), exam_bank=Bank())
    gen = svc.ask_stream("给我出三道判断题", origin="http", key_id="cafebabe")
    first = next(gen)
    assert first["type"] == "route" and first["route"] == "exam"
    gen.close()                                     # 模拟客户端断开
    rows = _read(logdir)
    assert len(rows) == 1
    assert rows[0]["route"] == "exam" and rows[0]["routes"] == ["exam"]
    assert rows[0]["error"] == "client_disconnect"


def test_tests_default_dir_is_temp(monkeypatch):
    """pytest 进程不写仓库 logs/（假答案会污染真实统计）。"""
    monkeypatch.delenv("RAG_LOG_DIR", raising=False)
    assert "futurx_usage_test" in str(usage_log.log_dir())


def test_lock_timing_separates_wait_from_hold(logdir):
    """检索锁：排队等待与持锁时长分别记入 locks，不互相混算。"""
    with usage_log.record("q", "eval") as rec:
        rec.note_lock("retrieval", 120.0, 340.0)
        rec.note_lock("retrieval", 30.0, 210.0)      # 同请求第二次检索：累加
        rec.finish({"route": "thesis", "status": "ok", "answer": "x"})
    row = _read(logdir)[0]
    lock = row["locks"]["retrieval"]
    assert lock["lock_wait_ms"] == 150.0
    assert lock["retrieval_ms"] == 550.0
    assert lock["locks"] == 2


def test_lock_timing_after_finish_is_ignored(logdir):
    with usage_log.record("q", "eval") as rec:
        rec.finish({"route": "exam", "status": "ok", "answer": "x"})
        rec.note_lock("retrieval", 999.0, 999.0)
    assert _read(logdir)[0]["locks"] == {}


def test_note_lock_without_current_record_is_noop():
    usage_log.note_lock("retrieval", 1.0, 2.0)       # 不抛异常


def test_retrieve_result_reports_lock_timing(monkeypatch):
    """真实调用 RAGService.retrieve_result：检索耗时照常落 locks.retrieval_ms（不再有锁等待）。"""
    import time as _t

    import rag.query as query

    svc = object.__new__(query.RAGService)
    from threading import local
    svc._request_state = local()
    svc._retrieval_query = lambda q: q
    svc.fetch_k = 20
    svc.where = None
    svc.expand_context = False
    svc.collection_name = "x"

    def search(q, **kwargs):
        _t.sleep(0.02)
        return q
    svc.retriever = type("R", (), {"search": staticmethod(search)})()

    with usage_log.record("q", "eval") as rec:
        assert svc.retrieve_result("hello") == "hello"
        rec.finish({"route": "thesis", "status": "ok", "answer": "x"})
    rows = _read(Path(usage_log.log_dir()))
    lock = [r for r in rows if r.get("locks")][0]["locks"]["retrieval"]
    assert lock["locks"] == 1
    assert lock["retrieval_ms"] >= 20
    assert lock["lock_wait_ms"] == 0.0              # 已取消全局检索锁


def test_tracing_llm_reports_every_call(logdir):
    """课程管线的 llm 被 TracingLLM 包着：chat 与 chat_stream 都必须上报到调用日志。"""
    from rag.course.llm import LLMResult, Trace, TracingLLM

    class Inner:
        def chat(self, step, messages, **kw):
            return LLMResult(content="x", prompt_tokens=11, completion_tokens=2, latency_s=0.01)

        def chat_stream(self, step, messages, *, model, on_delta):
            on_delta("a")
            return LLMResult(content="a", prompt_tokens=33, completion_tokens=44, latency_s=0.02)

    with usage_log.record("q", "eval") as rec:
        t = TracingLLM(Inner(), Trace())
        t.chat("select", [], model="m1")
        t.chat_stream("answer", [], model="m2", on_delta=lambda _x: None)
        rec.finish({"route": "course", "status": "ok", "answer": "x"})
    calls = _read(logdir)[0]["llm_calls"]
    assert [(c["step"], c["model"]) for c in calls] == [("select", "m1"), ("answer", "m2")]
    assert calls[0]["prompt_tokens"] == 11 and calls[1]["completion_tokens"] == 44


def test_tracing_llm_does_not_double_report(logdir):
    """内层会自己上报（OpenAICompatClient）时，包装器不能再记一遍，否则 token 翻倍。"""
    from rag.course.llm import LLMResult, TracingLLM, Trace

    class SelfReporting:
        reports_usage = True

        def chat(self, step, messages, **kw):
            res = LLMResult(content="x", prompt_tokens=1, completion_tokens=2, latency_s=0.01)
            usage_log.record_llm(step, "m", 1, 2, 0.01)
            return res

    with usage_log.record("q", "eval") as rec:
        TracingLLM(SelfReporting(), Trace()).chat("select", [], model="m")
        rec.finish({"answer": "x"})
    assert len(_read(logdir)[0]["llm_calls"]) == 1


def test_stream_log_survives_threadpool_resume(logdir):
    """回归：Starlette 把同步生成器交给线程池逐段 next()，每段都是新的上下文副本，
    生成器内 / 其后台线程里设置的 contextvar 会丢。流式请求必须逐段重新激活记录，
    否则只记到第一条 route 调用（线上实测 route 之外的 select/filter/answer 全丢）。"""
    import anyio
    from anyio import to_thread

    from rag.course.llm import LLMResult
    from rag.unified import UnifiedService

    def report(step, p_tok, c_tok):
        usage_log.record_llm(step, "m", p_tok, c_tok, 0.01)   # 模拟 OpenAICompatClient 的内部上报

    class Router:
        def chat(self, step, messages, **kw):
            import json as _j
            report(step, 100, 10)
            return LLMResult(content=_j.dumps({"route": "course", "routes": ["course"],
                                              "complexity": "simple", "reason": "t"}),
                             prompt_tokens=100, completion_tokens=10, latency_s=0.01)

    def ask(question, ctx, llm, **kw):
        report("select", 200, 50)                              # 模拟管线内三步调用
        report("filter", 300, 60)
        return {"question": question, "answer": {"text": "答案", "sources": []}, "trace": {"calls": []},
                "selected_concepts": [], "filtered": [], "recalled": [], "fallback": None}

    svc = UnifiedService(router_client=Router(), course_llm=object(), course_ctx=object())
    import rag.course.pipeline as pipeline
    orig = pipeline.ask
    pipeline.ask = ask
    try:
        def _next(it):
            try:
                return True, next(it)
            except StopIteration:
                return False, None

        async def consume():
            gen = svc.ask_stream("什么是数据要素", origin="http", key_id="feedface")
            out = []
            while True:
                more, ev = await to_thread.run_sync(_next, gen)   # 每段都在线程池新上下文里恢复
                if not more:
                    break
                out.append(ev)
            return out

        events = anyio.run(consume)
    finally:
        pipeline.ask = orig
    assert events[0]["type"] == "route"
    row = _read(logdir)[0]
    steps = [c["step"] for c in row["llm_calls"]]
    assert "route" in steps, "route 未记录"
    assert "select" in steps and "filter" in steps, f"后台线程里的调用丢失（contextvar 未跨线程传递）：{steps}"
