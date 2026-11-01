"""课程问答 HTTP 接口（rag.course.api）测试：全部离线，LLM 用 ScriptedLLM，不读取 .env 里的真实 key。"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import rag.api
from rag.course import api as course_api
from rag.course.llm import LLMError, LLMResult
from test_course_pipeline import SEL2, ScriptedLLM, ctx, finish_native, ordered, refs_for, tc, verdicts  # noqa: F401


def _client(ctx, llm):
    return TestClient(course_api.build_app(ctx, llm))


def _full_script(ctx):
    items = ordered(ctx, SEL2)
    keep = items[0]
    return keep, [
        tc("search_vocab", query="数据要素"),
        finish_native(*SEL2),
        verdicts(*[(r, i.chunk_id == keep.chunk_id) for r, i in zip(refs_for(items), items)]),
        LLMResult(content="数据要素是参与价值创造的数据资源[S1]。", prompt_tokens=100, completion_tokens=20),
    ]


def test_health_reports_book_and_llm_state(ctx):
    body = _client(ctx, None).get("/health").json()
    assert body == {"status": "ok", "book_version": ctx.chunkset.book_version,
                    "chunks": len(ctx.chunkset.chunks), "llm_configured": False}
    assert _client(ctx, ScriptedLLM([])).get("/health").json()["llm_configured"] is True


def test_ask_full_run_returns_pipeline_result(ctx):
    keep, script = _full_script(ctx)
    llm_ = ScriptedLLM(script)
    r = _client(ctx, llm_).post("/course/ask", json={"question": "  什么是数据要素？ "})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "什么是数据要素？" and body["fallback"] is None
    assert [c["chunk_id"] for c in body["citations"]] == [keep.chunk_id]
    assert body["answer"]["text"] == f"数据要素是参与价值创造的数据资源[{keep.chunk_id}]。"
    assert [c["step"] for c in body["trace"]["calls"]] == ["select", "select", "filter", "answer"]
    assert body["trace"]["totals"]["calls"] == 4 and llm_.script == []


def test_ask_can_omit_trace_and_passes_strategy_and_active_concepts(ctx):
    _, script = _full_script(ctx)
    llm_ = ScriptedLLM(script)
    body = _client(ctx, llm_).post("/course/ask", json={
        "question": "q", "include_trace": False, "strategy": "A", "active_concepts": ["C01-0002"]}).json()
    assert "trace" not in body and body["answer"] is not None
    assert "C01-0002" in llm_.calls[0]["messages"][0]["content"]                 # 已激活概念进入选词 prompt


def test_fallback_is_200_with_insufficient_evidence(ctx):
    llm_ = ScriptedLLM([finish_native(("C99-0000", "question_concept"))])
    r = _client(ctx, llm_).post("/course/ask", json={"question": "q"})
    assert r.status_code == 200 and r.json()["fallback"]["reason"] == "no_valid_concepts"


def test_dry_run_needs_no_llm_and_never_calls_it(ctx):
    llm_ = ScriptedLLM([])                                                         # 任何调用都会因脚本用尽而失败
    body = _client(ctx, llm_).post("/course/ask", json={
        "question": "q", "dry_run": True,
        "select": [{"concept_id": "C01-0001"}, {"concept_id": "C01-0003", "role": "related_concept"}]}).json()
    assert [s["concept_id"] for s in body["selected_concepts"]] == ["C01-0001", "C01-0003"]
    assert set(body["dry_run"]["strategies"]) == {"A", "B"} and llm_.calls == []
    r = _client(ctx, None).post("/course/ask", json={"question": "q", "dry_run": True})
    assert r.status_code == 200


def test_no_llm_configured_returns_503_for_real_questions(ctx):
    r = _client(ctx, None).post("/course/ask", json={"question": "什么是数据要素？"})
    assert r.status_code == 503 and r.json()["detail"]["error"]["type"] == "service_unavailable"


def test_upstream_llm_error_returns_502(ctx):
    def boom(step, messages):
        raise LLMError("[select] LLM 调用失败：APIConnectionError: ***")
    r = _client(ctx, ScriptedLLM([boom])).post("/course/ask", json={"question": "q"})
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == {"message": "[select] LLM 调用失败：APIConnectionError: ***",
                                           "type": "upstream_error"}


@pytest.mark.parametrize("payload, status", [
    ({"question": ""}, 422),
    ({"question": "   "}, 400),
    ({"question": "问" * 501}, 422),
    ({"question": "q", "strategy": "C"}, 422),
    ({"question": "q", "dry_run": True, "select": [{"concept_id": "C01-0001", "role": "boss"}]}, 422),
    ({}, 422),
])
def test_request_validation(ctx, payload, status):
    llm_ = ScriptedLLM([])
    assert _client(ctx, llm_).post("/course/ask", json=payload).status_code == status
    assert llm_.calls == []                                                        # 非法请求不产生任何 LLM 调用


def test_api_keys_enforced_when_configured(ctx, monkeypatch):
    monkeypatch.setattr(rag.api, "API_KEYS", ["k-test"])
    client = _client(ctx, None)
    req = {"question": "q", "dry_run": True}
    assert client.post("/course/ask", json=req).status_code == 401
    assert client.post("/course/ask", json=req, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/course/ask", json=req, headers={"Authorization": "Bearer k-test"}).status_code == 200
    assert client.get("/health").status_code == 200                                # 探活不鉴权


def test_build_app_without_llm_config_falls_back_to_none(ctx, monkeypatch):
    monkeypatch.setattr(course_api.config, "COURSE_LLM_API_BASE", "")              # 不构造真实客户端
    app = course_api.build_app(ctx)
    assert app.state.llm is None


def test_response_is_json_serializable_and_has_no_raw_refs(ctx):
    keep, script = _full_script(ctx)
    body = _client(ctx, ScriptedLLM(script)).post("/course/ask", json={"question": "q"}).json()
    blob = json.dumps(body, ensure_ascii=False)
    assert "[S1]" not in body["answer"]["text"] and keep.chunk_id in blob
