"""OpenAI 兼容接口契约测试：鉴权、流式、错误码。全部离线。"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api as api


class FakeService:
    """替身 RAGService：不加载模型、不联网。"""

    def __init__(self, answer="扎根理论有三級编码 [来源1]。", has_llm=True):
        self.answer_text = answer
        self.has_llm = has_llm
        self.questions = []

    def answer(self, query, top_k=4):
        self.questions.append((query, top_k))
        if not self.has_llm:
            return None, [{"source": "a.pdf", "section": "扎根", "distance": 0.2}], ["片段"]
        return self.answer_text, [{"source": "a.pdf", "section": "扎根", "distance": 0.2}], ["片段"]

    def open_stream(self, query, top_k=4):
        if not self.has_llm:
            raise RuntimeError("未配置 LLM_API_KEY")
        self.questions.append((query, top_k))
        sources = [{"source": "a.pdf", "section": "扎根", "distance": 0.2}]
        return ["片段"], sources, iter(["扎根", "理论", "三级编码"])


@pytest.fixture
def app_client(monkeypatch):
    # 本机 .env 可能已配置 API_KEYS；测试默认在“无鉴权”前提下跑，
    # 鉴权相关的两个用例自行 monkeypatch 覆盖。
    monkeypatch.setattr(api, "API_KEYS", [])

    def _make(svc=None):
        app = FastAPI()
        app.state.service = svc or FakeService()
        app.include_router(api.router)
        return TestClient(app)

    return _make


def _body(**kw):
    body = {"model": "ai-li-teacher", "messages": [{"role": "user", "content": "三级编码是什么？"}]}
    body.update(kw)
    return body


def test_models_listed(app_client):
    r = app_client().get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"]


def test_chat_completion_openai_shape(app_client):
    svc = FakeService()
    r = app_client(svc).post("/v1/chat/completions", json=_body())
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"].startswith("扎根理论")
    assert data["sources"][0]["source"] == "a.pdf"
    assert svc.questions[-1][0] == "三级编码是什么？"


def test_top_k_passthrough(app_client):
    svc = FakeService()
    app_client(svc).post("/v1/chat/completions", json=_body(top_k=6))
    assert svc.questions[-1][1] == 6


def test_stream_emits_sse_then_done(app_client):
    r = app_client().post("/v1/chat/completions", json=_body(stream=True))
    assert r.status_code == 200
    lines = [ln for ln in r.text.split("\n\n") if ln.startswith("data: ")]
    payloads = [ln[len("data: "):] for ln in lines]
    assert payloads[-1] == "[DONE]"
    deltas = [json.loads(p) for p in payloads[:-1]]
    assert deltas[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(d["choices"][0]["delta"].get("content", "") for d in deltas)
    assert text == "扎根理论三级编码"
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"
    assert deltas[-1]["sources"][0]["source"] == "a.pdf"


def test_missing_user_message_rejected(app_client):
    r = app_client().post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400


def test_no_llm_returns_503(app_client):
    r = app_client(FakeService(has_llm=False)).post("/v1/chat/completions", json=_body())
    assert r.status_code == 503


def test_multipart_content_text_extracted(app_client):
    svc = FakeService()
    body = _body(messages=[{"role": "user", "content": [{"type": "text", "text": "从分段数组来的问题"}]}])
    app_client(svc).post("/v1/chat/completions", json=body)
    assert svc.questions[-1][0] == "从分段数组来的问题"


def test_auth_enforced_when_keys_configured(app_client, monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", ["secret-1"])
    client = app_client()
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret-1"})
    assert ok.status_code == 200
    assert client.post("/v1/chat/completions", json=_body()).status_code == 401


def test_auth_open_when_no_keys(app_client, monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", [])
    assert app_client().get("/v1/models").status_code == 200
