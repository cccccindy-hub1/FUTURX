"""多轮对话（/v1/chat/completions 带历史）与向量化批处理调度的离线测试。全部 mock，不联网。"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api as api
import rag.enhance as enhance
from rag.embedding import BatchedEmbedder
from rag.query import RAGService


# ---------------------------------------------------------------- 历史消息解析

def _msgs(*items):
    return [api.ChatMessage(role=r, content=c) for r, c in items]


def test_split_history_keeps_prior_turns_and_ignores_system():
    q, history = api._split_history(_msgs(
        ("system", "我们自己的 system 消息"),
        ("user", "数据资产估值有哪些方法？"),
        ("assistant", "主要有成本法、收益法、市场法。"),
        ("user", "第二种具体怎么算？"),
    ))
    assert q == "第二种具体怎么算？"
    assert history == [("user", "数据资产估值有哪些方法？"),
                       ("assistant", "主要有成本法、收益法、市场法。")]


def test_split_history_single_turn_has_no_history():
    assert api._split_history(_msgs(("user", "三级编码是什么？"))) == ("三级编码是什么？", [])


def test_split_history_returns_all_client_turns():
    """_split_history 只负责拆分，截断交给 _merge_history（服务端历史与客户端历史要统一截）。"""
    items = []
    for i in range(5):
        items += [("user", f"问题{i}"), ("assistant", f"回答{i}")]
    items.append(("user", "最后的问题"))
    q, history = api._split_history(_msgs(*items))
    assert q == "最后的问题"
    assert len(history) == 10


def test_merge_history_caps_turns(monkeypatch):
    monkeypatch.setattr(api, "CHAT_HISTORY_MAX_TURNS", 2)
    server = []
    for i in range(5):
        server += [("user", f"问题{i}"), ("assistant", f"回答{i}")]
    got = api._merge_history(server, [])
    assert [h[0] for h in got] == ["user", "assistant", "user", "assistant"]
    assert got[0][1] == "问题3" and got[-1][1] == "回答4"


def test_merge_history_prefers_client_when_it_sends_turns():
    """客户端自己发了多轮时以客户端为准，避免服务端历史重复一遍。"""
    server = [("user", "旧问题"), ("assistant", "旧回答")]
    client = [("user", "新问题1"), ("assistant", "新回答1"), ("user", "新问题2")]
    assert api._merge_history(server, client) == client


def test_merge_history_uses_server_when_client_sends_one_turn():
    server = [("user", "旧问题"), ("assistant", "旧回答")]
    assert api._merge_history(server, []) == server


def test_merge_history_caps_chars_from_newest(monkeypatch):
    """字数上限从最近往前累加：超了丢更早的轮次，最近的必须保留。"""
    monkeypatch.setattr(api, "CHAT_HISTORY_MAX_TURNS", 6)
    monkeypatch.setattr(api, "CHAT_HISTORY_MAX_CHARS", 100)
    hist = [("user", "a" * 300), ("assistant", "b" * 10), ("user", "c" * 10)]
    got = api._merge_history(hist, [])
    assert got == [("assistant", "b" * 10), ("user", "c" * 10)]


def test_split_history_skips_empty_and_unknown_roles():
    q, history = api._split_history(_msgs(
        ("user", "第一问"),
        ("tool", "工具输出"),
        ("assistant", "   "),
        ("assistant", "第一答"),
        ("user", "第二问"),
    ))
    assert q == "第二问"
    assert history == [("user", "第一问"), ("assistant", "第一答")]


# ---------------------------------------------------------------- 追问改写（检索用）

class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


class RewriteLLM:
    """记录调用的假 LLM；指定 rewrites 时按顺序返回，否则原样返回最后一句。"""

    def __init__(self, rewrites=None, fail=False):
        self.rewrites, self.fail, self.calls = list(rewrites or []), fail, []
        self.chat = type("Chat", (), {"completions": self})()
        self.completions = self

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if self.fail:
            raise RuntimeError("upstream down")
        return _Resp(self.rewrites.pop(0) if self.rewrites else "数据资产估值的收益法具体怎么算")


def test_contextualize_rewrites_followup():
    llm = RewriteLLM(rewrites=["数据资产估值的收益法具体怎么算"])
    got = enhance.contextualize_query(
        "第二种具体怎么算？",
        [("user", "数据资产估值有哪些方法？"), ("assistant", "成本法、收益法、市场法。")],
        llm, "rewrite-model")
    assert got == "数据资产估值的收益法具体怎么算"
    assert llm.calls[0]["model"] == "rewrite-model"
    assert "成本法、收益法、市场法" in llm.calls[0]["messages"][1]["content"]


def test_contextualize_falls_back_on_failure():
    llm = RewriteLLM(fail=True)
    assert enhance.contextualize_query("第二种具体怎么算？", [("user", "前文")], llm, "m") == "第二种具体怎么算？"


def test_contextualize_without_history_or_llm_is_noop():
    assert enhance.contextualize_query("第二个呢", [], RewriteLLM(), "m") == "第二个呢"
    assert enhance.contextualize_query("第二个呢", [("user", "前文")], None, "m") == "第二个呢"


def test_needs_contextualize_only_for_short_or_anaphoric():
    assert enhance.needs_contextualize("第二种具体怎么算？")
    assert not enhance.needs_contextualize(
        "我想研究地方政府公共数据开放对企业创新的影响，应该采用什么样的研究设计？")
    assert enhance.needs_contextualize(
        "我想研究地方政府公共数据开放对企业创新的影响，那这个具体怎么操作？")


# ---------------------------------------------------------------- 传给模型的 messages

HISTORY = [("user", "数据资产估值有哪些方法？"), ("assistant", "成本法、收益法、市场法。")]


def _svc_with_capture():
    """构造不加载模型的 RAGService，截获将要发给 LLM 的 messages。"""
    svc = object.__new__(RAGService)
    from threading import local
    svc._request_state = local()
    captured = {}

    class LLM:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()
            self.completions = self

        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return iter([type("C", (), {"choices": [type("Ch", (), {
                "delta": type("D", (), {"content": "回答"})()})()]})()])

    svc.llm = LLM()
    svc._llm_kwargs = lambda messages: {"model": "m", "messages": messages, "stream": True}
    svc.prepare = lambda q, k, **kw: RAGService.prepare(svc, q, k, **kw)
    svc.retrieve_result = lambda q, k: type("R", (), {
        "documents": ["资料正文"], "metadatas": [{"source": "a.pdf", "section": "s"}],
        "distances": [0.2]})()
    return svc, captured


def test_history_passed_to_model_and_search_uses_rewritten_query():
    svc, captured = _svc_with_capture()
    seen = []
    svc.retrieve_result = lambda q, k: (seen.append(q) or type("R", (), {
        "documents": ["资料正文"], "metadatas": [{"source": "a.pdf", "section": "s"}],
        "distances": [0.2]})())
    docs, sources, deltas = svc.open_stream("第二种具体怎么算？", 4, history=HISTORY,
                                            search_query="数据资产估值的收益法具体怎么算")
    list(deltas)
    msgs = captured["kwargs"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1] == {"role": "user", "content": "数据资产估值有哪些方法？"}
    assert msgs[2]["content"].startswith("成本法")
    # 检索用的是改写后的完整问题；送给模型的本轮问题仍是原话，且资料在同一轮
    assert seen == ["数据资产估值的收益法具体怎么算"]
    assert "第二种具体怎么算？" in msgs[-1]["content"]
    assert "资料正文" in msgs[-1]["content"]


def test_no_history_keeps_two_message_shape():
    svc, captured = _svc_with_capture()
    _, _, deltas = svc.open_stream("三级编码是什么？", 4)
    list(deltas)
    msgs = captured["kwargs"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_prepare_from_accepts_history_kwarg():
    from rag.retrieval import RetrievalResult
    svc = object.__new__(RAGService)
    res = RetrievalResult(documents=["d"], metadatas=[{"source": "s"}], distances=[0.1])
    _, _, msgs = svc.prepare_from("问题", res, history=HISTORY)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------- HTTP 端点串起来

class FakeService:
    def __init__(self):
        self.calls = []
        self.llm = RewriteLLM(rewrites=["数据资产估值的收益法具体怎么算"])

    def answer(self, query, top_k=4, *, history=None, search_query=None):
        self.calls.append({"query": query, "history": history, "search_query": search_query})
        return "回答", [], []

    def open_stream(self, query, top_k=4, *, history=None, search_query=None):
        self.calls.append({"query": query, "history": history, "search_query": search_query})
        return [], [], iter(["回答"])


def _client(svc):
    app = FastAPI()
    app.state.service = svc
    app.include_router(api.router)
    return TestClient(app)


def test_chat_endpoint_sends_history_and_rewritten_search_query(monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", [])
    svc = FakeService()
    body = {"model": "ai-li-teacher", "messages": [
        {"role": "system", "content": "调用方的 system，应忽略"},
        {"role": "user", "content": "数据资产估值有哪些方法？"},
        {"role": "assistant", "content": "成本法、收益法、市场法。"},
        {"role": "user", "content": "第二种具体怎么算？"},
    ]}
    r = _client(svc).post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    call = svc.calls[-1]
    assert call["query"] == "第二种具体怎么算？"
    assert call["history"] == [("user", "数据资产估值有哪些方法？"),
                               ("assistant", "成本法、收益法、市场法。")]
    assert call["search_query"] == "数据资产估值的收益法具体怎么算"


def test_chat_endpoint_without_history_does_not_rewrite(monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", [])
    svc = FakeService()
    _client(svc).post("/v1/chat/completions", json={
        "model": "ai-li-teacher", "messages": [{"role": "user", "content": "三级编码是什么？"}]})
    assert svc.calls[-1]["history"] == []
    assert svc.calls[-1]["search_query"] == "三级编码是什么？"
    assert svc.llm.calls == []           # 单轮不发起改写调用


# ---------------------------------------------------------------- 向量化批处理

class _StubModel:
    def __init__(self):
        self.batches = []
        self.lock = threading.Lock()

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        with self.lock:
            self.batches.append(list(texts))
        return np.array([[float(len(t))] for t in texts], dtype=np.float32)


def test_batcher_merges_concurrent_single_encodes():
    model = _StubModel()
    b = BatchedEmbedder(model, max_batch=32, max_chars=1000, max_wait_ms=30)
    with cf.ThreadPoolExecutor(16) as ex:
        out = list(ex.map(b.encode, ["x" * 4] * 16))
    assert out == [[4.0]] * 16
    assert len(model.batches) < 16                       # 确实合并了
    assert b.batch_calls == len(model.batches)


def test_batcher_keeps_oversized_text_alone():
    model = _StubModel()
    b = BatchedEmbedder(model, max_batch=8, max_chars=10, max_wait_ms=5)
    assert b.encode("y" * 50) == [50.0]
    assert model.batches == [["y" * 50]]


def test_batcher_propagates_errors():
    class Boom:
        def encode(self, texts, normalize_embeddings=True):
            raise RuntimeError("model down")

    b = BatchedEmbedder(Boom(), max_wait_ms=1)
    with pytest.raises(RuntimeError):
        b.encode("x")


def test_batcher_single_encode_returns_flat_vector():
    """替身模型只返回单条向量（非二维）时也不能把字符当成向量。"""
    class Flat:
        def encode(self, texts, normalize_embeddings=True):
            return [0.1, 0.2]

    assert BatchedEmbedder(Flat(), max_wait_ms=1).encode("x") == [0.1, 0.2]
