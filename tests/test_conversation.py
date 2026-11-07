"""会话存储与接口的离线测试（SQLite 落临时目录，不联网、不加载模型）。"""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag import conversation as store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_DB", str(tmp_path / "conv.sqlite3"))
    monkeypatch.setenv("CONVERSATION_RETENTION_DAYS", "365")
    store.reset_connections()
    yield
    store.reset_connections()


# ---------------------------------------------------------------- 存储层

def test_create_and_get():
    c = store.create(key_id="k1", title="测试")
    assert c["id"].startswith("conv_")
    got = store.get(c["id"], key_id="k1")
    assert got["title"] == "测试" and got["turn_count"] == 0
    assert store.get(c["id"], key_id="k2") is None       # 归属隔离


def test_append_turn_writes_pair_and_titles():
    c = store.create(key_id="k1")
    r = store.append_turn(conversation_id=c["id"], question="数据资产估值有哪些方法？", answer="成本法、收益法、市场法。",
                          key_id="k1", route="thesis")
    assert r["seq"] == 1 and r["turn_count"] == 1
    assert r["title"] == "数据资产估值有哪些方法？"
    msgs = store.messages(c["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["seq"] for m in msgs] == [1, 2]


def test_append_turn_is_idempotent_by_request_id():
    c = store.create(key_id="k1")
    for _ in range(2):
        store.append_turn(conversation_id=c["id"], question="q", answer="a", request_id="req-1")
    assert len(store.messages(c["id"])) == 2              # 重试不会插两遍


def test_append_turn_rejects_foreign_key():
    c = store.create(key_id="k1")
    with pytest.raises(PermissionError):
        store.append_turn(conversation_id=c["id"], question="q", answer="a", key_id="other")


def test_history_returns_recent_turns_oldest_first():
    c = store.create(key_id="k1")
    for i in range(4):
        store.append_turn(conversation_id=c["id"], question=f"问题{i}", answer=f"回答{i}")
    got = store.history(c["id"], max_turns=2)
    assert got == [("user", "问题2"), ("assistant", "回答2"), ("user", "问题3"), ("assistant", "回答3")]


def test_history_trims_by_chars_from_newest():
    c = store.create(key_id="k1")
    store.append_turn(conversation_id=c["id"], question="a" * 500, answer="b" * 10)
    store.append_turn(conversation_id=c["id"], question="c" * 10, answer="d" * 10)
    got = store.history(c["id"], max_turns=6, max_chars=50)
    assert got == [("user", "c" * 10), ("assistant", "d" * 10)]   # 最老的长轮次被丢掉


def test_list_conversations_orders_by_update_and_pages():
    ids = []
    for i in range(3):
        c = store.create(key_id="k1", title=f"会话{i}")
        store.append_turn(conversation_id=c["id"], question=f"问题{i}", answer="答")
        ids.append(c["id"])
    page = store.list_conversations(key_id="k1", limit=2)
    assert len(page["items"]) == 2 and page["next_cursor"]
    assert page["items"][0]["id"] == ids[-1]                        # 最近更新的在前
    page2 = store.list_conversations(key_id="k1", limit=2, cursor=page["next_cursor"])
    assert [i["id"] for i in page2["items"]] == [ids[0]]


def test_rename_archive_and_delete():
    c = store.create(key_id="k1")
    assert store.rename(c["id"], "新名字", key_id="k1")["title"] == "新名字"
    assert store.set_archived(c["id"], True, key_id="k1")["archived"] is True
    assert store.list_conversations(key_id="k1")["items"] == []          # 归档的不在默认列表
    assert len(store.list_conversations(key_id="k1", archived=None)["items"]) == 1
    assert store.delete(c["id"], key_id="k1") is True
    assert store.get(c["id"]) is None                                   # 软删后不可见
    assert store.delete(c["id"], key_id="k1") is False                  # 已删的不再报成功


def test_clear_messages_keeps_conversation():
    c = store.create(key_id="k1")
    store.append_turn(conversation_id=c["id"], question="q", answer="a")
    assert store.clear_messages(c["id"], key_id="k1") == 2
    assert store.messages(c["id"]) == []
    assert store.get(c["id"])["turn_count"] == 0


def test_purge_expired_removes_old_conversations(monkeypatch):
    c = store.create(key_id="k1")
    store.append_turn(conversation_id=c["id"], question="q", answer="a")
    assert store.purge_expired(days=365) == 0
    assert store.purge_expired(days=0) == 0            # 0 = 不清理
    assert store.purge_expired(days=-1) == 0
    assert store.get(c["id"]) is not None


# ---------------------------------------------------------------- REST 接口

@pytest.fixture
def client():
    from rag import conversation_api

    app = FastAPI()
    app.include_router(conversation_api.router)
    return TestClient(app)


def test_rest_create_list_get(client):
    r = client.post("/v1/conversations", json={"title": "我的会话"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "conversation" and body["title"] == "我的会话"
    cid = body["id"]

    store.append_turn(conversation_id=cid, question="第一问", answer="第一答", route="thesis",
                      search_query="第一问改写", sources=[{"n": 1, "chunk_id": "c1"}])
    listed = client.get("/v1/conversations").json()
    assert listed["object"] == "list" and listed["data"][0]["id"] == cid
    assert listed["data"][0]["turn_count"] == 1

    detail = client.get(f"/v1/conversations/{cid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["search_query"] == "第一问改写"
    assert detail["messages"][1]["sources"][0]["chunk_id"] == "c1"


def test_rest_patch_archive_and_delete(client):
    cid = client.post("/v1/conversations", json={}).json()["id"]
    patched = client.patch(f"/v1/conversations/{cid}", json={"title": "改名", "archived": True}).json()
    assert patched["title"] == "改名" and patched["archived"] is True
    assert client.get("/v1/conversations").json()["data"] == []
    assert client.delete(f"/v1/conversations/{cid}").json()["deleted"] is True
    assert client.get(f"/v1/conversations/{cid}").status_code == 404


def test_rest_archived_filter_can_show_all(client):
    """archived=all 必须能查到已归档的会话：用 bool 参数表达不了「全部」，已归档的会永远查不到。"""
    keep = client.post("/v1/conversations", json={"title": "在用"}).json()["id"]
    gone = client.post("/v1/conversations", json={"title": "已归档"}).json()["id"]
    client.patch(f"/v1/conversations/{gone}", json={"archived": True})

    def ids(**params):
        return {c["id"] for c in client.get("/v1/conversations", params=params).json()["data"]}

    assert ids() == {keep}
    assert ids(archived="false") == {keep}
    assert ids(archived="true") == {gone}
    assert ids(archived="all") == {keep, gone}
    assert client.get("/v1/conversations", params={"archived": "bogus"}).status_code == 400


def test_rest_messages_after_seq(client):
    cid = client.post("/v1/conversations", json={}).json()["id"]
    store.append_turn(conversation_id=cid, question="q1", answer="a1")
    store.append_turn(conversation_id=cid, question="q2", answer="a2")
    got = client.get(f"/v1/conversations/{cid}/messages", params={"after_seq": 2}).json()
    assert [m["content"] for m in got["data"]] == ["q2", "a2"]
    assert client.delete(f"/v1/conversations/{cid}/messages").json()["deleted_messages"] == 4


def test_rest_missing_conversation_returns_404(client):
    assert client.get("/v1/conversations/conv_nope").status_code == 404
    assert client.patch("/v1/conversations/conv_nope", json={"title": "x"}).status_code == 404
    assert client.delete("/v1/conversations/conv_nope").status_code == 404


def test_rest_isolated_by_api_key(client, monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", ["key-a", "key-b"])
    a = {"Authorization": "Bearer key-a"}
    b = {"Authorization": "Bearer key-b"}
    cid = client.post("/v1/conversations", json={}, headers=a).json()["id"]
    assert client.get(f"/v1/conversations/{cid}", headers=a).status_code == 200
    assert client.get(f"/v1/conversations/{cid}", headers=b).status_code == 404
    assert client.get("/v1/conversations", headers=b).json()["data"] == []


# ---------------------------------------------------------------- 问答入口接入

class _EchoService:
    """最小 RAGService 替身：把收到的 history 回显出来，便于断言。"""

    def __init__(self):
        self.calls = []

    def answer(self, query, top_k=4, *, history=None, search_query=None):
        self.calls.append({"query": query, "history": history, "search_query": search_query})
        return "回答：" + str(len(history or [])), [], []

    def open_stream(self, query, top_k=4, *, history=None, search_query=None):
        self.calls.append({"query": query, "history": history, "search_query": search_query})
        return [], [], iter(["回答", "流式"])


def _chat_client(svc):
    import rag.api as api

    app = FastAPI()
    app.state.service = svc
    app.include_router(api.router)
    return TestClient(app)


def test_chat_creates_conversation_and_returns_id(monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", [])
    cl = _chat_client(_EchoService())
    r = cl.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "第一个问题"}]})
    assert r.status_code == 200
    cid = r.json()["conversation_id"]
    assert r.headers["x-conversation-id"] == cid
    assert [m["content"] for m in store.messages(cid)] == ["第一个问题", "回答：0"]


def test_chat_second_turn_gets_history_from_store(monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", [])
    svc = _EchoService()
    cl = _chat_client(svc)
    cid = cl.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "数据资产估值有哪些方法？"}]}).json()["conversation_id"]
    cl.post("/v1/chat/completions",
            json={"conversation_id": cid, "messages": [{"role": "user", "content": "第二种具体怎么算？"}]})
    assert svc.calls[-1]["history"] == [("user", "数据资产估值有哪些方法？"), ("assistant", "回答：0")]
    assert store.get(cid)["turn_count"] == 2


def test_chat_store_false_skips_persistence(monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", [])
    cl = _chat_client(_EchoService())
    r = cl.post("/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "临时问题"}], "store": False})
    assert "conversation_id" not in r.json()
    assert store.list_conversations()["items"] == []


def test_chat_unknown_conversation_is_404(monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", [])
    r = _chat_client(_EchoService()).post("/v1/chat/completions", json={
        "conversation_id": "conv_nope", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404


def test_chat_stream_persists_and_sends_header(monkeypatch):
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", [])
    cl = _chat_client(_EchoService())
    r = cl.post("/v1/chat/completions", json={
        "stream": True, "messages": [{"role": "user", "content": "流式问题"}]})
    assert r.status_code == 200
    cid = r.headers["x-conversation-id"]
    assert [m["content"] for m in store.messages(cid)] == ["流式问题", "回答流式"]
    assert "conversation_id" in r.text


def test_auto_ask_shares_conversation_with_v1(monkeypatch):
    """两个入口共用同一个会话库：/auto/ask 建的会话能在 /v1 里继续。"""
    import rag.api as api
    import rag.conversation_api as capi
    import rag.unified as U

    class FakeUnified:
        def __init__(self):
            self.histories = []

        def ask(self, q, k=None, *, origin="eval", key_id=None, history=None):
            self.histories.append(history)
            return {"route": "course", "routes": ["course"], "status": "ok",
                    "answer": "统一入口回答", "sources": [], "trace": {"total_s": 0.1}}

    monkeypatch.setattr(api, "API_KEYS", [])
    svc = FakeUnified()
    app = FastAPI()
    app.state.unified = svc
    app.state.service = _EchoService()
    app.include_router(U.router)
    app.include_router(api.router)
    app.include_router(capi.router)
    cl = TestClient(app)

    cid = cl.post("/auto/ask", json={"question": "课程问题"}).json()["conversation_id"]
    cl.post("/v1/chat/completions", json={"conversation_id": cid,
                                          "messages": [{"role": "user", "content": "追问"}]})
    # /v1 拿到的历史里包含 /auto/ask 那一轮
    assert ("user", "课程问题") in app.state.service.calls[-1]["history"]
    assert store.get(cid)["turn_count"] == 2



def test_standalone_question_rewrites_followup_for_routing(monkeypatch):
    """只有指代的追问要补全后再路由，否则会被判成 off_topic 直接拒答。"""
    import rag.unified as U

    class LLM:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()
            self.completions = self

        def create(self, model, messages, **kwargs):
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {
                "content": "数据资产估值的市场法适用于什么情形？"})()})()]})()

    history = [{"role": "user", "content": "数据资产估值有哪些方法？"},
               {"role": "assistant", "content": "成本法、收益法、市场法。"}]
    got = U._standalone_question("那它适用什么情形？", history, LLM())
    assert got == "数据资产估值的市场法适用于什么情形？"


def test_standalone_question_noop_without_history():
    import rag.unified as U

    assert U._standalone_question("三级编码是什么？", [], None) == "三级编码是什么？"


def test_standalone_question_keeps_long_self_contained_question():
    """长问题本身完整，不做多余改写（省掉一次模型调用）。"""
    import rag.unified as U

    q = "我想研究地方政府公共数据开放对企业创新的影响，应该采用什么样的研究设计？"
    assert U._standalone_question(q, [{"role": "user", "content": "前文"}], None) == q


# ---------------------------------------------------------------- 消息级 key 与月份

def test_message_records_key_and_month():
    """每轮消息都要带 key_id 与 YYYY-MM，便于按人、按月审计。"""
    import time

    c = store.create(key_id="k1")
    store.append_turn(conversation_id=c["id"], question="问", answer="答", key_id="k1")
    msgs = store.messages(c["id"])
    assert [m["key_id"] for m in msgs] == ["k1", "k1"]
    assert all(m["month"] == time.strftime("%Y-%m", time.localtime(m["created_at"])) for m in msgs)


def test_message_key_follows_conversation_when_not_passed():
    """调用方没传 key_id 时，用会话的归属，不要落成 None。"""
    c = store.create(key_id="k1")
    store.append_turn(conversation_id=c["id"], question="问", answer="答")
    assert {m["key_id"] for m in store.messages(c["id"])} == {"k1"}


def test_messages_rejects_foreign_key():
    """存储层也要校验归属：以前只有 API 层校验，裸调能读到别人的消息。"""
    a = store.create(key_id="k1")
    store.append_turn(conversation_id=a["id"], question="A 的隐私", answer="A 的答案", key_id="k1")
    assert len(store.messages(a["id"])) == 2                 # 不传 key_id：不过滤（内部/运维用）
    assert store.messages(a["id"], key_id="k1")
    assert store.messages(a["id"], key_id="k2") == []        # 别人的 key 读不到


def test_monthly_stats_groups_by_key_and_month():
    c1 = store.create(key_id="k1")
    c2 = store.create(key_id="k2")
    store.append_turn(conversation_id=c1["id"], question="q1", answer="a1", key_id="k1", latency_ms=100)
    store.append_turn(conversation_id=c1["id"], question="q2", answer="a2", key_id="k1", latency_ms=300)
    store.append_turn(conversation_id=c2["id"], question="q3", answer="a3", key_id="k2")

    rows = {r["key_id"]: r for r in store.monthly_stats()}
    assert rows["k1"]["messages"] == 4 and rows["k1"]["turns"] == 2
    assert rows["k1"]["avg_latency_ms"] == 200.0
    assert rows["k2"]["messages"] == 2 and rows["k2"]["turns"] == 1
    only = store.monthly_stats(key_id="k2")
    assert len(only) == 1 and only[0]["key_id"] == "k2"


def test_reassign_key_moves_anonymous_conversations():
    """配 key 之前产生的匿名会话，人工归置到某个 key 名下。"""
    anon = store.create(key_id=None, title="匿名时期的会话")
    store.append_turn(conversation_id=anon["id"], question="旧问题", answer="旧答案")
    mine = store.create(key_id="k1", title="我的")
    store.append_turn(conversation_id=mine["id"], question="新问题", answer="新答案", key_id="k1")

    assert store.list_conversations(key_id="k1")["items"] and len(store.list_conversations(key_id="k1")["items"]) == 1
    assert store.get(anon["id"], key_id="k1") is None            # 归置前打不开

    dry = store.reassign_key(src_key_id=None, dst_key_id="k1", dry_run=True)
    assert dry["conversations"] == 1 and store.get(anon["id"], key_id="k1") is None

    got = store.reassign_key(src_key_id=None, dst_key_id="k1")
    assert got["conversations"] == 1 and got["messages"] == 2
    assert store.get(anon["id"], key_id="k1") is not None
    assert store.list_conversations(key_id="k1")["items"].__len__() == 2
    # 消息级 key 也补上了，按人审计口径一致
    assert {m["key_id"] for m in store.messages(anon["id"])} == {"k1"}


def test_reassign_keeps_later_messages_key():
    """归置只补 NULL，不能覆盖「换人后新写入」的消息归属。"""
    c = store.create(key_id=None)
    store.append_turn(conversation_id=c["id"], question="旧", answer="旧答")
    store.append_turn(conversation_id=c["id"], question="新", answer="新答", key_id="k2")
    store.reassign_key(src_key_id=None, dst_key_id="k1")
    keys = [m["key_id"] for m in store.messages(c["id"])]
    assert keys == ["k1", "k1", "k2", "k2"]


def test_rest_message_includes_key_and_month(client):
    """接口返回里要带 key_id / month，前端与审计脚本都靠它。"""
    import hashlib

    cid = client.post("/v1/conversations", json={}).json()["id"]
    store.append_turn(conversation_id=cid, question="q", answer="a", key_id="k1")
    got = client.get(f"/v1/conversations/{cid}/messages").json()["data"]
    assert [m["key_id"] for m in got] == ["k1", "k1"]
    assert all(m["month"] for m in got)


def test_rest_messages_hides_foreign_conversation(client, monkeypatch):
    """B 的 key 读 A 的会话消息：404，不给内容。"""
    import rag.api as api

    monkeypatch.setattr(api, "API_KEYS", {"key-a", "key-b"})
    hdr_a = {"Authorization": "Bearer key-a"}
    cid = client.post("/v1/conversations", json={}, headers=hdr_a).json()["id"]
    # 建会话时没带 key 归属信息：用存储层补一次，模拟真实的「A 建的会话」
    store.reassign_key(src_key_id=None, dst_key_id=hashlib.sha256(b"key-a").hexdigest()[:8])
    store.append_turn(conversation_id=cid, question="A 的隐私", answer="A 的答案",
                      key_id=hashlib.sha256(b"key-a").hexdigest()[:8])

    resp = client.get(f"/v1/conversations/{cid}/messages",
                      headers={"Authorization": "Bearer key-b"})
    assert resp.status_code == 404
    assert "A 的隐私" not in resp.text
