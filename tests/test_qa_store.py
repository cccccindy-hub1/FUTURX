"""问答永久记录（rag.qa_store）与对话记忆（router standalone）。全部 mock：不调用真实模型、不写真库（RAG_QA_DB 指向临时文件）。"""
from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from rag import qa_store, usage_log
from rag.course.llm import LLMResult
from rag.router import ROUTER_HISTORY_ADDENDUM, ROUTER_SYSTEM, classify

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rag.unified import UnifiedService, router as unified_router  # noqa: E402


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    path = tmp_path / "qa.sqlite"
    monkeypatch.setenv("RAG_QA_DB", str(path))
    monkeypatch.setenv("RAG_LOG_DIR", str(tmp_path / "usage"))
    qa_store._inited.discard(str(path))
    return path


def rows(path, sql="SELECT * FROM qa_log ORDER BY id"):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


class RouterLLM:
    """记录每次路由调用的消息；按脚本返回。"""

    def __init__(self, routes=("course",), standalone=None):
        self.routes, self.standalone, self.calls = list(routes), standalone, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages})
        obj = {"routes": self.routes, "complexity": "simple", "count": 1, "chapter": None, "reason": "t"}
        if self.standalone is not None:
            obj["standalone"] = self.standalone
        return LLMResult(content=json.dumps(obj, ensure_ascii=False))


class AnsLLM:
    """课程 / 判例 / 政策等作答与追问的统一替身：记录每次的 step 与 user 内容。"""

    def __init__(self):
        self.calls = []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages})
        if step == "followup":
            return LLMResult(content=json.dumps({"followups": ["还有什么区别？", "怎么应用？"]}, ensure_ascii=False))
        return LLMResult(content="答案[S1]")


# ---------------------------------------------------------------- 标识与历史

@pytest.mark.parametrize("raw,ok", [("u-abc_123", True), ("A" * 64, True), ("A" * 65, False), ("带中文", False), ("a b", False),
                                    ("a;drop", False), ("", False), (None, False), ("../x", False)])
def test_user_id_validation(raw, ok):
    assert (qa_store.clean_id(raw) == str(raw) and raw not in (None, "")) is ok
    assert qa_store.resolve_user_id(raw, "abcd1234") == (str(raw) if ok else "anon-abcd1234")


def test_anonymous_ids_fall_back_to_key_fingerprint_or_none():
    assert qa_store.resolve_user_id(None, "deadbeef") == "anon-deadbeef" and qa_store.resolve_user_id("bad id", None) == "anon-none"


def test_history_keeps_last_three_rounds_truncates_and_starts_with_user():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"第{i}条" + "字" * 900} for i in range(10)]
    h = qa_store.normalize_history(msgs)
    assert len(h) == 6 and h[0]["role"] == "user" and h[-1]["role"] == "assistant" and all(len(m["content"]) <= 500 for m in h)
    assert h[0]["content"].startswith("第4条") and h[-1]["content"].startswith("第9条")


def test_history_drops_garbage_system_messages_and_a_leading_assistant():
    h = qa_store.normalize_history([{"role": "system", "content": "x"}, {"role": "assistant", "content": "开场白"}, {"role": "user", "content": "问"},
                                    {"role": "assistant", "content": ""}, {"role": "tool", "content": "t"}, {"role": "assistant", "content": "答"}, 5, None])
    assert h == [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]
    assert qa_store.normalize_history("nope") == [] and qa_store.normalize_history(None) == []


def test_request_meta_prefers_body_user_then_header_and_validates_session():
    m = qa_store.request_meta({"x-user-id": "hdr-user", "x-session-id": "s-1"}, body_user="body_user", history=None)
    assert (m["user_id"], m["session_id"]) == ("body_user", "s-1")
    m = qa_store.request_meta({"x-user-id": "hdr-user", "x-session-id": "bad session!"}, body_user="含 空格", history=None)
    assert (m["user_id"], m["session_id"]) == ("hdr-user", "")


def test_uuid_wins_over_user_and_header_fallbacks_are_kept():
    assert qa_store.request_meta({}, body_user="old", body_uuid="u-new", history=None)["user_id"] == "u-new"
    assert qa_store.request_meta({"x-user-uuid": "u-hdr"}, body_user="old", history=None)["user_id"] == "u-hdr"
    assert qa_store.request_meta({"x-user-uuid": "u-hdr"}, body_user="body-user", body_uuid="u-body", history=None)["user_id"] == "u-body"
    assert qa_store.request_meta({"x-user-id": "legacy"}, history=None)["user_id"] == "legacy"       # 老调用方不变
    assert qa_store.request_meta({}, body_uuid="bad id!", body_user="old", history=None)["user_id"] == "old"


@pytest.mark.parametrize("raw, want", [("张三", "张三"), ("  李 四 ", "李 四"), ("x" * 99, "x" * 32), (None, ""), ("", ""),
                                       ("王\n五", "王五"), ("%E5%BC%A0%E4%B8%89", "张三"), ("A" * 8 + "%zz", "A" * 8 + "%zz")])
def test_name_is_cleaned_percent_decoded_and_capped(raw, want):
    assert qa_store.clean_name(raw) == want


def test_user_name_is_stored_separately_from_user_id(db):
    qa_store.write(_record(), user_id="u-uuid", user_name="张三")
    qa_store.write(_record(request_id="r2"), user_id="u-uuid")
    a, b = rows(db)
    assert (a["user_id"], a["user_name"]) == ("u-uuid", "张三") and b["user_name"] == ""


def test_old_database_without_user_name_gets_the_column_added(db):
    """模拟改版前建的库：除 user_name 外列都齐（真实老库就是这样），迁移后照常写入。"""
    old = "\n".join(line for line in qa_store.SCHEMA.splitlines() if "user_name" not in line)
    con = sqlite3.connect(db)
    con.executescript(old)
    con.commit()
    con.close()
    qa_store._inited.discard(str(db))
    assert qa_store.write(_record(), user_id="u", user_name="张三")
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(qa_log)")}
    assert "user_name" in cols


# ---------------------------------------------------------------- 写入、去重、失败隔离

def _record(**kw):
    base = {"request_id": "r1", "key_id": "k1234567", "question": "什么是数据确权", "route": "course", "routes": ["course"],
            "answer": "答案", "followups": ["a？"], "sources": [{"n": 1, "path": "p"}], "status": "ok", "error": None,
            "llm_calls": [{"step": "route"}], "duration_ms": 12.5, "origin": "http"}
    base.update(kw)
    return base


def test_write_creates_tables_and_a_full_row(db):
    ph = qa_store.prompt_hash("系统提示")
    assert qa_store.write(_record(), user_id="u-1", session_id="s-1", history=[{"role": "user", "content": "x"}], standalone="完整问题",
                          prompts={ph: ("route", "系统提示")})
    r = rows(db)[0]
    assert (r["request_id"], r["user_id"], r["key_id"], r["session_id"], r["origin"], r["input_prompt"], r["standalone_question"]) == \
        ("r1", "u-1", "k1234567", "s-1", "http", "什么是数据确权", "完整问题")
    assert json.loads(r["system_prompt_hashes_json"]) == {"route": ph} and json.loads(r["history_json"])[0]["content"] == "x"
    assert r["output"] == "答案" and json.loads(r["routes"]) == ["course"] and r["latency_ms"] == 12.5 and len(r["created_at"]) == 19
    assert [x["text"] for x in rows(db, "SELECT * FROM prompts")] == ["系统提示"]


def test_identical_prompts_are_stored_once_and_referenced_by_hash(db):
    ph = qa_store.prompt_hash("同一个提示")
    for i in range(3):
        qa_store.write(_record(request_id=f"r{i}"), user_id="u", prompts={ph: ("route", "同一个提示")})
    assert len(rows(db)) == 3 and len(rows(db, "SELECT * FROM prompts")) == 1
    assert {json.loads(r["system_prompt_hashes_json"])["route"] for r in rows(db)} == {ph}


def test_existing_database_and_rows_are_never_overwritten_or_dropped(db):
    qa_store.write(_record(request_id="old"), user_id="u")
    qa_store._inited.clear()                                              # 模拟重启：再次连接同一个库
    qa_store.write(_record(request_id="new"), user_id="u")
    assert [r["request_id"] for r in rows(db)] == ["old", "new"]
    assert sqlite3.connect(db).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_write_failure_is_isolated_and_only_warns(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("RAG_QA_DB", str(blocker / "sub" / "qa.sqlite"))           # 父路径是个文件：必然失败
    with caplog.at_level("WARNING"):
        assert qa_store.write(_record(), user_id="u") is False
    assert any("qa_store 写入失败" in r.message for r in caplog.records)


def test_a_broken_database_never_breaks_the_request_or_the_jsonl_log(monkeypatch, tmp_path):
    monkeypatch.setattr(qa_store, "write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    svc = UnifiedService(router_client=RouterLLM(("off_topic",)))
    out = svc.ask("今天天气怎么样", origin="http", key_id="k1")
    assert out["status"] == "refused"
    assert list((tmp_path / "usage").glob("*.jsonl"))                                  # usage_log 照常落盘


def test_key_material_is_never_stored(db):
    qa_store.write(_record(), user_id="u")
    blob = json.dumps([dict(r) for r in rows(db)], ensure_ascii=False)
    assert "Bearer" not in blob and "sk-" not in blob


def test_concurrent_writes_are_all_stored(db):
    import threading
    ts = [threading.Thread(target=lambda i=i: qa_store.write(_record(request_id=f"c{i}"), user_id="u")) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(rows(db)) == 20


# ---------------------------------------------------------------- 统一入口：非流式 / 流式各写一条；各种终态也要记

def test_non_streaming_ask_writes_exactly_one_row_with_prompts_and_identity(db):
    llm = AnsLLM()
    svc = UnifiedService(router_client=RouterLLM(("thesis",)), course_llm=llm)
    svc.adaptive = type("A", (), {"run": lambda s, q, k, c, **kw: {"status": "ok", "answer": "论文答案", "sources": [{"source": "a.pdf"}], "contexts": [], "trace": {}}})()
    svc.ask("怎么写开题报告", origin="http", key_id="kid12345", user_id="u-web", session_id="s-9")
    (r,) = rows(db)
    assert (r["user_id"], r["session_id"], r["key_id"], r["route"], r["status"], r["output"]) == ("u-web", "s-9", "kid12345", "thesis", "ok", "论文答案")
    assert json.loads(r["system_prompt_hashes_json"]) == {}                                         # 论文路径的替身客户端没有登记 prompt：不编造
    assert json.loads(r["sources_json"])[0]["source"] == "a.pdf"


def test_router_system_prompt_is_stored_by_hash(db):
    svc = UnifiedService(router_client=RouterLLM(("off_topic",)))
    svc.router_client = _Reporting(RouterLLM(("off_topic",)))
    svc.ask("今天天气怎么样", origin="http", key_id="k", user_id="u")
    (r,) = rows(db)
    hashes = json.loads(r["system_prompt_hashes_json"])
    assert hashes["route"] == qa_store.prompt_hash(ROUTER_SYSTEM)
    assert rows(db, "SELECT text FROM prompts WHERE step='route'")[0]["text"] == ROUTER_SYSTEM


class _Reporting:
    """模拟 OpenAICompatClient：调用前登记 system prompt（真实客户端在 rag.course.llm 里做同样的事）。"""
    reports_usage = True

    def __init__(self, inner):
        self.inner = inner

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        from rag.course.llm import _note_prompt
        _note_prompt(step, messages)
        return self.inner.chat(step, messages, model=model, tools=tools, json_mode=json_mode)


def test_streaming_writes_exactly_one_row_even_with_followups(db):
    llm = AnsLLM()
    svc = UnifiedService(router_client=_Reporting(RouterLLM(("thesis",))), course_llm=llm)
    svc.adaptive = type("A", (), {"run": lambda s, q, k, c, **kw: {"status": "ok", "answer": "论文答案", "sources": [], "contexts": [], "trace": {}}})()
    ev = list(svc.ask_stream("怎么写开题报告", origin="http", key_id="k", user_id="u-s", session_id="s-s"))
    assert ev[-1]["type"] in ("done", "followups")
    (r,) = rows(db)
    assert (r["user_id"], r["session_id"], r["status"], r["output"]) == ("u-s", "s-s", "ok", "论文答案") and json.loads(r["followups_json"])
    assert "route" in json.loads(r["system_prompt_hashes_json"])


def test_refusal_error_and_exam_hits_are_recorded_too(db):
    UnifiedService(router_client=RouterLLM(("off_topic",))).ask("天气", origin="http", user_id="u")
    class Boom(UnifiedService):
        def _thesis(self, *a, **k):
            raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        Boom(router_client=RouterLLM(("thesis",))).ask("怎么写开题", origin="http", user_id="u")
    got = rows(db)
    assert [r["status"] for r in got] == ["refused", "error"] and got[1]["error"] == "RuntimeError"


def test_exam_prematch_hit_is_recorded_with_zero_llm_calls(db):
    from tests.test_exam_solve import JUDGE, make_bank
    svc = UnifiedService(router_client=RouterLLM(("course",)), exam_bank=make_bank())
    svc.ask(f"判断这道题是否正确：{JUDGE}", origin="http", user_id="u-ex")
    (r,) = rows(db)
    assert r["route"] == "exam" and r["status"] == "ok" and json.loads(r["llm_calls_json"]) == [] and "正确" in r["output"]


def test_http_user_and_session_headers_reach_the_database(db):
    """多轮由 Cindy 的 conversation_id 负责；问答记录只取 X-User-Id / X-Session-Id 两个头（store=false 时不建会话）。"""
    app = FastAPI()
    svc = UnifiedService(router_client=_Reporting(RouterLLM(("off_topic",))))
    app.state.unified = svc
    app.include_router(unified_router)
    c = TestClient(app)
    c.post("/auto/ask", json={"question": "考试中常考哪种题型", "store": False}, headers={"X-User-Id": "u-web", "X-Session-Id": "sess-1"})
    c.post("/auto/ask", json={"question": "你好", "store": False}, headers={"X-User-Id": "hdr-user", "Authorization": "Bearer sk-secret-123"})
    a, b = rows(db)
    assert (a["user_id"], a["session_id"]) == ("u-web", "sess-1")
    assert b["user_id"] == "hdr-user" and b["key_id"] == usage_log.key_id("sk-secret-123") and "sk-secret" not in json.dumps(dict(b))


def test_http_uuid_and_name_from_body_and_headers_reach_the_database(db):
    """新增入参：uuid 记进 user_id 列，user_name 单独一列。请求体直接写中文；请求头须百分号编码（HTTP 头不支持中文）。"""
    from urllib.parse import quote

    app = FastAPI()
    app.state.unified = UnifiedService(router_client=_Reporting(RouterLLM(("off_topic",))))
    app.include_router(unified_router)
    c = TestClient(app)
    c.post("/auto/ask", json={"question": "问题一", "store": False, "uuid": "stu-001", "user_name": "张三"})
    c.post("/auto/ask", json={"question": "问题二", "store": False},
           headers={"X-User-Uuid": "stu-002", "X-User-Name": quote("李四")})
    c.post("/auto/ask", json={"question": "问题三", "store": False, "uuid": "stu-003", "user_name": "王五"},
           headers={"X-User-Name": quote("应被请求体覆盖")})
    a, b, d = rows(db)
    assert (a["user_id"], a["user_name"]) == ("stu-001", "张三")
    assert (b["user_id"], b["user_name"]) == ("stu-002", "李四")
    assert (d["user_id"], d["user_name"]) == ("stu-003", "王五")


def test_openai_compatible_endpoint_takes_uuid_and_name_too(db):
    from rag.api import router as api_router

    app = FastAPI()
    app.state.service = type("S", (), {"answer": lambda self, q, k, **kw: ("论文答案", [], [])})()
    app.include_router(api_router)
    TestClient(app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "store": False,
                                                       "uuid": "stu-9", "user_name": "赵六"})
    (r,) = rows(db)
    assert (r["user_id"], r["user_name"], r["output"]) == ("stu-9", "赵六", "论文答案")


def test_dirty_uuid_and_over_long_name_never_fail_the_request(db):
    """身份只用于记录：非法 uuid 退化为匿名，姓名截断；都不该把 200 变成 4xx/5xx。"""
    app = FastAPI()
    app.state.unified = UnifiedService(router_client=_Reporting(RouterLLM(("off_topic",))))
    app.include_router(unified_router)
    resp = TestClient(app).post("/auto/ask", json={"question": "问题", "store": False, "uuid": "bad id!", "user_name": "李" * 99})
    assert resp.status_code == 200
    (r,) = rows(db)
    assert r["user_id"] == "anon-none" and r["user_name"] == "李" * 32


def test_export_csv_includes_user_name_and_filters_by_it(db, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("export_qa", Path(__file__).resolve().parents[1] / "scripts" / "export_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    qa_store.write(_record(request_id="r1"), user_id="u1", user_name="张三")
    qa_store.write(_record(request_id="r2"), user_id="u2", user_name="李四")
    out = tmp_path / "qa.csv"
    assert mod.export(db, out, user_name="张三") == 1
    text = out.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].split(",")[3] == "user_name" and "张三" in text and "李四" not in text


def test_export_works_on_a_database_that_still_lacks_user_name(db, tmp_path):
    """导出脚本对老库也要能跑（只读、不迁移）。"""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("export_qa", Path(__file__).resolve().parents[1] / "scripts" / "export_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    con = sqlite3.connect(db)
    con.executescript("\n".join(line for line in qa_store.SCHEMA.splitlines() if "user_name" not in line))
    con.execute("INSERT INTO qa_log(request_id,user_id,created_at,input_prompt) VALUES ('r0','u0','2026-11-01T00:00:00','老问题')")
    con.commit()
    con.close()
    out = tmp_path / "old.csv"
    assert mod.export(db, out) == 1
    assert "老问题" in out.read_text(encoding="utf-8-sig")


def test_openai_compatible_endpoint_logs_user_and_history_but_not_the_key(db):
    from rag.api import router as api_router
    app = FastAPI()

    class Svc:
        def answer(self, q, k, **kw):
            return "论文答案", [{"source": "a"}], []
    app.state.service = Svc()
    app.include_router(api_router)
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    TestClient(app).post("/v1/chat/completions", json={"messages": msgs, "user": "oa-user", "store": False}, headers={"X-Session-Id": "s-oa"})
    (r,) = rows(db)
    assert (r["user_id"], r["session_id"], r["input_prompt"], r["output"]) == ("oa-user", "s-oa", "q2", "论文答案")
    assert [m["content"] for m in json.loads(r["history_json"])] == ["q1", "a1"]


def test_ask_endpoint_accepts_optional_history_and_logs_it(db):
    from rag import serve
    import inspect
    assert "history" in serve.AskRequest.model_fields and "user" in serve.AskRequest.model_fields
    class Req:
        history = [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧答案"}]
        user = "ask-user"
    class R:
        headers = {"x-session-id": "s-ask"}
    meta = serve._qa_meta(R(), Req())
    assert meta["user_id"] == "ask-user" and meta["session_id"] == "s-ask" and len(meta["history"]) == 2
    usage_log.simple_record("新问题", origin="http", key_id_value="k", route="thesis", answer="答", qa_meta=meta)
    (r,) = rows(db)
    assert (r["user_id"], r["session_id"], r["output"]) == ("ask-user", "s-ask", "答") and len(json.loads(r["history_json"])) == 2


# ---------------------------------------------------------------- 对话记忆：路由改写 standalone

def test_without_history_router_prompt_and_messages_are_exactly_as_before():
    r = RouterLLM(("course",))
    d = classify("什么是数据确权", r)
    m = r.calls[0]["messages"]
    assert m[0]["content"] == ROUTER_SYSTEM and m[1]["content"] == "什么是数据确权" and "standalone" not in m[0]["content"] and d.standalone == ""
    r2 = RouterLLM(("course",))
    classify("什么是数据确权", r2, history=[])
    assert r2.calls[0]["messages"] == m


def test_with_history_router_gets_history_and_returns_standalone_in_one_call():
    hist = [{"role": "user", "content": "什么是数据确权"}, {"role": "assistant", "content": "数据确权是……"}]
    r = RouterLLM(("course",), standalone="数据确权在考试中常考哪种题型")
    d = classify("考试中常考哪种题型", r, history=hist)
    assert len(r.calls) == 1 and r.calls[0]["messages"][0]["content"] == ROUTER_SYSTEM + ROUTER_HISTORY_ADDENDUM
    user = r.calls[0]["messages"][1]["content"]
    assert "【对话历史】" in user and "用户：什么是数据确权" in user and "助手：数据确权是" in user and user.endswith("考试中常考哪种题型")
    assert d.standalone == "数据确权在考试中常考哪种题型" and d.route == "course"


@pytest.mark.parametrize("bad", [None, "", "   ", 123, ["x"], {"a": 1}])
def test_unusable_standalone_falls_back_to_the_original_question(bad):
    r = RouterLLM(("course",), standalone=bad)
    d = classify("考试中常考哪种题型", r, history=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert d.standalone == ""


def test_router_failure_with_history_falls_back_to_rules_and_the_original_question():
    class Down:
        def chat(self, *a, **k):
            raise TimeoutError("t")
    d = classify("什么是数据确权", Down(), history=[{"role": "user", "content": "q"}])
    assert d.source == "rules" and d.standalone == ""


def test_standalone_is_capped_and_ignored_without_history():
    r = RouterLLM(("course",), standalone="长" * 500)
    assert len(classify("q", r, history=[{"role": "user", "content": "h"}]).standalone) == 200
    assert classify("q", RouterLLM(("course",), standalone="不该被采用"), history=None).standalone == ""


# ---------------------------------------------------------------- 回归

def test_no_history_end_to_end_result_is_identical_to_before_the_feature():
    llm_a, llm_b = RouterLLM(("off_topic",)), RouterLLM(("off_topic",))
    a = UnifiedService(router_client=llm_a).ask("今天天气怎么样")
    b = UnifiedService(router_client=llm_b).ask("今天天气怎么样", history=None, user_id=None, session_id=None)
    strip = lambda o: {k: v for k, v in o.items() if k != "trace"}
    assert strip(a) == strip(b) and llm_a.calls == llm_b.calls and "standalone" not in a["trace"]["router"]


def test_prematch_still_uses_the_original_question_and_skips_the_router():
    from tests.test_exam_solve import JUDGE, make_bank
    r = RouterLLM(("course",), standalone="不应被使用")
    svc = UnifiedService(router_client=r, exam_bank=make_bank())
    out = svc.ask(f"判断这道题是否正确：{JUDGE}", history=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert r.calls == [] and out["route"] == "exam" and out["trace"]["llm_calls"] == 0


# ---------------------------------------------------------------- 导出脚本

def test_export_csv_has_bom_filters_by_date_and_user_and_hides_prompts_by_default(db, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    import export_qa
    ph = qa_store.prompt_hash("机密系统提示")
    qa_store.write(_record(request_id="a"), user_id="u-1", prompts={ph: ("route", "机密系统提示")})
    qa_store.write(_record(request_id="b"), user_id="u-2")
    con = sqlite3.connect(db)
    con.execute("UPDATE qa_log SET created_at='2026-10-01T10:00:00' WHERE request_id='b'")
    con.commit(), con.close()
    out = tmp_path / "o.csv"
    assert export_qa.export(db, out, user_id="u-1") == 1
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and "机密系统提示".encode() not in raw and "system_prompts" not in raw.decode("utf-8-sig")
    out2 = tmp_path / "o2.csv"
    assert export_qa.export(db, out2, since="2026-10-01", until="2026-10-01") == 1
    out3 = tmp_path / "o3.csv"
    assert export_qa.export(db, out3, user_id="u-1", with_prompts=True) == 1
    rows3 = list(csv.DictReader(open(out3, encoding="utf-8-sig")))
    assert json.loads(rows3[0]["system_prompts"]) == {"route": "机密系统提示"}
    with pytest.raises(SystemExit):
        export_qa.export(db, out3)                                           # 已存在：拒绝覆盖
    with pytest.raises(SystemExit):
        export_qa.export(tmp_path / "none.sqlite", tmp_path / "x.csv")       # 库不存在


# ---------------------------------------------------------------- 网页

def test_web_sends_the_user_id_header_and_has_no_recording_notice():
    import re
    from pathlib import Path
    js = Path("rag/web/app.js").read_text(encoding="utf-8")
    html = Path("rag/web/index.html").read_text(encoding="utf-8")
    assert "headers['X-User-Id'] = userId();" in js and "localStorage.getItem(UID_STORE)" in js
    assert "payload.conversation_id = conversationId" in js                       # 多轮由会话 id 负责（Cindy 的实现）
    assert "history: history.slice()" not in js and html.count('id="new-chat"') == 1
    for text in (html, js):
        assert not re.search(r"(记录|保存|存储|留存)[^。\n]{0,8}(对话|聊天|提问|问答)用于|对话[^。\n]{0,6}(被|将被|会被)[^。\n]{0,4}(记录|保存)", text)


def test_prompt_first_seen_is_kept_when_the_same_prompt_is_seen_again(db):
    ph = qa_store.prompt_hash("提示")
    qa_store.write(_record(request_id="a"), user_id="u", prompts={ph: ("route", "提示")})
    con = sqlite3.connect(db)
    con.execute("UPDATE prompts SET first_seen='2026-01-01T00:00:00'")
    con.commit(), con.close()
    qa_store.write(_record(request_id="b"), user_id="u", prompts={ph: ("route", "提示")})
    assert rows(db, "SELECT first_seen FROM prompts")[0]["first_seen"] == "2026-01-01T00:00:00"


def test_real_openai_client_registers_its_system_prompt_before_each_call_chat_and_stream(db, monkeypatch):
    """OpenAICompatClient 自己登记 system prompt（网络层用 mock 替换，不发任何请求）。"""
    from rag.course import llm as L

    class Resp:
        class _C:
            message = type("M", (), {"content": "ok", "tool_calls": None, "reasoning_content": None})()
        choices = [_C()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "completion_tokens_details": None})()

    client = L.OpenAICompatClient.__new__(L.OpenAICompatClient)
    client._client = type("X", (), {"chat": type("Y", (), {"completions": type("Z", (), {"create": staticmethod(lambda **k: Resp())})()})()})()
    client._reasoning = {}
    client._redact = lambda s: s
    monkeypatch.setattr(L, "parse_response", lambda resp, dt: L.LLMResult(content="ok"))
    rec = usage_log.start("q", "http", "k")
    with usage_log.activate(rec):
        client.chat("route", [{"role": "system", "content": "ROUTER 提示"}, {"role": "user", "content": "q"}], model="m")
    assert list(rec.prompts.values()) == [("route", "ROUTER 提示")]
