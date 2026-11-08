"""问答永久记录（SQLite）：每个请求一行 qa_log，system prompt 按 sha256 去重存进 prompts。

与 rag.usage_log 的关系：usage_log 的 JSONL 格式、字段、90 天保留策略都不动；这里只在它落盘的同一时刻（RequestLog.finish /
simple_record）旁路再写一条，且写失败只记 warning、绝不影响主请求。

- 库路径：RAG_QA_DB，默认 <项目根>/data/qa_log.sqlite；pytest 进程默认写临时目录（假答案不能混进真库）；
- 只用 CREATE TABLE IF NOT EXISTS：不删表、不覆盖已有库文件；WAL 模式；写入加线程锁；
- 不存 API key 原文，只存 key_id（sha256 前 8 位）；
- user_id / session_id：限长 64、只允许 [A-Za-z0-9_-]，不合法的 user_id 记为 anon-<key_id>，不合法的 session_id 记空；
- user_name：调用方传的提问者姓名，中文可用，去掉首尾空白与换行、限长 32 字；不传记空；
- 历史（history_json）最多 6 条、每条最多 500 字，超出截断。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "qa_log.sqlite"
ID_MAX = 64
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
NAME_MAX = 32
HISTORY_MAX_MESSAGES = 6          # 最近 3 轮
HISTORY_MAX_CHARS = 500
SOURCES_MAX = 20
_TEST_ENV = "PYTEST_CURRENT_TEST"

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL DEFAULT '',
    key_id TEXT,
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    origin TEXT,
    input_prompt TEXT NOT NULL,
    history_json TEXT NOT NULL DEFAULT '[]',
    standalone_question TEXT NOT NULL DEFAULT '',
    route TEXT,
    routes TEXT NOT NULL DEFAULT '[]',
    system_prompt_hashes_json TEXT NOT NULL DEFAULT '{}',
    output TEXT NOT NULL DEFAULT '',
    followups_json TEXT NOT NULL DEFAULT '[]',
    sources_json TEXT NOT NULL DEFAULT '[]',
    status TEXT,
    error TEXT,
    llm_calls_json TEXT NOT NULL DEFAULT '[]',
    latency_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_qa_log_created ON qa_log(created_at);
CREATE INDEX IF NOT EXISTS idx_qa_log_user ON qa_log(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_qa_log_request ON qa_log(request_id);
CREATE TABLE IF NOT EXISTS prompts (
    hash TEXT PRIMARY KEY,
    step TEXT NOT NULL,
    text TEXT NOT NULL,
    first_seen TEXT NOT NULL
);
"""

# 旧库升级用：(列名, ADD COLUMN 用的定义)。新库由 SCHEMA 直接建好，这里是幂等的补充。
MIGRATIONS = [
    ("user_name", "user_name TEXT NOT NULL DEFAULT ''"),
]

_lock = threading.Lock()
_inited: set[str] = set()


# ---------------------------------------------------------------- 路径与连接

def db_path() -> Path:
    env = os.environ.get("RAG_QA_DB")
    if env:
        return Path(env)
    if os.environ.get(_TEST_ENV):
        return Path(tempfile.gettempdir()) / "futurx_qa_test" / "qa_log.sqlite"
    return DEFAULT_DB


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5.0)
    if str(path) not in _inited:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)                 # 只有 IF NOT EXISTS：已有库、已有数据原样保留
        _migrate(con)                             # 老库补列（只加列不重建，旧行取默认值）
        _inited.add(str(path))
    return con


def _migrate(con: sqlite3.Connection) -> None:
    """给旧版本建的库补上后来新增的列；只做 ADD COLUMN，不删不改已有数据。"""
    have = {r[1] for r in con.execute("PRAGMA table_info(qa_log)")}
    for col, ddl in MIGRATIONS:
        if col not in have:
            with con:
                con.execute(f"ALTER TABLE qa_log ADD COLUMN {ddl}")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- 标识与历史

def clean_id(raw: Any) -> str:
    """合法的 user_id / session_id 原样返回，否则返回空串。"""
    s = str(raw).strip() if raw is not None else ""
    return s if ID_RE.match(s) else ""


def clean_name(raw: Any) -> str:
    """提问者姓名：去掉控制字符与首尾空白后限长 NAME_MAX；空值返回空串。中文、空格、点的姓名都保留。

    走请求头时非 ASCII 不合法（HTTP 头只能是 latin-1 字节），所以头里允许传百分号编码
    （张三 → %E5%BC%A0%E4%B8%89）；请求体里的 user_name 直接写中文即可。
    """
    s = str(raw) if raw is not None else ""
    s = _unlatin(s)
    if "%" in s:
        s = unquote(s)
    s = "".join(ch for ch in s if ch.isprintable() and ch not in "\r\n\t").strip()
    return s[:NAME_MAX]


def _unlatin(s: str) -> str:
    """HTTP 头按 latin-1 解码，客户端直接发 UTF-8 姓名会变成乱码（张三 → å¼ å ）。
    整串都能用 latin-1 还原且还原后是合法 UTF-8 时按 UTF-8 解释；否则原样返回。"""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def resolve_user_id(raw: Any, key_id: str | None) -> str:
    """用户标识；没有或不合法时记 anon-<key_id>（没有 key 则 anon-none）。"""
    return clean_id(raw) or f"anon-{clean_id(key_id) or 'none'}"


def normalize_history(messages: Any) -> list[dict]:
    """只留 user / assistant 的文本消息，取最近 HISTORY_MAX_MESSAGES 条，每条截断到 HISTORY_MAX_CHARS 字；
    首条若是 assistant 则丢弃（保持 用户→助手 成对，路由看得懂）。非法输入返回空列表。"""
    out: list[dict] = []
    if not isinstance(messages, (list, tuple)):
        return out
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = getattr(m, "role", None), getattr(m, "content", None)
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content.strip()[:HISTORY_MAX_CHARS]})
    out = out[-HISTORY_MAX_MESSAGES:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def prompt_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def system_text(messages: Iterable[dict]) -> str:
    return "\n\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") == "system")


# ---------------------------------------------------------------- 写入

def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def write(record: dict, *, user_id: str, session_id: str = "", history: list | None = None, standalone: str = "",
          prompts: dict[str, tuple[str, str]] | None = None, user_name: str = "") -> bool:
    """record：usage_log 的那条记录（dict）。prompts：{hash: (step, text)}。成功返回 True；任何失败只记 warning。"""
    try:
        path = db_path()
        hashes = {}
        for h, (step, _text) in (prompts or {}).items():
            hashes[step] = h
        row = (
            str(record.get("request_id") or ""), user_id, clean_name(user_name), record.get("key_id"), session_id,
            now_iso(), record.get("origin"), str(record.get("question") or ""), _j(history or []), (standalone or "")[:500],
            record.get("route"), _j(list(record.get("routes") or [])), _j(hashes),
            str(record.get("answer") or ""), _j(list(record.get("followups") or [])), _j(list(record.get("sources") or [])[:SOURCES_MAX]),
            record.get("status"), record.get("error"), _j(list(record.get("llm_calls") or [])), record.get("duration_ms"),
        )
        with _lock:
            con = _connect(path)
            try:
                with con:
                    for h, (step, text) in (prompts or {}).items():
                        con.execute("INSERT OR IGNORE INTO prompts(hash, step, text, first_seen) VALUES (?,?,?,?)", (h, step, text, now_iso()))
                    con.execute(
                        "INSERT INTO qa_log(request_id,user_id,user_name,key_id,session_id,created_at,origin,input_prompt,history_json,standalone_question,"
                        "route,routes,system_prompt_hashes_json,output,followups_json,sources_json,status,error,llm_calls_json,latency_ms) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            finally:
                con.close()
        return True
    except Exception as e:  # noqa: BLE001 - 记录失败绝不影响主请求
        log.warning("qa_store 写入失败（%s），本条丢弃", type(e).__name__)
        return False


def log_request(rec, record: dict) -> None:
    """usage_log.RequestLog.finish 的旁路挂钩：rec 是 RequestLog，record 是刚写进 JSONL 的那条。"""
    try:
        meta = getattr(rec, "qa_meta", None) or {}
        write(record, user_id=resolve_user_id(meta.get("user_id"), record.get("key_id")), session_id=clean_id(meta.get("session_id")),
              history=meta.get("history"), standalone=meta.get("standalone") or "", prompts=dict(getattr(rec, "prompts", {}) or {}),
              user_name=meta.get("user_name") or "")
    except Exception as e:  # noqa: BLE001
        log.warning("qa_store 挂钩失败（%s）", type(e).__name__)


def log_simple(record: dict, meta: dict | None) -> None:
    """usage_log.simple_record（/ask、/v1/chat/completions）的旁路挂钩。"""
    try:
        meta = meta or {}
        write(record, user_id=resolve_user_id(meta.get("user_id"), record.get("key_id")), session_id=clean_id(meta.get("session_id")),
              history=meta.get("history"), standalone="", prompts=meta.get("prompts"), user_name=meta.get("user_name") or "")
    except Exception as e:  # noqa: BLE001
        log.warning("qa_store 挂钩失败（%s）", type(e).__name__)


# ---------------------------------------------------------------- 读取（「我之前问过什么」）

def recent_questions(user_id: str, *, session_id: str | None = None, limit: int = 20) -> dict:
    """这个人问过的最近 limit 个问题（新的在前），以及总条数。只读，不写库。

    session_id 有值时只算这个会话；没有值时算这个人的全部提问。
    按 user_id + 问题去重（同一句连问几次只列一次，位置取最近那次），避免把重试刷屏。
    库不存在 / 读失败返回空结果（调用方按「还没有提问记录」作答），不影响主流程。
    """
    empty = {"items": [], "total": 0}
    uid = clean_id(user_id)
    if not uid:
        return empty
    limit = max(1, min(int(limit or 20), 100))
    where = " FROM qa_log WHERE user_id = ?"
    args: list[Any] = [uid]
    if session_id:
        where += " AND session_id = ?"
        args.append(clean_id(session_id))
    try:
        con = _connect(db_path())
    except Exception as e:  # noqa: BLE001 - 读不到就当作没有记录
        log.warning("qa_store 读取失败（%s）", type(e).__name__)
        return empty
    try:
        rows = con.execute("SELECT input_prompt, created_at" + where + " ORDER BY id DESC LIMIT ?",
                           [*args, limit]).fetchall()
        total = con.execute("SELECT COUNT(*)" + where, args).fetchone()[0] or 0
    except Exception as e:  # noqa: BLE001 - 老库缺列等情况
        log.warning("qa_store 查询失败（%s）", type(e).__name__)
        return empty
    finally:
        con.close()
    seen: set[str] = set()
    items: list[dict] = []
    for prompt, created_at in rows:
        q = " ".join(str(prompt or "").split())          # 题干里可能有换行
        if not q or q in seen:                           # 同一句连问几次只列一次（位置取最近那次）
            continue
        seen.add(q)
        items.append({"question": q, "created_at": str(created_at or "")})
    return {"items": items, "total": total}


def request_meta(headers, *, body_user: Any = None, body_uuid: Any = None, body_name: Any = None,
                 history: Any = None) -> dict:
    """从请求头 / 请求体取调用方身份与历史。

    uuid：body 的 uuid 优先，其次 X-User-Uuid 头，最后退回 OpenAI 标准字段 user / X-User-Id（保持旧调用方不变）；
    user_name：body 的 user_name 优先，其次 X-User-Name 头（中文可用，不参与 user_id 校验）；
    session：X-Session-Id。
    """
    get = headers.get if hasattr(headers, "get") else (lambda k, d=None: d)
    ident = body_uuid if body_uuid not in (None, "") else get("x-user-uuid")
    return {"user_id": clean_id(ident) or clean_id(body_user) or clean_id(get("x-user-id")),
            "user_name": clean_name(body_name if body_name not in (None, "") else get("x-user-name")),
            "session_id": clean_id(get("x-session-id")),
            "history": normalize_history(history)}


__all__ = ["db_path", "write", "log_request", "log_simple", "request_meta", "resolve_user_id", "clean_id", "clean_name",
           "normalize_history", "prompt_hash", "system_text", "recent_questions", "SCHEMA", "MIGRATIONS", "NAME_MAX"]
