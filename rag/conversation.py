"""会话存储：把每一轮问答（用户问题 + 助手回答）按会话落盘，供 API 查询。

与 rag/usage_log 的分工：
- 本模块是**对话档案**（事实来源）：谁在什么时候问了什么、答了什么、引了哪些来源。长期保留。
- usage_log 是**计量日志**（延迟 / token / 路由 / 失败率），按天 JSONL、默认 90 天。

存储：SQLite（WAL）。单进程部署，写入量小；WAL 让读接口不阻塞正在写回答的请求。
连接按线程持有（threading.local）：流式回答在 Starlette 的线程池里逐段恢复，跨线程共用
一个 connection 会报 "SQLite objects created in a thread..."。

归属：会话记 key_id（API key 的 sha256 前 8 位，见 rag.usage_log.key_id）与 user_id（调用方传的 uuid）。
查询按 (key_id, user_id) 过滤：A 的 key、A 的 uuid 都看不到 B 的会话。
- user_id 为 None：不按人过滤（脚本 / 运维用）；
- user_id 为 ""：只取「没有人员标识」的老会话（key 内共享），供未传身份的老调用方；
- user_id 有值：只取这个人的会话。
未配置 API_KEYS 时 key_id 为 None，该实例内所有匿名调用共享；一人一 key 时 key 已经隔离，不传 uuid 也不串号。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "rag" / "conversation_db" / "conversations.sqlite3"

_TITLE_CHARS = 40
_MAX_CONTENT = 20000           # 单条消息截断（防止超长粘贴撑爆库；正常问答远小于此）

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    key_id      TEXT,
    user_id     TEXT,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0,
    deleted_at  REAL
);
CREATE INDEX IF NOT EXISTS ix_conv_owner ON conversations (key_id, deleted_at, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    route           TEXT,
    status          TEXT,
    error           TEXT,
    search_query    TEXT,
    sources         TEXT,
    latency_ms      REAL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    key_id          TEXT,
    month           TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_msg_seq ON messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages (conversation_id, seq);
"""
# (key_id, month) 与 (key_id, user_id) 的索引放在 _migrate 里建：老库此时还没有这些列，
# 写在这里会让 executescript 整段失败，连建表都做不成。

# 后加的列：老库要就地补，不能只改 _SCHEMA（CREATE TABLE IF NOT EXISTS 对已有表不生效）。
# key_id：这一轮是哪个 API key 问的。会话级已有，但会话可能被改归属（迁移/转让），
#   消息级记一份才是「当时的调用方」的真实记录，按人审计不依赖会话表。
# month：YYYY-MM，按周期（月）统计不用每次算时间，直接命中索引。
_MIGRATIONS = (
    ("messages", "key_id", "TEXT"),
    ("messages", "month", "TEXT"),
    # 调用方传的 uuid：共享一把 key 时按人隔离会话，见模块头「归属」
    ("conversations", "user_id", "TEXT"),
)


def db_path() -> Path:
    """库文件路径；每次读取环境变量，便于测试与部署覆盖。"""
    return Path(os.environ.get("CONVERSATION_DB") or DEFAULT_DB)


def retention_days() -> int:
    """会话保留天数；0 表示永久保留。"""
    try:
        return max(0, int(os.environ.get("CONVERSATION_RETENTION_DAYS", "365")))
    except ValueError:
        return 365


_local = threading.local()
_init_lock = threading.Lock()
_ready: set[str] = set()


def _connect() -> sqlite3.Connection:
    path = db_path()
    key = str(path)
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "key", None) == key:
        return conn
    if conn is not None:                      # 测试切换了路径：关掉旧连接
        try:
            conn.close()
        except sqlite3.Error:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # busy_timeout 要最先设：后面切 WAL / 建表都可能撞上别的连接，没有它就直接抛 database is locked
    conn.execute("PRAGMA busy_timeout=5000")
    with _init_lock:                           # 建表只需一次，但多线程同时启动时要串行
        if key not in _ready:
            # 切 WAL 要独占锁：空库从 delete 模式切换时尤其如此。放在锁外会让两个线程同时首连就报
            # database is locked（测试用全新临时库时必现），所以和建表一起放进初始化锁。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            _migrate(conn)
            _ready.add(key)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn, _local.key = conn, key
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """给老库补列并回填。幂等：列已存在则跳过。

    回填规则：
    - messages.key_id：优先取本会话的 key_id（同一次对话都是同一个人问的）；
      老数据的会话 key_id 也可能是 NULL，那就保持 NULL。
    - messages.month：从 created_at 算 YYYY-MM（按本地时区，与使用者的「月份」直觉一致）。
    """
    for table, column, ctype in _MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column in cols:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
        logger.info("会话库迁移：%s.%s 已新增", table, column)

    conn.execute(
        "UPDATE messages SET key_id = (SELECT key_id FROM conversations c WHERE c.id = messages.conversation_id) "
        "WHERE key_id IS NULL AND EXISTS (SELECT 1 FROM conversations c "
        "WHERE c.id = messages.conversation_id AND c.key_id IS NOT NULL)"
    )
    # 只补 month 为空的行，避免每次启动重算全表
    rows = conn.execute("SELECT id, created_at FROM messages WHERE month IS NULL").fetchall()
    for r in rows:
        conn.execute("UPDATE messages SET month = ? WHERE id = ?", (_month_of(r["created_at"]), r["id"]))
    conn.execute("CREATE INDEX IF NOT EXISTS ix_msg_key_month ON messages (key_id, month)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_conv_owner_user "
                 "ON conversations (key_id, user_id, deleted_at, updated_at DESC)")


def _month_of(ts: float | None = None) -> str:
    """时间戳 → 'YYYY-MM'（本地时区）。"""
    return time.strftime("%Y-%m", time.localtime(_now() if ts is None else ts))


def reset_connections() -> None:
    """关掉本线程连接（测试用）。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _local.conn = None
    _local.key = None


# ---------------------------------------------------------------- 读写

def _owner_sql(key_id: str | None, user_id: str | None) -> tuple[str, list[Any]]:
    """conversations 的归属过滤条件（key_id + user_id），返回 (SQL 片段, 参数)。

    key_id  None：不按 key 过滤（脚本 / 运维）
    user_id None：不按人过滤（脚本 / 运维）
    user_id ""：只看没有人员标识的老会话（未传 uuid 的老调用方；迁移后是 NULL）
    user_id 有值：只看这个人的会话
    """
    sql, args = "", []
    if key_id is not None:
        sql += " AND key_id = ?"
        args.append(key_id)
    if user_id is not None:
        if user_id == "":
            sql += " AND COALESCE(user_id, '') = ''"
        else:
            sql += " AND user_id = ?"
            args.append(user_id)
    return sql, args


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return "conv_" + uuid.uuid4().hex


def create(*, key_id: str | None = None, title: str | None = None,
           conversation_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """建会话；conversation_id 由调用方指定（便于客户端自己生成）。
    user_id：调用方的人员标识（uuid），共享 key 时按人隔离。"""
    cid = conversation_id or _new_id()
    now = _now()
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (id, key_id, user_id, title, created_at, updated_at, turn_count, archived) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
        (cid, key_id, user_id or None, (title or None), now, now),
    )
    return {"id": cid, "key_id": key_id, "user_id": user_id or None, "title": title, "created_at": now,
            "updated_at": now, "turn_count": 0, "archived": False}


def get(conversation_id: str, *, key_id: str | None = None,
        user_id: str | None = None) -> dict[str, Any] | None:
    """取会话；key_id / user_id 不为 None 时校验归属（别人的会话当作不存在）。"""
    sql = "SELECT * FROM conversations WHERE id = ? AND deleted_at IS NULL"
    args: list[Any] = [conversation_id]
    extra, more = _owner_sql(key_id, user_id)
    row = _connect().execute(sql + extra, args + more).fetchone()
    return _conv(row) if row else None


def ensure(conversation_id: str | None, *, key_id: str | None = None,
           user_id: str | None = None) -> dict[str, Any]:
    """有则取、无则建；已删除或不属于该 key / 该人时抛 LookupError（调用方转 404）。"""
    if conversation_id:
        got = get(conversation_id, key_id=key_id, user_id=user_id)
        if got is None:
            raise LookupError(conversation_id)
        return got
    return create(key_id=key_id, user_id=user_id)


def list_conversations(*, key_id: str | None = None, limit: int = 20,
                       cursor: str | None = None, archived: bool | None = False,
                       user_id: str | None = None) -> dict[str, Any]:
    """按最近更新倒序分页。cursor 为上一页最后一条的 "updated_at:id"。
    user_id 语义同 get：None 不过滤，"" 只看无人员标识的老会话，有值只看这个人。"""
    limit = max(1, min(int(limit), 100))
    sql = "SELECT * FROM conversations WHERE deleted_at IS NULL"
    args: list[Any] = []
    extra, more = _owner_sql(key_id, user_id)
    sql += extra
    args += more
    if archived is not None:
        sql += " AND archived = ?"
        args.append(1 if archived else 0)
    if cursor:
        ts, _, last_id = cursor.partition(":")
        try:
            sql += " AND (updated_at < ? OR (updated_at = ? AND id < ?))"
            args += [float(ts), float(ts), last_id]
        except ValueError:
            pass
    sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    args.append(limit + 1)                     # 多取一条判断是否还有下一页
    rows = _connect().execute(sql, args).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    nxt = f"{rows[-1]['updated_at']}:{rows[-1]['id']}" if has_more and rows else None
    return {"items": [_conv(r) for r in rows], "next_cursor": nxt}


def messages(conversation_id: str, *, after_seq: int | None = None, limit: int = 200,
             key_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
    """取消息；after_seq 用于增量拉取（前端断线重连）。

    key_id / user_id 不为 None 时校验会话归属，不属于该 key 或该人直接返回空列表。
    以前只有 API 层校验归属，存储层裸着——任何新接口忘了先查就会串号。
    """
    if (key_id is not None or user_id is not None) and get(conversation_id, key_id=key_id, user_id=user_id) is None:
        return []
    limit = max(1, min(int(limit), 500))
    sql = "SELECT * FROM messages WHERE conversation_id = ?"
    args: list[Any] = [conversation_id]
    if after_seq is not None:
        sql += " AND seq > ?"
        args.append(int(after_seq))
    sql += " ORDER BY seq ASC LIMIT ?"
    args.append(limit)
    return [_msg(r) for r in _connect().execute(sql, args).fetchall()]


def monthly_stats(*, key_id: str | None = None, months: int = 12) -> list[dict[str, Any]]:
    """按 (key_id, 月) 统计：消息数、提问轮数、平均耗时。

    回答「每人每月用了多少」，走 ix_msg_key_month 索引，比翻 messages 快得多。
    """
    sql = ("SELECT key_id, month, COUNT(*) AS messages, "
           "SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS turns, "
           "AVG(CASE WHEN role = 'assistant' THEN latency_ms END) AS avg_latency_ms "
           "FROM messages WHERE month IS NOT NULL")
    args: list[Any] = []
    if key_id is not None:
        sql += " AND key_id = ?"
        args.append(key_id)
    sql += " GROUP BY key_id, month ORDER BY month DESC, key_id LIMIT ?"
    args.append(max(1, int(months)) * (1 if key_id else 500))
    return [{"key_id": r["key_id"], "month": r["month"], "messages": r["messages"],
             "turns": r["turns"],
             "avg_latency_ms": None if r["avg_latency_ms"] is None else round(r["avg_latency_ms"], 1)}
            for r in _connect().execute(sql, args).fetchall()]


def reassign_key(*, src_key_id: str | None, dst_key_id: str, dry_run: bool = False) -> dict[str, int]:
    """把一批会话（及其消息）改挂到另一个 key。

    用途：配置 API_KEYS 之前产生、key_id 为 NULL 的匿名会话谁都打不开，只能人工归置。
    src_key_id 传 None 表示选所有 key_id 为 NULL 的会话。
    """
    conn = _connect()
    if src_key_id is None:
        where, args = "key_id IS NULL", []
    else:
        where, args = "key_id = ?", [src_key_id]
    ids = [r["id"] for r in conn.execute(f"SELECT id FROM conversations WHERE {where}", args).fetchall()]
    if dry_run or not ids:
        return {"conversations": len(ids), "messages": 0, "dry_run": int(bool(dry_run))}
    marks = ",".join("?" * len(ids))
    conn.execute("BEGIN IMMEDIATE")
    try:
        n = conn.execute(f"SELECT COUNT(*) AS n FROM messages WHERE conversation_id IN ({marks})",
                         ids).fetchone()["n"]
        conn.execute(f"UPDATE conversations SET key_id = ? WHERE id IN ({marks})", [dst_key_id] + ids)
        # 只补 key_id 为 NULL 的消息，避免覆盖后来「换人后新写入」的记录
        conn.execute(f"UPDATE messages SET key_id = ? WHERE conversation_id IN ({marks}) AND key_id IS NULL",
                     [dst_key_id] + ids)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {"conversations": len(ids), "messages": n, "dry_run": 0}


def history(conversation_id: str, *, max_turns: int, max_chars: int | None = None) -> list[tuple[str, str]]:
    """取最近 max_turns 轮的 (role, content)，供送模型。

    按**整轮**截断（user+assistant 一起保留或一起丢），否则字数超限时会留下一条没有对应
    问题的助手回答，模型会看到断裂的上下文。max_chars 从最新一轮往前累加。
    """
    rows = _connect().execute(
        "SELECT role, content, seq FROM messages WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
        (conversation_id, max(1, int(max_turns)) * 2),
    ).fetchall()
    # 从新到旧分组：assistant 与其前面的 user 属于同一轮（seq 相邻、user 在前）
    turns: list[list[tuple[str, str]]] = []
    pending: list[tuple[str, str]] = []
    for row in rows:
        item = (row["role"], row["content"] or "")
        if row["role"] == "assistant":
            pending = [item]
        elif row["role"] == "user" and pending:
            pending.insert(0, item)
            turns.append(pending)
            pending = []
        elif row["role"] == "user":                      # 落单的 user（上一轮失败没回答）
            turns.append([item])
            pending = []
    picked: list[list[tuple[str, str]]] = []
    used = 0
    for turn in turns:
        size = sum(len(text) for _, text in turn)
        if picked and max_chars is not None and used + size > max_chars:
            break
        picked.append(turn)
        used += size
    return [msg for turn in reversed(picked) for msg in turn]


def append_turn(*, conversation_id: str, question: str, answer: str, key_id: str | None = None,
                route: str | None = None, status: str = "ok", error: str | None = None,
                search_query: str | None = None, sources: Iterable[dict] | None = None,
                latency_ms: float | None = None, prompt_tokens: int | None = None,
                completion_tokens: int | None = None, request_id: str | None = None,
                user_id: str | None = None) -> dict[str, Any]:
    """追加一轮：user 消息 + assistant 消息，同一事务内完成（避免只写进半轮）。

    request_id 作 user 消息 id：同一请求重试不会重复插入（已存在则跳过）。
    """
    conn = _connect()
    now = _now()
    conv = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    if conv is None:
        raise LookupError(conversation_id)
    if key_id is not None and conv["key_id"] is not None and conv["key_id"] != key_id:
        raise PermissionError(conversation_id)
    # 共享一把 key 时再校验人：别人的会话绝不能被写进来
    if user_id is not None and (conv["user_id"] or "") != (user_id or ""):
        raise PermissionError(conversation_id)

    seq = (conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM messages WHERE conversation_id = ?",
                        (conversation_id,)).fetchone()["s"]) + 1
    uid = request_id or uuid.uuid4().hex
    if request_id:                              # 同一请求重试：已落过就直接返回，不重复插入
        exists = conn.execute("SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
                              (request_id, conversation_id)).fetchone()
        if exists:
            return {"conversation_id": conversation_id, "seq": seq, "title": conv["title"],
                    "turn_count": conv["turn_count"], "duplicate": True}
    src_json = json.dumps(list(sources or []), ensure_ascii=False)
    answer = (answer or "")[:_MAX_CONTENT]
    q = (question or "")[:_MAX_CONTENT]
    title = conv["title"] or _title_of(q)
    # 消息级也记 key：会话的归属可能被迁移（老数据归置、换人接手），
    # 而「这一轮当时是谁问的」不该跟着变。
    owner = key_id if key_id is not None else conv["key_id"]
    month = _month_of(now)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, seq, role, content, created_at, route, status, "
            "error, search_query, sources, latency_ms, prompt_tokens, completion_tokens, key_id, month) "
            "VALUES (?, ?, ?, 'user', ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, NULL, ?, ?)",
            (uid, conversation_id, seq, q, now, route, status, search_query, "[]", owner, month),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, seq, role, content, created_at, route, status, "
            "error, search_query, sources, latency_ms, prompt_tokens, completion_tokens, key_id, month) "
            "VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, conversation_id, seq + 1, answer, now, route, status, error,
             src_json, latency_ms, prompt_tokens, completion_tokens, owner, month),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ?, turn_count = turn_count + 1, title = ? WHERE id = ?",
            (now, title, conversation_id),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {"conversation_id": conversation_id, "seq": seq, "title": title, "turn_count": conv["turn_count"] + 1}


def rename(conversation_id: str, title: str, *, key_id: str | None = None,
           user_id: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
    if get(conversation_id, key_id=key_id, user_id=user_id) is None:
        return None
    conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                 ((title or "")[:200], _now(), conversation_id))
    return get(conversation_id, key_id=key_id, user_id=user_id)


def set_archived(conversation_id: str, archived: bool, *, key_id: str | None = None,
                 user_id: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
    if get(conversation_id, key_id=key_id, user_id=user_id) is None:
        return None
    conn.execute("UPDATE conversations SET archived = ?, updated_at = ? WHERE id = ?",
                 (1 if archived else 0, _now(), conversation_id))
    return get(conversation_id, key_id=key_id, user_id=user_id)


def delete(conversation_id: str, *, key_id: str | None = None, user_id: str | None = None) -> bool:
    """软删除：列表与详情不再可见，数据保留到清理任务；便于误删恢复。"""
    conn = _connect()
    if get(conversation_id, key_id=key_id, user_id=user_id) is None:
        return False
    conn.execute("UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE id = ?",
                 (_now(), _now(), conversation_id))
    return True


def clear_messages(conversation_id: str, *, key_id: str | None = None,
                   user_id: str | None = None) -> int | None:
    """清空消息但保留会话（重开话题）；会话不存在或不属于该 key 返回 None。"""
    conn = _connect()
    if get(conversation_id, key_id=key_id, user_id=user_id) is None:
        return None
    n = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                     (conversation_id,)).fetchone()["n"]
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("UPDATE conversations SET turn_count = 0, updated_at = ? WHERE id = ?",
                     (_now(), conversation_id))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return n


def purge_expired(*, days: int | None = None) -> int:
    """物理删除超过保留期的会话（软删的也一并清）；days=0 表示不清理。返回删除条数。"""
    keep = retention_days() if days is None else days
    if keep <= 0:
        return 0
    cutoff = _now() - keep * 86400
    conn = _connect()
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM conversations WHERE updated_at < ? OR (deleted_at IS NOT NULL AND deleted_at < ?)",
        (cutoff, cutoff)).fetchall()]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM conversations WHERE id IN ({marks})", ids)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(ids)


# ---------------------------------------------------------------- 行转换

def _conv(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "title": row["title"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "turn_count": row["turn_count"],
            "archived": bool(row["archived"]), "key_id": row["key_id"],
            "user_id": row["user_id"]}


def _msg(row: sqlite3.Row) -> dict[str, Any]:
    try:
        sources = json.loads(row["sources"] or "[]")
    except ValueError:
        sources = []
    return {"id": row["id"], "conversation_id": row["conversation_id"], "seq": row["seq"],
            "role": row["role"], "content": row["content"], "created_at": row["created_at"],
            "route": row["route"], "status": row["status"], "error": row["error"],
            "search_query": row["search_query"], "sources": sources,
            "latency_ms": row["latency_ms"], "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "key_id": row["key_id"], "month": row["month"]}


def _title_of(question: str) -> str:
    text = " ".join((question or "").split())
    return text[:_TITLE_CHARS] or "新会话"


__all__ = ["db_path", "retention_days", "create", "get", "ensure", "list_conversations", "messages",
           "history", "append_turn", "rename", "set_archived", "delete", "clear_messages",
           "purge_expired", "reset_connections", "monthly_stats", "reassign_key"]
