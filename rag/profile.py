"""学习者画像：记住这个人问过哪些知识点、关注哪些章，下一次作答时给一句「接着上次」的上下文。

为什么单独一个库：会话库（rag.conversation）管「聊了什么」，问答记录（rag.qa_store）管「审计」，
这里管「这个人现在在哪儿」——三者用途不同，混在一张表里会让会话清理连学习状态一起删掉。

记录什么（只记能解释来源的事实，不做推断）：
- concept：这个人问过 / 被讲解过的课程概念（来自课程管线的 selected_concepts），记次数与最近时间；
- chapter：判例 / 试题命中的章，记次数与最近时间；
- 不记答题对错：产品链路目前没有「学生作答后判分」的入口，凭空记账就是编数据。

怎么用：`summary(user_id)` 给一段不超过两行的中文，拼进作答提示词的「关系层」；
读不到 / 没记录时返回空串，调用方什么都不加（不产生「我记得你」这类编造的亲近感）。

并发与失败：SQLite + WAL，写失败只记 warning，绝不影响主请求；路径可用 PROFILE_DB 覆盖，测试指向临时文件。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "profile.sqlite"
CONCEPT_MAX = 8            # 每人最多留多少个概念（超出按最近时间淘汰）
CHAPTER_MAX = 5
SUMMARY_MAX_ITEMS = 3      # 拼进提示词时最多列几个概念

SCHEMA = """
CREATE TABLE IF NOT EXISTS learner_concept (
    user_id     TEXT NOT NULL,
    concept     TEXT NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 1,
    first_at    REAL NOT NULL,
    last_at     REAL NOT NULL,
    PRIMARY KEY (user_id, concept)
);
CREATE TABLE IF NOT EXISTS learner_chapter (
    user_id     TEXT NOT NULL,
    chapter     INTEGER NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 1,
    last_at     REAL NOT NULL,
    PRIMARY KEY (user_id, chapter)
);
"""

_lock = threading.Lock()
_inited: set[str] = set()


def db_path() -> Path:
    import os

    return Path(os.environ.get("PROFILE_DB") or DEFAULT_DB)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5.0)
    if str(path) not in _inited:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        _inited.add(str(path))
    return con


def reset(path: str | Path | None = None) -> None:
    """测试用：忘掉「已初始化」标记（换库文件时必须调）。"""
    if path is None:
        _inited.clear()
    else:
        _inited.discard(str(path))


def _clean(user_id: Any) -> str:
    from .qa_store import clean_id

    return clean_id(user_id)


def notice_question(user_id: Any, *, concepts: list[str] | None = None, chapter: int | None = None) -> None:
    """记一次提问：`concepts` 是课程管线选中的概念名，`chapter` 是判例 / 试题命中的章。

    只记名字与章号，不记原文、不记答案；同名概念只累加次数与最近时间。
    任何失败都只记 warning —— 学习状态是加分项，不能因为它让回答失败。
    """
    uid = _clean(user_id)
    if not uid:
        return
    now = time.time()
    names = [str(c).strip()[:40] for c in (concepts or []) if str(c or "").strip()]
    if not names and chapter is None:
        return
    try:
        with _lock:
            con = _connect(db_path())
            try:
                with con:
                    for name in dict.fromkeys(names):
                        con.execute(
                            "INSERT INTO learner_concept(user_id, concept, hits, first_at, last_at) VALUES (?,?,1,?,?) "
                            "ON CONFLICT(user_id, concept) DO UPDATE SET hits = hits + 1, last_at = excluded.last_at",
                            (uid, name, now, now))
                    if chapter is not None:
                        con.execute(
                            "INSERT INTO learner_chapter(user_id, chapter, hits, last_at) VALUES (?,?,1,?) "
                            "ON CONFLICT(user_id, chapter) DO UPDATE SET hits = hits + 1, last_at = excluded.last_at",
                            (uid, int(chapter), now))
                    _trim(con, uid)
            finally:
                con.close()
    except Exception as e:  # noqa: BLE001 - 画像写失败不影响回答
        log.warning("profile 写入失败（%s），本条丢弃", type(e).__name__)


def _trim(con: sqlite3.Connection, uid: str) -> None:
    """每人只留最近 CONCEPT_MAX 个概念、CHAPTER_MAX 个章，避免长期用户越积越多。"""
    # 同一轮选出的概念往往超过 8 个：按命中次数排序更稳，其次按最近时间
    con.execute(
        "DELETE FROM learner_concept WHERE user_id = ? AND concept NOT IN "
        "(SELECT concept FROM learner_concept WHERE user_id = ? ORDER BY last_at DESC LIMIT ?)",
        (uid, uid, CONCEPT_MAX))
    con.execute(
        "DELETE FROM learner_chapter WHERE user_id = ? AND chapter NOT IN "
        "(SELECT chapter FROM learner_chapter WHERE user_id = ? ORDER BY last_at DESC LIMIT ?)",
        (uid, uid, CHAPTER_MAX))


def state(user_id: Any) -> dict:
    """这个人的学习状态：{"concepts": [{concept, hits, last_at}], "chapters": [int]}，新的在前。"""
    empty = {"concepts": [], "chapters": []}
    uid = _clean(user_id)
    if not uid:
        return empty
    try:
        con = _connect(db_path())
    except Exception as e:  # noqa: BLE001
        log.warning("profile 读取失败（%s）", type(e).__name__)
        return empty
    try:
        concepts = [{"concept": r[0], "hits": r[1], "last_at": r[2]} for r in con.execute(
            "SELECT concept, hits, last_at FROM learner_concept WHERE user_id = ? ORDER BY last_at DESC LIMIT ?",
            (uid, CONCEPT_MAX))]
        chapters = [r[0] for r in con.execute(
            "SELECT chapter FROM learner_chapter WHERE user_id = ? ORDER BY last_at DESC LIMIT ?",
            (uid, CHAPTER_MAX))]
    except Exception as e:  # noqa: BLE001 - 老库缺表等
        log.warning("profile 查询失败（%s）", type(e).__name__)
        return empty
    finally:
        con.close()
    return {"concepts": concepts, "chapters": chapters}


def summary(user_id: Any) -> str:
    """拼进提示词的关系层：一句话说明这个人问过什么、关注哪章。没有记录返回空串。

    只陈述已经发生的事（问过什么），不做任何推断（不说「你基础薄弱」）——推断没有依据，
    写进提示词会让模型顺着编。只取问过两次以上的概念（「反复问的才是主题」），
    问过一次的不算——课程管线一次会选出好几个概念，全列出来等于把词表塞进提示词。
    """
    st = state(user_id)
    repeated = [c["concept"] for c in st["concepts"] if c["hits"] >= 2]
    st = {"concepts": [{"concept": c} for c in repeated], "chapters": st["chapters"]}
    names = [c["concept"] for c in st["concepts"]][:SUMMARY_MAX_ITEMS]
    if not names and not st["chapters"]:
        return ""
    parts = []
    if names:
        parts.append("之前问过：" + "、".join(names))
    if st["chapters"]:
        parts.append("关注过的章：" + "、".join(f"第{c}章" for c in st["chapters"][:2]))
    return ("【这个学生的情况】" + "；".join(parts)
            + "。回答时可以顺带把它和上面这些内容联系起来，但不要生硬复述、不要推断他的水平。")


__all__ = ["notice_question", "state", "summary", "db_path", "reset", "SCHEMA", "CONCEPT_MAX", "CHAPTER_MAX"]
