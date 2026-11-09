"""学习者画像：记住这个人问过哪些知识点、关注哪些章，下一次作答时给一句「接着上次」的上下文。

为什么单独一个库：会话库（rag.conversation）管「聊了什么」，问答记录（rag.qa_store）管「审计」，
这里管「这个人现在在哪儿」——三者用途不同，混在一张表里会让会话清理连学习状态一起删掉。

记录什么（只记能解释来源的事实，不做推断）：
- concept：这个人问过 / 被讲解过的课程概念（来自课程管线的 selected_concepts），记次数与最近时间；
- chapter：判例 / 试题命中的章，记次数与最近时间；
- mastery：做题的判分结果（question_id → 最近一次得分与是否答对）。只在学生真的提交了答案、
  真的跑了判分之后才写，没有提交就没有记录——这一层的前提是「有分可判」，不是推断。

怎么用：`summary(user_id)` 给一段不超过两行的中文，拼进作答提示词的「关系层」；
读不到 / 没记录时返回空串，调用方什么都不加（不产生「我记得你」这类编造的亲近感）。

并发与失败：SQLite + WAL，写失败只记 warning，绝不影响主请求；路径可用 PROFILE_DB 覆盖，测试指向临时文件。
"""
from __future__ import annotations

import json
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
SCORE_MAX = 50             # 每个学生最多留多少条判分记录
SUMMARY_MAX_ITEMS = 3      # 拼进提示词时最多列几个概念
WEAK_MAX_ITEMS = 2         # 「还没答对」最多列几个：列多了像在念错题本

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
CREATE TABLE IF NOT EXISTS learner_mastery (
    user_id     TEXT NOT NULL,
    question_id TEXT NOT NULL,
    chapter     INTEGER,
    concepts    TEXT NOT NULL DEFAULT '[]',   -- JSON 数组：这道题挂接的概念名
    score       REAL NOT NULL,                -- 1–5 分
    correct     INTEGER NOT NULL,             -- 0 / 1
    method      TEXT NOT NULL DEFAULT '',     -- llm | lexical
    last_at     REAL NOT NULL,
    PRIMARY KEY (user_id, question_id)
);
CREATE TABLE IF NOT EXISTS learner_profile (
    user_id     TEXT PRIMARY KEY,
    level       TEXT NOT NULL DEFAULT '',     -- 本科 / 硕士 / 博士 / 教师
    major       TEXT NOT NULL DEFAULT '',     -- 编辑出版、公共管理……
    field       TEXT NOT NULL DEFAULT '',     -- 学科门类：新闻传播、管理学……
    stage       TEXT NOT NULL DEFAULT '',     -- 选题 / 开题 / 写作 / 答辩
    updated_at  REAL NOT NULL
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


def notice_score(user_id: Any, *, question_id: str, score: float, correct: bool,
                 concepts: list[str] | None = None, chapter: int | None = None,
                 method: str = "") -> None:
    """记一次判分结果：同一道题只留最近一次的分数（重做以最后一次为准）。

    先决条件是真的提交了作答并且真的判了分——调用方（rag.grade）只在有学生答案时才会调到这里，
    所以这里写下的每一个分数都能解释来源，不是推断出来的「掌握程度」。
    写失败只记 warning，与 notice_question 同口径。
    """
    uid = _clean(user_id)
    qid = str(question_id or "").strip()[:64]
    if not uid or not qid:
        return
    now = time.time()
    names = [str(c).strip()[:40] for c in (concepts or []) if str(c or "").strip()]
    try:
        with _lock:
            con = _connect(db_path())
            try:
                with con:
                    con.execute(
                        "INSERT INTO learner_mastery(user_id, question_id, chapter, concepts, score, correct, method, last_at) "
                        "VALUES (?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(user_id, question_id) DO UPDATE SET chapter = excluded.chapter, "
                        "concepts = excluded.concepts, score = excluded.score, correct = excluded.correct, "
                        "method = excluded.method, last_at = excluded.last_at",
                        (uid, qid, int(chapter) if chapter else None, json.dumps(names, ensure_ascii=False),
                         float(score), 1 if correct else 0, str(method or "")[:16], now))
                    _trim_scores(con, uid)
            finally:
                con.close()
    except Exception as e:  # noqa: BLE001 - 画像写失败不影响判分结果
        log.warning("profile 判分写入失败（%s），本条丢弃", type(e).__name__)


def _trim_scores(con: sqlite3.Connection, uid: str) -> None:
    """判分记录只留最近 SCORE_MAX 条，避免长期做题的人越积越多。"""
    con.execute(
        "DELETE FROM learner_mastery WHERE user_id = ? AND question_id NOT IN "
        "(SELECT question_id FROM learner_mastery WHERE user_id = ? ORDER BY last_at DESC LIMIT ?)",
        (uid, uid, SCORE_MAX))


def weaknesses(user_id: Any, limit: int = SUMMARY_MAX_ITEMS) -> list[str]:
    """还没掌握的概念：最近一次判分没答对的那几道题挂接的概念。

    只按「同一道题重做答对」来消除，不做时间衰减——没有依据的遗忘曲线等于编数据。
    同一道题的重做以最后一次为准（notice_score 是 upsert），所以答对之后自然就不在这里了。
    """
    uid = _clean(user_id)
    if not uid:
        return []
    try:
        con = _connect(db_path())
    except Exception as e:  # noqa: BLE001
        log.warning("profile 读取失败（%s）", type(e).__name__)
        return []
    try:
        rows = con.execute(
            "SELECT concepts FROM learner_mastery WHERE user_id = ? AND correct = 0 "
            "ORDER BY last_at DESC LIMIT ?", (uid, SCORE_MAX)).fetchall()
    except Exception as e:  # noqa: BLE001 - 老库缺表
        log.warning("profile 判分查询失败（%s）", type(e).__name__)
        return []
    finally:
        con.close()
    out: list[str] = []
    for (raw,) in rows:
        try:
            names = json.loads(raw or "[]")
        except (TypeError, ValueError):
            continue
        for n in names if isinstance(names, list) else []:
            name = str(n or "").strip()
            if name and name not in out:
                out.append(name)
        if len(out) >= limit:
            break
    return out[:limit]


def state(user_id: Any) -> dict:
    """这个人的学习状态：概念 / 章 / 判分记录，新的在前。"""
    empty = {"concepts": [], "chapters": [], "scores": []}
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
        scores = [{"question_id": r[0], "chapter": r[1], "concepts": json.loads(r[2] or "[]"),
                   "score": r[3], "correct": bool(r[4]), "method": r[5], "last_at": r[6]}
                  for r in con.execute(
                      "SELECT question_id, chapter, concepts, score, correct, method, last_at "
                      "FROM learner_mastery WHERE user_id = ? ORDER BY last_at DESC LIMIT ?",
                      (uid, SCORE_MAX))]
    except Exception as e:  # noqa: BLE001 - 老库缺表等
        log.warning("profile 查询失败（%s）", type(e).__name__)
        return empty
    finally:
        con.close()
    return {"concepts": concepts, "chapters": chapters, "scores": scores}


def notice_identity(user_id: Any, *, level: str = "", major: str = "", field: str = "", stage: str = "") -> None:
    """记学生自述的身份与阶段：学历、专业、学科门类、论文阶段。

    只写学生自己说过的（「我是编辑出版专业的本科生」「论文在选题阶段」），不做推断：
    没说过就留空，不会被「猜」成某个专业。同一个人多次自述时，非空字段覆盖、空字段保留旧值。
    """
    uid = _clean(user_id)
    if not uid:
        return
    vals = {"level": _clip(level, 12), "major": _clip(major, 24), "field": _clip(field, 24), "stage": _clip(stage, 12)}
    if not any(vals.values()):
        return
    try:
        with _lock:
            con = _connect(db_path())
            try:
                with con:
                    old = _read_identity(con, uid)
                    merged = {k: (vals[k] or old.get(k, "")) for k in vals}
                    con.execute(
                        "INSERT INTO learner_profile(user_id, level, major, field, stage, updated_at) VALUES (?,?,?,?,?,?) "
                        "ON CONFLICT(user_id) DO UPDATE SET level = excluded.level, major = excluded.major, "
                        "field = excluded.field, stage = excluded.stage, updated_at = excluded.updated_at",
                        (uid, merged["level"], merged["major"], merged["field"], merged["stage"], time.time()))
            finally:
                con.close()
    except Exception as e:  # noqa: BLE001 - 档案写失败不影响回答
        log.warning("profile 身份写入失败（%s），本条丢弃", type(e).__name__)


def _clip(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _read_identity(con: sqlite3.Connection, uid: str) -> dict:
    try:
        row = con.execute("SELECT level, major, field, stage FROM learner_profile WHERE user_id = ?",
                          (uid,)).fetchone()
    except Exception:  # noqa: BLE001 - 老库没有这张表
        return {}
    return dict(zip(("level", "major", "field", "stage"), row)) if row else {}


def identity(user_id: Any) -> dict:
    """学生自述的身份：{"level", "major", "field", "stage"}；没说过就是空字符串。"""
    empty = {"level": "", "major": "", "field": "", "stage": ""}
    uid = _clean(user_id)
    if not uid:
        return empty
    try:
        con = _connect(db_path())
    except Exception as e:  # noqa: BLE001
        log.warning("profile 读取失败（%s）", type(e).__name__)
        return empty
    try:
        return {**empty, **_read_identity(con, uid)}
    finally:
        con.close()


def summary(user_id: Any) -> str:
    """拼进提示词的关系层：说明这个人是谁（自述）、问过什么、关注哪章、哪几处还没掌握。

    只陈述已经发生的事（自述的专业、问过什么、哪道题没答对），不做任何推断（不说「你基础薄弱」）——
    推断没有依据，写进提示词会让模型顺着编。只取问过两次以上的概念（「反复问的才是主题」），
    问过一次的不算——课程管线一次会选出好几个概念，全列出来等于把词表塞进提示词。
    「还没掌握」只来自真实的判分记录；没有交过作业的学生这一段完全不出现。
    """
    st = state(user_id)
    repeated = [c["concept"] for c in st["concepts"] if c["hits"] >= 2]
    names = repeated[:SUMMARY_MAX_ITEMS]
    weak = [w for w in weaknesses(user_id) if w not in names][:WEAK_MAX_ITEMS]
    who = identity(user_id)
    if not names and not st["chapters"] and not weak and not any(who.values()):
        return ""
    parts = []
    line = _identity_line(who)
    if line:
        parts.append(line)
    if names:
        parts.append("之前问过：" + "、".join(names))
    if weak:
        parts.append("做过题但还没答对：" + "、".join(weak))
    if st["chapters"]:
        parts.append("关注过的章：" + "、".join(f"第{c}章" for c in st["chapters"][:2]))
    return ("【这个学生的情况】" + "；".join(parts)
            + "。回答时按他的专业与阶段举例选材，不要生硬复述、不要推断他的水平。")


def _identity_line(who: dict) -> str:
    """把自述身份拼成一句自然的话：本科生，编辑出版专业（新闻传播），选题阶段。"""
    level, major, field, stage = (who.get(k, "") for k in ("level", "major", "field", "stage"))
    head = f"{level}生" if level in ("本科", "硕士", "博士") else level
    if major and field and field not in major:
        major = f"{major}专业（{field}）"
    elif major and "专业" not in major:
        major = f"{major}专业"
    bits = [x for x in (head, major) if x]
    line = "，".join(bits) if len(bits) > 1 else "".join(bits)
    if stage:
        line = (line + "，") if line else ""
        line += f"{stage}阶段"
    return line


__all__ = ["notice_question", "notice_score", "notice_identity", "identity", "state", "summary",
           "weaknesses", "db_path", "reset", "SCHEMA", "CONCEPT_MAX", "CHAPTER_MAX", "SCORE_MAX"]
