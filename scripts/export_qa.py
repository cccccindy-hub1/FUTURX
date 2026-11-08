"""导出问答永久记录（SQLite → CSV，UTF-8 带 BOM，Excel 直接打开不乱码）。只读数据库，不改任何数据。

用法：
  python scripts/export_qa.py --out qa.csv [--db 路径] [--since 2026-11-01] [--until 2026-11-30] [--user-id u-xxxx] [--user-name 张三] [--with-prompts]
  --since / --until 含首尾两天（按 created_at 的日期）；不给就是全部。
  默认不导出 system prompt（只导出它们的 hash 引用）；加 --with-prompts 才多出一列“system_prompts”，内容为 {步骤: 全文}。
  不含 API key（库里本来就只有 key_id 指纹）。--out 已存在时拒绝覆盖，除非加 --force。
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = ["id", "request_id", "user_id", "user_name", "key_id", "session_id", "created_at", "origin", "input_prompt",
           "history_json", "standalone_question", "route", "routes", "system_prompt_hashes_json", "output", "followups_json",
           "sources_json", "status", "error", "llm_calls_json", "latency_ms"]


def _columns(con: sqlite3.Connection) -> list[str]:
    """按库里实际存在的列取数：老库还没跑过迁移（没有 user_name）时该列用空串占位，导出不会因此报错。"""
    have = {r[1] for r in con.execute("PRAGMA table_info(qa_log)")}
    return [c for c in COLUMNS if c in have]


def query(con: sqlite3.Connection, since: str | None, until: str | None, user_id: str | None,
          user_name: str | None = None) -> list[sqlite3.Row]:
    sql, args = "SELECT " + ",".join(_columns(con)) + " FROM qa_log WHERE 1=1", []
    if since:
        sql += " AND created_at >= ?"
        args.append(since)
    if until:
        sql += " AND created_at < ?"
        args.append(_next_day(until))
    if user_id:
        sql += " AND user_id = ?"
        args.append(user_id)
    if user_name:
        sql += " AND user_name = ?"
        args.append(user_name)
    return con.execute(sql + " ORDER BY created_at, id", args).fetchall()


def _next_day(day: str) -> str:
    import datetime as dt
    return (dt.date.fromisoformat(day[:10]) + dt.timedelta(days=1)).isoformat()


def export(db: Path, out: Path, *, since=None, until=None, user_id=None, user_name=None, with_prompts=False, force=False) -> int:
    if not db.is_file():
        raise SystemExit(f"数据库不存在：{db}")
    if out.exists() and not force:
        raise SystemExit(f"输出文件已存在，拒绝覆盖：{out}（加 --force 才覆盖）")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = query(con, since, until, user_id, user_name)
        texts = {r["hash"]: (r["step"], r["text"]) for r in con.execute("SELECT hash, step, text FROM prompts")} if with_prompts else {}
    finally:
        con.close()
    cols = COLUMNS + (["system_prompts"] if with_prompts else [])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            have = set(r.keys())
            line = [r[c] if c in have else "" for c in COLUMNS]
            if with_prompts:
                hashes = json.loads(r["system_prompt_hashes_json"] or "{}")
                line.append(json.dumps({step: texts.get(h, ("", ""))[1] for step, h in hashes.items()}, ensure_ascii=False))
            w.writerow(line)
    return len(rows)


def main(argv=None) -> int:
    from rag import qa_store
    ap = argparse.ArgumentParser(description="导出问答永久记录为 CSV")
    ap.add_argument("--db", default=None, help="SQLite 路径，默认 RAG_QA_DB 或 data/qa_log.sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--since"), ap.add_argument("--until"), ap.add_argument("--user-id")
    ap.add_argument("--user-name", help="按提问者姓名筛选（精确匹配）")
    ap.add_argument("--with-prompts", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    n = export(Path(a.db) if a.db else qa_store.db_path(), Path(a.out), since=a.since, until=a.until, user_id=a.user_id,
               user_name=a.user_name, with_prompts=a.with_prompts, force=a.force)
    print(f"已导出 {n} 条 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
