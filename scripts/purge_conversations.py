"""清理过期会话（rag/conversation.py）。

默认按 CONVERSATION_RETENTION_DAYS（365）删除超过保留期的会话，含已被软删的。
建议用 cron 每天跑一次：
    0 4 * * *  cd /home/FUTURX && /root/miniconda3/envs/futurx/bin/python scripts/purge_conversations.py

用法：
    python scripts/purge_conversations.py            # 按配置的保留期清理
    python scripts/purge_conversations.py --days 90  # 临时指定
    python scripts/purge_conversations.py --dry-run  # 只统计条数
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import conversation as store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="清理过期会话")
    ap.add_argument("--days", type=int, default=None, help="保留天数（默认取 CONVERSATION_RETENTION_DAYS）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    a = ap.parse_args(argv)

    days = store.retention_days() if a.days is None else a.days
    print(f"会话库：{store.db_path()}")
    print(f"保留期：{days or '永久'} 天")
    if days <= 0:
        print("保留期为 0（永久），不执行清理。")
        return 0
    if a.dry_run:
        rows = store.list_conversations(limit=100, archived=None)["items"]
        print(f"（dry-run）当前未删除的会话：{len(rows)} 条，未做任何改动。")
        return 0
    n = store.purge_expired(days=days)
    print(f"已删除 {n} 条过期会话。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
