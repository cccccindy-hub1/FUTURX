"""生成 API key 并追加到 .env 的 API_KEYS（先备份 .env）。

只提供工具，不改运行中的服务：写出后需自行决定是否启用与重启。
单 key 共享时所有调用记录同一个 key_id；要区分用户就按人发 key（本脚本打印 学号→key 分配表）。

用法：
    python scripts/gen_api_keys.py --count 5 --students 2023001,2023002,2023003,2023004,2023005
    python scripts/gen_api_keys.py --count 3 --prefix sk-futurx --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.usage_log import key_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
LINE = re.compile(r"^API_KEYS=(.*)$", re.MULTILINE)


def new_key(prefix: str) -> str:
    return f"{prefix}-{secrets.token_urlsafe(28)}"


def existing_keys(text: str) -> list[str]:
    m = LINE.search(text)
    if not m:
        return []
    return [k.strip() for k in m.group(1).split(",") if k.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 API key 并追加到 .env（先备份）")
    ap.add_argument("--count", type=int, required=True, help="生成数量")
    ap.add_argument("--prefix", default="sk-futurx", help="key 前缀（默认 sk-futurx）")
    ap.add_argument("--students", default="", help="学号列表，逗号分隔；与 --count 数量一致时逐个分配")
    ap.add_argument("--env", default=str(ENV), help=".env 路径（默认项目根）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写 .env")
    a = ap.parse_args(argv)
    if a.count < 1:
        raise SystemExit("--count 至少为 1")
    env = Path(a.env)
    if not env.is_file():
        raise SystemExit(f"{env} 不存在")

    text = env.read_text(encoding="utf-8")
    old = existing_keys(text)
    fresh = [new_key(a.prefix) for _ in range(a.count)]
    merged = old + fresh

    students = [s.strip() for s in a.students.split(",") if s.strip()]
    if students and len(students) != len(fresh):
        raise SystemExit(f"--students 给了 {len(students)} 个，与 --count {len(fresh)} 不一致")
    mapping = [(students[i] if students else f"#{len(old) + i + 1}", fresh[i]) for i in range(len(fresh))]

    print(f"已有 key：{len(old)} 条；本次新增：{len(fresh)} 条")
    print(f"{'学号':<12}{'key':<46}key_id")
    for stu, key in mapping:
        print(f"{stu:<12}{key:<46}{key_id(key)}")

    if a.dry_run:
        print("\n[dry-run] 未写入 .env")
        return 0

    backup = env.with_name(f".env.bak.{int(time.time())}")
    shutil.copy2(env, backup)
    os_mode = env.stat().st_mode
    line = "API_KEYS=" + ",".join(merged)
    if LINE.search(text):
        text = LINE.sub(lambda _m: line, text, count=1)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
    env.write_text(text, encoding="utf-8")
    env.chmod(os_mode)
    print(f"\n已写入 {env}（原文件备份为 {backup.name}）")
    print("服务仍在用旧配置：确认后自行 systemctl restart futurx 才会生效。")
    if mapping:
        out = ROOT / "logs" / f"api_keys_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([{"student": s, "key": k, "key_id": key_id(k)} for s, k in mapping],
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"分配表已另存：{out}（logs/ 已在 .gitignore）")
        except OSError as e:
            print(f"分配表保存失败（{type(e).__name__}），请自行抄录上面的表格")
    return 0


if __name__ == "__main__":
    sys.exit(main())
