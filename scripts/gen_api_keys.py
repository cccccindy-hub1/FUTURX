"""生成 API key（按人发放时一个班上千个）。

两种落盘方式：
  --keys-file  写到单独的密钥文件（推荐，千级时 .env 放不下/难维护），
               格式为每行一个 key，# 开头是注释；
                不给则追加到 .env 的 API_KEYS（先备份 .env）。

分配表（学号 -> key）总写到 logs/api_keys_<时间>.json 与 .csv，logs/ 已在 .gitignore。
单 key 共享时所有调用记录同一个 key_id；要区分用户就按人发 key。

用法：
    python scripts/gen_api_keys.py --count 5 --students 2023001,2023002,2023003,2023004,2023005
    python scripts/gen_api_keys.py --count 1000 --keys-file /etc/futurx/api_keys.txt
    python scripts/gen_api_keys.py --count 3 --prefix sk-futurx --dry-run
"""
from __future__ import annotations

import argparse
import csv
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


def read_keys_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 API key（写入密钥文件或 .env）")
    ap.add_argument("--count", type=int, required=True, help="生成数量")
    ap.add_argument("--prefix", default="sk-futurx", help="key 前缀（默认 sk-futurx）")
    ap.add_argument("--students", default="", help="学号列表，逗号分隔；与 --count 数量一致时逐个分配")
    ap.add_argument("--env", default=str(ENV), help=".env 路径（默认项目根）")
    ap.add_argument("--keys-file", default="", help="密钥文件路径；给了就写这里，不碰 .env")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写任何文件")
    a = ap.parse_args(argv)
    if a.count < 1:
        raise SystemExit("--count 至少为 1")

    to_file = bool(a.keys_file)
    if to_file:
        target = Path(a.keys_file).expanduser()
        old = read_keys_file(target)
    else:
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
    # 上千条时不要全部刷屏，只在终端看前 20 条，完整表在下面的文件里
    show = mapping if len(mapping) <= 50 else mapping[:20]
    print(f"{'学号':<12}{'key':<46}key_id")
    for stu, key in show:
        print(f"{stu:<12}{key:<46}{key_id(key)}")
    if len(mapping) > len(show):
        print(f"… 其余 {len(mapping) - len(show)} 条见分配表文件")

    if a.dry_run:
        print("\n[dry-run] 未写入任何文件")
        return 0

    if to_file:
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_name(f"{target.name}.bak.{int(time.time())}") if target.is_file() else None
        if backup:
            shutil.copy2(target, backup)
        header = (f"# API keys（{time.strftime('%Y-%m-%d %H:%M:%S')} 生成，共 {len(merged)} 条）\n"
                  "# 每行一个 key；# 开头为注释。改动后需重启服务。\n")
        target.write_text(header + "\n".join(merged) + "\n", encoding="utf-8")
        os.chmod(target, 0o600)          # 只有属主可读写
        print(f"\n已写入 {target}（共 {len(merged)} 条，权限 600）")
        if backup:
            print(f"原文件备份为 {backup.name}")
        print(f"记得在 .env 里设置 API_KEYS_FILE={target} 后 restart 才会生效。")
    else:
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
        if len(merged) > 200:
            print("提示：key 超过 200 条时这一行会很臃肿，建议改用 --keys-file 放到单独文件。")
    print("服务仍在用旧配置：确认后自行 systemctl restart futurx 才会生效。")
    if mapping:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = ROOT / "logs" / f"api_keys_{stamp}.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([{"student": s, "key": k, "key_id": key_id(k)} for s, k in mapping],
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            # 同时给一份 CSV：分发给学生时直接按行剪贴/导入表格更方便
            csv_path = out.with_suffix(".csv")
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["学号", "key", "key_id"])
                w.writerows(mapping)
            print(f"分配表已另存：{out} 与 {csv_path.name}（logs/ 已在 .gitignore）")
        except OSError as e:
            print(f"分配表保存失败（{type(e).__name__}），请自行抄录上面的表格")
    return 0


if __name__ == "__main__":
    sys.exit(main())
