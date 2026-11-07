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


def _lookup_by_student(student: str) -> list[str]:
    """从 logs/api_keys_*.json 里按学号找回 key（分配表是唯一的反查途径）。"""
    found = []
    for f in sorted((ROOT / "logs").glob("api_keys_*.json")):
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for r in rows:
            if str(r.get("student")) == student:
                found.append(r.get("key", ""))
    return [k for k in found if k]


def revoke(argv: list[str]) -> int:
    """撤销 key：从密钥文件（或 .env）里删掉匹配的行。

    支持三种定位方式：--key 明文、--student 学号（查分配表）、--key-id 指纹。
    删完不必重启：服务端按 mtime 热加载，默认 5 秒内生效。
    """
    ap = argparse.ArgumentParser(description="撤销 API key")
    ap.add_argument("--key", default="", help="要撤销的 key 明文")
    ap.add_argument("--student", default="", help="按学号撤销（查 logs/api_keys_*.json）")
    ap.add_argument("--key-id", default="", help="按 key_id（sha256 前 8 位）撤销")
    ap.add_argument("--keys-file", default="", help="密钥文件路径；不给则改 .env")
    ap.add_argument("--env", default=str(ENV), help=".env 路径（默认项目根）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    a = ap.parse_args(argv)

    targets: set[str] = set()
    if a.key:
        targets.add(a.key.strip())
    if a.student:
        got = _lookup_by_student(a.student)
        if not got:
            raise SystemExit(f"分配表里找不到学号 {a.student}，请直接用 --key")
        targets.update(got)
    if not targets and not a.key_id:
        raise SystemExit("需要 --key / --student / --key-id 之一")

    to_file = bool(a.keys_file)
    path = Path(a.keys_file).expanduser() if to_file else Path(a.env)
    if not path.is_file():
        raise SystemExit(f"{path} 不存在")
    old = read_keys_file(path) if to_file else existing_keys(path.read_text(encoding="utf-8"))

    if a.key_id:
        targets.update(k for k in old if key_id(k) == a.key_id.strip())
    if not targets:
        raise SystemExit("没有匹配到任何 key（文件里不存在？）")

    kept = [k for k in old if k not in targets]
    removed = len(old) - len(kept)
    print(f"待撤销 {len(targets)} 把：")
    for k in targets:
        print(f"  {k[:14]}…  key_id={key_id(k)}")
    print(f"撤销后剩余 {len(kept)} 把")

    if a.dry_run:
        print("[dry-run] 未写入")
        return 0
    if removed == 0:
        print("文件里没有这些 key（可能已撤销），未做改动。")
        return 0

    if to_file:
        backup = path.with_name(f"{path.name}.bak.{int(time.time())}")
        shutil.copy2(path, backup)
        header = f"# API keys（{time.strftime('%Y-%m-%d %H:%M:%S')} 撤销 {removed} 把，共 {len(kept)} 条）\n" \
                 "# 每行一个 key；# 开头为注释。改动后需重启服务。\n"
        path.write_text(header + "\n".join(kept) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
        print(f"已写入 {path}（备份 {backup.name}）")
    else:
        env = path
        backup = env.with_name(f".env.bak.{int(time.time())}")
        shutil.copy2(env, backup)
        text = env.read_text(encoding="utf-8")
        env.write_text(LINE.sub(lambda _m: "API_KEYS=" + ",".join(kept) if kept else "API_KEYS=",
                                text, count=1), encoding="utf-8")
        print(f"已写入 {env}（备份 {backup.name}）")

    print("服务端按 mtime 热加载：默认 5 秒内生效，无需重启。")
    print("注意：被撤销 key 创建的会话仍归属原 key_id，撤销后无人能访问（可 reassign 或删除）。")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "revoke":
        return revoke(argv[1:])
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
