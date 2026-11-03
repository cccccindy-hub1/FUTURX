"""政策库命令行（政策库不建向量库，服务启动时直接读 docx；这里只用来看切块结果）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.policy.cli --dry-run [--doc /path/数据政策汇编.docx]
      解析并切块，只打印统计：块总数、条款块数、各地区块数分布……不写任何文件。
  python -m rag.policy.cli export --out DIR [--doc ...]
      另把 files.jsonl / chunks.jsonl / stats.json 写到 DIR，供人工审阅（DIR 必须是新目录或空目录）。
docx 路径默认取 POLICY_DOC（.env 可覆盖），缺省 rag/policy_data/数据政策汇编.docx。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import POLICY_DOC
from .corpus import build_corpus, corpus_stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag.policy.cli", description="政策库（数据政策汇编）切块检查")
    ap.add_argument("cmd", nargs="?", choices=["export"], help="export：另写出 JSONL 供审阅")
    ap.add_argument("--doc", default=None, help="汇编 docx 路径，默认取 POLICY_DOC")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    ap.add_argument("--out", default=None, help="export 的输出目录（必须不存在或为空）")
    args = ap.parse_args(argv)
    doc = Path(args.doc) if args.doc else POLICY_DOC
    if not doc.is_file():
        print(f"政策汇编文件不存在：{doc}（用 --doc 或 POLICY_DOC 指定）", file=sys.stderr)
        return 2
    corpus = build_corpus(doc)
    stats = corpus_stats(corpus)
    print(f"块总数：{stats['chunks']}　条款块：{stats['article_chunks']}　文件数：{stats['files']}")
    print("各地区块数：" + "　".join(f"{k} {v}" for k, v in sorted(stats["chunks_by_region"].items(), key=lambda kv: -kv[1])))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run or args.cmd != "export":
        return 0
    if not args.out:
        print("export 需要 --out DIR", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"输出目录非空，拒绝写入：{out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("files.jsonl", corpus["files"]), ("chunks.jsonl", corpus["chunks"])):
        (out / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
