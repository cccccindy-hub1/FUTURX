"""命令行入口。

用法（在项目根目录 FUTURX 下）：
  .venv/bin/python -m rag.cli build
  .venv/bin/python -m rag.cli ask "文献综述应该怎么写？"
  .venv/bin/python -m rag.cli serve             # 交互式 REPL（加载一次反复问）
  .venv/bin/python -m rag.cli serve --http      # FastAPI 服务（默认 127.0.0.1:8000）
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="nlp_ 学位论文指导知识库 RAG 管线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="解析 -> 分块 -> 向量化 -> 入库（全量重建）")

    ask = sub.add_parser("ask", help="问答：检索 +（可选）LLM 生成")
    ask.add_argument("question", nargs="+", help="问题文本")
    ask.add_argument("-k", "--top-k", type=int, default=None, help="返回片段数，默认取配置 TOP_K")

    serve = sub.add_parser("serve", help="常驻查询服务：模型与向量库加载一次，反复提问")
    serve.add_argument("--http", action="store_true", help="以 HTTP 服务方式运行（默认交互式 REPL）")
    serve.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址（默认 127.0.0.1）")
    serve.add_argument("--port", type=int, default=8000, help="HTTP 端口（默认 8000）")

    args = parser.parse_args()

    if args.cmd == "build":
        from .index import build

        build()
    elif args.cmd == "ask":
        from .query import answer

        q = " ".join(args.question)
        top_k = args.top_k if args.top_k else None
        ans, sources, docs = answer(q) if top_k is None else answer(q, top_k)

        if ans is None:
            print("（未设置 LLM_API_KEY，仅返回检索结果）\n")
            for i, (d, s) in enumerate(zip(docs, sources), 1):
                print(f"[来源{i}: {s['source']}]（{s['section']}，距离 {s['distance']:.3f}）")
                print(d)
                print("-" * 60)
            return

        print("=" * 60)
        print(ans)
        print("=" * 60)
        print("参考来源：")
        for s in sources:
            print(f"  - {s['source']}  [{s['section']}]  (距离 {s['distance']:.3f})")
    elif args.cmd == "serve":
        from .serve import run_http, run_repl

        if args.http:
            run_http(args.host, args.port)
        else:
            run_repl()


if __name__ == "__main__":
    main()
