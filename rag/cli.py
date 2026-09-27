"""命令行入口。

用法（在项目根目录 FUTURX 下）：
  .venv/bin/python -m rag.cli build
  .venv/bin/python -m rag.cli ask "文献综述应该怎么写？"
  .venv/bin/python -m rag.cli serve             # 交互式 REPL（加载一次反复问）
  .venv/bin/python -m rag.cli serve --http      # FastAPI 服务（默认 127.0.0.1:8000）
"""
from __future__ import annotations

import argparse
import json


def _add_retrieval_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-k", "--top-k", type=int, default=None, help="返回片段数，默认取配置 TOP_K"
    )
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "hybrid"], default=None,
        help="检索模式：dense=纯向量，bm25=纯关键词，hybrid=RRF 融合（默认取配置 RETRIEVAL_MODE）",
    )
    parser.add_argument(
        "--rerank", dest="rerank", action="store_true", default=None, help="启用重排序"
    )
    parser.add_argument(
        "--no-rerank", dest="rerank", action="store_false", help="禁用重排序"
    )
    parser.add_argument(
        "--fetch-k", type=int, default=None, help="融合前各路召回的候选数（默认取配置 FETCH_K）"
    )
    parser.add_argument(
        "--where", action="append", metavar="KEY=VALUE",
        help="元数据过滤，可重复，如 --where category=006/研究方法/扎根理论",
    )
    parser.add_argument(
        "--rewrite", dest="rewrite", action="store_true", default=None,
        help="用 LLM 改写查询后再检索（需配置 LLM_API_KEY）",
    )
    parser.add_argument(
        "--hyde", dest="use_hyde", action="store_true", default=None,
        help="用 HyDE（假设答案）检索（需配置 LLM_API_KEY）",
    )
    parser.add_argument(
        "--expand", dest="expand_context", action="store_true", default=None,
        help="父子分块：命中子块后展开为父块上下文（库需以 PARENT_CHILD_ENABLED 建）",
    )


def _service_kwargs(args) -> dict:
    from .config import (
        CONTEXT_EXPAND_ENABLED,
        FETCH_K,
        HYDE_ENABLED,
        QUERY_REWRITE_ENABLED,
        RERANK_ENABLED,
        RETRIEVAL_MODE,
    )
    from .retrieval import parse_where

    def pick(flag, default):
        return flag if flag is not None else default

    return {
        "mode": args.mode or RETRIEVAL_MODE,
        "rerank": pick(getattr(args, "rerank", None), RERANK_ENABLED),
        "where": parse_where(getattr(args, "where", None)),
        "fetch_k": args.fetch_k or FETCH_K,
        "rewrite": pick(getattr(args, "rewrite", None), QUERY_REWRITE_ENABLED),
        "use_hyde": pick(getattr(args, "use_hyde", None), HYDE_ENABLED),
        "expand_context": pick(getattr(args, "expand_context", None), CONTEXT_EXPAND_ENABLED),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="nlp_ 学位论文指导知识库 RAG 管线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="解析 -> 分块 -> 向量化 -> 入库（全量重建）")

    sub.add_parser("eval", help="检索质量评估：BM25 vs BGE-m3，Recall@k/MRR@k/nDCG@k")

    ask = sub.add_parser("ask", help="问答：检索 +（可选）LLM 生成")
    ask.add_argument("question", nargs="+", help="问题文本")
    ask.add_argument("--json", action="store_true", help="以 JSON 输出 {answer,sources,docs}")
    _add_retrieval_flags(ask)

    serve = sub.add_parser("serve", help="常驻查询服务：模型与向量库加载一次，反复提问")
    serve.add_argument("--http", action="store_true", help="以 HTTP 服务方式运行（默认交互式 REPL）")
    serve.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址（默认 127.0.0.1）")
    serve.add_argument("--port", type=int, default=8000, help="HTTP 端口（默认 8000）")
    _add_retrieval_flags(serve)

    args = parser.parse_args()

    if args.cmd == "build":
        from .index import build

        build()
    elif args.cmd == "ask":
        from .query import RAGService

        q = " ".join(args.question)
        top_k = args.top_k or None
        svc = RAGService(**_service_kwargs(args))
        ans, sources, docs = svc.answer(q) if top_k is None else svc.answer(q, top_k)

        if args.json:
            print(json.dumps(
                {"answer": ans, "sources": sources, "contexts": docs},
                ensure_ascii=False, indent=2,
            ))
            return

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
    elif args.cmd == "eval":
        from .eval import run

        run()
    elif args.cmd == "serve":
        from .serve import run_http, run_repl

        kwargs = _service_kwargs(args)
        if args.http:
            run_http(args.host, args.port, **kwargs)
        else:
            run_repl(**kwargs)


if __name__ == "__main__":
    main()
