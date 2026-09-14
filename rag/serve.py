"""常驻查询服务：模型与向量库只加载一次，反复提问，消除每次 ~18s 的重载开销。

用法（在项目根目录 FUTURX 下）：
  .venv/bin/python -m rag.cli serve             # 交互式 REPL
  .venv/bin/python -m rag.cli serve --http      # FastAPI HTTP 服务（默认 127.0.0.1:8000）
"""
from __future__ import annotations

from pydantic import BaseModel

from .config import TOP_K
from .query import RAGService


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


def _print_answer(ans, sources, docs) -> None:
    if ans is None:
        print("（未设置 LLM_API_KEY，仅返回检索结果）")
        for i, (d, s) in enumerate(zip(docs, sources), 1):
            print(f"\n[来源{i}: {s['source']}]（{s['section']}，距离 {s['distance']:.3f}）")
            print(d)
        return
    print("=" * 60)
    print(ans)
    print("=" * 60)
    print("参考来源：")
    for s in sources:
        print(f"  - {s['source']}  [{s['section']}]  (距离 {s['distance']:.3f})")


def run_repl() -> None:
    svc = RAGService()
    print("模型与向量库已加载。直接输入问题回车（输入 exit / quit 退出）。")
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        try:
            ans, sources, docs = svc.answer(q)
        except Exception as e:  # 单条查询失败不影响会话
            print(f"[错误] {e}")
            continue
        _print_answer(ans, sources, docs)


def run_http(host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI

    svc = RAGService()
    app = FastAPI(title="FUTURX RAG 查询服务")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest):
        ans, sources, docs = svc.answer(req.question, req.top_k)
        return {"answer": ans, "sources": sources, "contexts": docs}

    print(f"服务启动：http://{host}:{port}  （GET /health 探活，POST /ask 提问）")
    uvicorn.run(app, host=host, port=port, log_level="warning")
