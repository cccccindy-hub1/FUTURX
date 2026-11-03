"""常驻查询服务：模型与向量库只加载一次，反复提问，消除每次 ~18s 的重载开销。

用法（在项目根目录 FUTURX 下）：
  python -m rag.cli serve             # 交互式 REPL
  python -m rag.cli serve --http      # FastAPI HTTP 服务（默认 127.0.0.1:8000）
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from .config import MAX_CONCURRENT_REQUESTS, TOP_K
from .query import RAGService


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


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


def run_repl(**service_kwargs) -> None:
    svc = RAGService(**service_kwargs)
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


def load_course_backend():
    """启动时加载一次课程库（与 rag.course.api.build_app 一致）：返回 (CourseContext, LLM 客户端)。

    课程数据缺失、解析失败或 COURSE_LLM_* 未配置时返回 (None, None)，统一入口的 course 路由降级为
    unavailable，不影响论文 RAG 启动。日志只记异常类型名，避免泄露路径细节或 key。
    """
    import logging

    log = logging.getLogger(__name__)
    from .course.llm import LLMNotConfigured, OpenAICompatClient

    try:
        llm = OpenAICompatClient()
    except LLMNotConfigured:
        log.warning("课程库未启用：COURSE_LLM_* 未配置，/auto/ask 的 course 路由将返回 unavailable")
        return None, None
    try:
        from .course.pipeline import CourseContext

        return CourseContext.load(), llm
    except Exception as e:  # noqa: BLE001 - 课程数据问题不能拖垮论文 RAG
        log.warning("课程库未启用：加载失败（%s），/auto/ask 的 course 路由将返回 unavailable", type(e).__name__)
        return None, None


def load_exam_backend():
    """启动时加载一次试卷库（解析题库 docx + 学生知识集合，约 2 秒，不调用 LLM）。

    文件缺失或解析失败时返回 None，统一入口的 exam 路由降级为 unavailable、课程回答不附基础题。
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        from .exam import config as exam_config
        from .exam.store import ExamBank

        if not exam_config.EXAM_TEACHER_DOC.exists():
            log.warning("试卷库未启用：EXAM_TEACHER_DOC 未配置或文件不存在，/auto/ask 的 exam 路由将返回 unavailable")
            return None
        return ExamBank.build()
    except Exception as e:  # noqa: BLE001 - 试卷库问题不能拖垮其他知识库
        log.warning("试卷库未启用：加载失败（%s），/auto/ask 的 exam 路由将返回 unavailable", type(e).__name__)
        return None


def load_doc_libraries():
    """启动时直接读原始 docx 建内存索引（不用向量库、不用 bge-m3；解析结果按 docx 的 mtime+size 缓存成 JSON）。

    返回 (case_index, quote_index, policy_index)。docx 不存在 / 解析失败时对应项为 None，统一入口的 case / quote 路由降级为 unavailable，
    不影响其它路由启动。日志只记异常类型名。
    """
    import logging

    log = logging.getLogger(__name__)
    out = []
    for name, loader, label in (("case", "rag.cases.retrieval:CaseIndex", "判例库"), ("quote", "rag.quotes.retrieval:QuoteIndex", "名言库"),
                                ("policy", "rag.policy.retrieval:PolicyIndex", "政策库")):
        mod, cls = loader.split(":")
        try:
            import importlib
            idx = getattr(importlib.import_module(mod), cls).load()
            if idx is None:
                log.warning("%s未启用：原始 docx 不存在，%s 路由将返回 unavailable", label, name)
            else:
                log.info("%s已加载（%s，%.1fs）", label, idx.info.get("source"), idx.info.get("seconds", 0))
            out.append(idx)
        except Exception as e:  # noqa: BLE001 - 单个库的问题不能拖垮其它库
            log.warning("%s未启用：加载失败（%s），%s 路由将返回 unavailable", label, type(e).__name__, name)
            out.append(None)
    return tuple(out)


def build_app(**service_kwargs):
    """构造 FastAPI 应用（模型与向量库在启动时加载一次）。"""
    from pathlib import Path

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from .api import router as openai_router
    from .concurrency import RequestConcurrencyMiddleware

    svc = RAGService(**service_kwargs)
    app = FastAPI(title="FUTURX RAG 查询服务")
    app.state.service = svc
    app.add_middleware(RequestConcurrencyMiddleware, max_requests=MAX_CONCURRENT_REQUESTS)
    web_dir = Path(__file__).resolve().parent / "web"

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # 同源网页直接调用；浏览器端无法安全保存密钥，因此 /ask 不鉴权（仅 /v1/* 校验）。
    @app.post("/ask")
    def ask(req: AskRequest):
        ans, sources, docs = svc.answer(req.question, req.top_k)
        return {"answer": ans, "sources": sources, "contexts": docs}

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        """SSE 流式问答：先推 sources，再逐段推 delta，避免前端长时间空等。"""
        try:
            docs, sources, deltas = svc.open_stream(req.question, req.top_k)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        def gen():
            yield _sse({"type": "sources", "sources": sources, "contexts": docs})
            try:
                for text in deltas:
                    yield _sse({"type": "delta", "text": text})
            except Exception as e:  # 流已开始，只能以事件告知
                yield _sse({"type": "error", "message": f"生成中断：{e}"})
                return
            yield _sse({"type": "done"})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.include_router(openai_router)

    # 统一入口 POST /auto/ask：先路由再分发；路由客户端未配置时只用规则。课程库启动时加载一次，缺失则降级。
    from .router import build_router_client
    from .unified import UnifiedService, router as unified_router

    course_ctx, course_llm = load_course_backend()
    exam_bank = load_exam_backend()
    case_index, quote_index, policy_index = load_doc_libraries()
    app.state.unified = UnifiedService(rag=svc, router_client=build_router_client(),
                                       course_ctx=course_ctx, course_llm=course_llm, exam_bank=exam_bank,
                                       case_index=case_index, quote_index=quote_index, policy_index=policy_index)
    app.include_router(unified_router)
    if exam_bank is not None:
        from .exam.api import router as exam_router

        app.state.exam = exam_bank
        app.include_router(exam_router)

    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(web_dir / "index.html")

    return app


def run_http(host: str, port: int, **service_kwargs) -> None:
    import uvicorn

    app = build_app(**service_kwargs)
    print(
        f"服务启动：http://{host}:{port}\n"
        f"  Web 前端   GET  /\n"
        f"  探活       GET  /health\n"
        f"  原生问答   POST /ask\n"
        f"  统一入口   POST /auto/ask（LLM 路由 + 自适应）\n"
        f"  OpenAI 兼容 POST /v1/chat/completions   GET /v1/models"
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
