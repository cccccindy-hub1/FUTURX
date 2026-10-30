"""《数据要素》课程问答 HTTP 接口（独立服务，与 rag.serve 的 Chroma 检索服务分开部署）。

用法（在项目根目录 FUTURX 下）：
  python -m rag.course.cli serve [--host 127.0.0.1] [--port 8100]

- GET  /health        探活：教材版本、子块数、LLM 是否已配置
- POST /course/ask    四步问答（选词→召回→筛选→作答）；dry_run=true 时不调用 LLM，只估算召回（需 select）

鉴权沿用 rag.api.require_api_key：配置 API_KEYS 后要求 Authorization: Bearer <key>；未配置则不校验。
每个问题约 5 次付费 LLM 调用，对外开放前务必配置 API_KEYS。
LLM 调用失败返回 502（消息已由 OpenAICompatClient 脱敏，不含 key）；LLM 未配置时非 dry-run 请求返回 503。
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..api import require_api_key
from . import config
from .llm import LLMClient, LLMError, LLMNotConfigured
from .pipeline import CourseContext, ask, result_to_json
from .recall import DEFAULT_STRATEGY

router = APIRouter()

_FROM_ENV = object()


class SelectItem(BaseModel):
    concept_id: str
    role: Literal["question_concept", "related_concept"] = "question_concept"


class CourseAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.COURSE_QUESTION_MAX_CHARS)
    active_concepts: list[str] = Field(default_factory=list, max_length=50)
    strategy: Literal["A", "B"] = DEFAULT_STRATEGY
    dry_run: bool = False
    select: list[SelectItem] = Field(default_factory=list, max_length=50)   # 仅 dry_run 使用
    include_trace: bool = True


def _error(status: int, message: str, type_: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"message": message, "type": type_}})


@router.post("/course/ask", dependencies=[Depends(require_api_key)])
def course_ask(req: CourseAskRequest, request: Request):
    ctx: CourseContext = request.app.state.ctx
    llm: LLMClient | None = request.app.state.llm
    question = req.question.strip()
    if not question:
        raise _error(400, "question 不能为空", "invalid_request_error")
    if not req.dry_run and llm is None:
        raise _error(503, "未配置 COURSE_LLM_API_BASE / COURSE_LLM_API_KEY，只能 dry_run", "service_unavailable")
    try:
        result = ask(question, ctx=ctx, llm=None if req.dry_run else llm, active_concepts=req.active_concepts,
                     strategy=req.strategy, dry_run=req.dry_run,
                     select=[(s.concept_id, s.role) for s in req.select])
    except LLMError as e:
        raise _error(502, str(e), "upstream_error") from None
    out = result_to_json(result)
    if not req.include_trace:
        out.pop("trace", None)
    return out


def build_app(ctx: CourseContext | None = None, llm=_FROM_ENV) -> FastAPI:
    """构造应用：教材与本体在启动时加载一次。llm 不传则按环境变量构造 OpenAICompatClient（未配置时为 None）。"""
    if ctx is None:
        ctx = CourseContext.load()
    if llm is _FROM_ENV:
        from .llm import OpenAICompatClient
        try:
            llm = OpenAICompatClient()
        except LLMNotConfigured:
            llm = None
    app = FastAPI(title="FUTURX《数据要素》课程问答服务")
    app.state.ctx, app.state.llm = ctx, llm

    @app.get("/health")
    def health():
        return {"status": "ok", "book_version": ctx.chunkset.book_version, "chunks": len(ctx.chunkset.chunks),
                "llm_configured": llm is not None}

    app.include_router(router)
    return app


def run_http(host: str, port: int, **ctx_kwargs) -> None:
    import uvicorn

    app = build_app(CourseContext.load(**ctx_kwargs))
    print(
        f"课程问答服务启动：http://{host}:{port}\n"
        f"  探活      GET  /health\n"
        f"  课程问答  POST /course/ask\n"
        f"  LLM 已配置：{app.state.llm is not None}"
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
