"""OpenAI 兼容对外接口：把 AI 李老师当成一个「模型」暴露出去。

客户端只需把 base_url 指向本服务、model 填 API_MODEL_NAME，即可用任何
OpenAI SDK / 智能体平台直接调用检索增强后的论文指导能力。

- GET  /v1/models            列出可用模型
- POST /v1/chat/completions  对话补全（支持 stream=true 的 SSE 流式）
- POST /ask                  原生接口（保留原有单轮问答形状）

鉴权：配置 API_KEYS 后要求 Authorization: Bearer <key>；未配置则不校验。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import API_KEYS, API_MODEL_NAME, LLM_MODEL, TOP_K

router = APIRouter()


# ---------------------------------------------------------------- 鉴权

def require_api_key(request: Request) -> None:
    """配置了 API_KEYS 才校验，未配置时直接放行（便于本机调试）。"""
    if not API_KEYS:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


def _service(request: Request):
    svc = getattr(request.app.state, "service", None)
    if svc is None:  # pragma: no cover - 正常启动不会走到
        raise HTTPException(status_code=503, detail="服务未就绪")
    return svc


def key_fingerprint(request: Request) -> str | None:
    """请求所带 API key 的指纹（sha256 前 8 位），供调用日志区分调用方；无 key 返回 None。"""
    from .usage_log import key_id_from_header

    return key_id_from_header(request.headers.get("authorization"))


# ---------------------------------------------------------------- 请求模型

class ChatMessage(BaseModel):
    role: str
    # OpenAI 允许 content 为字符串或分段数组，这里都接住，只取文本部分
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model: str = API_MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    # 非标准扩展：允许客户端顺手指定检索片段数
    top_k: int | None = None
    user: str | None = None          # OpenAI 标准字段：终端用户标识（记入永久问答记录，限长 64、只含 [A-Za-z0-9_-]）


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in (None, "text"):
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return ""


def _last_user_question(messages: list[ChatMessage]) -> str:
    """取最后一条 user 消息作为检索问题（当前为单轮问答，不拼接历史）。"""
    for m in reversed(messages):
        if m.role == "user":
            q = _text_of(m.content).strip()
            if q:
                return q
    raise HTTPException(
        status_code=400,
        detail={"error": {"message": "messages 中缺少非空的 user 消息", "type": "invalid_request_error"}},
    )


def _chunk(cid: str, created: int, model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _qa_meta(req: "ChatCompletionRequest", request: Request) -> dict:
    """永久问答记录用：user（标准字段）/ X-User-Id、X-Session-Id，以及 messages 里最近几轮历史（不含最后那条 user 问题）。"""
    try:
        from .qa_store import normalize_history, request_meta

        msgs = [{"role": m.role, "content": _text_of(m.content)} for m in req.messages]
        last = max((i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"].strip()), default=-1)
        return request_meta(request.headers, body_user=req.user, history=normalize_history(msgs[:last] if last >= 0 else []))
    except Exception:  # noqa: BLE001
        return {}


def _log(question: str, request: Request, cid: str, model: str, answer: str, sources: list,
         t0: float, *, status: str = "ok", error: str | None = None,
         first_at: float | None = None, meta: dict | None = None) -> None:
    """OpenAI 兼容层的轻量调用日志（route 固定 thesis：本接口只走论文 RAG）。日志失败不影响响应。"""
    try:
        from .usage_log import simple_record

        simple_record(question, origin="http", key_id_value=key_fingerprint(request),
                      route="thesis", status=status, answer=answer, sources=sources,
                      started=t0, error=error, request_id=cid,
                      stages={"answer": round((time.time() - t0) * 1000, 1)},
                      time_to_first_delta_ms=None if first_at is None else round((first_at - t0) * 1000, 1),
                      qa_meta=meta)
    except Exception:  # noqa: BLE001 - 日志失败不影响响应
        pass


# ---------------------------------------------------------------- 路由

@router.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": API_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "futurx",
                # 便于客户端发现后端实际使用的生成模型
                "upstream_model": LLM_MODEL,
            }
        ],
    }


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest, request: Request):
    svc = _service(request)
    question = _last_user_question(req.messages)
    top_k = req.top_k or TOP_K
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model = req.model or API_MODEL_NAME
    t0 = time.time()
    meta = _qa_meta(req, request)

    if req.stream:
        try:
            _docs, sources, deltas = svc.open_stream(question, top_k)
        except RuntimeError as e:
            _log(question, request, cid, model, "", [], t0, status="unavailable", error="RuntimeError", meta=meta)
            raise HTTPException(status_code=503, detail={"error": {"message": str(e), "type": "service_unavailable"}})

        def event_stream() -> Iterator[bytes]:
            parts: list[str] = []
            first_at: float | None = None
            # 首帧只带 role，符合 OpenAI 流式约定
            yield b"data: " + json.dumps(
                _chunk(cid, created, model, {"role": "assistant", "content": ""}), ensure_ascii=False
            ).encode() + b"\n\n"
            try:
                for text in deltas:
                    if first_at is None:
                        first_at = time.time()
                    parts.append(text)
                    yield b"data: " + json.dumps(
                        _chunk(cid, created, model, {"content": text}), ensure_ascii=False
                    ).encode() + b"\n\n"
            except Exception as e:  # 流已开始，只能以错误帧告知
                yield b"data: " + json.dumps(
                    {"error": {"message": f"生成中断：{e}", "type": "upstream_error"}}, ensure_ascii=False
                ).encode() + b"\n\n"
                _log(question, request, cid, model, "".join(parts), sources, t0,
                     status="error", error=type(e).__name__, first_at=first_at, meta=meta)
                return
            # 末帧附带来源，标准客户端会忽略这个额外字段
            final = _chunk(cid, created, model, {}, finish="stop")
            final["sources"] = sources
            yield b"data: " + json.dumps(final, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
            _log(question, request, cid, model, "".join(parts), sources, t0, first_at=first_at, meta=meta)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    ans, sources, _docs = svc.answer(question, top_k)
    if ans is None:
        _log(question, request, cid, model, "", sources, t0, status="unavailable", meta=meta)
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "未配置 LLM_API_KEY，服务只能检索、不能生成", "type": "service_unavailable"}},
        )
    _log(question, request, cid, model, ans, sources, t0, meta=meta)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ans},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # 非标准扩展：检索来源，客户端可选择性展示
        "sources": sources,
    }
