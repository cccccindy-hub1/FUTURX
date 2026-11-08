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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import (API_KEYS, API_MODEL_NAME, CHAT_HISTORY_MAX_CHARS, CHAT_HISTORY_MAX_TURNS,
                     CONTEXTUALIZE_QUERY_ENABLED, CONTEXTUALIZE_TIMEOUT, LLM_MODEL, REWRITE_MODEL, TOP_K)

router = APIRouter()

_MISSING = object()          # 区分「没有这个属性」与「属性为 None」


# ---------------------------------------------------------------- 鉴权

def require_api_key(request: Request) -> None:
    """配置了 API_KEYS 才校验，未配置时直接放行（便于本机调试）。"""
    if not API_KEYS:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    # API_KEYS 是 frozenset：按人发 key（千级）时每个请求 O(1) 查找，不做线性扫描。
    if scheme.lower() != "bearer" or not token or token not in API_KEYS:
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
    # 非标准扩展：会话 id。传了就把本轮追加到该会话，并从库里取之前的历史；
    # 不传则每轮独立（无状态调用）；响应体与 X-Conversation-Id 头都会带回 id。
    conversation_id: str | None = Field(default=None, max_length=80)
    # 非标准扩展：true = 不落库（临时试探性提问）
    store: bool = True
    user: str | None = None          # OpenAI 标准字段：终端用户标识（记入永久问答记录，限长 64、只含 [A-Za-z0-9_-]）
    # 非标准扩展：调用方的人员标识与姓名。uuid 优先于 user / X-User-Id 记进 user_id 列；
    # user_name 单独存一列，中文可用（限长 32 字）。两者只用于问答记录，超长 / 含非法字符的在落库前清洗，不影响本次回答。
    uuid: str | None = None
    user_name: str | None = None


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
    """取最后一条 user 消息作为本轮问题。"""
    for m in reversed(messages):
        if m.role == "user":
            q = _text_of(m.content).strip()
            if q:
                return q
    raise HTTPException(
        status_code=400,
        detail={"error": {"message": "messages 中缺少非空的 user 消息", "type": "invalid_request_error"}},
    )


def _split_history(messages: list[ChatMessage]) -> tuple[str, list[tuple[str, str]]]:
    """拆出本轮问题与客户端发来的历史。

    返回 (本轮问题, [(role, content)])。历史只保留 user / assistant 两种角色、跳过空内容，
    系统消息按约定忽略（服务端用自己的提示词）。
    """
    question = _last_user_question(messages)
    # 找到最后一条非空 user 消息的位置，它之前的才算历史
    last = max(i for i, m in enumerate(messages)
               if m.role == "user" and _text_of(m.content).strip())
    pairs: list[tuple[str, str]] = []
    for m in messages[:last]:
        if m.role not in ("user", "assistant"):
            continue
        text = _text_of(m.content).strip()
        if text:
            pairs.append((m.role, text))
    return question, pairs


def _server_history(conversation_id: str | None) -> list[tuple[str, str]]:
    """从会话库取历史（已按轮数/字数截断）；读库失败不影响本轮回答。"""
    if not conversation_id or not CHAT_HISTORY_MAX_TURNS:
        return []
    try:
        from . import conversation as store

        return store.history(conversation_id, max_turns=CHAT_HISTORY_MAX_TURNS,
                             max_chars=CHAT_HISTORY_MAX_CHARS or None)
    except Exception:  # noqa: BLE001 - 会话库故障时退化为无历史
        return []


def _merge_history(server: list[tuple[str, str]], client: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """服务端历史与客户端 messages 合并，避免同一轮重复。

    规则：客户端发了多轮就以客户端为准（OpenAI 兼容平台的常见做法，历史由调用方掌握）；
    只发了本轮一句时用服务端历史。两者都只保留最近 CHAT_HISTORY_MAX_TURNS 轮，
    并按字数上限从最近往前截。
    """
    pairs = client if len(client) > 1 else server
    if CHAT_HISTORY_MAX_TURNS:
        pairs = pairs[-CHAT_HISTORY_MAX_TURNS * 2:]
    if not CHAT_HISTORY_MAX_CHARS:
        return pairs
    kept: list[tuple[str, str]] = []
    used = 0
    for role, text in reversed(pairs):
        if kept and used + len(text) > CHAT_HISTORY_MAX_CHARS:
            break
        kept.append((role, text))
        used += len(text)
    return list(reversed(kept))


def _search_query(question: str, history: list[tuple[str, str]], svc) -> str:
    """检索用的问题：有历史且本轮像追问时，先改写成独立问题，失败则退回本轮原问。"""
    if not history or not CONTEXTUALIZE_QUERY_ENABLED:
        return question
    from .enhance import contextualize_query, needs_contextualize

    if not needs_contextualize(question):
        return question
    llm = getattr(svc, "llm", None)
    # 用改写小模型 + 关思考 + 短超时：这一步在首字之前，不能让追问整体变慢。
    return contextualize_query(question, history, llm, REWRITE_MODEL,
                               reasoning_effort="none", timeout=CONTEXTUALIZE_TIMEOUT)


def _chunk(cid: str, created: int, model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _qa_meta(req: "ChatCompletionRequest", request: Request, history: list[tuple[str, str]],
             conversation_id: str | None) -> dict:
    """永久问答记录用：uuid（本次新增）/ user 标准字段 / X-User-Id；姓名取 user_name 或 X-User-Name；
    会话号取 X-Session-Id，没有就用 conversation_id；本轮实际用到的历史。"""
    try:
        from .qa_store import clean_id, request_meta

        meta = request_meta(request.headers, body_user=req.user, body_uuid=req.uuid, body_name=req.user_name,
                            history=[{"role": r, "content": c} for r, c in history or []])
        if not meta.get("session_id") and conversation_id:
            meta["session_id"] = clean_id(conversation_id)
        return meta
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


def _resolve_conversation(conversation_id: str | None, request: Request, store: bool) -> str | None:
    """校验/创建会话，返回可用的会话 id（不落库时返回 None）。

    显式传入但不存在 → 404（客户端拼错了 id 不该被静默新建一个，否则历史会莫名丢失）。
    """
    if not store:
        return None
    from . import conversation as cstore

    owner = key_fingerprint(request)
    if conversation_id:
        try:
            cstore.ensure(conversation_id, key_id=owner)
        except LookupError:
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"conversation not found: {conversation_id}",
                "type": "invalid_request_error", "code": "not_found"}}) from None
        return conversation_id
    return cstore.create(key_id=owner)["id"]


def _save_turn(conversation_id: str | None, *, question: str, answer: str, request: Request,
               route: str, status: str, error: str | None, search_query: str | None,
               sources: list, latency_ms: float, request_id: str) -> None:
    """把一轮问答写进会话库；任何失败都只记日志，绝不影响已经发出去的回答。"""
    if not conversation_id:
        return
    try:
        from . import conversation as cstore

        cstore.append_turn(conversation_id=conversation_id, question=question, answer=answer,
                           key_id=key_fingerprint(request), route=route, status=status, error=error,
                           search_query=search_query, sources=sources, latency_ms=round(latency_ms * 1000, 1),
                           request_id=request_id)
    except Exception as e:  # noqa: BLE001 - 落库失败不影响响应
        import logging

        logging.getLogger(__name__).warning("会话落库失败（%s），本轮未保存", type(e).__name__)


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest, request: Request):
    svc = _service(request)
    question, client_history = _split_history(req.messages)
    top_k = req.top_k or TOP_K
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model = req.model or API_MODEL_NAME
    t0 = time.time()
    conversation_id = _resolve_conversation(req.conversation_id, request, req.store)
    history = _merge_history(_server_history(conversation_id), client_history)
    extra_headers = {"X-Conversation-Id": conversation_id} if conversation_id else None
    meta = _qa_meta(req, request, history, conversation_id)

    if req.stream:
        # 只有明确没有 LLM 客户端时才提前 503；替身服务（测试或其它实现）不声明 llm 属性时照常走流。
        if getattr(svc, "llm", _MISSING) is None:
            # 配置缺失是启动期事实，直接 503，别让客户端等一个必然失败的流
            _log(question, request, cid, model, "", [], t0, status="unavailable", error="RuntimeError", meta=meta)
            raise HTTPException(status_code=503, detail={"error": {
                "message": "未配置 LLM_API_KEY，服务只能检索、不能生成", "type": "service_unavailable"}})

        def event_stream() -> Iterator[bytes]:
            parts: list[str] = []
            first_at: float | None = None
            search_query: str | None = None
            sources: list = []
            saved = False

            def save(status: str, error: str | None = None) -> None:
                nonlocal saved
                if saved:
                    return
                saved = True
                _save_turn(conversation_id, question=question, answer="".join(parts), request=request,
                           route="thesis", status=status, error=error, search_query=search_query,
                           sources=sources, latency_ms=time.time() - t0, request_id=cid)

            # 首帧只带 role，符合 OpenAI 流式约定。
            # 先发这一帧再做改写 / 检索：这两步要几秒，旧实现连响应头都要等它们做完，
            # 客户端看起来像“服务没反应”。现在响应头与首帧立刻返回，随后的等待是正常的生成等待。
            yield b"data: " + json.dumps(
                _chunk(cid, created, model, {"role": "assistant", "content": ""}), ensure_ascii=False
            ).encode() + b"\n\n"
            try:
                search_query = _search_query(question, history, svc)
                _docs, sources, deltas = svc.open_stream(question, top_k, history=history,
                                                         search_query=search_query)
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
                    save("error", type(e).__name__)
                    return
            except GeneratorExit:                 # 客户端断开：保留已生成的半截，标记 partial
                save("partial", "client_disconnect")
                raise
            except Exception as e:  # noqa: BLE001 - 检索失败也走错误帧，只带类型名
                yield b"data: " + json.dumps(
                    {"error": {"message": f"检索失败：{type(e).__name__}", "type": "upstream_error"}},
                    ensure_ascii=False).encode() + b"\n\n"
                _log(question, request, cid, model, "", [], t0, status="error", error=type(e).__name__, meta=meta)
                save("error", type(e).__name__)
                return
            # 末帧附带来源，标准客户端会忽略这个额外字段
            final = _chunk(cid, created, model, {}, finish="stop")
            final["sources"] = sources
            if conversation_id:
                final["conversation_id"] = conversation_id
            yield b"data: " + json.dumps(final, ensure_ascii=False).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
            _log(question, request, cid, model, "".join(parts), sources, t0, first_at=first_at, meta=meta)
            save("ok")

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **(extra_headers or {})}
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    search_query = _search_query(question, history, svc)
    ans, sources, _docs = svc.answer(question, top_k, history=history, search_query=search_query)
    if ans is None:
        _log(question, request, cid, model, "", sources, t0, status="unavailable", meta=meta)
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "未配置 LLM_API_KEY，服务只能检索、不能生成", "type": "service_unavailable"}},
        )
    _log(question, request, cid, model, ans, sources, t0, meta=meta)
    _save_turn(conversation_id, question=question, answer=ans, request=request, route="thesis",
               status="ok", error=None, search_query=search_query, sources=sources,
               latency_ms=time.time() - t0, request_id=cid)
    out = {
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
    if conversation_id:
        out["conversation_id"] = conversation_id
    return JSONResponse(content=out, headers=extra_headers)
