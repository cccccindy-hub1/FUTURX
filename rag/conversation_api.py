"""会话管理 REST 接口（OpenAI 风格：对象带 object 字段，分页用 cursor）。

- GET    /v1/conversations                会话列表（按最近更新倒序，cursor 分页）
- POST   /v1/conversations                新建会话（可选 title）
- GET    /v1/conversations/{id}           会话详情（可带最近若干条消息）
- PATCH  /v1/conversations/{id}           改名 / 归档
- DELETE /v1/conversations/{id}           删除（软删）
- GET    /v1/conversations/{id}/messages  消息列表（after_seq 增量拉取）
- DELETE /v1/conversations/{id}/messages  清空消息（保留会话）

归属：按 (API key 指纹, 调用方 uuid) 过滤，别人的会话返回 404（不泄露存在性）。
uuid 取 X-User-Uuid / X-User-Id 请求头（与问答接口的入参同名，取头是因为本组接口只有 GET/PATCH/DELETE）。
没传 uuid 时只看到「没有人员标识」的老会话，不会串到别人的会话里。
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import conversation as store
from .api import key_fingerprint, require_api_key

router = APIRouter()


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    # 允许客户端自己生成 id（离线创建后同步）；不传由服务端生成
    id: str | None = Field(default=None, max_length=80)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    archived: bool | None = None


def _not_found(cid: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"error": {
        "message": f"conversation not found: {cid}", "type": "invalid_request_error", "code": "not_found"}})


def _owner(request: Request) -> str | None:
    return key_fingerprint(request)


def _user(request: Request) -> str:
    """调用方 uuid（X-User-Uuid 优先，其次 X-User-Id）；没传返回 ""（只看无人员标识的老会话）。"""
    from .qa_store import clean_id

    h = request.headers
    return clean_id(h.get("x-user-uuid")) or clean_id(h.get("x-user-id")) or ""


def _conv_obj(row: dict[str, Any]) -> dict[str, Any]:
    """对外形状：object=conversation，时间为 unix 秒，另给 ISO 便于展示。"""
    return {
        "id": row["id"],
        "object": "conversation",
        "title": row["title"],
        "created_at": int(row["created_at"]),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(row["created_at"])),
        "updated_at": int(row["updated_at"]),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(row["updated_at"])),
        "turn_count": row["turn_count"],
        "archived": row["archived"],
    }


def _msg_obj(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "object": "conversation.message",
        "conversation_id": row["conversation_id"],
        "seq": row["seq"],
        "role": row["role"],
        "content": row["content"],
        "created_at": int(row["created_at"]),
        "route": row["route"],
        "status": row["status"],
        "error": row["error"],
        "search_query": row["search_query"],
        "sources": row["sources"],
        "latency_ms": row["latency_ms"],
        # 审计用：哪个 key 问的（指纹，不是原文）、YYYY-MM
        "key_id": row.get("key_id"),
        "month": row.get("month"),
    }


@router.get("/v1/conversations", dependencies=[Depends(require_api_key)])
def list_conversations(request: Request, limit: int = Query(20, ge=1, le=100),
                       cursor: str | None = None,
                       archived: str = Query("false", description="false=只看未归档（默认）/ true=只看已归档"
                                                             " / all=全部")):
    # false / true / all：用字符串接收，None 才能传给存储层表示「不按归档状态过滤」。
    # 用 bool | None 时 Query 无法表达「全部」——不传和传空都落到默认值，已归档的会话永远查不到。
    flag = str(archived).strip().lower()
    if flag in ("all", "any", "none", ""):
        only = None
    elif flag in ("true", "1", "yes"):
        only = True
    elif flag in ("false", "0", "no"):
        only = False
    else:
        raise HTTPException(status_code=400, detail={"error": {
            "message": "archived 只能是 true / false / all", "type": "invalid_request_error"}})
    got = store.list_conversations(key_id=_owner(request), limit=limit, cursor=cursor, archived=only,
                                  user_id=_user(request))
    return {
        "object": "list",
        "data": [_conv_obj(c) for c in got["items"]],
        "next_cursor": got["next_cursor"],
        "has_more": got["next_cursor"] is not None,
    }


@router.post("/v1/conversations", dependencies=[Depends(require_api_key)])
def create_conversation(req: ConversationCreate, request: Request):
    try:
        row = store.create(key_id=_owner(request), title=req.title, conversation_id=req.id,
                           user_id=_user(request))
    except Exception as e:  # noqa: BLE001 - 同 id 已存在等
        raise HTTPException(status_code=409, detail={"error": {
            "message": f"无法创建会话：{type(e).__name__}", "type": "invalid_request_error",
            "code": "already_exists"}}) from None
    return _conv_obj(row)


@router.get("/v1/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def get_conversation(conversation_id: str, request: Request,
                     include_messages: bool = True, limit: int = Query(100, ge=1, le=500)):
    row = store.get(conversation_id, key_id=_owner(request), user_id=_user(request))
    if row is None:
        raise _not_found(conversation_id)
    out = _conv_obj(row)
    if include_messages:
        out["messages"] = [_msg_obj(m) for m in store.messages(conversation_id, limit=limit,
                                                               key_id=_owner(request), user_id=_user(request))]
    return out


@router.patch("/v1/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def update_conversation(conversation_id: str, req: ConversationUpdate, request: Request):
    owner, user = _owner(request), _user(request)
    row = store.get(conversation_id, key_id=owner, user_id=user)
    if row is None:
        raise _not_found(conversation_id)
    if req.title is not None:
        row = store.rename(conversation_id, req.title, key_id=owner, user_id=user)
    if req.archived is not None:
        row = store.set_archived(conversation_id, req.archived, key_id=owner, user_id=user)
    return _conv_obj(row or store.get(conversation_id, key_id=owner, user_id=user))


@router.delete("/v1/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def delete_conversation(conversation_id: str, request: Request):
    if not store.delete(conversation_id, key_id=_owner(request), user_id=_user(request)):
        raise _not_found(conversation_id)
    return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}


@router.get("/v1/conversations/{conversation_id}/messages", dependencies=[Depends(require_api_key)])
def list_messages(conversation_id: str, request: Request, after_seq: int | None = Query(None, ge=0),
                  limit: int = Query(200, ge=1, le=500)):
    if store.get(conversation_id, key_id=_owner(request), user_id=_user(request)) is None:
        raise _not_found(conversation_id)
    # key_id / user_id 一起传下去：存储层也校验归属，不只靠上面的 get()
    items = [_msg_obj(m) for m in store.messages(conversation_id, after_seq=after_seq, limit=limit,
                                                key_id=_owner(request), user_id=_user(request))]
    return {"object": "list", "data": items,
            "next_cursor": None if len(items) < limit else items[-1]["seq"]}


@router.delete("/v1/conversations/{conversation_id}/messages", dependencies=[Depends(require_api_key)])
def clear_messages(conversation_id: str, request: Request):
    """清空消息但保留会话本身（学生想重开一个话题时用）。"""
    n = store.clear_messages(conversation_id, key_id=_owner(request), user_id=_user(request))
    if n is None:
        raise _not_found(conversation_id)
    return {"id": conversation_id, "object": "conversation.messages.cleared", "deleted_messages": n}


__all__ = ["router"]
