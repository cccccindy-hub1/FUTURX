"""试卷库 HTTP 接口（挂在 rag.serve 上，试卷库加载成功时才注册）。不调用 LLM。

- POST /exam/search        自然语言找题：{"query": "出3道数据资产的基础题"}；章 / 题型 / 是否只要基础题可显式给出
- POST /exam/by_concepts   按知识点找题（课程问答后的「来几道基础题」）：{"concepts": ["数据资产"], "k": 3}
- GET  /exam/question/{id} 取单题；include_answer=true 时附参考答案与评分要点

默认不返回答案（先练后看）。鉴权沿用 rag.api.require_api_key。
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..api import require_api_key
from .store import ExamBank

router = APIRouter()

QType = Literal["choice", "judge", "term", "short", "essay", "case"]


class ExamSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=10)
    chapter: int | None = Field(default=None, ge=1, le=11)
    qtypes: list[QType] = Field(default_factory=list)
    basic_only: bool | None = None
    include_answer: bool = False
    include_knowledge: bool = False


class ByConceptsRequest(BaseModel):
    concepts: list[str] = Field(min_length=1, max_length=20)
    k: int = Field(default=3, ge=1, le=10)
    chapter: int | None = Field(default=None, ge=1, le=11)
    qtypes: list[QType] = Field(default_factory=list)
    basic_only: bool = True
    exclude_ids: list[str] = Field(default_factory=list, max_length=200)
    include_answer: bool = False
    include_knowledge: bool = False


def _bank(request: Request) -> ExamBank:
    bank = getattr(request.app.state, "exam", None)
    if bank is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "试卷库未加载", "type": "service_unavailable"}})
    return bank


@router.post("/exam/search", dependencies=[Depends(require_api_key)])
def exam_search(req: ExamSearchRequest, request: Request):
    bank = _bank(request)
    r = bank.search(req.query, req.k, chapter=req.chapter, qtypes=tuple(req.qtypes), basic_only=req.basic_only)
    want = req.include_answer or r["intent"]["want_answer"]
    kp = bank.kp_view if req.include_knowledge else None
    return {"status": r["status"], "strategy": r["strategy"], "intent": r["intent"],
            "questions": [it.view(include_answer=want, kp_lookup=kp) for it in r["items"]]}


@router.post("/exam/by_concepts", dependencies=[Depends(require_api_key)])
def exam_by_concepts(req: ByConceptsRequest, request: Request):
    bank = _bank(request)
    items = bank.by_concepts(req.concepts, req.k, chapter=req.chapter, qtypes=tuple(req.qtypes),
                             basic_only=req.basic_only, exclude_ids=req.exclude_ids)
    kp = bank.kp_view if req.include_knowledge else None
    return {"status": "ok" if items else "no_match",
            "questions": [it.view(include_answer=req.include_answer, kp_lookup=kp) for it in items]}


@router.get("/exam/question/{question_id}", dependencies=[Depends(require_api_key)])
def exam_question(question_id: str, request: Request, include_answer: bool = False):
    bank = _bank(request)
    it = bank.by_id.get(question_id)
    if it is None:
        raise HTTPException(status_code=404, detail={"error": {"message": "题目不存在", "type": "not_found"}})
    return it.view(include_answer=include_answer, kp_lookup=bank.kp_view)
