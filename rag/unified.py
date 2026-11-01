"""统一入口：POST /auto/ask —— 先路由，再分发到对应知识库。现有 /ask、/v1/*、/course/ask 不变。

- off_topic：固定拒答文案，不检索、不调用任何作答模型；
- policy：政策检索尚未上线，返回占位（接口与路由已打通）；
- course：走 rag.course.pipeline.ask（需要课程上下文与课程 LLM，未加载时返回 unavailable）；
  试卷库已加载时，另附 practice：本题选中概念对应的 1–3 道【基础】题（只给题目 ID 与题干，不额外调用 LLM），
  供前端追问「要不要来几道基础题」；
- exam：走 rag.exam 试卷库检索（不调用作答模型），默认只给题目不给答案，问题里要答案 / 解析时才附上；
- thesis：走 rag.adaptive.AdaptiveRAG（simple / complex 由路由给出）。
返回里带 route / complexity / reason / router 耗时 / 各路径 trace，便于调试。
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import config
from .adaptive import AdaptiveRAG
from .api import require_api_key
from .router import RouteDecision, classify

router = APIRouter()


class AutoAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = None
    include_trace: bool = True


class UnifiedService:
    def __init__(self, *, rag=None, router_client=None, course_ctx=None, course_llm=None, exam_bank=None):
        self.rag = rag
        self.adaptive = AdaptiveRAG(rag) if rag is not None else None
        self.router_client = router_client
        self.course_ctx, self.course_llm = course_ctx, course_llm
        self.exam = exam_bank

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        question = question.strip()
        t0 = time.perf_counter()
        d: RouteDecision = classify(question, self.router_client)
        out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                               "answer": None, "sources": [], "status": "ok"}
        trace: dict[str, Any] = {"router": d.to_dict()}
        if d.route == "off_topic":
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT)
        elif d.route == "policy":
            out.update(status="not_available", answer=config.ROUTER_POLICY_TEXT)
        elif d.route == "course":
            self._course(question, out, trace)
        elif d.route == "exam":
            self._exam(question, out, trace)
        else:
            self._thesis(question, top_k, d.complexity, out, trace)
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        out["trace"] = trace
        return out

    def _thesis(self, question, top_k, complexity, out, trace) -> None:
        if self.adaptive is None:
            out.update(status="unavailable", answer="论文指导服务未加载。")
            return
        r = self.adaptive.run(question, top_k, complexity)
        trace["adaptive"] = r.pop("trace")
        out.update(status=r["status"], answer=r["answer"], sources=r["sources"])
        out["contexts"] = r["contexts"]

    def _course(self, question, out, trace) -> None:
        if self.course_ctx is None or self.course_llm is None:
            out.update(status="unavailable", answer="课程问答服务未加载（需要教材/本体数据与 COURSE_LLM_* 配置）。")
            return
        from .course.pipeline import ask, result_to_json
        res = result_to_json(ask(question, ctx=self.course_ctx, llm=self.course_llm))
        trace["course"] = res.pop("trace", None)
        ans = res.get("answer")
        if ans:
            out.update(answer=ans.get("text"), sources=res.get("citations", []))
        else:
            out.update(status="insufficient", answer=config.ADAPTIVE_INSUFFICIENT_TEXT)
        grade = {v["chunk_id"]: v["grade"] for v in res.get("filtered", [])}
        out["course"] = {k: res.get(k) for k in ("selected_concepts", "fallback")}
        # 每个召回块的来源（ontology/bm25/both）与筛选等级，便于调试与评测
        out["course"]["practice"] = self._practice(res.get("selected_concepts") or [])
        out["course"]["recalled"] = [{"chunk_id": r["chunk_id"], "source": r.get("source"), "grade": grade.get(r["chunk_id"]),
                                      "path": " › ".join(r["path"][-3:])} for r in res.get("recalled", [])]


    def _practice(self, selected: list[dict]) -> list[dict]:
        """课程选中概念 -> 试卷库 1–3 道基础题（选择 / 判断 / 名词解释）。试卷库未加载或无对应题时为空。"""
        if self.exam is None or self.course_ctx is None:
            return []
        by_id = self.course_ctx.concepts.by_id
        names = list(dict.fromkeys(by_id[s["concept_id"]].name for s in selected if s.get("concept_id") in by_id))
        items = self.exam.by_concepts(names, config.EXAM_PRACTICE_MAX, basic_only=True)
        return [{"question_id": it.q.question_id, "qtype_cn": it.q.qtype_cn, "stem": it.q.stem} for it in items]

    def _exam(self, question, out, trace) -> None:
        if self.exam is None:
            out.update(status="unavailable", answer="试卷库未加载（需要 EXAM_TEACHER_DOC 与学生知识集合）。")
            return
        from .exam.store import format_items
        t0 = time.perf_counter()
        r = self.exam.search(question)
        want = r["intent"]["want_answer"]
        trace["exam"] = {"strategy": r["strategy"], "intent": r["intent"], "bm25_top": r.get("bm25_top"),
                         "latency_s": round(time.perf_counter() - t0, 4)}
        if r["status"] != "ok":
            out.update(status="no_match", answer=config.EXAM_NO_MATCH_TEXT)
            out["exam"] = {"questions": []}
            return
        out["answer"] = format_items(r["items"], include_answer=want)
        out["exam"] = {"questions": [it.view(include_answer=want) for it in r["items"]]}

@router.post("/auto/ask", dependencies=[Depends(require_api_key)])
def auto_ask(req: AutoAskRequest, request: Request):
    svc: UnifiedService | None = getattr(request.app.state, "unified", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "统一入口未就绪", "type": "service_unavailable"}})
    try:
        out = svc.ask(req.question, req.top_k)
    except Exception as e:  # noqa: BLE001 - 下游错误统一转 502；只回类型名，避免泄露细节
        raise HTTPException(status_code=502, detail={"error": {"message": f"下游调用失败：{type(e).__name__}",
                                                               "type": "upstream_error"}}) from None
    if not req.include_trace:
        out.pop("trace", None)
    return out
