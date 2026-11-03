"""统一入口：POST /auto/ask —— 先路由，再分发到对应知识库。现有 /ask、/v1/*、/course/ask 不变。

- off_topic：固定拒答文案，不检索、不调用任何作答模型；
- policy：政策检索尚未上线，返回占位（接口与路由已打通）；
- course：走 rag.course.pipeline.ask（需要课程上下文与课程 LLM，未加载时返回 unavailable）；
  试卷库已加载时，另附 practice：本题选中概念对应的 1–3 道【基础】题（只给题目 ID 与题干，不额外调用 LLM），
  供前端追问「要不要来几道基础题」；
- exam：走 rag.exam 试卷库检索（不调用作答模型），默认只给题目不给答案，问题里要答案 / 解析时才附上；
- thesis：走 rag.adaptive.AdaptiveRAG（simple / complex 由路由给出）；
- case：判例库（点名直达 / 章节优先 / 按案聚合；「第 X 章有哪些判例」列表模式不调用模型），作答 1 次；
- quote：名言库（按讲话检索，模型只挑段落编号，程序原文照录），共 1 次挑段落调用；
- routes 有两个库（仅 course / case / quote 之间）：各库并行检索、各自筛选，只调**一次**作答（rag.combined）。
单库判例、名言不流式，结果放进 done；组合问答只有一次作答，照常流式，materials 事件带库别 kind。
返回里带 route（= routes[0]）/ routes / complexity / reason / router 耗时 / 各路径 trace，便于调试。
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
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


class _Cancelled(Exception):
    """客户端断开后用来中止 pipeline 回调链（内部使用，不对外暴露）。"""


CASE_UNAVAILABLE = "判例库服务未加载（需要先在服务器上建好判例向量库 rag/case_store）。"
QUOTE_UNAVAILABLE = "名言库服务未加载（需要先在服务器上建好名言向量库 rag/quote_store）。"


class UnifiedService:
    def __init__(self, *, rag=None, router_client=None, course_ctx=None, course_llm=None,
                 case_service=None, case_refs=None, case_llm=None, case_model=None,
                 quote_service=None, quote_llm=None, quote_model=None, exam_bank=None):
        """case_service / quote_service：以判例库 / 名言库为目录的 RAGService（rag.query，store_dir / collection_name 可传参）；
        case_refs：判例目录（cases.jsonl 的行 -> rag.cases.retrieval.load_case_refs）；case_llm / quote_llm：作答 / 挑段落用的
        OpenAICompatClient（缺省沿用课程库的 course_llm）。没有传入对应服务时，该路由返回 unavailable，不影响其它库。"""
        self.rag = rag
        self.adaptive = AdaptiveRAG(rag) if rag is not None else None
        self.router_client = router_client
        self.course_ctx, self.course_llm = course_ctx, course_llm
        self.exam = exam_bank                         # 试卷库（线上功能）：exam 路由与课程回答附带的基础题
        self.case_service, self.case_llm = case_service, case_llm or course_llm
        self.quote_service, self.quote_llm = quote_service, quote_llm or course_llm
        from .course import config as cc
        self.case_model = case_model or cc.COURSE_ANSWER_MODEL
        self.quote_model = quote_model or cc.COURSE_SELECT_MODEL
        self.case_retriever = None
        self.case_refs = case_refs or []
        if case_service is not None:
            from .cases.retrieval import CaseRetriever
            self.case_retriever = CaseRetriever(case_service, self.case_refs)
        self.quote_retriever = None
        if quote_service is not None:
            from .quotes.retrieval import QuoteRetriever
            self.quote_retriever = QuoteRetriever(quote_service)

    # ---------------------------------------------------------------- 可用性
    def _available(self, routes: list[str]) -> list[str]:
        """routes 里哪些库没有加载。"""
        bad = []
        for r in routes:
            if r == "course" and (self.course_ctx is None or self.course_llm is None):
                bad.append(r)
            elif r == "case" and (self.case_retriever is None or self.case_llm is None):
                bad.append(r)
            elif r == "quote" and (self.quote_retriever is None or self.quote_llm is None):
                bad.append(r)
        return bad

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        question = question.strip()
        t0 = time.perf_counter()
        d: RouteDecision = classify(question, self.router_client)
        out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                               "answer": None, "sources": [], "status": "ok"}
        trace: dict[str, Any] = {"router": d.to_dict()}
        out["routes"], out["count"], out["chapter"] = list(d.routes), d.count, d.chapter
        if d.combined:
            self._combined(question, d, out, trace)
        elif d.route == "off_topic":
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT)
        elif d.route == "policy":
            out.update(status="not_available", answer=config.ROUTER_POLICY_TEXT)
        elif d.route == "course":
            self._course(question, out, trace)
        elif d.route == "exam":
            self._exam(question, out, trace)
        elif d.route == "case":
            self._case(question, d, out, trace)
        elif d.route == "quote":
            self._quote(question, d, out, trace)
        else:
            self._thesis(question, top_k, d.complexity, out, trace)
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        out["trace"] = trace
        return out

    def ask_stream(self, question: str, top_k: int | None = None) -> Iterator[dict[str, Any]]:
        """流式版 ask：生成事件字典。顺序 route →（课程路由才有）stage / materials / delta → done。

        课程路由的 pipeline 是回调式的，这里用后台线程 + 队列转成生成器；非课程路由不流式，完整结果直接放进 done。
        正文对学生只显示序号：delta 与 done.answer 里的引用都是按首次出现编号的 [1] [2]，不出现 [S#] 和 chunk_id；
        服务端实时维护 S# → 序号的映射，一个被拆在两片里的 [S12] 先缓冲到闭合再输出。
        materials 在某块第一次被引用（分配序号）时发出，带 n、chunk_id、章节路径；done 带最终的 sources（按序号排列，只含被引用的块）。
        生成中途出错发 error 事件，只带异常类型名。调用方提前断开（生成器被 close）时通知后台线程尽快收尾。"""
        question = question.strip()
        t0 = time.perf_counter()
        d: RouteDecision = classify(question, self.router_client)
        yield {"type": "route", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "count": d.count,
               "chapter": d.chapter, "reason": d.reason, "router_s": round(d.latency_s, 3)}
        if d.combined and not self._available(d.routes):
            yield from self._stream_combined(question, d, t0)
            return
        if d.route != "course" or d.combined or self.course_ctx is None or self.course_llm is None:
            out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                                   "answer": None, "sources": [], "status": "ok", "routes": list(d.routes)}
            trace: dict[str, Any] = {"router": d.to_dict()}
            try:
                if d.combined:
                    self._combined(question, d, out, trace)        # 其中某个库未加载：返回 unavailable
                elif d.route == "off_topic":
                    out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT)
                elif d.route == "policy":
                    out.update(status="not_available", answer=config.ROUTER_POLICY_TEXT)
                elif d.route == "course":
                    self._course(question, out, trace)        # 课程库未加载：返回 unavailable
                elif d.route == "exam":
                    self._exam(question, out, trace)
                elif d.route == "case":
                    self._case(question, d, out, trace)       # 单库判例：不流式，结果放进 done
                elif d.route == "quote":
                    self._quote(question, d, out, trace)      # 单库名言：不流式，结果放进 done
                else:
                    self._thesis(question, top_k, d.complexity, out, trace)
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                yield {"type": "error", "error": type(e).__name__}
                return
            trace["total_s"] = round(time.perf_counter() - t0, 4)
            out["trace"] = trace
            yield {"type": "done", **{k: out.get(k) for k in ("route", "routes", "complexity", "status", "answer", "sources", "exam")},
                   "trace": trace}
            return

        from .course.pipeline import ask, result_to_json
        q: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        _END = object()

        def on_event(ev: dict) -> None:
            if cancelled.is_set():                            # 客户端已断开：中止后续回调，让 pipeline 线程尽快抛出收尾
                raise _Cancelled()
            q.put(ev)

        def run() -> None:
            try:
                q.put(("result", ask(question, ctx=self.course_ctx, llm=self.course_llm, on_event=on_event)))
            except _Cancelled:
                pass
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                q.put(("error", type(e).__name__))
            finally:
                q.put(_END)

        threading.Thread(target=run, name="auto-ask-stream", daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, tuple):
                    kind, val = item
                    if kind == "error":
                        yield {"type": "error", "error": val}
                        continue
                    res = result_to_json(val)
                    trace = {"router": d.to_dict(), "course": res.pop("trace", None)}
                    trace["total_s"] = round(time.perf_counter() - t0, 4)
                    ans = res.get("answer")
                    if ans:
                        status, text, sources = "ok", ans.get("text"), ans.get("sources", [])
                    else:
                        status, text, sources = "insufficient", config.ADAPTIVE_INSUFFICIENT_TEXT, []
                    yield {"type": "done", "route": d.route, "complexity": d.complexity, "status": status,
                           "answer": text, "sources": sources,
                           "practice": self._practice(res.get("selected_concepts") or []), "trace": trace}
                else:
                    yield item
        finally:
            cancelled.set()

    # ---------------------------------------------------------------- 判例（单库，不流式）
    def _case(self, question, d, out, trace) -> None:
        if self.case_retriever is None or self.case_llm is None:
            out.update(status="unavailable", answer=CASE_UNAVAILABLE)
            return
        from .cases.answer import answer_cases
        from .cases.retrieval import NOT_FOUND_TEXT, render_case_list
        t0 = time.perf_counter()
        from .router import is_case_list_question
        if is_case_list_question(question):                      # 列表模式：直接列出该章判例名，不检索、不调用任何模型
            text = render_case_list(self.case_refs, d.chapter)
            out.update(answer=text, status="ok" if "共有" in text else "insufficient",
                       sources=[], list_mode=True)
            trace["case"] = {"mode": "list", "chapter": d.chapter, "llm_calls": 0, "total_s": round(time.perf_counter() - t0, 4)}
            return
        rewrite = (lambda q: _rewrite_with(self, q)) if getattr(self.case_service, "llm", None) else None
        got = self.case_retriever.retrieve(question, count=d.count, chapter=d.chapter, rewrite=rewrite)
        trace["case"] = {"mode": got["mode"], **got["trace"]}
        if not got["hits"]:
            out.update(status="insufficient", answer=NOT_FOUND_TEXT, sources=[])      # 没找到：不调用作答模型
            trace["case"]["llm_calls"] = 0
            return
        t1 = time.perf_counter()
        ans = answer_cases(question, got["hits"], self.case_llm, model=self.case_model)
        trace["case"].update(answer_s=round(time.perf_counter() - t1, 4), llm_calls=1, dropped=ans.dropped)
        out.update(answer=ans.text, sources=ans.sources)

    # ---------------------------------------------------------------- 名言（单库，不流式，不调用作答模型）
    def _quote(self, question, d, out, trace) -> None:
        if self.quote_retriever is None or self.quote_llm is None:
            out.update(status="unavailable", answer=QUOTE_UNAVAILABLE)
            return
        from .quotes.retrieval import NOT_FOUND_TEXT, pick_paragraphs, render_quote_answer, source_line
        rewrite = (lambda q: _rewrite_with(self, q, self.quote_service)) if getattr(self.quote_service, "llm", None) else None
        got = self.quote_retriever.retrieve(question, count=d.count, rewrite=rewrite)
        trace["quote"] = {"mode": got["mode"], **got["trace"]}
        if not got["hits"]:
            out.update(status="insufficient", answer=NOT_FOUND_TEXT, sources=[])        # 没找到：不调用挑段落模型
            trace["quote"]["llm_calls"] = 0
            return
        t1 = time.perf_counter()
        picks, info = pick_paragraphs(question, got["hits"], self.quote_llm, model=self.quote_model)
        trace["quote"].update(pick_s=round(time.perf_counter() - t1, 4), llm_calls=1, pick=info)
        out.update(answer=render_quote_answer(picks, numbered=True),                    # 程序原文照录，不含任何模型文字
                   sources=[{"n": k, "kind": "quote", "chunk_id": p.speech.speech_id, "path": source_line(p.speech),
                             "title": p.speech.title, "year": p.speech.year} for k, p in enumerate(picks, 1)])

    # ---------------------------------------------------------------- 组合（各库并行取材，一次作答）
    def _combined(self, question, d, out, trace) -> None:
        bad = self._available(d.routes)
        if bad:
            names = {"course": "课程库", "case": "判例库", "quote": "名言库"}
            out.update(status="unavailable", answer="、".join(names[b] for b in bad) + "服务未加载，暂时无法组合回答。")
            return
        from .combined import answer_combined, gather_materials
        from .course.config import COURSE_ANSWER_MODEL
        m = gather_materials(self, question, d)
        trace["combined"] = dict(m.traces)
        if not m.sections:
            out.update(status="insufficient", answer=config.ADAPTIVE_INSUFFICIENT_TEXT, sources=[])
            trace["combined"]["llm_calls"] = {"answer": 0}
            return
        t1 = time.perf_counter()
        titles = self.course_ctx.chapter_titles if self.course_ctx is not None else None
        ans = answer_combined(question, m, self.course_llm, model=COURSE_ANSWER_MODEL, titles=titles)
        trace["combined"].update(answer_s=round(time.perf_counter() - t1, 4), removed_quotes=ans.removed_quotes,
                                 missing=m.empty, sections=m.sections)
        out.update(answer=ans.text, sources=ans.sources)

    def _stream_combined(self, question, d, t0) -> Iterator[dict[str, Any]]:
        """组合问答的流式：只有一次作答，照常流式。事件：stage（各库取材进度）→ materials（带库别 kind）/ delta → done。
        讲话引号校验需要完整正文，所以只作用于 done 里的最终 answer（delta 先流出；前端在 done 时用最终正文覆盖）。"""
        from .combined import answer_combined, gather_materials
        from .course.config import COURSE_ANSWER_MODEL
        q: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        _END = object()

        def emit(ev: dict) -> None:
            if cancelled.is_set():
                raise _Cancelled()
            q.put(ev)

        def run() -> None:
            try:
                m = gather_materials(self, question, d, on_event=lambda e: emit(e) if e.get("type") == "stage" else None)
                emit({"type": "stage", "stage": "combined", "sections": m.sections, "missing": m.empty})
                if not m.sections:
                    q.put(("result", None, m))
                    return
                titles = self.course_ctx.chapter_titles if self.course_ctx is not None else None
                ans = answer_combined(question, m, self.course_llm, model=COURSE_ANSWER_MODEL, titles=titles,
                                      on_delta=lambda t: emit({"type": "delta", "text": t}),
                                      on_source=lambda e: emit({"type": "materials", **e}))
                q.put(("result", ans, m))
            except _Cancelled:
                pass
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                q.put(("error", type(e).__name__))
            finally:
                q.put(_END)

        threading.Thread(target=run, name="auto-ask-stream", daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, tuple):
                    if item[0] == "error":
                        yield {"type": "error", "error": item[1]}
                        continue
                    _, ans, m = item
                    trace = {"router": d.to_dict(), "combined": dict(m.traces), "total_s": round(time.perf_counter() - t0, 4)}
                    if ans is None:
                        yield {"type": "done", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "status": "insufficient",
                               "answer": config.ADAPTIVE_INSUFFICIENT_TEXT, "sources": [], "trace": trace}
                    else:
                        trace["combined"].update(removed_quotes=ans.removed_quotes, missing=m.empty, sections=m.sections)
                        yield {"type": "done", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "status": "ok",
                               "answer": ans.text, "sources": ans.sources, "trace": trace}
                else:
                    yield item
        finally:
            cancelled.set()

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
            out.update(answer=ans.get("text"), sources=ans.get("sources", []))      # [{n, chunk_id, path}]，按序号排列，只含正文引用过的块
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


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


@router.post("/auto/ask/stream", dependencies=[Depends(require_api_key)])
def auto_ask_stream(req: AutoAskRequest, request: Request):
    """SSE 流式统一入口：先推 route，课程路由再推 stage / delta / materials，最后 done（带最终 answer 与 sources）。
    delta 与 answer 里引用都是序号 [1] [2]（不含 [S#] 与 chunk_id）；materials 在序号分配时发出（n、chunk_id、path）。
    非课程路由不流式，完整结果直接放进 done。生成中途出错发 error 事件，只带异常类型名。
    并发名额由 RequestConcurrencyMiddleware 持有到整个流结束。"""
    svc: UnifiedService | None = getattr(request.app.state, "unified", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "统一入口未就绪", "type": "service_unavailable"}})

    def gen():
        events = svc.ask_stream(req.question, req.top_k)
        try:
            for ev in events:
                if ev.get("type") == "done" and not req.include_trace:
                    ev = {k: v for k, v in ev.items() if k != "trace"}
                yield _sse(ev)
        except Exception as e:  # noqa: BLE001 - 流已开始，只能以事件告知，且只带类型名
            yield _sse({"type": "error", "error": type(e).__name__})
        finally:
            events.close()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _rewrite_with(svc: "UnifiedService", q: str, store=None) -> str:
    """检索纠错时改写查询：复用论文库的 enhance.rewrite_query（需要该服务有 OpenAI 兼容 llm）。"""
    from .config import REWRITE_MODEL
    from .enhance import rewrite_query
    return rewrite_query(q, (store or svc.case_service).llm, REWRITE_MODEL)
