"""统一入口：POST /auto/ask —— 先路由，再分发到对应知识库。现有 /ask、/v1/*、/course/ask 不变。

- off_topic：固定拒答文案，不检索、不调用任何作答模型；
- policy：政策库（agentic：点名直达文件 / 条 → BM25 ≤15 块 → 模型选块 1 次 → 作答 1 次）；
- course：走 rag.course.pipeline.ask（需要课程上下文与课程 LLM，未加载时返回 unavailable）；
  试卷库已加载时，另附 practice：本题选中概念对应的 1–3 道【基础】题（只给题目 ID 与题干，不额外调用 LLM），
  供前端追问「要不要来几道基础题」；
- exam：走 rag.exam 试卷库检索（不调用作答模型），默认只给题目不给答案，问题里要答案 / 解析时才附上；
- thesis：走 rag.adaptive.AdaptiveRAG（simple / complex 由路由给出）；
- case：判例库（agentic：点名直达 / BM25 初筛 ≤12 案 / 模型选案 1 次 / 作答 1 次；「第 X 章有哪些判例」列表模式不调用模型）；
- quote：名言库（BM25 初筛段落 / 模型挑段落编号 1 次，程序原文照录，不再调用模型）；
- routes 有两个库（仅 course / case / quote / policy 之间）：各库并行检索、各自筛选，只调**一次**作答（rag.combined）。
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from . import usage_log
from .adaptive import AdaptiveRAG
from .api import require_api_key
from .router import RouteDecision, classify

router = APIRouter()


class AutoAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = None
    include_trace: bool = True
    # 会话：传了就把本轮追加到该会话，并从库里取之前几轮作为上下文（检索仍只用本轮问题）。
    conversation_id: str | None = Field(default=None, max_length=80)
    store: bool = True                       # false = 不落库（临时提问）


def _conversation_context(conversation_id: str | None, request: Request, store: bool) -> str | None:
    """校验/新建会话，返回可用的会话 id（不落库返回 None）。显式传入但不存在的 id → 404。"""
    if not store:
        return None
    from . import conversation as cstore
    from .api import key_fingerprint

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


def _history_messages(conversation_id: str | None) -> list[dict]:
    """会话历史转成 LLM messages（user/assistant 交替，最近的在后）。"""
    if not conversation_id:
        return []
    from . import conversation as cstore
    from .config import CHAT_HISTORY_MAX_CHARS, CHAT_HISTORY_MAX_TURNS

    try:
        pairs = cstore.history(conversation_id, max_turns=CHAT_HISTORY_MAX_TURNS,
                               max_chars=CHAT_HISTORY_MAX_CHARS or None)
    except Exception:  # noqa: BLE001 - 会话库故障时退化为单轮
        return []
    return [{"role": role, "content": text} for role, text in pairs]


def _standalone_question(question: str, history: list[dict] | None, llm) -> str:
    """把追问改写成独立问题，供**路由**判断（检索与作答仍用原话）。

    统一入口的路由按问题文本分类，像「那它适用什么情形？」这种只有指代的追问会被判成
    off_topic 直接拒答。这里复用检索用的改写器（rag.enhance.contextualize_query）拿到完整
    问题再分类；改写失败或不需要改写时原样返回。
    """
    if not history:
        return question
    from .config import CONTEXTUALIZE_QUERY_ENABLED, CONTEXTUALIZE_TIMEOUT, REWRITE_MODEL
    from .enhance import contextualize_query, needs_contextualize

    if not CONTEXTUALIZE_QUERY_ENABLED or not needs_contextualize(question):
        return question
    pairs = [(m.get("role"), m.get("content")) for m in history
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    if not pairs:
        return question
    return contextualize_query(question, pairs, llm, REWRITE_MODEL,
                               reasoning_effort="none", timeout=CONTEXTUALIZE_TIMEOUT)


def _save_turn(conversation_id: str | None, out: dict, request: Request) -> None:
    """把统一入口的一轮结果落库；失败只记日志。"""
    if not conversation_id:
        return
    try:
        from . import conversation as cstore
        from .api import key_fingerprint

        cstore.append_turn(conversation_id=conversation_id, question=out.get("question") or "",
                           answer=out.get("answer") or "", key_id=key_fingerprint(request),
                           route=out.get("route"), status=out.get("status") or "ok",
                           error=out.get("error"), sources=out.get("sources") or [],
                           latency_ms=(out.get("trace") or {}).get("total_s", 0) * 1000 or None)
    except Exception as e:  # noqa: BLE001 - 落库失败不影响响应
        import logging

        logging.getLogger(__name__).warning("会话落库失败（%s），本轮未保存", type(e).__name__)


@router.post("/auto/ask", dependencies=[Depends(require_api_key)])
def auto_ask(req: AutoAskRequest, request: Request):
    svc: UnifiedService | None = getattr(request.app.state, "unified", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "统一入口未就绪", "type": "service_unavailable"}})
    from .api import key_fingerprint

    conversation_id = _conversation_context(req.conversation_id, request, req.store)
    context = _history_messages(conversation_id)
    try:
        out = svc.ask(req.question, req.top_k, origin="http", key_id=key_fingerprint(request),
                      history=context)
    except Exception as e:  # noqa: BLE001 - 下游错误统一转 502；只回类型名，避免泄露细节
        raise HTTPException(status_code=502, detail={"error": {"message": f"下游调用失败：{type(e).__name__}",
                                                               "type": "upstream_error"}}) from None
    out["question"] = req.question
    _save_turn(conversation_id, out, request)
    out.pop("question", None)
    if conversation_id:
        out["conversation_id"] = conversation_id
    if not req.include_trace:
        out.pop("trace", None)
    headers = {"X-Conversation-Id": conversation_id} if conversation_id else None
    return JSONResponse(content=_jsonable(out), headers=headers)


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


@router.post("/auto/ask/stream", dependencies=[Depends(require_api_key)])
def auto_ask_stream(req: AutoAskRequest, request: Request):
    """SSE 流式统一入口：先推 route，课程路由再推 stage / delta / materials，最后 done（带最终 answer 与 sources）。
    delta 与 answer 里引用都是序号 [1] [2]（不含 [S#] 与 chunk_id）；materials 在序号分配时发出（n、chunk_id、path）。
    非课程路由不流式，完整结果直接放进 done。生成中途出错发 error 事件，只带异常类型名。
    并发名额由 RequestConcurrencyMiddleware 持有到整个流结束。
    带 conversation_id 时：历史只作为上下文交给作答模型，检索仍只用本轮问题；done 之后落库。"""
    svc: UnifiedService | None = getattr(request.app.state, "unified", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "统一入口未就绪", "type": "service_unavailable"}})

    from .api import key_fingerprint

    conversation_id = _conversation_context(req.conversation_id, request, req.store)
    context = _history_messages(conversation_id)

    def gen():
        final: dict | None = None
        events = svc.ask_stream(req.question, req.top_k, origin="http", key_id=key_fingerprint(request),
                                history=context)
        try:
            for ev in events:
                if ev.get("type") == "done":
                    final = {**ev, "question": req.question}
                    if conversation_id:
                        ev = {**ev, "conversation_id": conversation_id}
                if ev.get("type") == "done" and not req.include_trace:
                    ev = {k: v for k, v in ev.items() if k != "trace"}
                yield _sse(ev)
        except Exception as e:  # noqa: BLE001 - 流已开始，只能以事件告知，且只带类型名
            yield _sse({"type": "error", "error": type(e).__name__})
            if final is None:
                final = {"question": req.question, "status": "error", "error": type(e).__name__, "answer": ""}
        finally:
            events.close()
            _save_turn(conversation_id, final or {"question": req.question, "status": "error",
                                                  "error": "client_disconnect", "answer": ""}, request)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if conversation_id:
        headers["X-Conversation-Id"] = conversation_id
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


def _jsonable(obj):
    """把 trace / dataclass 之类的对象转成可 JSON 序列化的结构（复用 FastAPI 的编码器）。"""
    from fastapi.encoders import jsonable_encoder

    return jsonable_encoder(obj)


class _Cancelled(Exception):
    """客户端断开后用来中止 pipeline 回调链（内部使用，不对外暴露）。"""


CASE_UNAVAILABLE = "判例库未加载（服务器上缺少判例原始 docx，见 CASE_DOC）。"
PRACTICE_QTYPES = ("judge", "term")                       # 巩固练习允许的题型：判断题、名词解释
POLICY_UNAVAILABLE = "政策库未加载（服务器上缺少数据政策汇编 docx，见 POLICY_DOC）。"
QUOTE_UNAVAILABLE = "名言库未加载（服务器上缺少名言合集 docx，见 QUOTE_DOC）。"


class UnifiedService:
    def __init__(self, *, rag=None, router_client=None, course_ctx=None, course_llm=None,
                 case_index=None, quote_index=None, policy_index=None, lib_llm=None, pick_model=None, answer_model=None, exam_bank=None):
        """case_index / quote_index：启动时从 docx 读入的内存索引（rag.cases.retrieval.CaseIndex / rag.quotes.retrieval.QuoteIndex），
        不用向量库；lib_llm：判例 / 名言（以及后面的政策）选择与作答用的胜算云 OpenAICompatClient（缺省沿用课程库的 course_llm，绝不用论文库的 key）；
        pick_model：选案 / 挑段模型（Flash、关闭思考）；answer_model：作答模型（Pro、思考 low）。没有传入索引时，该路由返回 unavailable，不影响其它库。"""
        self.rag = rag
        self.adaptive = AdaptiveRAG(rag) if rag is not None else None
        self.router_client = router_client
        self.course_ctx, self.course_llm = course_ctx, course_llm
        self.exam = exam_bank                         # 试卷库（线上功能）：exam 路由与课程回答附带的基础题
        self._stems = None                            # 题干预匹配索引（解题先查题库），首次使用时建
        from .course import config as cc
        self.lib_llm = lib_llm or course_llm
        self.pick_model = pick_model or cc.LIB_PICK_MODEL
        self.answer_model = answer_model or cc.LIB_ANSWER_MODEL
        self.case_index, self.quote_index, self.policy_index = case_index, quote_index, policy_index
        self.case_refs = case_index.refs if case_index is not None else []
        self.case_retriever = self.quote_retriever = self.policy_retriever = None
        if case_index is not None and self.lib_llm is not None:
            from .cases.retrieval import CaseRetriever
            self.case_retriever = CaseRetriever(case_index, self.lib_llm, model=self.pick_model)
        if quote_index is not None and self.lib_llm is not None:
            from .quotes.retrieval import QuoteRetriever
            self.quote_retriever = QuoteRetriever(quote_index, self.lib_llm, model=self.pick_model)
        if policy_index is not None and self.lib_llm is not None:
            from .policy.retrieval import PolicyRetriever
            self.policy_retriever = PolicyRetriever(policy_index, self.lib_llm, model=self.pick_model)

    # ---------------------------------------------------------------- 可用性
    def _available(self, routes: list[str]) -> list[str]:
        """routes 里哪些库没有加载。"""
        bad = []
        for r in routes:
            if r == "course" and (self.course_ctx is None or self.course_llm is None):
                bad.append(r)
            elif r == "case" and (self.case_retriever is None):
                bad.append(r)
            elif r == "quote" and (self.quote_retriever is None):
                bad.append(r)
            elif r == "policy" and (self.policy_retriever is None):
                bad.append(r)
        return bad

    def ask(self, question: str, top_k: int | None = None, *, origin: str = "eval",
            key_id: str | None = None, history: list[dict] | None = None) -> dict[str, Any]:
        """origin/key_id：调用来源（http / eval）与 API key 指纹，只用于调用日志（需求5）。
        history：之前几轮的 [{"role","content"}]，只作为作答上下文，不参与检索与路由判断。"""
        with usage_log.record(question, origin, key_id) as rec:
            out = self._ask(question, top_k, history=history)
            rec.finish(out)
            return out

    def _ask(self, question: str, top_k: int | None = None, *, history: list[dict] | None = None) -> dict[str, Any]:
        question = question.strip()
        t0 = time.perf_counter()
        # 路由 / 题库匹配用「独立问题」：把「那它适用什么情形」补全成完整问句，
        # 否则只有指代的追问会被路由判成 off_topic 直接拒答。
        routed_q = _standalone_question(question, history, self.lib_llm)
        solved = self._prematch(routed_q)                          # 解题先查题库：命中就不路由、不调用任何模型
        if solved is not None:
            solved["trace"]["total_s"] = round(time.perf_counter() - t0, 4)
            solved["followups"] = self._followups(question, solved)      # 命中题库：零调用模板
            return solved
        solve = self._solve_flag(routed_q)
        d: RouteDecision = classify(routed_q, self.router_client)
        if solve:
            d = self._solve_route(d)
        out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                               "answer": None, "sources": [], "status": "ok"}
        trace: dict[str, Any] = {"router": d.to_dict()}
        out["routes"], out["count"], out["chapter"] = list(d.routes), d.count, d.chapter
        # 作答与检索用补全后的独立问题：各库的作答提示词都是「只依据材料回答」，
        # 只给历史不看指代时，模型会说「你问的它指代不明」（实测）。历史仍作为上下文一并传入。
        q = routed_q
        if d.combined:
            self._combined(q, d, out, trace, history)
        elif d.route == "off_topic":
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT)
        elif d.route == "policy":
            self._policy(q, out, trace, history)
        elif d.route == "course":
            self._course(q, out, trace, solve=solve, history=history)
        elif d.route == "exam":
            self._exam(q, out, trace)
        elif d.route == "case":
            self._case(q, d, out, trace, history)
        elif d.route == "quote":
            self._quote(q, d, out, trace)
        else:
            self._thesis(q, top_k, d.complexity, out, trace, history)
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        out["trace"] = trace
        out["followups"] = self._followups(question, out)
        return out

    # ---------------------------------------------------------------- 追问（需求3）
    _SKIP_FOLLOWUP = ("off_topic", "quote")           # 拒答无追问；讲话原文照录，不附追问

    def _followups(self, question: str, out: dict[str, Any]) -> list[str]:
        """追问（需求3）；耗时记入调用日志的 followup 阶段。"""
        t0 = time.perf_counter()
        try:
            return self._followups_impl(question, out)
        finally:
            usage_log.stage("followup", time.perf_counter() - t0)

    def _followups_impl(self, question: str, out: dict[str, Any]) -> list[str]:
        """按路由决定追问来源：exam 命中题库走零调用模板，其余走一次小调用。

        门控：仅 status == "ok" 且 answer 非空时生成。off_topic / quote / unavailable /
        insufficient / refused / 其它非 ok 状态一律返回空，且不发起任何模型调用。
        任何异常都静默返回空列表，不影响主答案。
        """
        if out.get("status") != "ok":
            return []
        answer = (out.get("answer") or "").strip()
        if not answer:
            return []
        if out.get("route") in self._SKIP_FOLLOWUP:
            return []
        try:
            from .followups import exam_followups, make_followups
            ex = out.get("exam") or {}
            questions = ex.get("questions") or []
            if out.get("route") == "exam" and len(questions) == 1:
                got = exam_followups(questions[0])
                if got:
                    return got
            return make_followups(question, answer, out.get("route") or "", self.lib_llm)
        except Exception:  # noqa: BLE001 - 追问失败不影响主答案
            return []

    def ask_stream(self, question: str, top_k: int | None = None, *, origin: str = "eval",
                   key_id: str | None = None,
                   history: list[dict] | None = None) -> Iterator[dict[str, Any]]:
        """流式版 ask：生成事件字典。顺序 route →（课程路由才有）stage / materials / delta → done。

        课程路由的 pipeline 是回调式的，这里用后台线程 + 队列转成生成器；非课程路由不流式，完整结果直接放进 done。
        正文对学生只显示序号：delta 与 done.answer 里的引用都是按首次出现编号的 [1] [2]，不出现 [S#] 和 chunk_id；
        服务端实时维护 S# → 序号的映射，一个被拆在两片里的 [S12] 先缓冲到闭合再输出。
        materials 在某块第一次被引用（分配序号）时发出，带 n、chunk_id、章节路径；done 带最终的 sources（按序号排列，只含被引用的块）。
        生成中途出错发 error 事件，只带异常类型名。调用方提前断开（生成器被 close）时通知后台线程尽快收尾。
        origin/key_id 只用于调用日志（需求5）。

        注意：Starlette 把同步生成器交给线程池逐段 next()，每次恢复都在新的上下文里，
        生成器内设置的 contextvar 会被丢弃（流式日志只剩第一条 route 调用）。所以整条链都显式传 rec，
        不依赖 contextvar。"""
        rec = usage_log.start(question, origin, key_id)
        inner = self._ask_stream(question, top_k, rec, history)
        try:
            while True:
                # 关键：每段恢复都在线程池的新上下文里，必须在本段内重新激活 rec，
                # 内层的 LLM 上报（以及它再起的后台线程 via bind）才看得到这条记录。
                with usage_log.activate(rec):
                    try:
                        ev = next(inner)
                    except StopIteration:
                        break
                if ev.get("type") == "route":                # route 先到、done 可能因断开来不了
                    rec.route_name = ev.get("route")
                    rec.routes = list(ev.get("routes") or [])
                    rec.complexity = ev.get("complexity")
                yield ev
        except GeneratorExit:                                # 客户端断开：落已得结果后正常收尾
            inner.close()
            rec.finish(error="client_disconnect")
            raise
        except Exception as e:  # noqa: BLE001 - 已开始流式，异常转事件由端点处理
            rec.finish(error=type(e).__name__)
            raise
        finally:
            usage_log.clear()
        rec.finish(rec.pending_out)                          # 最终结果由 _ask_stream 在 done 时留在 rec 上

    def _ask_stream(self, question: str, top_k: int | None, rec, history: list[dict] | None = None) -> Iterator[dict[str, Any]]:
        question = question.strip()
        t0 = time.perf_counter()
        # 同 _ask：路由与题库匹配用补全后的独立问题，作答与检索资料仍对应本轮原话。
        routed_q = _standalone_question(question, history, self.lib_llm)
        solved = self._prematch(routed_q)                          # 解题先查题库：命中就不路由、不调用任何模型
        if solved is not None:
            solved["trace"]["total_s"] = round(time.perf_counter() - t0, 4)
            yield {"type": "route", "route": "exam", "routes": ["exam"], "complexity": "simple", "count": 1, "chapter": None,
                   "reason": solved["reason"], "router_s": 0.0}
            yield {"type": "done", **{k: solved.get(k) for k in ("route", "routes", "complexity", "status", "answer", "sources", "exam")},
                   "trace": solved["trace"]}
            got = self._followups(question, solved)                # exam 命中题库：零调用模板
            if got:
                solved["followups"] = got
                yield {"type": "followups", "items": got}
            rec.pending_out = dict(solved)
            return
        solve = self._solve_flag(routed_q)
        d: RouteDecision = classify(routed_q, self.router_client)
        if solve:
            d = self._solve_route(d)
        yield {"type": "route", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "count": d.count,
               "chapter": d.chapter, "reason": d.reason, "router_s": round(d.latency_s, 3)}
        if d.combined and not self._available(d.routes):
            yield from self._stream_combined(question, d, t0, rec, history)
            return
        if d.route != "course" or d.combined or self.course_ctx is None or self.course_llm is None:
            out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                                   "answer": None, "sources": [], "status": "ok", "routes": list(d.routes)}
            trace: dict[str, Any] = {"router": d.to_dict()}
            try:
                if d.combined:
                    self._combined(routed_q, d, out, trace, history)        # 其中某个库未加载：返回 unavailable
                elif d.route == "off_topic":
                    out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT)
                elif d.route == "policy":
                    self._policy(routed_q, out, trace, history)    # 单库政策：不流式，结果放进 done
                elif d.route == "course":
                    self._course(routed_q, out, trace, solve=solve, history=history)        # 课程库未加载：返回 unavailable
                elif d.route == "exam":
                    self._exam(routed_q, out, trace)
                elif d.route == "case":
                    self._case(routed_q, d, out, trace, history)       # 单库判例：不流式，结果放进 done
                elif d.route == "quote":
                    self._quote(routed_q, d, out, trace)      # 单库名言：不流式，结果放进 done
                else:
                    self._thesis(routed_q, top_k, d.complexity, out, trace, history)
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                rec.pending_out = {"route": d.route, "routes": list(d.routes), "complexity": d.complexity,
                                   "status": "error", "answer": "", "sources": []}
                rec.finish(error=type(e).__name__)
                yield {"type": "error", "error": type(e).__name__}
                return
            trace["total_s"] = round(time.perf_counter() - t0, 4)
            out["trace"] = trace
            yield {"type": "done", **{k: out.get(k) for k in ("route", "routes", "complexity", "status", "answer", "sources", "exam")},
                   "trace": trace}
            got = self._followups(question, out)                   # done 先发，追问后到；失败静默省略
            if got:
                out["followups"] = got
                yield {"type": "followups", "items": got}
            rec.pending_out = dict(out)
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
                with usage_log.activate(rec):                    # 后台线程里重新激活，pipeline 的 LLM 调用才记得到
                    q.put(("result", ask(question, ctx=self.course_ctx, llm=self.course_llm,
                                         on_event=on_event, solve=solve, history=history)))
            except _Cancelled:
                pass
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                q.put(("error", type(e).__name__))
            finally:
                q.put(_END)

        threading.Thread(target=usage_log.bind(run), name="auto-ask-stream", daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, tuple):
                    kind, val = item
                    if kind == "error":
                        rec.finish(error=str(val))
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
                    got = self._followups(question, {"route": d.route, "status": status, "answer": text, "exam": None})
                    if got:
                        yield {"type": "followups", "items": got}
                    rec.pending_out = {"route": d.route, "routes": [d.route], "complexity": d.complexity,
                                       "status": status, "answer": text, "sources": sources,
                                       "followups": got or [], "trace": trace}
                else:
                    if item.get("type") == "delta":
                        rec.first_delta()
                    elif item.get("type") == "route":
                        rec.route_name, rec.routes, rec.complexity = (
                            item.get("route"), list(item.get("routes") or []), item.get("complexity"))
                    yield item
        finally:
            cancelled.set()

    # ---------------------------------------------------------------- 判例（单库，不流式）
    def _case(self, question, d, out, trace, history: list[dict] | None = None) -> None:
        if self.case_retriever is None:
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
        got = self.case_retriever.retrieve(question, count=d.count, chapter=d.chapter)
        trace["case"] = {"mode": got["mode"], **got["trace"]}
        if not got["hits"]:
            out.update(status="insufficient", answer=NOT_FOUND_TEXT, sources=[])      # 没找到：不调用作答模型
            trace["case"]["answer_calls"] = 0
            return
        t1 = time.perf_counter()
        ans = answer_cases(question, got["hits"], self.lib_llm, model=self.answer_model, history=history)
        trace["case"].update(answer_s=round(time.perf_counter() - t1, 4), answer_calls=1, dropped=ans.dropped)
        out.update(answer=ans.text, sources=ans.sources)

    # ---------------------------------------------------------------- 名言（单库，不流式，不调用作答模型）
    def _quote(self, question, d, out, trace) -> None:
        if self.quote_retriever is None:
            out.update(status="unavailable", answer=QUOTE_UNAVAILABLE)
            return
        from .quotes.retrieval import NOT_FOUND_TEXT, render_quote_answer, source_line
        got = self.quote_retriever.retrieve(question, count=d.count)
        trace["quote"] = {"mode": got["mode"], **got["trace"]}
        picks = got["picks"]
        if not picks:
            out.update(status="insufficient", answer=NOT_FOUND_TEXT, sources=[])        # 没找到：不再调用任何模型
            return
        out.update(answer=render_quote_answer(picks, numbered=True),                    # 程序原文照录，不含任何模型文字
                   sources=[{"n": k, "kind": "quote", "chunk_id": p.speech.speech_id, "path": source_line(p.speech),
                             "title": p.speech.title, "year": p.speech.year} for k, p in enumerate(picks, 1)])

    # ---------------------------------------------------------------- 政策（单库，不流式）
    def _policy(self, question, out, trace, history: list[dict] | None = None) -> None:
        if self.policy_retriever is None:
            out.update(status="unavailable", answer=POLICY_UNAVAILABLE)
            return
        from .policy.retrieval import NOT_FOUND_TEXT, answer_policy
        got = self.policy_retriever.retrieve(question)
        trace["policy"] = {"mode": got["mode"], **got["trace"]}
        if not got["hits"]:
            out.update(status="insufficient", answer=NOT_FOUND_TEXT, sources=[])        # 没找到：不调用作答模型
            trace["policy"]["answer_calls"] = 0
            return
        t1 = time.perf_counter()
        ans = answer_policy(question, got["hits"], self.lib_llm, model=self.answer_model, history=history)
        trace["policy"].update(answer_s=round(time.perf_counter() - t1, 4), answer_calls=1, removed_quotes=ans.removed_quotes)
        out.update(answer=ans.text, sources=ans.sources)

    # ---------------------------------------------------------------- 组合（各库并行取材，一次作答）
    def _combined(self, question, d, out, trace, history: list[dict] | None = None) -> None:
        bad = self._available(d.routes)
        if bad:
            names = {"course": "课程库", "case": "判例库", "quote": "名言库", "policy": "政策库"}
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
        ans = answer_combined(question, m, self.course_llm, model=COURSE_ANSWER_MODEL, titles=titles,
                              history=history)
        trace["combined"].update(answer_s=round(time.perf_counter() - t1, 4), removed_quotes=ans.removed_quotes,
                                 missing=m.empty, sections=m.sections)
        out.update(answer=ans.text, sources=ans.sources)

    def _stream_combined(self, question, d, t0, rec, history: list[dict] | None = None) -> Iterator[dict[str, Any]]:
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
                with usage_log.activate(rec):                    # 组合路由同样在后台线程里调 LLM
                    m = gather_materials(self, question, d,
                                         on_event=lambda e: emit(e) if e.get("type") == "stage" else None)
                    emit({"type": "stage", "stage": "combined", "sections": m.sections, "missing": m.empty})
                    if not m.sections:
                        q.put(("result", None, m))
                        return
                    titles = self.course_ctx.chapter_titles if self.course_ctx is not None else None
                    ans = answer_combined(question, m, self.course_llm, model=COURSE_ANSWER_MODEL, titles=titles,
                                          history=history,
                                          on_delta=lambda t: emit({"type": "delta", "text": t}),
                                          on_source=lambda e: emit({"type": "materials", **e}))
                    q.put(("result", ans, m))
            except _Cancelled:
                pass
            except Exception as e:  # noqa: BLE001 - 只带异常类型名
                q.put(("error", type(e).__name__))
            finally:
                q.put(_END)

        threading.Thread(target=usage_log.bind(run), name="auto-ask-stream", daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, tuple):
                    if item[0] == "error":
                        rec.finish(error=str(item[1]))
                        yield {"type": "error", "error": item[1]}
                        continue
                    _, ans, m = item
                    trace = {"router": d.to_dict(), "combined": dict(m.traces), "total_s": round(time.perf_counter() - t0, 4)}
                    if ans is None:
                        yield {"type": "done", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "status": "insufficient",
                               "answer": config.ADAPTIVE_INSUFFICIENT_TEXT, "sources": [], "trace": trace}
                        rec.pending_out = {"route": d.route, "routes": list(d.routes), "complexity": d.complexity,
                                           "status": "insufficient", "answer": config.ADAPTIVE_INSUFFICIENT_TEXT,
                                           "sources": [], "followups": [], "trace": trace}
                    else:
                        trace["combined"].update(removed_quotes=ans.removed_quotes, missing=m.empty, sections=m.sections)
                        yield {"type": "done", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "status": "ok",
                               "answer": ans.text, "sources": ans.sources, "trace": trace}
                        got = self._followups(question, {"route": d.route, "status": "ok", "answer": ans.text, "exam": None})
                        if got:
                            yield {"type": "followups", "items": got}
                        rec.pending_out = {"route": d.route, "routes": list(d.routes), "complexity": d.complexity,
                                           "status": "ok", "answer": ans.text, "sources": ans.sources,
                                           "followups": got or [], "trace": trace}
                else:
                    if item.get("type") == "delta":
                        rec.first_delta()
                    yield item
        finally:
            cancelled.set()

    def _thesis(self, question, top_k, complexity, out, trace, history: list[dict] | None = None) -> None:
        if self.adaptive is None:
            out.update(status="unavailable", answer="论文指导服务未加载。")
            return
        r = self.adaptive.run(question, top_k, complexity, history=history)
        trace["adaptive"] = r.pop("trace")
        out.update(status=r["status"], answer=r["answer"], sources=r["sources"])
        out["contexts"] = r["contexts"]

    def _course(self, question, out, trace, solve: bool = False, history: list[dict] | None = None) -> None:
        if self.course_ctx is None or self.course_llm is None:
            out.update(status="unavailable", answer="课程问答服务未加载（需要教材/本体数据与 COURSE_LLM_* 配置）。")
            return
        from .course.pipeline import ask, result_to_json
        res = result_to_json(ask(question, ctx=self.course_ctx, llm=self.course_llm, solve=solve,
                                 history=history))
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


    # ---------------------------------------------------------------- 解题：先查题库
    def _solve_flag(self, question: str) -> bool:
        """带具体题干的解题问题（没命中题库时才会走到这里）：课程作答的提示词因此追加“先给结论”一条。"""
        from .exam.solve import is_solve_question
        return is_solve_question(question)

    @staticmethod
    def _solve_route(d: RouteDecision) -> RouteDecision:
        """路由把带具体题干的解题问题判成 exam（找题）时改送课程库；其它路由（课程、判例……）保持原样。"""
        if d.route == "exam":
            d.routes, d.route, d.reason = ["course"], "course", (d.reason + "；带具体题干的解题，改送课程库")[:80]
        return d

    def _prematch(self, question: str) -> dict[str, Any] | None:
        """解题预匹配（在路由之前）：问题去掉外壳后题干 ≥12 字，与题库题干完全包含或相似度 ≥0.9 → 直接给题目 + 答案 + 解析。
        耗时 ms 级，写进 trace.prematch；命中时 0 次模型调用（连路由也不调）。没命中返回 None，走原来的路由。"""
        if self.exam is None:
            return None
        from .exam.solve import StemMatcher, is_solve_question
        if not is_solve_question(question):
            return None
        if self._stems is None:
            self._stems = StemMatcher(self.exam.items)
        m = self._stems.match(question)
        info = {k: m[k] for k in ("hit", "similarity", "how", "stem_chars", "candidates", "ms")}
        if m["hit"] is None:
            return None
        from .exam.store import format_items
        it = self.exam.items[m["hit"]]
        views = [it.view(include_answer=True)]
        views[0]["solve"] = True
        info["hit"] = it.q.question_id
        return {"route": "exam", "routes": ["exam"], "complexity": "simple", "reason": "题库命中，直接给答案", "count": 1, "chapter": None,
                "status": "ok", "answer": format_items([it], include_answer=True), "sources": [], "exam": {"questions": views, "solve": True},
                "trace": {"prematch": info, "llm_calls": 0}}

    def _practice(self, selected: list[dict]) -> list[dict]:
        """课程选中概念 -> 试卷库 1–3 道巩固练习（判断 / 名词解释，不含选择题）。试卷库未加载或没有符合的题时为空列表，
        前端据此不显示巩固练习这一块。"""
        if self.exam is None or self.course_ctx is None:
            return []
        by_id = self.course_ctx.concepts.by_id
        names = list(dict.fromkeys(by_id[s["concept_id"]].name for s in selected if s.get("concept_id") in by_id))
        # 巩固练习只给判断题和名词解释：网页上不显示选择题的选项，学生没法作答，所以单选 / 多选等选择类题型一律排除。
        # 只在这里限定题型；试卷库直接问题目（“给我第三章选择题”、/exam/*）走另一条路径，仍然照常返回选择题。
        items = self.exam.by_concepts(names, config.EXAM_PRACTICE_MAX, qtypes=PRACTICE_QTYPES)
        items = [it for it in items if it.q.qtype in PRACTICE_QTYPES]          # 再保险：即使上游题型过滤失效也不放选择题
        # 答案、评分要点与解析由程序原样带出（网页默认折叠），不经过模型
        return [{"question_id": it.q.question_id, "qtype_cn": it.q.qtype_cn, "stem": it.q.stem,
                 "answer": it.q.answer, "analysis": it.q.analysis} for it in items]

    def _exam(self, question, out, trace) -> None:
        if self.exam is None:
            out.update(status="unavailable", answer="试卷库未加载（需要 EXAM_TEACHER_DOC 与学生知识集合）。")
            return
        from .exam.store import format_items
        t0 = time.perf_counter()
        r = self.exam.search(question)
        trace["exam"] = {"strategy": r["strategy"], "intent": r["intent"], "bm25_top": r.get("bm25_top"),
                         "latency_s": round(time.perf_counter() - t0, 4)}
        if r["status"] != "ok":
            out.update(status="no_match", answer=config.EXAM_NO_MATCH_TEXT)
            out["exam"] = {"questions": []}
            return
        # 答案始终随题带出（网页默认折叠）；正文只排题干与选项，答案放在 exam.questions[*].answer / analysis
        out["answer"] = format_items(r["items"], include_answer=False)
        out["exam"] = {"questions": [it.view(include_answer=True) for it in r["items"]]}
