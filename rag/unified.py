"""统一入口：POST /auto/ask —— 先路由，再分发到对应知识库。现有 /ask、/v1/*、/course/ask 不变。

- meta：问助手自身（能做什么 / 知识库有哪些内容）-> 按实际加载的库拼说明，不检索、不调用模型；
- off_topic：超出知识库范围 -> 不检索，直接用大模型自身知识作答（rag.unified._general）；
  模型未配置时退回固定拒答文案；
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
import logging
import queue
import threading
import time
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from . import profile as learner
from . import qa_store, student, usage_log
from .adaptive import AdaptiveRAG
from .api import require_api_key
from .persona import EXAMPLE_DIALOGUE, GREETING_SHORT
from .router import (RouteDecision, classify, looks_ack, looks_greeting, looks_progress, looks_recall,
                     looks_whoami)

router = APIRouter()
logger = logging.getLogger(__name__)

_UNSET = object()          # 区分「还没建过通用作答客户端」与「建过但未配置（None）」


class AutoAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = None
    include_trace: bool = True
    # 会话：传了就把本轮追加到该会话，并从库里取之前几轮作为上下文（检索仍只用本轮问题）。
    conversation_id: str | None = Field(default=None, max_length=80)
    store: bool = True                       # false = 不落库（临时提问）
    # 永久问答记录用：uuid = 调用方人员标识（优先于 X-User-Id）；user_name = 提问者姓名（中文可用，限长 32 字）。
    # 两者只用于问答记录，超长 / 含非法字符的在落库前清洗，不影响本次回答。
    uuid: str | None = None
    user_name: str | None = None


def _conversation_context(conversation_id: str | None, request: Request, store: bool,
                          uuid_value: str | None = None) -> str | None:
    """校验/新建会话，返回可用的会话 id（不落库返回 None）。显式传入但不存在的 id、或不属于本人 → 404。

    归属按 (key_id, uuid)：共享一把 key 时学生之间互不可见；没传 uuid 的老调用方按 "" 处理。
    """
    if not store:
        return None
    from . import conversation as cstore
    from .api import conversation_owner

    owner, user = conversation_owner(request, uuid_value)
    if conversation_id:
        try:
            cstore.ensure(conversation_id, key_id=owner, user_id=user or "")
        except LookupError:
            raise HTTPException(status_code=404, detail={"error": {
                "message": f"conversation not found: {conversation_id}",
                "type": "invalid_request_error", "code": "not_found"}}) from None
        return conversation_id
    return cstore.create(key_id=owner, user_id=user or "")["id"]


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
    from .config import CONTEXTUALIZE_QUERY_ENABLED, CONTEXTUALIZE_TIMEOUT
    from .enhance import contextualize_query, needs_contextualize

    if not CONTEXTUALIZE_QUERY_ENABLED or not needs_contextualize(question):
        return question
    pairs = [(m.get("role"), m.get("content")) for m in history
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    if not pairs:
        return question
    # 统一入口的改写走胜算云客户端（lib_llm）：REWRITE_MODEL 没单独配置时默认沿用论文库的模型名，
    # 在胜算云上可能不存在而静默失败（退回原问题）；这里改用与追问胶囊同一个、确定可用的小模型。
    import os
    from .course.config import LIB_PICK_MODEL
    model = os.environ.get("REWRITE_MODEL") or LIB_PICK_MODEL
    return contextualize_query(question, pairs, llm, model,
                               reasoning_effort="none", timeout=CONTEXTUALIZE_TIMEOUT)


def _qa_ids(fn, request: Request, conversation_id: str | None, req=None) -> dict:
    """永久问答记录（rag.qa_store）的调用方标识：uuid（body 优先，其次 X-User-Uuid / X-User-Id 当 uuid 用）；
    姓名取 user_name 或 X-User-Name；会话号取 X-Session-Id，没有就用 conversation_id。
    只在服务实现接受这些参数时才传（测试替身 / 其它实现的 ask 签名里没有它们）。"""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "user_id" not in params or "session_id" not in params:
        return {}
    meta = qa_store.request_meta(request.headers,
                                 body_uuid=getattr(req, "uuid", None),
                                 body_name=getattr(req, "user_name", None))
    out = {"user_id": meta["user_id"], "session_id": meta["session_id"] or qa_store.clean_id(conversation_id)}
    if "user_name" in params:
        out["user_name"] = meta["user_name"]
    return out


def _save_turn(conversation_id: str | None, out: dict, request: Request,
               uuid_value: str | None = None) -> None:
    """把统一入口的一轮结果落库；失败只记日志。"""
    if not conversation_id:
        return
    try:
        from . import conversation as cstore
        from .api import conversation_owner

        owner, user = conversation_owner(request, uuid_value)
        cstore.append_turn(conversation_id=conversation_id, question=out.get("question") or "",
                           answer=out.get("answer") or "", key_id=owner,
                           route=out.get("route"), status=out.get("status") or "ok",
                           error=out.get("error"), sources=out.get("sources") or [],
                           latency_ms=(out.get("trace") or {}).get("total_s", 0) * 1000 or None,
                           user_id=user or "")
    except Exception as e:  # noqa: BLE001 - 落库失败不影响响应
        import logging

        logging.getLogger(__name__).warning("会话落库失败（%s），本轮未保存", type(e).__name__)


@router.post("/auto/ask", dependencies=[Depends(require_api_key)])
def auto_ask(req: AutoAskRequest, request: Request):
    svc: UnifiedService | None = getattr(request.app.state, "unified", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "统一入口未就绪", "type": "service_unavailable"}})
    from .api import key_fingerprint

    conversation_id = _conversation_context(req.conversation_id, request, req.store, req.uuid)
    context = _history_messages(conversation_id)
    try:
        out = svc.ask(req.question, req.top_k, origin="http", key_id=key_fingerprint(request),
                      history=context, **_qa_ids(svc.ask, request, conversation_id, req))
    except Exception as e:  # noqa: BLE001 - 下游错误统一转 502；只回类型名，避免泄露细节
        raise HTTPException(status_code=502, detail={"error": {"message": f"下游调用失败：{type(e).__name__}",
                                                               "type": "upstream_error"}}) from None
    out["question"] = req.question
    _save_turn(conversation_id, out, request, req.uuid)
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

    conversation_id = _conversation_context(req.conversation_id, request, req.store, req.uuid)
    context = _history_messages(conversation_id)

    def gen():
        final: dict | None = None
        events = svc.ask_stream(req.question, req.top_k, origin="http", key_id=key_fingerprint(request),
                                history=context, **_qa_ids(svc.ask_stream, request, conversation_id, req))
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
                                                  "error": "client_disconnect", "answer": ""}, request, req.uuid)

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


# meta 路由（问助手自身）的固定片段。能力清单按实际加载情况拼装，见 UnifiedService._meta。
# 人设与开场白统一来自 rag.persona（角色卡），此处不再另写一份。
META_INTRO = "我是 AI 李老师，数字治理科研团队的学术数字人。"
META_SCOPES = [
    ("thesis", "学位论文写作与指导", "选题、开题报告、文献综述、研究设计、研究方法（扎根理论 / 问卷 / 访谈 / 统计）、"
                                    "论文结构与格式、答辩、学术规范"),
    ("course", "《数据要素》课程知识", "数据要素、数据确权与数据产权、数据价值、数据政策、数据产品、数据市场、"
                                     "数据交易、数据安全、隐私保护、数据治理"),
    ("exam", "课程试题", "按知识点或题型出练习题，也能查题库里某道题的参考答案、解析与评分要点"),
    ("policy", "数据政策法规", "数据安全法、个人信息保护法、数据二十条等法律与政策文件的条文、适用范围、发布时间"),
    ("case", "数据法律判例", "具体案件的案情、裁判要点，以及按章节列判例"),
    ("quote", "总书记关于数字领域的讲话名言", "按主题查讲话原文、金句，原文照录不改写"),
]
META_TAIL = ("回答都会标注依据来源：课程与论文类给页码 / 章节，政策与判例给文件名与条款，讲话给年份与场合。"
             "直接输入问题就行；追问时可以接着上一轮问。")
META_GREETING = GREETING_SHORT
# meta 各路径的短回复：确认 / 收尾一句话，并按上文给一个下一步；没有上文就不给，不硬凑。
ACK_REPLY = "好，需要接着问什么直接说。"
ACK_REPLY_WITH_NEXT = "好。接着可以问「{next}」，或者直接把下一个问题发给我。"
ACK_REPLY_WITH_TOPIC = "好。接着可以问「{next}」，或者直接说下一个问题。"
# 上一个问题问的是什么，决定「下一步」推荐什么方向（只影响文案，不检索）
_ACK_NEXT = {
    "course": "这个概念怎么考", "thesis": "这个方法在论文里怎么写",
    "policy": "这条规定的适用范围", "case": "这个判例的裁判要点", "quote": "同一主题还有哪些讲话",
    "exam": "这道题的参考答案", "meta": "你能做什么", "off_topic": "再问一个具体问题",
}
_ROUTE_CN = {"course": "课程知识", "thesis": "论文写作", "policy": "政策法规", "exam": "课程试题",
             "case": "判例", "quote": "讲话名言", "meta": "助手自身", "off_topic": "库外通用问答"}

# 上一轮问的是什么概念：去掉疑问外壳后剩下的部分（「什么是数据确权」→「数据确权」）。
# 只做字符串处理，不查词表、不调模型：确认语必须零调用返回，答错一句闲话不值得多花一次请求。
_TOPIC_PREFIX = ("请问一下", "请问", "想问一下", "我想问一下", "想问", "问一下", "老师", "请教一下", "请教",
                 "给我讲讲", "给我说说", "讲讲", "说说", "介绍一下", "那它和", "它和", "那么", "那")
_TOPIC_HEAD = ("具体是指什么", "指的是什么", "是指什么", "是什么意思", "是什么", "什么叫", "什么是", "啥是")
# 疑问词出现的位置就是概念的结尾：截到它前面，「数据确权为什么难」就只剩「数据确权」
_TOPIC_MARK = ("为什么", "为何", "怎么", "如何", "是什么", "什么", "有哪些", "哪些", "包括", "吗", "呢")
_TOPIC_TAIL = "？?。！!，,、；;：: 　"
_TOPIC_VERB_TAIL = ("规定了什么", "规定了", "包括什么", "包括", "涉及", "指什么", "指的是", "是指", "指")
# 尾部这些虚词不成词，剥掉再判断
_TOPIC_FILLER = ("呢", "吗", "啊", "呀", "吧", "嘛", "的", "了", "得", "地", "着", "过", "和", "与", "及", "或")
_TOPIC_EMPTY = ("它", "这个", "那个", "这些", "上述内容", "以上", "这个知识点")
# 像不像一个「可以接着问的概念」：太长、带动作词或数字量词的，多半是整句请求，不是概念
_TOPIC_BAD = ("出", "几道", "给我", "帮我", "写", "改", "翻译", "举", "列", "生成", "总结", "判断", "题")


def _clean_topic(text: str) -> str:
    """一句话里抽出「问的是什么」；不像个概念（太短、只是代词）就返回空串。"""
    s = (text or "").strip()
    if not s:
        return ""
    changed = True
    while changed:                                  # 「请问一下」+「什么是」可能连着两层
        changed = False
        for p in _TOPIC_PREFIX:
            if s.startswith(p):
                s, changed = s[len(p):].lstrip(_TOPIC_TAIL), True
                break
        for a in _TOPIC_HEAD:
            if s.startswith(a):
                s, changed = s[len(a):].lstrip(_TOPIC_TAIL), True
                break
    # 截到最早的疑问词：它后面是「怎么问」，前面才是「问的是什么」
    cut = min((s.find(m) for m in _TOPIC_MARK if s.find(m) > 0), default=-1)
    if cut > 0:
        s = s[:cut]
    s = s.strip(_TOPIC_TAIL)
    # 先剥疑问句尾部的动词 / 助词，再剥语气词——「规定了」剥掉「了」之后就没法整词匹配了
    for _ in range(2):
        for _, v in sorted(((len(v), v) for v in _TOPIC_VERB_TAIL), reverse=True):
            if s.endswith(v):
                s = s[: -len(v)]
        for _, f in sorted(((len(f), f) for f in _TOPIC_FILLER), reverse=True):
            if s.endswith(f) and len(s) > len(f) + 1:
                s = s[: -len(f)]
        s = s.strip(_TOPIC_TAIL)
    if not (2 <= len(s) <= 16) or s in _TOPIC_EMPTY:
        return ""
    if any(b in s for b in _TOPIC_BAD):            # 「出三道数据资产的判断题」是整句请求，不是概念
        return ""
    return s


def _last_question(history: list[dict] | None, depth: int = 6) -> str:
    """上一轮真正的提问：跳过「好的 / 谢谢 / 你好」这类确认与招呼，往回找最近一个有内容的问题。

    不跳的话，连着两次确认会让第二次只能拿「你能做什么」当下一步（线上实测）。
    """
    for m in reversed((history or [])[-depth * 2:]):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = str(m.get("content") or "").strip()
        if not text:
            continue
        try:
            from .router import looks_ack, looks_greeting
            if looks_ack(text) or looks_greeting(text):
                continue
        except Exception:  # noqa: BLE001 - 判不出来就当作正常提问
            pass
        return text
    return ""


def _last_route(history: list[dict] | None, depth: int = 6) -> str:
    """上一轮问的是哪一类：取 history 里最后一条真正的提问，用关键词规则判一次（零调用、不联网）。

    不看助手回答的措辞（各库句式不统一，认不准），只看学生上一句问的是什么；
    规则判不出来（返回 off_topic 又没命中关键词）时也照样返回，调用方据此给通用建议。
    """
    last = _last_question(history, depth)
    if not last:
        return ""
    try:
        from .router import rule_route
        return rule_route(last).route
    except Exception:  # noqa: BLE001 - 判不出来就不带上下文，不影响回答
        return ""


# 带具体概念的下一步（{topic} 由上一轮的问题抽出；抽不出概念时退回下面 _ACK_NEXT 的通用文案）
_ACK_NEXT_TOPIC = {
    "course": "{topic}在考试里怎么考",
    "thesis": "{topic}在论文里怎么写",
    "policy": "{topic}的适用范围",
    "case": "{topic}这类案件的裁判要点",
    "quote": "关于{topic}的讲话还有哪些",
    "exam": "{topic}这类题怎么答",
    "off_topic": "{topic}和课程内容有什么关系",
}


def _ack_reply(history: list[dict] | None) -> str:
    """确认 / 收尾语的回法：没有上文就一句短的；有上文就顺着上文的具体概念给一个下一步。

    只给类型不给概念的版本（「接着可以问「这个概念怎么考」」）等于没说，学生还得自己想问什么；
    所以优先用上一轮问题里抽出的概念拼一句，抽不出来才退回类型文案。
    """
    last = _last_question(history)
    if not last:
        return ACK_REPLY
    try:
        from .router import rule_route

        route = rule_route(last).route
    except Exception:  # noqa: BLE001
        return ACK_REPLY
    topic = _clean_topic(last)
    tpl = _ACK_NEXT_TOPIC.get(route)
    if topic and tpl:
        return ACK_REPLY_WITH_TOPIC.format(next=tpl.format(topic=topic))
    return ACK_REPLY_WITH_NEXT.format(next=_ACK_NEXT.get(route, "下一个问题"))
# 问「我之前问过哪些」时最多列多少条：只是给个回顾，不把几十条翻出来。
RECALL_MAX = 20

# 纯打招呼用短欢迎语（用户只是打个招呼，不该甩一整篇清单）；问了能力才给完整清单。
_GREETING_RE = __import__("re").compile(
    r"^[\s，。！？!?,.]*(?:你好|您好|哈喽|嗨|hi|hello|hey|早上好|上午好|中午好|下午好|晚上好|早安|晚安|在吗|在不在)"
    r"[\s，。！？!?,.]*(?:呀|啊|哈|呢|嘛|哦|呦|哟)*(?:[\s，。！？!?,.]{0,3}"
    r"(?:你好|您好|哈喽|嗨|hi|hello|hey|早上好|上午好|中午好|下午好|晚上好|早安|晚安|在吗|在不在)"
    r"[\s，。！？!?,.]*(?:呀|啊|哈|呢|嘛|哦|呦|哟)*)*[\s。！？!?,.]*$", __import__("re").I)

GENERAL_UNAVAILABLE = "这个问题超出知识库范围，而通用作答模型未配置（需要 ROUTER_API_BASE / ROUTER_API_KEY）。"

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
            key_id: str | None = None, history: list[dict] | None = None, user_id: str | None = None,
            session_id: str | None = None, user_name: str | None = None) -> dict[str, Any]:
        """origin/key_id：调用来源（http / eval）与 API key 指纹，只用于调用日志（需求5）。
        history：之前几轮的 [{"role","content"}]，只作为作答上下文，不参与检索与路由判断。
        user_id/session_id/user_name：永久问答记录（rag.qa_store）用；都可省略。"""
        with usage_log.record(question, origin, key_id) as rec:
            rec.qa_meta = {"user_id": user_id, "session_id": session_id, "user_name": user_name,
                           "history": qa_store.normalize_history(history)}
            out = self._ask(question, top_k, history=history, rec=rec, user_id=user_id,
                            session_id=session_id, key_id=key_id)
            rec.finish(out)
            return out

    def _ask(self, question: str, top_k: int | None = None, *, history: list[dict] | None = None,
             rec=None, user_id: str | None = None, session_id: str | None = None,
             key_id: str | None = None) -> dict[str, Any]:
        question = question.strip()
        t0 = time.perf_counter()
        # 路由 / 题库匹配用「独立问题」：把「那它适用什么情形」补全成完整问句，
        # 否则只有指代的追问会被路由判成 off_topic 直接拒答。
        routed_q = _standalone_question(question, history, self.lib_llm)
        if rec is not None and routed_q != question:
            rec.qa_meta["standalone"] = routed_q
        solved, by_ref = self._prematch_with_history(question, routed_q, history)   # 解题先查题库：命中就不路由、不调用任何模型
        if solved is not None:
            if self._explain_wanted(question, routed_q, by_ref):
                return self._explain(question, solved, t0, history)   # 要讲解：参考答案 + 课程讲解
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
        who = user_id or key_id or (rec.key_id if rec is not None else None)     # 画像按人记，口径同问答记录
        if who and self._note_identity(who, question):
            # 纯自述（「我是编辑出版专业的本科生」）：记下来，回一句确认，不检索、不调模型。
            # 以前这句会被判成 off_topic，模型把它当问题答，讲了一段考研方向；
            # 紧接着问选题，回答里全是公共管理的例子（实测）。
            out.update(route="meta", status="ok", sources=[],
                       answer=student.confirm_text(learner.identity(who)))
            out["trace"] = {"router": d.to_dict(), "identity": learner.identity(who), "llm_calls": 0,
                            "total_s": round(time.perf_counter() - t0, 4)}
            out["followups"] = []
            return out
        prof = learner.summary(who)                                              # 没有记录时是空串，提示词逐字不变
        if d.combined:
            self._combined(q, d, out, trace, history, profile=prof)
        elif d.route == "off_topic":
            self._general(q, out, trace, history)          # 超出知识库：用大模型自身能力答，不检索
        elif d.route == "policy":
            self._policy(q, out, trace, history, profile=prof)
        elif d.route == "course":
            self._course(q, out, trace, solve=solve, history=history, profile=prof)
        elif d.route == "exam":
            self._exam(q, out, trace)
        elif d.route == "case":
            self._case(q, d, out, trace, history, profile=prof)
        elif d.route == "quote":
            self._quote(q, d, out, trace)
        elif d.route == "meta":
            self._meta(out, trace, q, {"user_id": user_id, "session_id": session_id, "key_id": key_id},
                       history=history)
        else:
            self._thesis(q, top_k, d.complexity, out, trace, history, profile=prof)
        self._note_learning(who, out, d)
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        out["trace"] = trace
        out["followups"] = self._followups(question, out)
        return out

    # ---------------------------------------------------------------- 学习者画像（P1）
    def _note_identity(self, who: str | None, question: str) -> bool:
        """认出「我是本科生 编辑出版专业」这类自述并记进画像；返回是否只是自我介绍。

        只认明确的说法（见 rag.student），认不出就什么都不做。返回 True 时调用方应当直接回一句确认，
        因为这句话不是问题，用模型去答只会得到一段「这个专业怎么样」的泛泛介绍。
        """
        if not who:
            return False
        try:
            got = student.parse(question)
            title = student.parse_title(question)          # 「论文题目是…」：只认显式说法
            if not any(got.values()) and not title:
                return False
            learner.notice_identity(who, **got, title=title)
            return student.only_intro(question)
        except Exception as e:  # noqa: BLE001 - 认身份失败按普通问题处理
            logger.warning("身份识别失败（%s）", type(e).__name__)
            return False

    def _note_learning(self, who: str | None, out: dict[str, Any], d: RouteDecision) -> None:
        """把这一轮涉及的概念 / 章记进画像（rag.profile）。

        只记事实：课程管线选中的概念名、判例 / 试题命中的章号；不记对错、不推断水平。
        写失败只记 warning（profile 内部已兜底），绝不影响本次回答。
        """
        if not who:
            return
        concepts = [self._concept_name(c) for c in ((out.get("course") or {}).get("selected_concepts") or [])
                    if isinstance(c, dict)]
        concepts = [c for c in concepts if c]
        chapter = d.chapter if ("case" in (d.routes or []) or d.route in ("case", "exam")) else None
        if not concepts and chapter is None:
            return
        try:
            learner.notice_question(who, concepts=concepts, chapter=chapter)
        except Exception as e:  # noqa: BLE001 - 画像只是加分项
            logger.warning("画像记录失败（%s）", type(e).__name__)

    def _concept_name(self, item: dict) -> str:
        """selected_concepts 的元素只有 concept_id（名称在课程词表里），换成名字；换不到就用 id。"""
        cid = item.get("concept_id") or ""
        if not cid:
            return ""
        if self.course_ctx is not None:
            try:
                return self.course_ctx.concepts.by_id[cid].name
            except Exception:  # noqa: BLE001 - 词表里没有这个 id：退回 id
                pass
        return cid

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
        if (out.get("trace") or {}).get("meta", {}).get("recall"):
            return []                                  # 「我之前问过什么」只做回顾，别顺带推销别的知识点
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
                   key_id: str | None = None, history: list[dict] | None = None, user_id: str | None = None,
                   session_id: str | None = None, user_name: str | None = None) -> Iterator[dict[str, Any]]:
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
        rec.qa_meta = {"user_id": user_id, "session_id": session_id, "user_name": user_name,
                       "history": qa_store.normalize_history(history)}
        inner = self._ask_stream(question, top_k, rec, history, user_id=user_id, session_id=session_id, key_id=key_id)
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

    def _ask_stream(self, question: str, top_k: int | None, rec, history: list[dict] | None = None,
                    *, user_id: str | None = None, session_id: str | None = None,
                    key_id: str | None = None) -> Iterator[dict[str, Any]]:
        question = question.strip()
        orig = question
        t0 = time.perf_counter()
        # 同 _ask：路由与题库匹配用补全后的独立问题，作答与检索资料仍对应本轮原话。
        routed_q = _standalone_question(question, history, self.lib_llm)
        if routed_q != question:
            rec.qa_meta["standalone"] = routed_q
        solved, by_ref = self._prematch_with_history(question, routed_q, history)   # 解题先查题库：命中就不路由、不调用任何模型
        explain = solved if solved is not None and self._explain_wanted(question, routed_q, by_ref) else None
        if solved is not None and explain is None:
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
        if explain is not None:                                    # 题库命中 + 要讲解：路由跳过，直接走课程流式讲解
            solve: bool | str = "explain"
            d = RouteDecision("course", "simple", "题库命中，按课程讲解", "prematch", 0.0, routes=["course"])
            question = self._explain_question(explain)
            yield {"type": "route", "route": "course", "routes": ["exam", "course"], "complexity": "simple", "count": 1,
                   "chapter": None, "reason": d.reason, "router_s": 0.0}
        else:
            solve = self._solve_flag(routed_q)
            d = classify(routed_q, self.router_client)
            if solve:
                d = self._solve_route(d)
            yield {"type": "route", "route": d.route, "routes": list(d.routes), "complexity": d.complexity, "count": d.count,
                   "chapter": d.chapter, "reason": d.reason, "router_s": round(d.latency_s, 3)}
        who = user_id or key_id or getattr(rec, "key_id", None) or None   # 画像按人记，口径同问答记录
        prof = learner.summary(who) if explain is None else ""            # 没有记录时是空串，提示词逐字不变
        if explain is None and d.combined and not self._available(d.routes):
            yield from self._stream_combined(question, d, t0, rec, history, profile=prof)
            return
        if explain is None and d.route == "off_topic":
            # 超出知识库：不检索，直接用大模型自身知识作答。生成是流式的，逐字推给客户端，
            # 不能等整段写完再放进 done——通用问答也要几十秒，用户会以为卡住。
            yield from self._stream_general(routed_q, d, t0, rec, history)
            return
        if explain is None and (d.route != "course" or d.combined or self.course_ctx is None or self.course_llm is None):
            out: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                                   "answer": None, "sources": [], "status": "ok", "routes": list(d.routes)}
            trace: dict[str, Any] = {"router": d.to_dict()}
            try:
                if d.combined:
                    self._combined(routed_q, d, out, trace, history, profile=prof)   # 其中某个库未加载：返回 unavailable
                elif d.route == "policy":
                    self._policy(routed_q, out, trace, history, profile=prof)    # 单库政策：不流式，结果放进 done
                elif d.route == "course":
                    self._course(routed_q, out, trace, solve=solve, history=history,
                                 profile=prof)        # 课程库未加载：返回 unavailable
                elif d.route == "exam":
                    self._exam(routed_q, out, trace)
                elif d.route == "case":
                    self._case(routed_q, d, out, trace, history, profile=prof)       # 单库判例：不流式，结果放进 done
                elif d.route == "quote":
                    self._quote(routed_q, d, out, trace)      # 单库名言：不流式，结果放进 done
                elif d.route == "meta":
                    self._meta(out, trace, routed_q, {"user_id": user_id, "session_id": session_id,
                                                      "key_id": key_id or getattr(rec, "key_id", None)},
                               history=history)
                else:
                    self._thesis(routed_q, top_k, d.complexity, out, trace, history, profile=prof)
                self._note_learning(who, out, d)
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
                    # 检索与作答用补全后的独立问题（与非流式一致）；讲解时 question 已是程序拼好的“题干 + 参考答案”
                    q.put(("result", ask(question if explain is not None else routed_q, ctx=self.course_ctx,
                                         llm=self.course_llm, on_event=on_event, solve=solve, history=history,
                                         profile=prof)))
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
                    if explain is not None:
                        trace["prematch"] = explain["trace"]["prematch"]
                    trace["total_s"] = round(time.perf_counter() - t0, 4)
                    ans = res.get("answer")
                    if ans:
                        status, text, sources = "ok", ans.get("text"), ans.get("sources", [])
                    else:
                        status, text, sources = "insufficient", config.ADAPTIVE_INSUFFICIENT_TEXT, []
                    routes = ["exam", "course"] if explain is not None else [d.route]
                    extra = {"routes": routes, "exam": explain["exam"], "explain": True} if explain is not None else {}
                    yield {"type": "done", "route": d.route, "complexity": d.complexity, "status": status,
                           "answer": text, "sources": sources,
                           "practice": [] if explain is not None else self._practice(res.get("selected_concepts") or []),
                           "trace": trace, **extra}
                    got = self._followups(orig if explain is not None else question,
                                          {"route": d.route, "status": status, "answer": text, "exam": None})
                    if got:
                        yield {"type": "followups", "items": got}
                    rec.pending_out = {"route": d.route, "routes": routes, "complexity": d.complexity,
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
    def _case(self, question, d, out, trace, history: list[dict] | None = None, profile: str = "") -> None:
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
        ans = answer_cases(question, got["hits"], self.lib_llm, model=self.answer_model, history=history,
                           profile=profile)
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

    # ---------------------------------------------------------------- 通用作答（超出知识库，不检索）
    def _general_client(self):
        """通用作答用的客户端：复用路由的胜算云配置（绝不用论文库的 key）；未配置返回 None。

        懒加载 + 缓存：只有真的问到知识库以外的问题才建连接，正常问答不受影响。
        """
        if getattr(self, "_general_llm", _UNSET) is not _UNSET:
            return self._general_llm
        self._general_llm = None
        try:
            from .course.llm import OpenAICompatClient
            self._general_llm = OpenAICompatClient(
                config.ROUTER_API_BASE, config.ROUTER_API_KEY,
                timeout=config.GENERAL_TIMEOUT, reasoning={"general": config.GENERAL_REASONING})
        except Exception as e:  # noqa: BLE001 - 未配置 / 依赖缺失：退回拒答文案
            import logging
            logging.getLogger(__name__).info("通用作答模型未配置（%s），超出知识库时仍按拒答处理", type(e).__name__)
        return self._general_llm

    def _general(self, question: str, out: dict, trace: dict, history: list[dict] | None = None,
                 on_delta=None) -> None:
        """超出知识库范围：不检索，直接用大模型自身知识答，并明确标注「非知识库内容」。

        提示词禁止编造页码 / 条文号 / 案号，避免把通用回答伪装成有据可查的库内答案。
        模型未配置时退回原来的拒答文案，行为与改动前一致。
        """
        client = self._general_client()
        if client is None:
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT, sources=[])
            trace["general"] = {"configured": False}
            return
        msgs = [{"role": "system", "content": config.GENERAL_SYSTEM}]
        for m in (history or [])[-6:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": str(m["content"])[:500]})
        msgs.append({"role": "user", "content": question})
        t0 = time.perf_counter()
        try:
            if on_delta is not None and hasattr(client, "chat_stream"):
                res = client.chat_stream("general", msgs, model=config.GENERAL_MODEL, on_delta=on_delta)
                text = res.content
            else:
                text = client.chat("general", msgs, model=config.GENERAL_MODEL).content
        except Exception as e:  # noqa: BLE001 - 通用作答失败不该 500，退回拒答并记类型名
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT, sources=[])
            trace["general"] = {"configured": True, "error": type(e).__name__}
            return
        text = (text or "").strip()
        if not text:
            out.update(status="refused", answer=config.ROUTER_REFUSAL_TEXT, sources=[])
            trace["general"] = {"configured": True, "error": "EmptyAnswer"}
            return
        out.update(answer=text, sources=[], status="ok", general=True)
        trace["general"] = {"configured": True, "model": config.GENERAL_MODEL, "llm_calls": 1,
                            "answer_s": round(time.perf_counter() - t0, 4)}

    # ---------------------------------------------------------------- 助手自身（不检索、不调用模型）
    def _meta(self, out: dict, trace: dict, question: str = "", owner: dict | None = None,
              history: list[dict] | None = None) -> None:
        """「你能做什么 / 知识库有哪些内容」：按实际加载的库拼能力清单。

        不检索、不调用任何模型：这类问题问的是系统本身，检索出来的教材片段只会答偏。
        某个库没加载（数据缺失 / key 未配置）时如实不列，避免承诺做不到的能力。
        按问法分六种回答（都不甩长清单，除非真的问能力）：
        - 纯打招呼（「你好」）→ 一句欢迎语；
        - 确认 / 收尾（「好的 了解了」「谢谢」）→ 一句短的，并按上文给一个下一步（没有上文就不给）；
        - 问刚才怎么答的（「你的回答链路是怎样的」）→ 讲本轮实际路径，不背能力清单；
        - 问我是谁（「我是谁」「你记得我是谁」）→ 回学生档案，见 _whoami；
        - 问之前问过什么 → 按问答记录列，见 _recall；
        - 问能力 / 知识库 → 完整清单。
        owner 是本次请求的调用方身份（user_id / session_id / key_id），按参数传，不放实例上——
        服务实例是所有请求共用的，放实例上会让并发的两个人读到对方的身份。
        history 是本轮之前的对话：用来判断「这句确认是在确认什么」，只影响文案，不影响检索。
        """
        names = {"thesis": "论文", "course": "课程", "exam": "试卷", "policy": "政策", "case": "判例", "quote": "名言"}
        ready = [r for r, *_ in META_SCOPES if self._route_ready(r)]
        q = (question or "").strip()
        if looks_whoami(q):
            self._whoami(out, trace, owner)
            return
        if looks_recall(q):
            self._recall(out, trace, owner)
            return
        if looks_ack(q):
            out.update(answer=_ack_reply(history), sources=[])
            trace["meta"] = {"ack": True, "llm_calls": 0}
            return
        if looks_progress(q):
            out.update(answer=self._progress_reply(history), sources=[])
            trace["meta"] = {"progress": True, "llm_calls": 0}
            return
        if _GREETING_RE.match(q):
            if not ready:
                out.update(status="unavailable", answer="知识库尚未加载，暂时无法作答。", sources=[])
                trace["meta"] = {"scopes": [], "greeting": True}
                return
            out.update(answer=META_GREETING, sources=[])
            trace["meta"] = {"scopes": ready, "greeting": True, "llm_calls": 0}
            return
        lines = [META_INTRO, "我能回答以下范围内的内容："]
        shown: list[str] = []
        for i, (route, title, detail) in enumerate(META_SCOPES, 1):
            if not self._route_ready(route):
                continue                      # 没加载的库不列，避免让学生问了才发现不可用
            shown.append(names[route])
            lines.append(f"{i}. {title}：{detail}。")
        if not shown:
            out.update(status="unavailable", answer="知识库尚未加载，暂时无法作答。", sources=[])
            trace["meta"] = {"scopes": []}
            return
        body = "\n".join(lines)                                  # 列举项之间用单换行，读起来紧凑
        tail = META_TAIL + "\n\n您想了解哪方面？可以直接提问，例如「什么是数据确权」「数据安全法第二十一条规定了什么」。"
        out.update(answer=body + "\n\n" + tail, sources=[])
        trace["meta"] = {"scopes": shown, "llm_calls": 0}

    def _recall(self, out: dict, trace: dict, owner: dict | None = None) -> None:
        """「我之前都问过你哪些内容」：按问答记录（rag.qa_store）列出这个人的提问，零检索、零模型调用。

        user_id 来自请求（uuid / X-User-Uuid / user / X-User-Id），取不到就按本次请求的 key 指纹退回
        anon-<key>——与写入时的口径一致，没有标识的旧记录也归在 anon 里。
        session_id 有值时只列当前会话（「我们刚才聊了什么」），没有值时列这个人的全部提问。
        """
        from . import qa_store

        meta = owner or {}
        user_id = qa_store.resolve_user_id(meta.get("user_id"), meta.get("key_id"))
        session_id = qa_store.clean_id(meta.get("session_id"))
        got = qa_store.recent_questions(user_id, session_id=session_id, limit=RECALL_MAX)
        scope = "本次对话" if session_id else "此前"
        if not got["items"]:
            out.update(answer=f"{scope}还没有提问记录。直接问我就行，例如「什么是数据确权」。", sources=[])
            trace["meta"] = {"recall": {"user_id": user_id, "session_id": session_id, "count": 0, "llm_calls": 0}}
            return
        total = got["total"]
        lines = [f"{scope}一共问过 {total} 个问题" + (
            f"，列出最近 {len(got['items'])} 个：" if total > len(got["items"]) else "：")]
        for i, it in enumerate(got["items"], 1):
            q = " ".join(str(it["question"]).split())                  # 题干里可能有换行
            lines.append(f"{i}. {q if len(q) <= 40 else q[:40] + '…'}（{it['created_at'][:10]}）")
        lines.append("想接着某个问题继续问，把问题再发一遍就行。")
        out.update(answer="\n".join(lines), sources=[], followups=[])
        trace["meta"] = {"recall": {"user_id": user_id, "session_id": session_id,
                                    "count": len(got["items"]), "total": total, "llm_calls": 0}}

    def _whoami(self, out: dict, trace: dict, owner: dict | None = None) -> None:
        """「我是谁」：回系统记住的这个人，不是助手自我介绍。

        只念已经记下的事实：自述身份、问过的主题、做过的题（判分记录）。一条都没有就如实说
        「还没记下你的信息」，并说明怎么让我记住——不编一个身份，也不改答成「我是 AI 李老师」。
        零检索、零模型调用。
        """
        from . import qa_store

        meta = owner or {}
        user_id = qa_store.resolve_user_id(meta.get("user_id"), meta.get("key_id"))
        who = learner.identity(user_id)
        line = learner._identity_line(who)
        st = learner.state(user_id)
        names = [c["concept"] for c in st["concepts"]][:3]     # 只念最近几个主题，不是你问过的全部
        scores = st["scores"]
        weak = learner.weaknesses(user_id)

        lines: list[str] = []
        if line:
            lines.append(f"我记得的你是：{line}。")
            title = (who.get("title") or "").strip()
            if title:
                lines.append(f"你在做的论文题目是：{title}。")
        else:
            lines.append("我这边还没有记下你的身份信息。")
        if names:
            lines.append("你之前问过：" + "、".join(names) + "。")
        if scores:
            lines.append(f"你交过 {len(scores)} 道题，其中"
                         + (("还没答对的有：" + "、".join(weak) + "。") if weak else "都答对了。"))
        if not line:
            lines.append("如果想让我按你的专业和阶段来答，可以直接说一句，比如「我是编辑出版专业的本科生，"
                         "论文在选题阶段」，我记下来之后每一轮都会用上。")
        if len(lines) == 1:
            lines.append("你也可以问我之前问过什么，我按记录给你列。")
        out.update(answer="\n".join(lines), sources=[], followups=[])
        trace["meta"] = {"whoami": {"user_id": user_id, "identity": who, "concepts": names,
                                    "scores": len(scores), "llm_calls": 0}}

    def _progress_reply(self, history: list[dict] | None = None) -> str:
        """「你的回答链路是怎样的」：讲这一轮实际是怎么走的，不背能力清单。

        用上一轮实际命中的 route 说具体路径；没有上文就说通用路径。零检索、零调用。
        """
        prev = _last_route(history)
        head = "你这一轮的问题是这样处理的：" if prev else "一般来说，问题是这样处理的："
        lines = [head,
                 "1. 先看是不是题库里的原题（带题干或指代第几道题）：命中就直接给题目与答案，不检索；",
                 "2. 否则用一次路由判断该由哪个库回答（论文 / 课程 / 政策 / 试题 / 判例 / 讲话 / 助手自身 / 库外）；",
                 "3. 再从对应库里检索并挑出材料，只依据材料作答，正文标注来源；",
                 "4. 材料不够就说资料不足，不硬编；库外问题用通用知识回答并标明不来自知识库。"]
        if prev:
            lines.append(f"刚才那一轮走的是「{_ROUTE_CN.get(prev, prev)}」这条路径。")
        return "\n".join(lines)

    def _route_ready(self, route: str) -> bool:
        """该库现在能不能真的作答（没加载 / 没配 key 都算不能）。"""
        if route == "course":
            return self.course_ctx is not None and self.course_llm is not None
        if route == "thesis":
            return self.adaptive is not None
        if route == "exam":
            return self.exam is not None
        if route == "policy":
            return self.policy_retriever is not None
        if route == "case":
            return self.case_retriever is not None
        if route == "quote":
            return self.quote_retriever is not None
        return False

    # ---------------------------------------------------------------- 政策（单库，不流式）
    def _policy(self, question, out, trace, history: list[dict] | None = None, profile: str = "") -> None:
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
        ans = answer_policy(question, got["hits"], self.lib_llm, model=self.answer_model, history=history,
                            profile=profile)
        trace["policy"].update(answer_s=round(time.perf_counter() - t1, 4), answer_calls=1, removed_quotes=ans.removed_quotes)
        out.update(answer=ans.text, sources=ans.sources)

    # ---------------------------------------------------------------- 组合（各库并行取材，一次作答）
    def _combined(self, question, d, out, trace, history: list[dict] | None = None, profile: str = "") -> None:
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
                              history=history, profile=profile)
        trace["combined"].update(answer_s=round(time.perf_counter() - t1, 4), removed_quotes=ans.removed_quotes,
                                 missing=m.empty, sections=m.sections)
        out.update(answer=ans.text, sources=ans.sources)

    def _stream_general(self, question, d, t0, rec, history: list[dict] | None = None) -> Iterator[dict[str, Any]]:
        """超出知识库的流式作答：逐字推 delta，最后 done 带完整正文。

        与课程路径同样用后台线程 + 队列：LLM 客户端是同步的，直接在生成器里等会占住线程池。
        模型未配置或调用失败时不 500，退回拒答文案（done 里 status=refused）。
        """
        q: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        _END = object()
        parts: list[str] = []

        def emit(ev: dict) -> None:
            if cancelled.is_set():
                raise _Cancelled()
            q.put(ev)

        def on_delta(text: str) -> None:
            parts.append(text)
            emit({"type": "delta", "text": text})

        def run() -> None:
            try:
                with usage_log.activate(rec):
                    self._general(question, pending, trace_holder, history, on_delta=on_delta)
            except _Cancelled:
                pass
            except Exception as e:  # noqa: BLE001 - 只带类型名
                q.put(("error", type(e).__name__))
            finally:
                q.put(_END)

        pending: dict[str, Any] = {"route": d.route, "complexity": d.complexity, "reason": d.reason,
                                   "answer": None, "sources": [], "status": "ok", "routes": list(d.routes)}
        trace_holder: dict[str, Any] = {}
        threading.Thread(target=usage_log.bind(run), name="auto-ask-general", daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    break
                if isinstance(item, tuple):
                    if item[0] == "error":
                        rec.finish(error=str(item[1]))
                        yield {"type": "error", "error": item[1]}
                        continue
                if item.get("type") == "delta":
                    rec.first_delta()
                yield item
        finally:
            cancelled.set()
        trace = {"router": d.to_dict(), "general": trace_holder.get("general") or {},
                 "total_s": round(time.perf_counter() - t0, 4)}
        answer = pending.get("answer") or "".join(parts)
        status = pending.get("status") or "ok"
        yield {"type": "done", "route": d.route, "routes": list(d.routes), "complexity": d.complexity,
               "status": status, "answer": answer, "sources": [], "general": status == "ok", "trace": trace}
        if status == "ok":
            got = self._followups(question, {"route": d.route, "status": "ok", "answer": answer, "exam": None})
            if got:
                yield {"type": "followups", "items": got}
                pending["followups"] = got
        rec.pending_out = {**pending, "answer": answer, "status": status, "sources": [], "trace": trace}

    def _stream_combined(self, question, d, t0, rec, history: list[dict] | None = None,
                         profile: str = "") -> Iterator[dict[str, Any]]:
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
                                          history=history, profile=profile,
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

    def _thesis(self, question, top_k, complexity, out, trace, history: list[dict] | None = None,
                profile: str = "") -> None:
        if self.adaptive is None:
            out.update(status="unavailable", answer="论文指导服务未加载。")
            return
        r = self.adaptive.run(question, top_k, complexity, history=history, profile=profile)
        trace["adaptive"] = r.pop("trace")
        out.update(status=r["status"], answer=r["answer"], sources=r["sources"])
        out["contexts"] = r["contexts"]

    def _course(self, question, out, trace, solve: bool = False, history: list[dict] | None = None,
                profile: str = "") -> None:
        if self.course_ctx is None or self.course_llm is None:
            out.update(status="unavailable", answer="课程问答服务未加载（需要教材/本体数据与 COURSE_LLM_* 配置）。")
            return
        from .course.pipeline import ask, result_to_json
        res = result_to_json(ask(question, ctx=self.course_ctx, llm=self.course_llm, solve=solve,
                                 history=history, profile=profile))
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
    def _prematch_with_history(self, question: str, routed_q: str, history) -> tuple[dict[str, Any] | None, bool]:
        """先看是否指代上一轮列出的某道题（“讲解一下第四道题”，纯程序取题干），再按改写后的问题预匹配。
        返回 (命中结果或 None, 是否由指代命中)。"""
        from .exam.solve import referenced_stem
        stem = referenced_stem(question, history) if history else None
        if stem:
            hit = self._prematch(stem, force=True)
            if hit is not None:
                return hit, True
        return self._prematch(routed_q), False

    def _explain_wanted(self, question: str, routed_q: str, by_ref: bool) -> bool:
        """指代某道题时默认讲解（除非只问答案）；直接贴题干时要出现“怎么理解 / 讲解”等说法才讲解。"""
        if by_ref:
            from .exam.solve import wants_answer_only
            return not wants_answer_only(question) and self._wants_explain(question, routed_q, "讲解")
        return self._wants_explain(question, routed_q)

    def _wants_explain(self, *questions: str) -> bool:
        """命中题库后学生要的是讲解（怎么理解 / 讲解 / 为什么……）而不只是答案；课程库不可用时退回只给答案。"""
        if self.course_ctx is None or self.course_llm is None:
            return False
        from .exam.solve import wants_explanation
        return any(wants_explanation(q) for q in questions if q)

    def _explain_question(self, solved: dict) -> str:
        """交给课程作答的问题：程序拼上题干、选项、参考答案与评分要点（讲解必须与参考答案一致）。"""
        qv = (solved.get("exam") or {}).get("questions") or [{}]
        v = qv[0]
        opts = v.get("options") or {}
        lines = [f"请从课程角度讲解这道{v.get('qtype_cn') or ''}题：{v.get('stem', '')}"]
        if opts:
            lines.append("选项：" + "；".join(f"{k}. {opts[k]}" for k in sorted(opts)))
        lines.append(f"参考答案：{v.get('answer', '')}")
        if v.get("analysis"):
            lines.append(f"评分要点与解析：{v['analysis']}")
        return "\n".join(lines)

    def _explain(self, orig: str, solved: dict, t0: float, history: list[dict] | None = None) -> dict[str, Any]:
        """非流式：题库命中 + 要讲解 → 课程讲解（选概念 + 筛选 + 作答，路由跳过），附上题目与参考答案。"""
        out: dict[str, Any] = {"route": "course", "routes": ["exam", "course"], "complexity": "simple",
                               "reason": "题库命中，按课程讲解", "answer": None, "sources": [], "status": "ok",
                               "count": 1, "chapter": None, "explain": True, "exam": solved["exam"]}
        trace: dict[str, Any] = {"prematch": solved["trace"]["prematch"]}
        self._course(self._explain_question(solved), out, trace, solve="explain", history=history)
        out["exam"] = solved["exam"]
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        out["trace"] = trace
        out["followups"] = self._followups(orig, out)
        return out

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

    def _prematch(self, question: str, force: bool = False) -> dict[str, Any] | None:
        """解题预匹配（在路由之前）：问题去掉外壳后题干 ≥12 字，与题库题干完全包含或相似度 ≥0.9 → 直接给题目 + 答案 + 解析。
        耗时 ms 级，写进 trace.prematch；命中时 0 次模型调用（连路由也不调）。没命中返回 None，走原来的路由。"""
        if self.exam is None:
            return None
        from .exam.solve import StemMatcher, is_prematch_candidate
        if not force and not is_prematch_candidate(question):      # force：题干由程序从上一轮取出，不必再判外壳
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
