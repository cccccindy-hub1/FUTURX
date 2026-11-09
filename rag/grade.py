"""学生作答判分：把「交上来的答案」和题库参考答案对齐，用 rag.judge 打分，再回写学习者画像。

为什么要有这一层：以前判分器（rag/judge.py）只在离线评测里用，学生做题没有任何反馈，
画像也只能记「你问过什么」，说不了「你哪里不会」。这里补上产品链路缺的那一段：

    学生提交答案 → 取题库参考答案 → 判分（1–5 分 + correct）→ 回写 profile → 下一轮作答带上一句「还没答对」

只做能解释来源的事：
- 答案与参考答案都来自题库本身，判分用的是现成的 judge_answer（LLM 评审，失败降级词法）；
- 没有学生答案就不判、不写画像——不猜「他大概不会」；
- 判分失败（模型抖动、题库里没有这道题）如实返回错误码，不编一个分数出来。

模型分工沿用既有约定：判分走 lib_llm（胜算云），模型用 LIB_PICK_MODEL（判断类用小模型），
因为判分是「按标准答案核对要点」，不需要作答模型那样的长思考。
"""
from __future__ import annotations

import logging
from typing import Any

from .judge import judge_answer

log = logging.getLogger(__name__)

ANSWER_MAX_CHARS = 2000          # 学生作答长度上限（超过截断，避免把长文塞进判分提示）
GRADE_TEXT = {5: "完全答对", 4: "基本答对", 3: "部分答对", 2: "只沾边", 1: "答错"}


def _clean(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def grade(bank, question_id: str, answer: str, *, user_id: str | None = None,
          llm=None, model: str | None = None) -> dict:
    """判一道题。返回 {"ok", "question_id", "score", "correct", "reason", "method", ...}。

    ok=False 时带 error（not_found / empty_answer / no_reference），调用方据此给 HTTP 状态码，
    不把「没判成」伪装成「答错了」。
    """
    qid = str(question_id or "").strip()
    text = _clean(answer, ANSWER_MAX_CHARS)
    if bank is None:
        return {"ok": False, "error": "no_bank", "question_id": qid}
    if not text:
        return {"ok": False, "error": "empty_answer", "question_id": qid}
    item = getattr(bank, "by_id", {}).get(qid)
    if item is None:
        return {"ok": False, "error": "not_found", "question_id": qid}
    q = item.q
    ref = (q.answer or "").strip()
    if not ref:
        return {"ok": False, "error": "no_reference", "question_id": qid}

    model = model or _pick_model()
    res = judge_answer(_stem_with_options(q), text, ref, _ChatClient(llm, model) if llm is not None else None, model)
    if res.method == "lexical" and llm is None:
        # 没配判分模型：退回词法比对也能给出分数，但要说清楚这不是老师阅卷
        log.info("判分未配置模型，使用词法比对：%s", qid)
    out = {
        "ok": True,
        "question_id": qid,
        "chapter": q.chapter,
        "chapter_title": q.chapter_title,
        "qtype_cn": q.qtype_cn,
        "score": float(res.score),
        "correct": bool(res.correct),
        "reason": res.reason,
        "method": res.method,
        "verdict": GRADE_TEXT.get(int(round(res.score)), "已评分"),
        "reference": ref,
        "analysis": q.analysis,
        "concepts": list(item.concepts or []),
    }
    if user_id:
        _remember(user_id, item, res)
    return out


def _stem_with_options(q) -> str:
    """判分时给模型的题面：题干 + 选项（选择题没有选项就没法判断学生说的是不是某一块）。"""
    if q.options:
        opts = "；".join(f"{k}. {v}" for k, v in q.options.items())
        return f"{q.stem}\n选项：{opts}"
    return q.stem


def _pick_model() -> str:
    from .course.config import LIB_PICK_MODEL

    return LIB_PICK_MODEL


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_NS(message=_NS(content=content))]


class _ChatClient:
    """把课程库的 OpenAICompatClient 包成 rag.judge 认得的形状（llm.chat.completions.create）。

    判分器是照 OpenAI SDK 写的（chat.completions.create），课程 / 判例 / 政策库用的是自带的
    chat(step, messages, ...)；直接传后者会 AttributeError。这里只做一次适配，并把 step 标成 grade，
    让调用日志能把判分耗时与作答分开统计。
    """

    def __init__(self, inner, model: str):
        self._inner = inner
        self._model = model
        if inner is None:
            return

        def create(**kwargs):
            messages = kwargs.get("messages") or []
            res = inner.chat("grade", messages, model=kwargs.get("model") or self._model)
            return _Resp(res.content or "")

        self.chat = _NS(completions=_NS(create=create))


def _remember(user_id: str, item, res) -> None:
    """把这次判分写进画像。失败只记 warning：判分结果本身仍然正常返回。"""
    try:
        from . import profile as learner

        learner.notice_score(user_id, question_id=item.q.question_id, score=res.score, correct=res.correct,
                             concepts=list(item.concepts or []), chapter=item.q.chapter, method=res.method)
    except Exception as e:  # noqa: BLE001 - 画像只是加分项
        log.warning("判分回写画像失败（%s）", type(e).__name__)


__all__ = ["grade", "GRADE_TEXT", "ANSWER_MAX_CHARS"]
