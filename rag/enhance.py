"""查询增强：查询改写与 HyDE（Hypothetical Document Embeddings）。

两者都依赖 LLM，因此默认关闭，且在没有 LLM 客户端时**原样返回**查询——
调用方无需分支，行为与不启用时一致。

- `rewrite_query`：把口语化/含歧义的问题改写成更适合 BM25/向量检索的中文关键词式查询。
- `hyde`：让模型先写一段「假设性的答案」，用这段文字去检索（Embeddings 更贴近答案区）。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的问题改写成一段更适合在学术资料库中做关键词检索的中文查询："
    "补全省略的主语与限定词，展开同义/近义术语（如「扎根」→「扎根理论 三级编码」），"
    "保留专有名词。只输出改写后的查询本身，不要解释、不要引号、不要换行。控制在 60 字以内。"
)

_HYDE_SYSTEM = (
    "你是数字治理方向的研究方法参考书。针对用户的问题，写一段约 150 字的、"
    "看起来像标准答案的中文段落（可以使用领域术语和分点式表述）。"
    "只输出这段假设性答案本身，不要前缀、不要解释。"
)


def rewrite_query(query: str, llm, model: str) -> str:
    """返回改写后的查询；llm 为 None 或调用失败时返回原查询。"""
    if llm is None or not query.strip():
        return query
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or query
    except Exception as e:  # 增强失败不能影响主流程
        logger.warning("查询改写失败，回退原查询：%s", e)
        return query


def hyde(query: str, llm, model: str) -> str:
    """返回假设性答案文本；llm 为 None 或调用失败时返回空串（调用方应回退原查询）。"""
    if llm is None or not query.strip():
        return ""
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("HyDE 生成失败，回退原查询：%s", e)
        return ""


_CONTEXTUALIZE_SYSTEM = (
    "你是多轮对话的追问改写器。根据对话历史，把用户最后一句问话改写成不依赖上下文、"
    "可以单独拿去检索的完整问题：补全省略的主语、指代（「第二种」「这个」「它」）和限定条件，"
    "保留专有名词与关键术语。只输出改写后的问句本身，不要解释、不要引号、不要换行，"
    "不要回答问题。若最后一句本身已经完整，原样输出。"
)


# 出现这些词时，问题很可能依赖上一轮（指代、序数、继续追问）。
# 只收明确的指代/承接词：「该」「其」「此」单独出现太常见（应该、尤其、因此），会误判。
_ANAPHORA = ("这", "那", "它们", "第二种", "第一种", "第三种", "上述", "前面", "上面",
             "再", "继续", "换成", "同样", "具体怎么", "怎么做", "为什么呢")


def needs_contextualize(question: str, max_self_contained_chars: int = 20) -> bool:
    """判断本轮问题是否值得为检索改写：很短的问题或含指代/追问词才改。

    长问题通常自带完整语境，改写既无收益又多一次模型调用（会直接拉长首字时间）。
    阈值放宽到 20 字：像「三级编码是什么」这种短问句本身完整，改写基本原样返回，
    多花一次小调用但不会改变检索结果，代价可接受。
    """
    q = question.strip()
    if len(q) <= max_self_contained_chars:
        return True
    return any(w in q for w in _ANAPHORA)


def _complete(llm, model: str, system: str, user: str, kwargs: dict) -> str:
    """调一次模型拿文本，兼容两种客户端。

    论文库用 OpenAI SDK（`llm.chat.completions.create`）；课程 / 判例 / 政策库用
    `OpenAICompatClient`（`llm.chat(step, messages, model=...)`，返回 LLMResult.content）。
    以前只认第一种，统一入口的追问改写因此静默失败退回原问题（路由把追问判成 off_topic）。
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    completions = getattr(getattr(llm, "chat", None), "completions", None)
    if completions is not None:
        resp = completions.create(model=model, messages=messages, **kwargs)
        return (resp.choices[0].message.content or "").strip()
    resp = llm.chat("contextualize", messages, model=model)
    return (getattr(resp, "content", "") or "").strip()


def contextualize_query(question: str, history: list[tuple[str, str]], llm, model: str,
                        *, reasoning_effort: str | None = None, timeout: float | None = None) -> str:
    """把多轮追问改写成独立问题（检索与路由用）；无法改写时返回原问题。

    history 为 [(role, content)] 的之前轮次（不含本轮问题）。改写只用于检索与路由判断：
    交给模型的对话仍是原文，避免改写偏差污染回答。
    """
    if llm is None or not question.strip() or not history:
        return question
    transcript = "\n".join(
        f"{'学生' if role == 'user' else '老师'}：{text}" for role, text in history if text.strip()
    )
    if not transcript:
        return question
    kwargs: dict = {"temperature": 0.0}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if timeout:
        kwargs["timeout"] = timeout
    try:
        out = _complete(llm, model, _CONTEXTUALIZE_SYSTEM,
                        f"对话历史：\n{transcript}\n\n最后一句问话：{question}", kwargs)
        out = out.strip().strip('"“”')
        return out or question
    except Exception as e:  # 改写失败不能影响主流程
        logger.warning("追问改写失败，回退原问题：%s", e)
        return question


def build_retrieval_query(query: str, llm, model: str, rewrite: bool = False, use_hyde: bool = False) -> str:
    """按开关组装最终的检索查询。

    HyDE 优先：命中时用假设答案去检索；否则在启用改写时用改写结果；都没有则原查询。
    """
    if use_hyde:
        text = hyde(query, llm, model)
        if text:
            return f"{query}\n{text}"
    if rewrite:
        return rewrite_query(query, llm, model)
    return query
