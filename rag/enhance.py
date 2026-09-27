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
