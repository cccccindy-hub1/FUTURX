"""Contextual Retrieval（Anthropic, 2024）：建库时让 LLM 为每个 chunk 写一两句「它在整篇文档里讲什么」，
拼在 chunk 前面再做向量化与 BM25（见 chunk.contextual_text）；入库和返回给 LLM 的仍是原文。

- 默认关闭（CONTEXTUAL_ENABLED=false）。开启后每个新 chunk 一次 LLM 调用，会打印每个文件的调用量；
- 结果按 (模型, 文档哈希, chunk 文本) 缓存到 CACHE_DIR/contextual_cache.json，内容不变的重建不再调用；
- 文档超过 CONTEXTUAL_DOC_MAX_CHARS 时，只取 chunk 所在位置附近的窗口；
- 文档放在 prompt 最前面：同一文档的多个 chunk 共享前缀，可命中服务端前缀缓存（DeepSeek 自动磁盘缓存）；
- 单个 chunk 失败只告警、上下文留空，不影响建库。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config

logger = logging.getLogger(__name__)

PROMPT = """<document>
{document}
</document>
下面是上述文档中的一个片段：
<chunk>
{chunk}
</chunk>
请用一两句话（不超过 80 字）说明这个片段在全文中的位置和主题（如所属章节、讨论的对象），用于提升检索效果。
只输出这段说明，不要复述片段内容。"""

_lock = threading.Lock()
_cache: dict[str, str] | None = None
_client = None


def _cache_path():
    return config.CACHE_DIR / "contextual_cache.json"


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_cache_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(json.dumps(_cache or {}, ensure_ascii=False), encoding="utf-8")


def _key(model: str, doc: str, chunk: str) -> str:
    doc_hash = hashlib.sha1(doc.encode("utf-8")).hexdigest()
    return hashlib.sha1(f"{model}\n{doc_hash}\n{chunk}".encode("utf-8")).hexdigest()


def doc_window(doc: str, chunk: str, max_chars: int) -> str:
    """文档不超过 max_chars 时原样返回；否则取以 chunk 为中心的 max_chars 字窗口。"""
    if len(doc) <= max_chars:
        return doc
    pos = doc.find(chunk[:40].strip())
    center = pos + len(chunk) // 2 if pos >= 0 else len(doc) // 2
    start = max(0, min(center - max_chars // 2, len(doc) - max_chars))
    return doc[start:start + max_chars]


def _get_client():
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise SystemExit("CONTEXTUAL_ENABLED=true 需要配置 LLM_API_KEY（或关闭 Contextual Retrieval）")
        from openai import OpenAI
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_API_BASE)
    return _client


def contextualize(doc: str, chunk: str, client, model: str) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(
            document=doc_window(doc, chunk, config.CONTEXTUAL_DOC_MAX_CHARS), chunk=chunk)}],
    )
    return (resp.choices[0].message.content or "").strip()


def add_contexts(doc_text: str, rows: list, client=None) -> dict:
    """给同一文档的 rows（(id, text, meta) 三元组）写入 meta["context"]；返回本次统计。"""
    model = config.CONTEXTUAL_MODEL
    cache = _load_cache()
    todo = [(r, _key(model, doc_text, r[1])) for r in rows]
    missing = [(r, k) for r, k in todo if k not in cache]
    stats = {"chunks": len(rows), "cached": len(rows) - len(missing), "llm_calls": len(missing), "failed": 0}
    if missing:
        client = client or _get_client()

        def run(item):
            row, key = item
            try:
                ctx = contextualize(doc_text, row[1], client, model)
            except Exception as e:  # noqa: BLE001 - 单个 chunk 失败不影响建库
                logger.warning("contextual 失败（%s）：%s", row[0], type(e).__name__)
                return key, None
            return key, ctx

        with ThreadPoolExecutor(max(1, config.CONTEXTUAL_WORKERS)) as ex:
            for key, ctx in ex.map(run, missing):
                if ctx is None:
                    stats["failed"] += 1
                    continue
                with _lock:
                    cache[key] = ctx
        _save_cache()
    for row, key in todo:
        row[2]["context"] = cache.get(key, "")
    logger.info("contextual：%d chunks，缓存 %d，LLM 调用 %d，失败 %d",
                stats["chunks"], stats["cached"], stats["llm_calls"], stats["failed"])
    return stats
