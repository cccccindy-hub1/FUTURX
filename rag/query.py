"""检索 + 生成。

- RAGService：常驻服务，模型与向量库只加载一次（供 serve 反复查询，消除重载开销）。
- retrieve() / answer()：模块级便捷函数，每次调用新建服务（一次性 CLI 用）。

检索增强（混合检索 / 重排序 / 元数据过滤）由 rag.retrieval 提供，默认关闭，
`mode="dense"` 时行为与旧的纯向量检索完全一致。
"""
from __future__ import annotations

import time

from openai import OpenAI

from .config import (
    COLLECTION_NAME,
    CONTEXT_EXPAND_ENABLED,
    EMBED_MODEL,
    FETCH_K,
    HYDE_ENABLED,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TIMEOUT,
    MAX_DISTANCE,
    QUERY_REWRITE_ENABLED,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_MODE,
    REWRITE_MODEL,
    RRF_K,
    STORE_DIR,
    TOP_K,
)
from .enhance import build_retrieval_query
from .retrieval import HybridRetriever, Reranker, load_parents

_SYSTEM_PROMPT = (
    "你是数字治理科研团队的学术论文指导助手。"
    "请仅依据下面给出的资料片段回答，并引用[来源N]标注出处；"
    "若资料不足以回答，请明确说明不足。回答使用中文，条理清晰、准确。"
)


class RAGService:
    def __init__(
        self,
        mode: str = RETRIEVAL_MODE,
        rerank: bool = RERANK_ENABLED,
        where: dict | None = None,
        fetch_k: int = FETCH_K,
        rewrite: bool = QUERY_REWRITE_ENABLED,
        use_hyde: bool = HYDE_ENABLED,
        expand_context: bool = CONTEXT_EXPAND_ENABLED,
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        # 优先走本地缓存：直连 HF 校验版本/权重会多等数秒，离线命中时没必要。
        # 缓存缺失再回退到联网下载，保证首次部署仍可用。
        try:
            self.model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(STORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.llm = (
            OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE, timeout=LLM_TIMEOUT, max_retries=1)
            if LLM_API_KEY
            else None
        )

        self.mode = mode
        self.where = where
        self.fetch_k = fetch_k
        self.rewrite = rewrite
        self.use_hyde = use_hyde
        self.expand_context = expand_context
        self._parents: dict | None = None
        self.retriever = HybridRetriever(
            self.collection, self._embed, mode=mode, rrf_k=RRF_K
        )
        if rerank:
            self.retriever.attach_reranker(Reranker(RERANK_MODEL))

    def _embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def _llm_kwargs(self, messages: list[dict]) -> dict:
        """生成请求的公共参数。

        推理模型默认会先产出「思考」token，而思考 token 不计入 content 增量，
        客户端只能干等（实测首字节 32s+）。这里按配置关掉思考，首字节降到 ~2.7s。
        """
        kwargs: dict = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        if LLM_REASONING_EFFORT:
            kwargs["reasoning_effort"] = LLM_REASONING_EFFORT
        return kwargs

    def _retrieval_query(self, query: str) -> str:
        """按开关做查询改写 / HyDE；无 LLM 时原样返回。"""
        if not (self.rewrite or self.use_hyde):
            return query
        return build_retrieval_query(
            query, self.llm, REWRITE_MODEL, rewrite=self.rewrite, use_hyde=self.use_hyde
        )

    def retrieve_result(self, query: str, top_k: int = TOP_K):
        """返回 RetrievalResult（含 debug 信息），供 answer 与评估使用。"""
        result = self.retriever.search(
            self._retrieval_query(query),
            top_k=top_k,
            fetch_k=max(self.fetch_k, top_k),
            where=self.where,
            collection_name=COLLECTION_NAME,
        )
        if self.expand_context:
            self._expand(result)
        return result

    def _expand(self, result) -> None:
        """父子分块：把命中子块替换为其父块全文，供生成阶段获得更完整上下文。

        同一父块下的多个子块只保留排名最靠前的那一条，避免同一段上下文重复送给 LLM。
        找不到父块（未用父子分块建库）时保留子块原文。
        """
        if self._parents is None:
            self._parents = load_parents(COLLECTION_NAME)
        if not self._parents:
            return
        seen: set[str] = set()
        keep: list[int] = []
        for k, meta in enumerate(result.metadatas):
            pid = meta.get("parent_id", "")
            parent = self._parents.get(pid)
            if parent:
                if pid in seen:
                    continue
                seen.add(pid)
                result.documents[k] = parent
            keep.append(k)
        n = len(result.metadatas)
        for name in ("ids", "documents", "metadatas", "distances", "scores"):
            vals = getattr(result, name)
            if len(vals) == n:
                setattr(result, name, [vals[k] for k in keep])

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        """保持 Chroma query() 的嵌套列表返回形状（兼容既有调用方）。"""
        return self.retrieve_result(query, top_k).as_chroma()

    def prepare(self, query: str, top_k: int = TOP_K):
        """检索 + 组装上下文，返回 (contexts, sources, messages)。

        供 answer / 流式生成 / OpenAI 兼容接口共用，避免提示词与过滤逻辑分叉。
        """
        result = self.retrieve_result(query, top_k)

        docs = result.documents
        metas = result.metadatas
        dists = result.distances

        # 可选：按余弦距离过滤噪声片段（MAX_DISTANCE>0 时生效，但至少保留 1 条）
        if MAX_DISTANCE > 0:
            kept = [k for k, d in enumerate(dists) if d <= MAX_DISTANCE]
            if kept:
                docs = [docs[k] for k in kept]
                metas = [metas[k] for k in kept]
                dists = [dists[k] for k in kept]

        sources = [
            {"source": m.get("source", ""), "section": m.get("section", ""), "distance": float(d)}
            for m, d in zip(metas, dists)
        ]
        ctx = "\n\n".join(
            f"[来源{i + 1}: {m.get('source', '')}]（{m.get('section', '')}）\n{d}"
            for i, (d, m) in enumerate(zip(docs, metas))
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{query}"},
        ]
        return docs, sources, messages

    def answer(self, query: str, top_k: int = TOP_K):
        self.timing: dict = {
            "retrieval": 0.0,
            "llm_ttft": 0.0,
            "llm_generation": 0.0,
            "llm_total": 0.0,
        }

        t0 = time.perf_counter()
        docs, sources, messages = self.prepare(query, top_k)
        self.timing["retrieval"] = time.perf_counter() - t0
        if not self.llm:
            return None, sources, docs

        t_llm0 = time.perf_counter()
        stream = self.llm.chat.completions.create(**self._llm_kwargs(messages))
        parts: list[str] = []
        first_token_at: float | None = None
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(text)
        t_llm_end = time.perf_counter()

        self.timing["llm_ttft"] = (
            (first_token_at - t_llm0) if first_token_at is not None else (t_llm_end - t_llm0)
        )
        self.timing["llm_generation"] = (
            t_llm_end - (first_token_at if first_token_at is not None else t_llm0)
        )
        self.timing["llm_total"] = t_llm_end - t_llm0
        return "".join(parts), sources, docs

    def open_stream(self, query: str, top_k: int = TOP_K):
        """流式生成，返回 (contexts, sources, delta_iter)。

        delta_iter 逐段产出增量文本；无 LLM 时抛 RuntimeError（由调用方转成 HTTP 错误）。
        """
        docs, sources, messages = self.prepare(query, top_k)
        if not self.llm:
            raise RuntimeError("未配置 LLM_API_KEY，无法生成回答")

        def _deltas():
            stream = self.llm.chat.completions.create(**self._llm_kwargs(messages))
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    yield text

        return docs, sources, _deltas()


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    return RAGService().retrieve(query, top_k)


def answer(query: str, top_k: int = TOP_K):
    return RAGService().answer(query, top_k)
