"""检索 + 生成。

- RAGService：常驻服务，模型与向量库只加载一次（供 serve 反复查询，消除重载开销）。
- retrieve() / answer()：模块级便捷函数，每次调用新建服务（一次性 CLI 用）。

检索增强（混合检索 / 重排序 / 元数据过滤）由 rag.retrieval 提供，默认关闭，
`mode="dense"` 时行为与旧的纯向量检索完全一致。
"""
from __future__ import annotations

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

        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(STORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE) if LLM_API_KEY else None

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
        """父子分块：把命中子块替换为其父块全文，供生成阶段获得更完整上下文。"""
        if self._parents is None:
            self._parents = load_parents(COLLECTION_NAME)
        if not self._parents:
            return
        for k, meta in enumerate(result.metadatas):
            parent = self._parents.get(meta.get("parent_id", ""))
            if not parent:
                continue
            off, length = meta.get("parent_offset"), meta.get("parent_length")
            if isinstance(off, int) and isinstance(length, int):
                result.documents[k] = parent[off : off + length]
            else:
                result.documents[k] = parent

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        """保持 Chroma query() 的嵌套列表返回形状（兼容既有调用方）。"""
        return self.retrieve_result(query, top_k).as_chroma()

    def answer(self, query: str, top_k: int = TOP_K):
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
        if not self.llm:
            return None, sources, docs

        ctx = "\n\n".join(
            f"[来源{i + 1}: {m.get('source', '')}]（{m.get('section', '')}）\n{d}"
            for i, (d, m) in enumerate(zip(docs, metas))
        )
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{query}"},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content, sources, docs


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    return RAGService().retrieve(query, top_k)


def answer(query: str, top_k: int = TOP_K):
    return RAGService().answer(query, top_k)
