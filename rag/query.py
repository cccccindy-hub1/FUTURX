"""检索 + 生成。

- RAGService：常驻服务，模型与向量库只加载一次（供 serve 反复查询，消除重载开销）。
- retrieve() / answer()：模块级便捷函数，每次调用新建服务（一次性 CLI 用）。
"""
from __future__ import annotations

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    STORE_DIR,
    TOP_K,
)

_SYSTEM_PROMPT = (
    "你是数字治理科研团队的学术论文指导助手。"
    "请仅依据下面给出的资料片段回答，并引用[来源N]标注出处；"
    "若资料不足以回答，请明确说明不足。回答使用中文，条理清晰、准确。"
)


class RAGService:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(STORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE) if LLM_API_KEY else None

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        q_emb = self.model.encode(query, normalize_embeddings=True).tolist()
        return self.collection.query(query_embeddings=[q_emb], n_results=top_k)

    def answer(self, query: str, top_k: int = TOP_K):
        res = self.retrieve(query, top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        sources = [
            {"source": m["source"], "section": m.get("section", ""), "distance": float(d)}
            for m, d in zip(metas, dists)
        ]
        if not self.llm:
            return None, sources, docs

        ctx = "\n\n".join(
            f"[来源{i + 1}: {m['source']}]（{m.get('section', '')}）\n{d}"
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
