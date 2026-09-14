"""检索 + 生成。

- 未设置 LLM_API_KEY 时，answer() 返回 (None, sources, docs)，仅做检索。
- 设置后走 OpenAI 兼容协议调用大模型生成最终回答。
"""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    STORE_DIR,
    TOP_K,
)


def _load():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    model, collection = _load()
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=[q_emb], n_results=top_k)


def answer(query: str, top_k: int = TOP_K):
    res = retrieve(query, top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    sources = [
        {
            "source": m["source"],
            "section": m.get("section", ""),
            "distance": float(d),
        }
        for m, d in zip(metas, dists)
    ]

    if not LLM_API_KEY:
        return None, sources, docs

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    ctx = "\n\n".join(
        f"[来源{i + 1}: {m['source']}]（{m.get('section', '')}）\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    )
    system = (
        "你是数字治理科研团队的学术论文指导助手。"
        "请仅依据下面给出的资料片段回答，并引用[来源N]标注出处；"
        "若资料不足以回答，请明确说明不足。回答使用中文，条理清晰、准确。"
    )
    user = f"资料：\n{ctx}\n\n问题：{query}"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, sources, docs
