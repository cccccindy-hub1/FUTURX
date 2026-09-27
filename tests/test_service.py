"""RAGService 端到端契约测试：确保不破坏 retrieve/answer 的既有形状。"""
from __future__ import annotations

import sys
import types

import chromadb
import pytest

from rag.query import RAGService
from tests.test_query import DOCS, IDS, TEXTS, METAS


@pytest.fixture(autouse=True)
def fake_sentence_transformers(stub_embedder):
    """注入假的 sentence_transformers 模块，避免安装/加载真实 torch 模型。"""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda *a, **kw: stub_embedder
    module.CrossEncoder = lambda *a, **kw: None
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", module)
        yield


@pytest.fixture
def service(isolated_dirs, stub_embedder):
    """在隔离目录中建好同名 collection，RAGService 会自己连上去。"""
    client = chromadb.PersistentClient(path=str(isolated_dirs["store"]))
    try:
        client.delete_collection("nlp_degree_thesis")
    except Exception:
        pass
    col = client.create_collection("nlp_degree_thesis", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=IDS,
        documents=TEXTS,
        metadatas=METAS,
        embeddings=stub_embedder.encode(TEXTS, normalize_embeddings=True).tolist(),
    )
    return col


def test_retrieve_keeps_nested_chroma_shape(service):
    svc = RAGService(mode="dense", rerank=False)
    res = svc.retrieve("扎根理论", top_k=2)
    for key in ("ids", "documents", "metadatas", "distances"):
        assert key in res
        assert isinstance(res[key], list) and isinstance(res[key][0], list)
    assert len(res["documents"][0]) == 2
    assert all(isinstance(d, float) for d in res["distances"][0])


def test_answer_returns_three_tuple_and_uses_llm(service, mock_llm):
    svc = RAGService(mode="dense", rerank=False)
    ans, sources, docs = svc.answer("扎根理论的三级编码", top_k=2)
    assert ans and isinstance(ans, str)
    assert docs and all(isinstance(d, str) for d in docs)
    for s in sources:
        assert set(("source", "section", "distance")) <= set(s)
        assert isinstance(s["distance"], float)


def test_answer_without_llm_returns_none(service, monkeypatch):
    monkeypatch.setattr("rag.query.LLM_API_KEY", "")
    svc = RAGService(mode="dense", rerank=False)
    ans, sources, docs = svc.answer("开题报告", top_k=2)
    assert ans is None
    assert docs


def test_hybrid_mode_still_satisfies_contract(service, mock_llm):
    svc = RAGService(mode="hybrid", rerank=False)
    ans, sources, docs = svc.answer("问卷数据分析", top_k=3)
    assert ans
    assert all(isinstance(s["distance"], float) for s in sources)


def test_where_filter_applied_in_service(service):
    svc = RAGService(mode="dense", rerank=False, where={"category": "002/写作"})
    res = svc.retrieve("写作", top_k=4)
    assert res["metadatas"][0], "过滤后仍应有命中"
    assert all(m["category"] == "002/写作" for m in res["metadatas"][0])
