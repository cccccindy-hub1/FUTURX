"""HybridRetriever 与 RAGService 的契约测试（用 stub embedder，不加载真模型）。"""
from __future__ import annotations

import chromadb
import pytest

from rag.retrieval import HybridRetriever, Reranker


DOCS = {
    "d1": "扎根理论的三级编码包括开放式编码、主轴编码和选择性编码。",
    "d2": "文献综述要围绕研究问题组织，避免简单罗列。",
    "d3": "开题报告需要明确研究问题、研究方法和预期贡献。",
    "d4": "调查问卷数据常用 spss、stata 等软件分析。",
}
IDS = list(DOCS)
TEXTS = [DOCS[i] for i in IDS]
METAS = [
    {"source": "a.pdf", "category": "006/研究方法", "section": "扎根理论", "filename": "a.pdf", "ext": ".pdf"},
    {"source": "b.pdf", "category": "002/写作", "section": "文献综述", "filename": "b.pdf", "ext": ".pdf"},
    {"source": "c.pdf", "category": "002/写作", "section": "开题", "filename": "c.pdf", "ext": ".pdf"},
    {"source": "d.pdf", "category": "006/研究方法", "section": "调查", "filename": "d.pdf", "ext": ".pdf"},
]


@pytest.fixture
def collection(isolated_dirs, stub_embedder):
    client = chromadb.PersistentClient(path=str(isolated_dirs["store"]))
    col = client.create_collection("test_col", metadata={"hnsw:space": "cosine"})
    embs = stub_embedder.encode(TEXTS, normalize_embeddings=True).tolist()
    col.add(ids=IDS, documents=TEXTS, metadatas=METAS, embeddings=embs)
    return col


def _embed_fn(stub_embedder):
    return lambda text: stub_embedder.encode(text, normalize_embeddings=True).tolist()


def test_dense_mode_returns_requested_count(collection, stub_embedder):
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="dense")
    res = r.search("扎根理论编码", top_k=2, collection_name="test_col")
    assert len(res.ids) == 2
    assert all(isinstance(d, float) for d in res.distances)


def test_bm25_mode_returns_hits_with_float_distances(collection, stub_embedder):
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="bm25")
    res = r.search("问卷 分析 软件", top_k=2, collection_name="test_col")
    assert res.ids, "BM25 应召回含关键词的片段"
    assert "d4" in res.ids
    assert all(isinstance(d, float) for d in res.distances), "BM25 无距离时必须回退为浮点"


def test_hybrid_mode_merges_both_signals(collection, stub_embedder):
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="hybrid")
    res = r.search("扎根理论 三级编码", top_k=3, collection_name="test_col")
    assert res.ids
    assert "dense" in res.debug and "bm25" in res.debug
    assert len(res.ids) <= 3


def test_where_filter_restricts_results(collection, stub_embedder):
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="dense")
    res = r.search("研究", top_k=4, where={"category": "002/写作"}, collection_name="test_col")
    assert res.ids
    assert all(m["category"] == "002/写作" for m in res.metadatas)


def test_as_chroma_shape_is_nested(collection, stub_embedder):
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="dense")
    res = r.search("编码", top_k=2, collection_name="test_col")
    chroma = res.as_chroma()
    for key in ("ids", "documents", "metadatas", "distances"):
        assert isinstance(chroma[key], list) and isinstance(chroma[key][0], list)
    assert len(chroma["documents"][0]) == len(res.ids)


def test_reranker_reorders_candidates(collection, stub_embedder, monkeypatch):
    """用假 CrossEncoder 验证重排会改变顺序，且不加载真模型。"""
    import rag.retrieval as ret

    class FakeCrossEncoder:
        def __init__(self, *_a, **_kw):
            pass

        def predict(self, pairs):
            # 把含「扎根」的对打高分
            return [10.0 if "扎根" in d else 0.1 for _, d in pairs]

    monkeypatch.setattr(ret, "Reranker", Reranker)  # 保持类不变
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="dense")
    rk = Reranker("fake-model")
    monkeypatch.setattr(rk, "_ensure", lambda: FakeCrossEncoder())
    r.attach_reranker(rk)

    res = r.search("随便的问题", top_k=2, fetch_k=4, collection_name="test_col")
    assert res.ids
    assert "扎根" in res.documents[0], "重排后最相关的片段应排第一"
    assert "rerank_scores" in res.debug


def test_missing_rank_bm25_degrades(monkeypatch, collection, stub_embedder):
    """rank_bm25 缺失时 bm25 模式应给出可读错误而非静默错误结果。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "rank_bm25":
            raise ImportError("no rank_bm25")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = HybridRetriever(collection, _embed_fn(stub_embedder), mode="bm25")
    with pytest.raises(ImportError):
        r.search("x", top_k=1, collection_name="test_col")
