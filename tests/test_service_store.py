"""RAGService 的向量库目录与集合名可传参：默认值与论文库完全一致，新库只用自己的目录。全部 mock。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import rag.query as q


class FakeCollection:
    def __init__(self, name):
        self.name = name


class FakeClient:
    opened: list = []                 # (path, 集合名)

    def __init__(self, path):
        self.path = path

    def get_collection(self, name):
        FakeClient.opened.append((str(self.path), name))
        return FakeCollection(name)


@pytest.fixture
def fake_env(monkeypatch, stub_embedder):
    FakeClient.opened = []
    chroma = ModuleType("chromadb")
    chroma.PersistentClient = FakeClient
    st = ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda *a, **k: stub_embedder
    st.CrossEncoder = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr(q, "LLM_API_KEY", "")
    return FakeClient


def test_default_store_and_collection_are_the_thesis_library(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "STORE_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(q, "COLLECTION_NAME", "nlp_degree_thesis")
    svc = q.RAGService(mode="dense", rerank=False)
    assert fake_env.opened == [(str(tmp_path / "chroma_db"), "nlp_degree_thesis")]            # 不传参 = 与以前完全一致
    assert svc.store_dir == tmp_path / "chroma_db" and svc.collection_name == "nlp_degree_thesis"


def test_explicit_store_and_collection_are_used_and_global_store_untouched(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "STORE_DIR", tmp_path / "chroma_db")
    case_dir = tmp_path / "case_store"
    svc = q.RAGService(mode="dense", rerank=False, store_dir=case_dir, collection_name="data_law_cases")
    assert fake_env.opened == [(str(case_dir), "data_law_cases")]
    assert all(str(tmp_path / "chroma_db") not in p for p, _ in fake_env.opened)              # 绝不打开论文库目录
    assert svc.collection.name == "data_law_cases"


def test_only_one_of_the_two_params_overrides_just_that_one(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "STORE_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(q, "COLLECTION_NAME", "nlp_degree_thesis")
    q.RAGService(mode="dense", rerank=False, collection_name="other")
    q.RAGService(mode="dense", rerank=False, store_dir=tmp_path / "x")
    assert fake_env.opened == [(str(tmp_path / "chroma_db"), "other"), (str(tmp_path / "x"), "nlp_degree_thesis")]


def test_retrieval_and_parent_lookup_use_the_instance_collection_name(fake_env, tmp_path, monkeypatch):
    svc = q.RAGService(mode="dense", rerank=False, store_dir=tmp_path / "quote_store", collection_name="xi_digital_quotes")
    seen = {}

    def fake_search(query, top_k, fetch_k, where, collection_name):
        seen["collection_name"] = collection_name
        return SimpleNamespace(ids=[], documents=[], metadatas=[], distances=[], scores=[], debug={})
    monkeypatch.setattr(svc.retriever, "search", fake_search)
    svc.retrieve_result("问题", 3)
    assert seen["collection_name"] == "xi_digital_quotes"                                       # BM25 缓存文件名据此区分，不会覆盖论文库的
    loaded = []
    monkeypatch.setattr(q, "load_parents", lambda name: (loaded.append(name), {})[1])
    svc.expand_context = True
    svc._expand(SimpleNamespace(metadatas=[{}], ids=["a"], documents=["d"], distances=[0.1], scores=[0.1]))
    assert loaded == ["xi_digital_quotes"]


def test_two_services_have_independent_stores(fake_env, tmp_path):
    a = q.RAGService(mode="dense", rerank=False, store_dir=tmp_path / "case_store", collection_name="data_law_cases")
    b = q.RAGService(mode="dense", rerank=False, store_dir=tmp_path / "quote_store", collection_name="xi_digital_quotes")
    assert a.collection.name == "data_law_cases" and b.collection.name == "xi_digital_quotes"
    assert a.store_dir != b.store_dir and a.retriever.collection is not b.retriever.collection


def test_new_stores_are_never_the_thesis_store_by_default():
    from rag import config as base
    from rag.cases import config as cc
    from rag.quotes import config as qc
    for d in (cc.CASE_STORE_DIR, qc.QUOTE_STORE_DIR):
        assert d.name in ("case_store", "quote_store") and d != base.STORE_DIR and base.STORE_DIR not in d.parents
    assert cc.CASE_COLLECTION == "data_law_cases" and qc.QUOTE_COLLECTION == "xi_digital_quotes"
    assert cc.CASE_COLLECTION != base.COLLECTION_NAME and qc.QUOTE_COLLECTION != base.COLLECTION_NAME


def test_instance_built_without_init_falls_back_to_thesis_collection(monkeypatch):
    """测试脚手架常用 object.__new__ 绕过 __init__：此时 collection_name / store_dir 必须仍落到论文库的全局值。"""
    svc = object.__new__(q.RAGService)
    assert svc.collection_name == q.COLLECTION_NAME and svc.store_dir == q.STORE_DIR
