"""分块 / 检索修复与 Contextual Retrieval 测试：全部离线（假 collection、假 reranker、假 LLM）。"""
from __future__ import annotations

import re

import pytest

import rag.contextual as contextual
import rag.index as index
from rag.chunk import chunk_parent_child, chunk_text, contextual_text, is_heading
from rag.query import RAGService
from rag.retrieval import HybridRetriever, RetrievalResult


# ---------------------------------------------------------------- 标题识别与内容保留

@pytest.mark.parametrize("line", ["12", "2019年研究者对30位受访者进行了深度访谈", "30位受访者参与了访谈",
                                  "1.5倍的增长", "1、访谈对象为30人。", "一、本研究采用扎根理论，"])
def test_numeric_body_lines_are_not_headings(line):
    assert not is_heading(line)


@pytest.mark.parametrize("line", ["1. 选题原则", "1、研究背景", "1.2 研究背景", "2.3.1 编码方法", "2.3.1、编码方法",
                                  "一、研究背景", "（二）文献综述", "第三章 研究设计", "参考文献"])
def test_real_headings_still_detected(line):
    assert is_heading(line)


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_no_content_is_lost_and_headings_stay_in_text():
    text = ("第三章 研究设计\n一、访谈\n2019年研究者对30位受访者进行了深度访谈\n"
            "1、访谈对象为30人。\n访谈持续三个月。\n\n二、编码\n扎根理论包括三级编码。")
    chunks = chunk_text(text, 600, 0)
    joined = "".join(t for t, _ in chunks)
    assert _compact(joined) == _compact(text)                     # 一个字都不少
    assert chunks[0][0].startswith("第三章 研究设计\n一、访谈\n2019年")  # 标题留在正文开头
    assert [s for _, s in chunks] == ["一、访谈", "二、编码"]


def test_consecutive_headings_do_not_make_heading_only_chunks():
    chunks = chunk_text("第一章 绪论\n一、背景\n（一）政策\n本文研究数字治理。", 600, 0)
    assert len(chunks) == 1 and chunks[0][1] == "（一）政策"
    assert chunks[0][0] == "第一章 绪论\n一、背景\n（一）政策\n本文研究数字治理。"


def test_page_numbers_dropped_and_english_wraps_joined():
    text = "Grounded theory is a systematic\n12\nmethodology in the social sciences.\n- 3 -\n第 4 页\n中文换行\n不加空格。"
    (body, _), = chunk_text(text, 600, 0)
    assert body == "Grounded theory is a systematic methodology in the social sciences.中文换行不加空格。"


def test_parent_child_children_have_no_id_marker_and_keep_headings():
    children, parents = chunk_parent_child("第一章 绪论\n\n本文研究数字治理。", "uid9", 600, 0, 1500)
    assert children[0]["text"] == "第一章 绪论\n本文研究数字治理。" and children[0]["parent_id"] == "uid9::p0"


def test_contextual_text_prefixes_title_section_and_context():
    meta = {"filename": "扎根理论讲义.pdf", "section": "二、编码", "context": "本段介绍三级编码。"}
    assert contextual_text("正文", meta) == "扎根理论讲义 › 二、编码\n本段介绍三级编码。\n正文"
    assert contextual_text("正文", {}) == "正文"


# ---------------------------------------------------------------- 检索：重排候选池与 BM25 元数据

class FakeCollection:
    """按预设顺序返回稠密结果；get 支持 ids / include。"""

    def __init__(self, docs: dict[str, str], dense_order: list[str]):
        self.docs, self.dense_order = docs, dense_order
        self.metas = {i: {"source": f"{i}.pdf", "filename": f"{i}.pdf", "section": ""} for i in docs}

    def query(self, query_embeddings, n_results, where=None):
        ids = self.dense_order[:n_results]
        return {"ids": [ids], "metadatas": [[self.metas[i] for i in ids]],
                "distances": [[0.1 * (k + 1) for k in range(len(ids))]]}

    def get(self, ids=None, include=()):
        ids = list(self.docs) if ids is None else [i for i in ids if i in self.docs]
        out = {"ids": ids}
        if "documents" in include:
            out["documents"] = [self.docs[i] for i in ids]
        if "metadatas" in include:
            out["metadatas"] = [self.metas[i] for i in ids]
        return out


class FakeReranker:
    def rerank(self, query, docs, top_n):
        scored = sorted(((k, float("目标" in d)) for k, d in enumerate(docs)), key=lambda x: -x[1])
        return scored[:top_n]


def test_rerank_can_promote_candidates_beyond_top_k():
    docs = {f"d{k}": f"普通内容{k}" for k in range(5)} | {"gold": "这里写着目标答案"}
    col = FakeCollection(docs, ["d0", "d1", "d2", "d3", "gold"])        # 稠密排第 5
    r = HybridRetriever(col, lambda q: [0.0], mode="dense")
    r.attach_reranker(FakeReranker())
    res = r.search("问题", top_k=2, fetch_k=5)
    assert res.ids[0] == "gold" and len(res.ids) == 2


def test_without_reranker_dense_still_cuts_to_top_k():
    col = FakeCollection({f"d{k}": "x" for k in range(5)}, [f"d{k}" for k in range(5)])
    res = HybridRetriever(col, lambda q: [0.0], mode="dense").search("q", top_k=2, fetch_k=5)
    assert res.ids == ["d0", "d1"]


def test_hybrid_bm25_only_hits_get_metadata():
    docs = {"a": "数字治理概述", "b": "研究方法概述", "c": "扎根理论 三级 编码 开放式编码"}
    col = FakeCollection(docs, ["a", "b"])                              # c 只有 BM25 能命中
    res = HybridRetriever(col, lambda q: [0.0], mode="hybrid").search("开放式编码", top_k=3, fetch_k=2)
    assert "c" in res.ids
    assert res.metadatas[res.ids.index("c")]["source"] == "c.pdf"


def test_bm25_indexes_title_and_section():
    col = FakeCollection({k: f"正文{k}" for k in "abcde"}, list("abcde"))   # 文档太少时 BM25 的 IDF 为 0
    col.metas["b"]["filename"] = "扎根理论讲义.pdf"
    res = HybridRetriever(col, lambda q: [0.0], mode="bm25").search("扎根理论", top_k=2, fetch_k=5)
    assert res.ids == ["b"]


# ---------------------------------------------------------------- 父块展开

def test_expand_returns_full_parent_once_per_parent(monkeypatch):
    svc = object.__new__(RAGService)
    svc._parents = {"p1": "父块一的完整内容", "p2": "父块二的完整内容"}
    res = RetrievalResult(
        ids=["c1", "c2", "c3", "c4"], documents=["子1", "子2", "子3", "子4"],
        metadatas=[{"parent_id": "p1"}, {"parent_id": "p1"}, {"parent_id": "p2"}, {}],
        distances=[0.1, 0.2, 0.3, 0.4], scores=[4, 3, 2, 1])
    svc._expand(res)
    assert res.ids == ["c1", "c3", "c4"]
    assert res.documents == ["父块一的完整内容", "父块二的完整内容", "子4"]
    assert res.distances == [0.1, 0.3, 0.4] and res.scores == [4, 2, 1]


# ---------------------------------------------------------------- Contextual Retrieval

class FakeLLM:
    def __init__(self, fail_on: str | None = None):
        self.calls, self.fail_on = [], fail_on
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        chunk = prompt.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        if self.fail_on and self.fail_on in chunk:
            raise RuntimeError("upstream error")

        class R:  # 最小 OpenAI 响应形状
            choices = [type("C", (), {"message": type("M", (), {"content": f" 关于「{chunk[:4]}」的上下文 "})()})()]
        return R()


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(contextual, "_cache", None)
    monkeypatch.setattr(contextual, "_client", None)


def test_add_contexts_calls_llm_once_per_chunk_then_uses_cache(fresh_cache):
    rows = [("r1", "第一段内容", {}), ("r2", "第二段内容", {})]
    llm = FakeLLM()
    st = contextual.add_contexts("全文：第一段内容。第二段内容。", rows, client=llm)
    assert st == {"chunks": 2, "cached": 0, "llm_calls": 2, "failed": 0} and len(llm.calls) == 2
    assert rows[0][2]["context"] == "关于「第一段内」的上下文"
    assert llm.calls[0].startswith("<document>\n全文：")                  # 文档在前，利于前缀缓存
    contextual._cache = None                                                # 模拟新进程：从磁盘读缓存
    rows2 = [("r1", "第一段内容", {}), ("r2", "第二段内容", {})]
    st2 = contextual.add_contexts("全文：第一段内容。第二段内容。", rows2, client=FakeLLM())
    assert st2["llm_calls"] == 0 and rows2[1][2]["context"] == "关于「第二段内」的上下文"


def test_add_contexts_failure_leaves_empty_context_and_is_not_cached(fresh_cache):
    rows = [("r1", "好的段落", {}), ("r2", "坏的段落", {})]
    st = contextual.add_contexts("文档", rows, client=FakeLLM(fail_on="坏的"))
    assert st["failed"] == 1 and rows[1][2]["context"] == ""
    st2 = contextual.add_contexts("文档", [("r2", "坏的段落", {})], client=FakeLLM())
    assert st2["llm_calls"] == 1                                            # 失败的会重试


def test_doc_window_centers_on_chunk():
    doc = "甲" * 5000 + "目标片段" + "乙" * 5000
    win = contextual.doc_window(doc, "目标片段", 1000)
    assert len(win) == 1000 and "目标片段" in win
    assert contextual.doc_window("短文档", "短", 1000) == "短文档"


def test_missing_api_key_is_a_clear_error(fresh_cache, monkeypatch):
    monkeypatch.setattr(contextual.config, "LLM_API_KEY", "")
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        contextual.add_contexts("文档", [("r1", "段落", {})])


def test_parse_with_contextual_enabled_fills_context(fake_corpus, monkeypatch, fresh_cache):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    monkeypatch.setattr(index, "CONTEXTUAL_ENABLED", True)
    llm = FakeLLM()
    monkeypatch.setattr(contextual, "_client", llm)
    monkeypatch.setattr(index, "extract_files", _extract_none)
    rows, skipped, _ = index._parse(index._collect_files())
    assert rows and all(m["context"].startswith("关于「") for _, _, m in rows)
    assert len(llm.calls) == len(rows)
    assert contextual_text(rows[0][1], rows[0][2]).split("\n")[1] == rows[0][2]["context"]


def _extract_none(files):
    from rag import extract
    return extract.extract_files(files, doc_backend="none")


def test_signature_changes_with_contextual_flag(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    before = index._signature(files)
    monkeypatch.setattr(index, "CONTEXTUAL_ENABLED", True)
    assert index._signature(files) != before
