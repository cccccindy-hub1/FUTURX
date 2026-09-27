"""查询改写 / HyDE 测试：无 LLM 时安全回退，有 LLM 时按契约返回。"""
from __future__ import annotations

from rag.enhance import build_retrieval_query, hyde, rewrite_query
from tests.conftest import StubLLM


def test_rewrite_without_llm_returns_original():
    assert rewrite_query("扎根怎么做", None, "m") == "扎根怎么做"


def test_hyde_without_llm_returns_empty():
    assert hyde("扎根怎么做", None, "m") == ""


def test_rewrite_uses_llm_output():
    llm = StubLLM("扎根理论 三级编码 操作步骤")
    assert rewrite_query("扎根怎么做", llm, "m") == "扎根理论 三级编码 操作步骤"


def test_hyde_returns_generated_text():
    llm = StubLLM("扎根理论的三级编码包括……")
    assert hyde("三级编码", llm, "m") == "扎根理论的三级编码包括……"


def test_rewrite_falls_back_on_llm_error():
    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    raise RuntimeError("network down")

    assert rewrite_query("原问题", Boom(), "m") == "原问题"


def test_build_retrieval_query_no_flags_is_identity():
    assert build_retrieval_query("q", None, "m") == "q"


def test_build_retrieval_query_prefers_hyde():
    llm = StubLLM("假设答案内容")
    out = build_retrieval_query("q", llm, "m", rewrite=True, use_hyde=True)
    assert "假设答案内容" in out


def test_build_retrieval_query_rewrite_only():
    llm = StubLLM("改写后的查询")
    assert build_retrieval_query("q", llm, "m", rewrite=True) == "改写后的查询"
