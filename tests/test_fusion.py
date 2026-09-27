"""RRF 融合与元数据解析的纯函数测试。"""
from __future__ import annotations

import pytest

from rag.retrieval import rrf_fuse, parse_where


def test_rrf_fuse_prefers_document_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = dict(rrf_fuse([dense, sparse], k=60))
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["c"]
    # a、b 均出现在两路中，应高于仅出现一次的 d
    assert fused["a"] > fused["d"]
    assert fused["b"] > fused["d"]


def test_rrf_fuse_single_list_is_identity_order():
    fused = rrf_fuse([["x", "y", "z"]], k=60)
    assert [i for i, _ in fused] == ["x", "y", "z"]


def test_rrf_fuse_scores_are_monotonic_decreasing():
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "b"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_fuse_is_deterministic_on_ties():
    a = rrf_fuse([["p", "q"], ["q", "p"]], k=60)
    b = rrf_fuse([["p", "q"], ["q", "p"]], k=60)
    assert a == b


def test_parse_where_none_and_empty():
    assert parse_where(None) is None
    assert parse_where([]) is None


def test_parse_where_builds_dict():
    assert parse_where(["category=006/研究方法", "ext=.pdf"]) == {
        "category": "006/研究方法",
        "ext": ".pdf",
    }


def test_parse_where_value_may_contain_equals():
    assert parse_where(["section=a=b"]) == {"section": "a=b"}


def test_parse_where_rejects_malformed():
    with pytest.raises(ValueError):
        parse_where(["no-equals-sign"])
    with pytest.raises(ValueError):
        parse_where(["=missing-key"])
