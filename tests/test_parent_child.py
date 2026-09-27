"""父子分块测试：子块带 header、父子映射一致、上下文可按偏移展开。"""
from __future__ import annotations

from rag.chunk import chunk_parent_child


def _sample_text(n_blocks: int = 12) -> str:
    blocks = []
    for i in range(n_blocks):
        blocks.append(f"第{i + 1}节 主题{i}\n\n" + f"这是第{i + 1}节的正文内容。" * 20)
    return "\n\n".join(blocks)


def test_parent_child_returns_both_levels():
    children, parents = chunk_parent_child(_sample_text(), "uid123", parent_size=300)
    assert children and parents
    assert all({"parent_id", "text", "section", "parent_offset", "parent_length"} <= set(c) for c in children)


def test_parent_ids_are_unique_and_child_headers_match():
    children, parents = chunk_parent_child(_sample_text(), "uid123", parent_size=300)
    parent_ids = [p["parent_id"] for p in parents]
    assert len(parent_ids) == len(set(parent_ids)), "父块 id 必须唯一"
    by_id = {p["parent_id"]: p["text"] for p in parents}
    for c in children:
        assert c["parent_id"] in by_id
        assert c["text"].startswith(f"〔{c['parent_id']}〕")


def test_child_offset_restores_body_from_parent():
    children, parents = chunk_parent_child(_sample_text(), "uid123", child_size=100, parent_size=300)
    by_id = {p["parent_id"]: p["text"] for p in parents}
    for c in children:
        parent = by_id[c["parent_id"]]
        body = c["text"].split("\n", 1)[1]
        assert parent[c["parent_offset"] : c["parent_offset"] + c["parent_length"]] == body


def test_children_are_smaller_than_parents():
    children, parents = chunk_parent_child(_sample_text(), "uid", child_size=150, parent_size=600)
    assert max(len(c["text"]) for c in children) < max(len(p["text"]) for p in parents)


def test_empty_text_yields_empty_results():
    assert chunk_parent_child("", "uid") == ([], [])
    assert chunk_parent_child("   \n\n  ", "uid") == ([], [])
