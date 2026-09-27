"""分块层测试（现有契约，回归保护）。"""
from __future__ import annotations

from rag.chunk import chunk_text, is_heading


def test_is_heading_recognizes_numbered_and_known_sections():
    assert is_heading("一、研究背景")
    assert is_heading("1. 选题原则")
    assert is_heading("（二）文献综述")
    assert is_heading("第二章 研究方法")
    assert is_heading("文献综述")
    assert not is_heading("这是一句普通的正文，不应该被当作标题。")


def test_is_heading_rejects_too_long_lines():
    assert not is_heading("一、" + "很长" * 30)


def test_chunk_text_returns_text_and_section_pairs():
    text = "第一章 绪论\n\n本文研究数字治理。\n\n第二章 研究方法\n\n采用案例研究法。"
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    assert chunks, "应切出至少一个 chunk"
    for body, section in chunks:
        assert isinstance(body, str) and body.strip()
        assert isinstance(section, str)


def test_chunk_text_respects_size_limit():
    body = "这是一个句子。" * 400  # 远大于 600
    chunks = chunk_text(body, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for text, _ in chunks:
        assert len(text) <= 200


def test_chunk_text_creates_overlap_between_neighbours():
    body = "内容" * 500
    chunks = chunk_text(body, chunk_size=100, overlap=30)
    assert len(chunks) >= 2
    first_tail = chunks[0][0][-30:]
    assert first_tail and first_tail in chunks[1][0]


def test_chunk_text_empty_input():
    assert chunk_text("", 600, 100) == []
    assert chunk_text("   \n\n  ", 600, 100) == []
