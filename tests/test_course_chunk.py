"""课程教材解析与结构感知分块（M1）测试：全部离线，用标准库现造 docx。"""
from __future__ import annotations

import zipfile
from xml.sax.saxutils import escape

import pytest

from rag.course import chunker
from rag.course.chunker import chunk_textbook, split_sentences
from rag.course.textbook import parse_docx

_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

_STYLES = (
    f'<w:styles {_NS}>'
    + "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{i}"><w:name w:val="heading {i}"/></w:style>'
        for i in range(1, 5)
    )
    + "</w:styles>"
)

DEF_TEXT = "数据要素是指根据特定生产需求，经过汇聚、整理、加工等环节形成的，能够投入到生产经营活动中、参与价值创造的数据资源及其衍生形态。" * 2
LONG_TEXT = "这是一个用于测试长段落按句切分的句子，长度大约有四十个字符左右。" * 40  # > PARA_SPLIT


def _p(text: str = "", style: str | None = None, textbox: str | None = None, outline: int | None = None) -> str:
    lvl = f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/>{lvl}</w:pPr>' if style else ""
    box = ""
    if textbox:
        box = (
            "<w:r><w:pict><w:txbxContent><w:p><w:r><w:t>"
            f"{escape(textbox)}</w:t></w:r></w:p></w:txbxContent></w:pict></w:r>"
        )
    run = f"<w:r><w:t>{escape(text)}</w:t></w:r>" if text else ""
    return f"<w:p>{ppr}{run}{box}</w:p>"


def _tbl(rows: list[list[str]]) -> str:
    trs = "".join(
        "<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{escape(c)}</w:t></w:r></w:p></w:tc>" for c in r) + "</w:tr>"
        for r in rows
    )
    return f"<w:tbl>{trs}</w:tbl>"


def _book(overrides: dict[int, str] | None = None) -> list[str]:
    """返回 body 元素列表；注释为该段落的 P 序号。overrides 按序号替换段落正文。"""
    o = overrides or {}
    t = lambda n, default: o.get(n, default)  # noqa: E731
    return [
        _p("数据要素流通导论", "Heading1"),                                   # P1 书名
        _p("目录"),                                                           # P2
        _p("第一章 数据要素"),                                                # P3 目录行
        _p("第一章 数据要素", "Heading1"),                                    # P4
        _p("第一节 数据要素的概念界定", "Heading2", outline=0),              # P5 直设大纲级别与样式冲突（书稿实况）
        _p("一、数据要素的内涵", "Heading3"),                                 # P6
        _p("（一）数据要素的概念演进", "Heading4"),                           # P7
        _p(t(8, "要素是经济学概念，指社会生产投入的原始性资源。")),           # P8 短段
        _p(t(9, DEF_TEXT)),                                                   # P9 定义段
        _p(""),                                                               # P10 空段
        _p("（二）相近概念辨析", "Heading4"),                                 # P11
        _p("1.数据与数据要素"),                                               # P12 编号子标题
        _p(t(13, "数据是对客观事物状态、特征及其变化的记录，是信息表达与传递的基本载体。")),  # P13
        _p("2.数据资源、数据资产与数据要素"),                                 # P14
        _p(t(15, "数据资源是具有价值创造潜力的数据的总称。")),                # P15
        _p("表1 相近概念对照"),                                               # P16 表题
        _tbl([["概念", "含义"], ["数据资源", "可再利用的数据集合"], ["数据资产", "可计量的数据资源"]]),  # T01
        _p(LONG_TEXT),                                                        # P17 长段
        _p("小结", "Heading2"),                                               # P18
        _p("本章阐释了数据要素的概念内涵与演进历程。"),                      # P19
        _p("引用文献"),                                                       # P20
        _p("[1]某某.数据的本质[J].网络法律评论,2017(2):3-18."),              # P21
        _p("第二章 数据权属", "Heading1"),                                    # P22
        _p("第一节 数据权属的制度功能", "Heading2"),                          # P23
        _p(t(24, "数据权属界定能够降低交易成本。"), textbox="图1 文本框说明"),  # P24
        _p("", "Heading2"),                                                   # P25 空标题
    ]


def _write(tmp_path, body: list[str], name: str = "book.docx"):
    path = tmp_path / name
    doc = f'<w:document {_NS}><w:body>{"".join(body)}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", _STYLES)
    return path


@pytest.fixture
def book(tmp_path):
    return parse_docx(_write(tmp_path, _book()))


def _unit(book, anchor):
    return next(u for u in book.units if u.anchor == anchor)


# ---------------------------------------------------------------- 解析

def test_anchor_numbering_counts_empty_paragraphs(book):
    # 与 python-docx Document.paragraphs 序号一致：空段也计数，表格不计数
    assert book.n_paragraphs == 25
    assert _unit(book, "P0009").text == DEF_TEXT
    assert _unit(book, "P0013").text.startswith("数据是对客观事物")
    assert not any(u.anchor == "P0010" for u in book.units)  # 空段不产出单元


def test_zones_exclude_front_and_references(book):
    assert _unit(book, "P0001").zone == "front"
    assert _unit(book, "P0003").zone == "front"  # 目录行虽像章标题，但无标题样式
    assert _unit(book, "P0019").zone == "summary"
    assert _unit(book, "P0021").zone == "references"
    assert _unit(book, "P0024").zone == "body"   # 下一章标题重置区域


def test_heading_style_wins_over_direct_outline_level(book):
    u = _unit(book, "P0005")
    assert u.level == 2 and u.zone == "body"


def test_heading_path_and_chapter(book):
    u = _unit(book, "P0009")
    assert u.chapter == 1
    assert u.path == ("第一章 数据要素", "第一节 数据要素的概念界定", "一、数据要素的内涵", "（一）数据要素的概念演进")
    assert u.section_id == u.block_id == "ch01.1.1.1"
    assert _unit(book, "P0024").chapter == 2


def test_numbered_subheading_extends_path_but_not_block(book):
    assert _unit(book, "P0012").kind == "subheading"
    u13, u15 = _unit(book, "P0013"), _unit(book, "P0015")
    assert u13.path[-1] == "1.数据与数据要素"
    assert u13.section_id == "ch01.1.1.2.i1" and u15.section_id == "ch01.1.1.2.i2"
    assert u13.block_id == u15.block_id == "ch01.1.1.2"


def test_table_kept_in_original_position(book):
    anchors = [u.anchor for u in book.units]
    assert anchors.index("P0015") < anchors.index("T01") < anchors.index("P0017")
    t = _unit(book, "T01")
    assert t.kind == "table" and t.rows[0] == ["概念", "含义"]
    assert t.section_id == "ch01.1.1.2.i2"


def test_textbox_is_separate_unit(book):
    assert _unit(book, "P0024").text == "数据权属界定能够降低交易成本。"
    assert _unit(book, "P0024.tb1").kind == "textbox"


def test_empty_heading_is_ignored(book):
    assert not any(u.kind == "heading" and not u.text for u in book.units)


def test_rejects_non_docx(tmp_path):
    bad = tmp_path / "bad.docx"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("word/document.xml", f"<w:document {_NS}/>")
    with pytest.raises(ValueError):
        parse_docx(bad)


# ---------------------------------------------------------------- 分块

@pytest.fixture
def cs(book):
    return chunk_textbook(book, version="t1")


def _chunk_with(cs, anchor):
    return [c for c in cs.chunks if anchor in c.anchors]


def test_front_and_references_not_chunked(cs):
    joined = "\n".join(c.text for c in cs.chunks)
    assert "目录" not in joined and "[1]某某" not in joined


def test_definition_paragraph_is_never_split(cs):
    hits = _chunk_with(cs, "P0009")
    assert len(hits) == 1 and DEF_TEXT in hits[0].text


def test_short_paragraph_merged_with_neighbour(cs):
    (c,) = _chunk_with(cs, "P0008")
    assert c.anchors == ["P0008", "P0009"]


def test_chunks_do_not_cross_subheadings(cs):
    (a,) = _chunk_with(cs, "P0013")
    (b,) = _chunk_with(cs, "P0015")
    assert a.chunk_id != b.chunk_id
    assert a.parent_id == b.parent_id  # 同一父块（样式标题小节）


def test_long_paragraph_split_by_sentence(cs):
    parts = _chunk_with(cs, "P0017")
    assert len(parts) > 1
    assert [c.split_part for c in parts] == list(range(len(parts)))
    assert all(len(c.text) <= chunker.CHILD_MAX for c in parts)
    assert "".join(c.text for c in parts) == LONG_TEXT


def test_caption_travels_with_its_table(cs):
    (t,) = _chunk_with(cs, "T01")
    assert t.anchors == ["P0016", "T01"]
    assert t.text.startswith("表1 相近概念对照\n概念 | 含义")
    assert all("P0016" not in c.anchors for c in cs.chunks if c.chunk_type != "table")


def test_table_and_summary_chunk_types(cs):
    assert {c.chunk_type for c in _chunk_with(cs, "T01")} == {"table"}
    assert {c.chunk_type for c in _chunk_with(cs, "P0019")} == {"summary"}


def test_children_are_contained_in_parent(cs):
    parents = {p.parent_id: p for p in cs.parents}
    for c in cs.chunks:
        p = parents[c.parent_id]
        assert set(c.anchors) <= set(p.anchors)
        for line in c.text.split("\n"):
            assert line in p.text


def test_parent_path_stops_at_styled_heading(cs):
    (c,) = _chunk_with(cs, "P0013")
    parent = next(p for p in cs.parents if p.parent_id == c.parent_id)
    assert parent.path[-1] == "（二）相近概念辨析"
    assert c.path[-1] == "1.数据与数据要素"


def test_embed_text_carries_heading_path(cs):
    (c,) = _chunk_with(cs, "P0013")
    assert c.embed_text.startswith("第一章 数据要素 › ")
    assert c.embed_text.endswith(c.text)
    assert not c.text.startswith("第一章")


def test_chunk_ids_unique_and_deterministic(book, cs):
    ids = [c.chunk_id for c in cs.chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in chunk_textbook(book, version="t1").chunks]
    assert all(i.startswith("de:t1:") for i in ids)


def test_edit_changes_only_local_chunk_ids(tmp_path, cs):
    edited = parse_docx(_write(tmp_path, _book({13: "数据是记录客观事物的符号。", 24: "改写后的第二章内容。"}), "e.docx"))
    after = {c.chunk_id for c in chunk_textbook(edited, version="t1").chunks}
    before = {c.chunk_id for c in cs.chunks}
    changed = before - after
    assert {c.section_id for c in cs.chunks if c.chunk_id in changed} == {"ch01.1.1.2.i1", "ch02.1"}


def test_version_changes_all_ids(book, cs):
    other = {c.chunk_id for c in chunk_textbook(book, version="t2").chunks}
    assert not other & {c.chunk_id for c in cs.chunks}


def test_oversized_parent_split_at_unit_boundaries(book, monkeypatch):
    monkeypatch.setattr(chunker, "PARENT_MAX", 300)
    cs = chunk_textbook(book, version="t1")
    long_parents = [p for p in cs.parents if p.block_id == "ch01.1.1.2"]
    assert len(long_parents) > 1
    anchors = [a for p in long_parents for a in p.anchors]
    assert len(anchors) == len(set(anchors))  # 单元不重复、不被切开


# ---------------------------------------------------------------- 分章目录（toc）

def _toc_book(tmp_path):
    body = [
        _p("第一章 数据要素", "Heading1"),                          # P1
        _p("第一节 数据要素的概念", "Heading2"),                    # P2
        _p("数据要素是参与价值创造的数据资源。"),                    # P3
        _p("引用文献"),                                              # P4 -> references
        _p("[1]某某.数据的本质[J].2017."),                           # P5
        _p("一、数据产品化的核心环节"),                              # P6 下一章目录行：在引用文献之后、下一章标题之前
        _p("第二章 数据产品", "Heading1"),                           # P7
        _p("第一节 数据产品的界定", "Heading2"),                     # P8
        _p("一、数据产品化的核心环节", "Heading3"),                  # P9 正文真实标题
        _p("数据产品化是把数据加工成可交付产品的过程。"),            # P10
        _p("引用文献"),                                              # P11
        _p("[2]某某.数据产品[J].2020."),                             # P12
    ]
    return parse_docx(_write(tmp_path, body, "toc.docx"))


def test_toc_lines_after_references_are_marked(tmp_path):
    book = _toc_book(tmp_path)
    assert _unit(book, "P0005").zone == "references"
    assert _unit(book, "P0006").zone == "toc"          # 与下一章正文标题同名的短行
    assert _unit(book, "P0012").zone == "references"   # 最后一章之后没有目录行，不误伤


def test_toc_lines_are_not_chunked(tmp_path):
    cs = chunk_textbook(_toc_book(tmp_path), version="t1")
    assert not any("P0006" in c.anchors for c in cs.chunks)
    assert any("P0010" in c.anchors for c in cs.chunks)


def test_toc_key_ignores_page_numbers_and_dots():
    from rag.course.textbook import toc_key

    assert toc_key("一、数据产品化的核心环节 ……12") == toc_key("一、数据产品化的核心环节")


# ---------------------------------------------------------------- 工具函数与隔离

def test_split_sentences_hard_splits_overlong_sentence():
    pieces = split_sentences("甲" * 25 + "。乙乙。", 10)
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces) == "甲" * 25 + "。乙乙。"


def test_course_dirs_isolated_from_thesis_store():
    import rag.config as base
    from rag.course import config as course

    assert course.COURSE_CACHE_DIR != base.CACHE_DIR
    assert course.COURSE_CACHE_DIR != base.STORE_DIR
    assert base.STORE_DIR not in course.COURSE_CACHE_DIR.parents
