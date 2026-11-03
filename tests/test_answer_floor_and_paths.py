"""作答下限提示词 + 来源路径补章名。"""
from types import SimpleNamespace as NS

from rag.course import prompts
from rag.course.answer import chapter_titles, source_entry


def _chunk(cid, path, chapter=None):
    return NS(chunk_id=cid, path=list(path), chapter=chapter)


def test_answer_prompt_requires_points_for_list_and_relation_questions():
    s = prompts.ANSWER_SYSTEM
    assert "必须分点展开" in s and "不能只给结论或只罗列名称" in s
    for kind in ("列举", "构成", "环节", "关系", "区别", "原因"):
        assert kind in s
    assert "一般不超过 400 字" in s                     # 上限保留
    assert "按需分点补充" not in s                     # 旧的「按需」写法会让模型只写一两句结论


def test_answer_prompt_total_split_total_and_closing_question():
    s = prompts.ANSWER_SYSTEM
    assert "AI 李老师" in s
    assert "“总—分—总”结构" in s and "收束要点" in s
    assert "最后单独一行，写一个问句" in s and "不加引用编号" in s
    assert "不写引言和总结段" not in s                 # 与「总分总」冲突的旧要求已去掉


def test_chapter_titles_from_chunks():
    chunks = [_chunk("de:1102:ch06.1:000:a", ["第六章 数据市场", "第一节 x"], 6),
              _chunk("de:1102:ch06.1.1:000:b", ["第一节 x", "一、y"], 6),
              _chunk("de:1102:ch11.3:000:c", ["第十一章 数据资产", "第三节 z"], 11),
              _chunk("de:1102:front:000:d", ["前言"], None)]
    assert chapter_titles(chunks) == {6: "第六章 数据市场", 11: "第十一章 数据资产"}


def test_source_path_gets_chapter_prefix_when_missing():
    titles = {1: "第一章 数据要素", 11: "第十一章 数据资产"}
    e = source_entry(2, _chunk("de:1102:ch01.1.1.1.i2:001:a9381835", ["一、数据要素的内涵", "（一）概念演进"]), titles)
    assert e == {"n": 2, "chunk_id": "de:1102:ch01.1.1.1.i2:001:a9381835",
                 "path": "第一章 数据要素 › 一、数据要素的内涵 › （一）概念演进"}
    # 已经以章名开头的不重复加
    e = source_entry(1, _chunk("de:1102:ch11.3.3:000:x", ["第十一章 数据资产", "第三节 实践路径"]), titles)
    assert e["path"] == "第十一章 数据资产 › 第三节 实践路径"


def test_source_path_unchanged_without_titles_or_unknown_chapter():
    c = _chunk("de:1102:ch07.2:000:x", ["第二节 定价"])
    assert source_entry(1, c)["path"] == "第二节 定价"
    assert source_entry(1, c, {1: "第一章 数据要素"})["path"] == "第二节 定价"
