"""判例单库作答：头部由程序填写、模型只写三项、引用统一成 [1] [2]、一次调用。假 LLM，不调用真实模型。"""
from __future__ import annotations

import json

import pytest

from rag.cases.answer import (CASE_ANSWER_SYSTEM, answer_cases, build_messages, header_lines, render_materials, source_path)
from rag.cases.retrieval import CaseHit, CaseRef
from rag.course.llm import LLMResult


def mk(n, chapter, title, meta=None, card="卡片全文", bodies=("正文块一", "正文块二")):
    return CaseHit(ref=CaseRef(f"case:{n:02d}", n, chapter, int(chapter[1:2] if chapter[1:2].isdigit() else 6), title, meta or {}),
                   score=0.9, card=card, bodies=list(bodies))


H1 = mk(3, "第六章 数据市场", "北京微梦创科诉云智联不正当竞争纠纷案（指导性案例262号）",
        {"case_number": "（2020）京73民终2980号", "court": "北京知识产权法院", "judgment_date": "2020-12-01", "source_url": "https://example.org/a"})
H2 = mk(4, "第二章 数据权属", "大数据搬家软件案", {})


class LLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "json_mode": json_mode, "model": model})
        return LLMResult(content=self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False))


def items(*rows):
    return {"cases": [{"id": i, "facts": f, "holding": h, "relation": r} for i, f, h, r in rows]}


# ---------------------------------------------------------------- 头部由程序填写

def test_header_lines_come_from_metadata_not_from_the_model():
    assert header_lines(H1) == ["【第六章 数据市场】北京微梦创科诉云智联不正当竞争纠纷案（指导性案例262号）",
                                "案号：（2020）京73民终2980号　审理法院：北京知识产权法院　裁判日期：2020-12-01"]
    assert header_lines(H2) == ["【第二章 数据权属】大数据搬家软件案"]                       # 字段缺失就不写，不让模型补


def test_model_cannot_overwrite_case_number_court_or_date():
    evil = items(("C1", "案情[C1]", "要点[C1]", "关系[C1]"))
    evil["cases"][0].update(case_number="（2099）假民初1号", court="假法院", title="假案名", date="2099-01-01")
    out = answer_cases("问", [H1], LLM(evil), model="m")
    assert "（2020）京73民终2980号" in out.text and "（2099）假民初1号" not in out.text and "假法院" not in out.text and "假案名" not in out.text


def test_header_is_verbatim_even_with_fullwidth_brackets_and_parentheses_in_title():
    out = answer_cases("问", [H1], LLM(items(("C1", "案情[C1]", "要点[C1]", "关系[C1]"))), model="m")
    assert out.text.startswith("【第六章 数据市场】北京微梦创科诉云智联不正当竞争纠纷案（指导性案例262号）\n案号：")          # 头部不被引用改写器动过


# ---------------------------------------------------------------- 模型只写三项，引用统一成 [1] [2]

def test_one_call_writes_all_cases_and_only_three_items_each():
    llm = LLM(items(("C1", "平台数据被抓取[C1]。", "构成不正当竞争[C1]。", "说明数据权益[C1]。"), ("C2", "软件搬运数据[C2]。", "公开数据也受保护[C2]。", "与问题相关[C2]。")))
    out = answer_cases("平台数据能抓吗", [H1, H2], llm, model="m")
    assert len(llm.calls) == 1 and llm.calls[0]["step"] == "case_answer" and llm.calls[0]["json_mode"] is True
    assert out.text == ("【第六章 数据市场】北京微梦创科诉云智联不正当竞争纠纷案（指导性案例262号）\n"
                        "案号：（2020）京73民终2980号　审理法院：北京知识产权法院　裁判日期：2020-12-01\n"
                        "基本案情：平台数据被抓取[1]。\n裁判要点：构成不正当竞争[1]。\n与你的问题的关系：说明数据权益[1]。\n\n"
                        "【第二章 数据权属】大数据搬家软件案\n基本案情：软件搬运数据[2]。\n裁判要点：公开数据也受保护[2]。\n与你的问题的关系：与问题相关[2]。")
    assert "[C" not in out.text and "case:" not in out.text


def test_numbers_follow_first_appearance_across_cases_not_c_numbers():
    """模型先写了 C2 再写 C1，但回答按判例顺序（C1、C2）输出；序号按正文里首次出现编号：C1 的条目在前，所以 C1 是 [1]。
    若把条目顺序倒过来（先 C2 后 C1 出现在正文里），则 C2 才是 [1]——这里验证的是「按出现顺序」而不是「按 C 编号」。"""
    llm = LLM(items(("C2", "第二个案情[C2]。", "要点[C2]。", "关系[C2]。"), ("C1", "第一个案情[C1]。", "要点[C1]。", "关系。")))
    out = answer_cases("问", [H1, H2], llm, model="m")
    assert "基本案情：第一个案情[1]。" in out.text and "基本案情：第二个案情[2]。" in out.text
    assert [(s["n"], s["chunk_id"]) for s in out.sources] == [(1, "case:03"), (2, "case:04")]
    # 条目按材料顺序输出：C1 的头部在前
    assert out.text.index("北京微梦创科") < out.text.index("大数据搬家软件案")


def test_numbering_is_by_first_appearance_when_only_the_second_case_is_cited_first():
    llm = LLM(items(("C1", "第一个案情没有引用。", "要点。", "关系。"), ("C2", "第二个案情[C2]。", "要点[C1]。", "关系。")))
    out = answer_cases("问", [H1, H2], llm, model="m")
    assert "基本案情：第二个案情[1]。" in out.text and "裁判要点：要点[2]。" in out.text           # C2 先出现 -> [1]，C1 随后 -> [2]
    assert [(s["n"], s["chunk_id"]) for s in out.sources] == [(1, "case:04"), (2, "case:03")]


def test_same_case_numbered_consistently_and_adjacent_duplicates_merged_within_an_item():
    out = answer_cases("问", [H1], LLM(items(("C1", "案情[C1][C1]。", "要点[C1]。", "关系[C1]。"))), model="m")
    assert out.text.count("[1]") == 3 and "[1][1]" not in out.text and out.text.count("[") == 3


def test_adjacent_merge_does_not_leak_across_items():
    """上一条目末尾引用了某案，下一条目开头引用同一案：它们之间隔着标题行，不能被当成「相邻重复」而吞掉。"""
    out = answer_cases("问", [H1], LLM(items(("C1", "案情[C1]", "[C1]要点以此为据。", "关系[C1]"))), model="m")
    assert out.text.count("[1]") == 3


@pytest.mark.parametrize("variant", ["[C1]", "［C1］", "【C1】", "[c1]", "[ C1 ]"])
def test_c_marker_variants_are_recognized(variant):
    out = answer_cases("问", [H1], LLM(items(("C1", f"案情{variant}。", "要点。", "关系。"))), model="m")
    assert "基本案情：案情[1]。" in out.text and "C1" not in out.text


def test_out_of_range_markers_are_removed_and_do_not_create_sources():
    out = answer_cases("问", [H1], LLM(items(("C1", "案情[C9][C1]。", "要点。", "关系。"))), model="m")
    assert "C9" not in out.text and "基本案情：案情[1]。" in out.text and [s["n"] for s in out.sources] == [1]


def test_ontology_style_and_chunk_id_markers_are_stripped_in_case_answers_too():
    out = answer_cases("问", [H1], LLM(items(("C1", "案情[本体][C1][de:0906:x]。", "要点[onto1]。", "关系。"))), model="m")
    assert "本体" not in out.text and "onto" not in out.text and "de:" not in out.text


# ---------------------------------------------------------------- 来源列表

def test_sources_only_cited_cases_with_chapter_title_case_number_and_link():
    out = answer_cases("问", [H1, H2], LLM(items(("C1", "案情[C1]。", "要点[C1]。", "关系。"), ("C2", "案情。", "要点。", "关系。"))), model="m")
    assert [s["n"] for s in out.sources] == [1]                                 # 只列正文引用过的判例（C2 没被引用）
    s = out.sources[0]
    assert s["kind"] == "case" and s["chunk_id"] == "case:03" and s["case_number"] == "（2020）京73民终2980号" and s["source_url"] == "https://example.org/a"
    assert s["path"] == source_path(H1) == "第六章 数据市场 › 北京微梦创科诉云智联不正当竞争纠纷案（指导性案例262号）（（2020）京73民终2980号）"
    assert s["title"].startswith("北京微梦创科")


# ---------------------------------------------------------------- 异常输出：不重试，退回程序填写的头部

@pytest.mark.parametrize("bad", ["这不是 JSON", {"cases": []}, {"nope": 1}, {"cases": [{"id": "C9", "facts": "x"}]}])
def test_unusable_output_still_shows_program_filled_headers_without_retry(bad):
    llm = LLM(bad)
    out = answer_cases("问", [H1, H2], llm, model="m")
    assert len(llm.calls) == 1                                                  # 不重试
    assert out.text.startswith("【第六章 数据市场】北京微梦创科诉云智联") and "【第二章 数据权属】大数据搬家软件案" in out.text
    assert "基本案情" not in out.text and out.sources == [] and out.dropped


def test_partially_missing_items_are_skipped_not_invented():
    out = answer_cases("问", [H1], LLM({"cases": [{"id": "C1", "facts": "只有案情[C1]。"}]}), model="m")
    assert "基本案情：只有案情[1]。" in out.text and "裁判要点" not in out.text and "与你的问题的关系" not in out.text


def test_duplicate_ids_in_model_output_use_the_first_one():
    out = answer_cases("问", [H1], LLM(items(("C1", "第一版[C1]。", "要点。", "关系。"), ("C1", "第二版[C1]。", "要点。", "关系。"))), model="m")
    assert "第一版" in out.text and "第二版" not in out.text and {"id": "C1", "reason": "unknown_or_duplicate"} in out.dropped


# ---------------------------------------------------------------- 提示词与材料

def test_prompt_forbids_writing_header_fields_and_asks_for_three_items():
    for w in ("不要编造案号、法院、日期", "基本案情", "裁判要点", "与你的问题的关系", "你不要写", "输出严格 JSON"):
        assert w in CASE_ANSWER_SYSTEM


def test_materials_are_numbered_c1_c2_with_card_and_bodies_and_respect_first_n():
    m = render_materials([H1, H2])
    assert m.startswith("[C1] 卡片全文\n正文块一\n正文块二") and "\n\n[C2] 卡片全文" in m
    assert render_materials([H2], first_n=3).startswith("[C3] ")                 # 组合问答里编号从 first_n 起
    msgs = build_messages("问题？", [H1])
    assert msgs[1]["content"].startswith("学生问题：问题？\n\n判例材料：\n[C1] ")


def test_material_text_is_truncated_per_case():
    big = mk(5, "第一章 数据要素", "长案", card="卡" * 5000, bodies=())
    assert len(render_materials([big], limit_chars=300)) <= 300 + len("[C1] ")
