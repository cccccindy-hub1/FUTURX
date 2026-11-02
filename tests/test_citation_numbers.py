"""学生可见引用序号：[S#] -> [n]（按首次出现编号）、重复与相邻合并、跨分片缓冲、sources 只含被引用的块、
正文不含 chunk_id、[onto1]/[本体] 被去掉、筛选提示词放宽、--ids。全部 mock，不发真实调用。"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.course import answer, config, prompts
from rag.course.answer import CitationRewriter, sanitize_citations
from rag.course.llm import LLMResult
from rag.course.recall import Recalled
from rag.unified import UnifiedService, router as unified_router
from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: F401

REFS = {f"S{i}": f"de:{config.BOOK_VERSION}:c{i}" for i in range(1, 15)}


def render(text, refs=REFS):
    return sanitize_citations(text, refs)


# ---------------------------------------------------------------- 序号按首次出现编号

def test_numbers_follow_first_appearance_not_s_number():
    text, cites, invalid, _ = render("先说[S5]，再说[S2]，最后[S9]。")
    assert text == "先说[1]，再说[2]，最后[3]。"                                   # S5 最先出现 -> 1，不是 5
    assert cites == [REFS["S5"], REFS["S2"], REFS["S9"]] and invalid == []


def test_same_chunk_reuses_its_number_across_the_text():
    text, cites, _, _ = render("甲[S3]。乙[S7]。丙[S3]。丁[S7][S3]。")
    assert text == "甲[1]。乙[2]。丙[1]。丁[2][1]。" and cites == [REFS["S3"], REFS["S7"]]    # 非相邻的重复引用不合并，序号不变


# ---------------------------------------------------------------- 相邻重复合并

@pytest.mark.parametrize("raw,expected", [
    ("事实甲[S3][S3]。", "事实甲[1]。"),                          # 紧挨着重复
    ("事实甲[S3]，[S3]。", "事实甲[1]，。"),                        # 只隔标点
    ("事实甲[S3] [S3]。", "事实甲[1]。"),                           # 只隔空白：合并时连中间的空格一起去掉
    ("事实甲[S3]。事实乙[S3]。", "事实甲[1]。事实乙[1]。"),            # 中间有实质内容：不算相邻，各保留
    ("[S3][S4][S3]", "[1][2][1]"),                                 # 中间隔了别的块：不算相邻
    ("甲[S3][本体][S3]", "甲[1]"),                                  # 中间隔着被去掉的标记：仍视为相邻，合并
])
def test_adjacent_duplicates_merged_only_when_truly_adjacent(raw, expected):
    assert render(raw)[0] == expected


# ---------------------------------------------------------------- 正文不含 chunk_id；非教材标记被去掉

def test_text_never_contains_chunk_id_even_when_model_writes_it():
    cid = REFS["S1"]
    text, cites, invalid, _ = render(f"见[S1]，也见[{cid}]和[de:0906:fake]。")
    assert "de:" not in text and cid not in text and text == "见[1]，也见和。"
    assert cites == [cid] and set(invalid) == {cid, "de:0906:fake"}              # 绕过编号直接写 chunk_id 一律不信任


@pytest.mark.parametrize("mark", ["[本体]", "[onto1]", "[onto2]", "[Onto1]", "[onto]", "【本体】", "［onto1］"])
def test_ontology_style_marks_are_stripped_from_text(mark):
    text, cites, invalid, onto = render(f"数据资产入表受财政部规定规制{mark}。另见[S1]。")
    assert "本体" not in text and "onto" not in text.lower() and text == "数据资产入表受财政部规定规制。另见[1]。"
    assert onto is True and invalid == [] and cites == [REFS["S1"]]            # 记录用过本体，但不算无效引用，也不计入 sources


def test_ordinary_brackets_like_chapter_paths_are_kept():
    text, *_ = render("参见[第一章 › 第一节]，依据[S1]；列表[1]不动。")
    assert text == "参见[第一章 › 第一节]，依据[1]；列表[1]不动。"


def test_out_of_range_and_placeholder_removed_and_recorded():
    text, cites, invalid, _ = render("甲[S99]乙[S#]丙[S1]")
    assert text == "甲乙丙[1]" and invalid == ["S99"] and cites == [REFS["S1"]]


# ---------------------------------------------------------------- 跨分片缓冲

def _stream(pieces, refs=REFS, on_assign=None):
    rw = CitationRewriter(refs, on_assign=on_assign)
    shown = [rw.feed(p) for p in pieces]
    shown.append(rw.finish())
    return shown, rw


@pytest.mark.parametrize("pieces", [
    ["数据要素[S1", "2]是资源。"],                        # [S12] 拆在 "[S1" | "2]" 两片
    ["甲[", "S12", "]乙"],                                # 拆成三片
    ["甲[S", "12]乙"],
    ["甲[S12", "]乙"],
    ["甲[S12]", "乙"],                                    # 不拆
    list("甲[S12]乙"),                                    # 逐字符
])
def test_split_marker_is_buffered_until_complete(pieces):
    shown, rw = _stream(pieces)
    assert "".join(shown) == "甲[1]乙" or "".join(shown) == "数据要素[1]是资源。"
    assert not any("S1" in s or "[S" in s for s in shown)                         # 任何一片都不会泄露半截的 [S1 / [S12
    assert rw.citations == [REFS["S12"]]


def test_incomplete_marker_is_held_back_not_emitted_early():
    rw = CitationRewriter(REFS)
    assert rw.feed("结论是[S1") == "结论是"                    # "[S1" 先扣住
    assert rw.feed("2]。") == "[1]。"
    assert rw.finish() == ""


def test_stream_result_identical_to_one_shot_for_every_split_point():
    raw = "要点甲[S5]，要点乙[S2][S2]，丙[本体]，丁[S5]；[onto1]戊[S12]。"
    expected, cites, invalid, _ = render(raw)
    for k in range(len(raw) + 1):
        shown, rw = _stream([raw[:k], raw[k:]])
        assert "".join(shown) == expected and rw.citations == cites, k          # 在任何位置切开，结果都与一次性处理逐字一致


def test_dangling_open_bracket_does_not_swallow_the_rest():
    shown, _ = _stream(["说明[这是一个没有闭合的括号", "，后面的话还要继续显示" + "x" * 150])
    out = "".join(shown)
    assert out.startswith("说明[这是一个没有闭合的括号，后面的话") and out.endswith("x" * 150)   # 超过缓冲上限就当普通文本放出


def test_unclosed_marker_at_end_is_flushed_as_text():
    shown, _ = _stream(["结尾是[S1"])
    assert "".join(shown) == "结尾是[S1"


def test_on_assign_called_once_per_new_chunk_in_order():
    seen = []
    _stream(["甲[S5]乙[S2]丙[S5]丁[S9]"], on_assign=lambda n, c: seen.append((n, c)))
    assert seen == [(1, REFS["S5"]), (2, REFS["S2"]), (3, REFS["S9"])]


# ---------------------------------------------------------------- answer_question：sources 只含被引用的块，带路径

def _kept(ctx_, *anchors):
    from test_course_pipeline import _items
    return _items(ctx_, *anchors)


def test_sources_only_cited_chunks_in_number_order_with_path(ctx):
    kept = _kept(ctx, "P0005", "P0008")[:2]
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="只引用第二块[S2]，不引用第一块。")]))
    assert ar.text == "只引用第二块[1]，不引用第一块。"
    assert ar.citations == [kept[1].chunk_id]
    assert ar.sources == [{"n": 1, "chunk_id": kept[1].chunk_id, "path": " › ".join(kept[1].path)}]
    assert kept[0].chunk_id not in json.dumps(ar.sources) and kept[1].chunk_id not in ar.text
    assert ar.sources[0]["path"].startswith("第一章") and " › " in ar.sources[0]["path"]       # 章名与小节路径


def test_sources_ordered_by_first_appearance_and_text_has_no_chunk_id(ctx):
    kept = _kept(ctx, "P0005", "P0008")[:2]
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="先[S2]后[S1]再[S2]。")]))
    assert [s["n"] for s in ar.sources] == [1, 2] and [s["chunk_id"] for s in ar.sources] == [kept[1].chunk_id, kept[0].chunk_id]
    assert ar.text == "先[1]后[2]再[1]。" and not re.search(r"de:", ar.text)


def test_no_citations_means_empty_sources(ctx):
    kept = _kept(ctx, "P0005")[:1]
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="没有引用的回答。")]))
    assert ar.sources == [] and ar.citations == [] and ar.text == "没有引用的回答。"


# ---------------------------------------------------------------- 管线 / 接口：sources 结构与 citations 兼容

@pytest.fixture(autouse=True)
def single_mode(request, monkeypatch):
    if "ctx" in request.fixturenames:
        request.getfixturevalue("ctx")
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(config, "COURSE_SELECT_FAST_PATH", False)
    monkeypatch.setattr(config, "COURSE_FILTER_SKIP_MAX", 0)


def _script(answer_text):
    def filt(step, msgs):
        refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
        return jtxt({"direct": refs[:1], "background": refs[1:]})
    pick = jtxt({"question_concepts": ["C01-0001"], "related_concepts": ["C01-0003"], "reason": "r"})
    return [pick, filt, LLMResult(content=answer_text)]


def _route(route="course"):
    return type("R", (), {"chat": lambda self, *a, **k: LLMResult(content=json.dumps({"route": route, "complexity": "simple", "reason": "t"}))})()


def test_auto_ask_sources_have_n_chunk_id_path_and_citations_still_available(ctx):
    out = UnifiedService(router_client=_route(), course_ctx=ctx, course_llm=ScriptedLLM(_script("定义见[S1]。又见[S1]。"))).ask("什么是数据要素")
    assert out["answer"] == "定义见[1]。又见[1]。"
    assert out["sources"] and set(out["sources"][0]) == {"n", "chunk_id", "path"} and out["sources"][0]["n"] == 1
    assert isinstance(out["sources"][0]["path"], str) and " › " in out["sources"][0]["path"]
    assert out["sources"][0]["chunk_id"].startswith(f"de:{config.BOOK_VERSION}:") and "de:" not in out["answer"]
    assert [c["chunk_id"] for c in out["sources"]] == [out["sources"][0]["chunk_id"]]       # 评测脚本用 c.get("chunk_id") 取 citations，照常可用


def test_pipeline_citations_keep_chunk_id_for_existing_consumers(ctx):
    from rag.course import pipeline
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=ScriptedLLM(_script("定义见[S1]。")))
    assert r["answer"]["citations"] == [c["chunk_id"] for c in r["citations"]] == [s["chunk_id"] for s in r["sources"]]
    assert r["citations"][0]["n"] == 1 and isinstance(r["citations"][0]["path"], list)       # 旧的 path 列表形式不变（cli 仍可 ' › '.join）


def test_eval_scripts_read_chunk_ids_from_sources(ctx):
    """评测脚本从 sources 取 chunk_id：指标（引用数、是否引用到 gold 块）与改造前一致。"""
    from rag.course import eval_e2e
    row = eval_e2e.run_one(UnifiedService(router_client=_route(), course_ctx=ctx, course_llm=ScriptedLLM(_script("甲[S1]，乙[S1]。"))),
                           {"id": "r11", "question": "什么是数据要素"}, {"calls": 0, "cap": 99})
    assert row["error"] == "" and len(row["citations"]) == 1 and row["citations"][0].startswith("de:")
    assert "de:" not in row["answer"]


# ---------------------------------------------------------------- SSE：delta 只有序号，materials 在分配序号时发出

def _app(svc):
    app = FastAPI()
    app.state.unified = svc
    app.include_router(unified_router)
    return TestClient(app)


def _events(resp):
    return [json.loads(b[6:]) for b in resp.text.split("\n\n") if b.startswith("data: ")]


class Chunked(ScriptedLLM):
    """作答走流式，按给定分片依次回调（可把 [S12] 拆在两片里）。"""

    def __init__(self, script, pieces):
        super().__init__(script)
        self.pieces = pieces

    def chat_stream(self, step, messages, *, model, on_delta):
        res = self.chat(step, messages, model=model)
        for p in self.pieces:
            on_delta(p)
        return res


def test_sse_delta_has_only_numbers_materials_before_first_marker_done_has_sources(ctx):
    pieces = ["定义是资源[S", "1]；再次提到[S1", "]。"]                      # [S1] 被拆在分片边界
    llm_ = Chunked(_script("".join(pieces)), pieces)
    ev = _events(_app(UnifiedService(router_client=_route(), course_ctx=ctx, course_llm=llm_)).post("/auto/ask/stream", json={"question": "什么是数据要素"}))
    kinds = [e["type"] for e in ev]
    deltas = [e["text"] for e in ev if e["type"] == "delta"]
    text = "".join(deltas)
    assert text == "定义是资源[1]；再次提到[1]。" and not any("S1" in d or "[S" in d or "de:" in d for d in deltas)   # 没有任何一片泄露半截 [S# 或 chunk_id
    mats = [e for e in ev if e["type"] == "materials"]
    assert len(mats) == 1 and mats[0]["n"] == 1 and mats[0]["path"] and mats[0]["chunk_id"].startswith("de:")
    first_marker = next(i for i, e in enumerate(ev) if e["type"] == "delta" and "[1]" in e["text"])
    assert ev.index(mats[0]) < first_marker                                # 序号分配时先发 materials
    done = ev[-1]
    assert done["type"] == "done" and done["answer"] == text and kinds.count("done") == 1
    assert [s["n"] for s in done["sources"]] == [1] and done["sources"][0]["path"] and "de:" not in done["answer"]


def test_sse_done_sources_only_cited_and_numbered_by_first_appearance(ctx):
    llm_ = Chunked(_script("先[S2]后[S1]"), ["先[S2]", "后[S1]"])
    ev = _events(_app(UnifiedService(router_client=_route(), course_ctx=ctx, course_llm=llm_)).post("/auto/ask/stream", json={"question": "什么是数据要素"}))
    done = ev[-1]
    nums = [(e["n"], e["chunk_id"]) for e in ev if e["type"] == "materials"]
    assert [n for n, _ in nums] == [1, 2] and [(s["n"], s["chunk_id"]) for s in done["sources"]] == nums
    assert done["answer"] == "先[1]后[2]"


def test_sse_ontology_marks_never_reach_client(ctx):
    pieces = ["资产入表受规定规制[onto", "1]；治理区别于管理[本体]。见[S1]。"]
    ev = _events(_app(UnifiedService(router_client=_route(), course_ctx=ctx, course_llm=Chunked(_script("".join(pieces)), pieces))).post(
        "/auto/ask/stream", json={"question": "什么是数据要素"}))
    shown = "".join(e["text"] for e in ev if e["type"] == "delta")
    assert "onto" not in shown and "本体" not in shown and shown == "资产入表受规定规制；治理区别于管理。见[1]。" and ev[-1]["answer"] == shown


# ---------------------------------------------------------------- 3. 作答提示词不让模型输出 [本体] / [onto1]

def test_answer_prompt_forbids_ontology_marks_and_relations_have_no_view_tag():
    sysm = prompts.ANSWER_SYSTEM
    assert "不要为它加任何方括号标记" in sysm and "[本体]" in sysm and "[onto1]" in sysm      # 明确禁止（这些字样只出现在「不要写」的语境里）
    assert "用 [本体] 标注" not in sysm
    rel = answer.render_relations([{"subject": "数据资产入表", "predicate_label": "受其规制", "predicate_id": "p", "object": "暂行规定",
                                    "subject_id": "C03-0071", "object_id": "C03-0072", "view": "onto1"}])
    assert rel == "数据资产入表—受其规制→暂行规定" and "onto1" not in rel and "C03-" not in rel          # 材料里不再带 [onto1] 视角标签，模型无从模仿


def test_relations_in_prompt_reach_model_without_onto_view_tag(ctx):
    kept = _kept(ctx, "P0005")[:1]
    rel = ctx.index.relations_among(["C01-0001", "C01-0003"])
    llm_ = ScriptedLLM([LLMResult(content="见[S1]")])
    answer.answer_question("q", kept, rel, llm_)
    user = llm_.calls[0]["messages"][1]["content"]
    assert "【本体结构信息】" in user and "[onto1]" not in user and "[onto2]" not in user


# ---------------------------------------------------------------- 1. 筛选提示词放宽（Pro / Flash 通用）

def test_filter_prompt_treats_components_of_composition_questions_as_direct():
    f = prompts.FILTER_SYSTEM
    assert all(w in f for w in ("构成", "组成", "环节", "类型", "方法", "机制", "区别"))
    assert "任一组成部分、环节、类型、方法或机制的段落都判 direct" in f
    assert "拿不准是否相关时判 background，不要判 irrelevant" in f and "只有明显在讲别的主题" in f
    assert '{"direct":["S1","S4"],"background":["S2"]}'.replace('{', '{{').replace('}', '}}') in f      # 仍是只输出编号的 JSON 格式
    assert "不写理由" in f and "每个编号最多出现一次" in f


def test_filter_prompt_still_formats_and_parses_id_only_json():
    from rag.course import passage_filter
    assert '"direct"' in prompts.FILTER_SYSTEM.format() and '"background"' in prompts.FILTER_SYSTEM.format()
    items, id_fmt = passage_filter._parse_verdicts('{"direct":["S1"],"background":["S2","S3"]}')
    assert id_fmt and [(i["id"], i["grade"]) for i in items] == [("S1", "direct"), ("S2", "background"), ("S3", "background")]


def test_check_filter_model_script_prompt_consistency_still_holds():
    import importlib
    cfm = importlib.import_module("test_check_filter_model").cfm
    assert cfm.build_messages()[0]["content"] == prompts.FILTER_SYSTEM


# ---------------------------------------------------------------- 4. --ids

def test_select_ids_keeps_order_and_rejects_unknown():
    from rag.course.eval_full_report import select_ids
    gold = [{"id": f"r{i}"} for i in range(11, 21)]
    assert [g["id"] for g in select_ids(gold, "r17, r13,r14")] == ["r13", "r14", "r17"]           # 保持评测集原顺序，容忍空格
    assert select_ids(gold, "") == gold and select_ids(gold, " , ") == gold                         # 不给就是全部
    with pytest.raises(SystemExit) as e:
        select_ids(gold, "r13,r99")
    assert "r99" in str(e.value)


def test_eval_e2e_main_runs_only_requested_ids(monkeypatch, tmp_path):
    import sys
    from rag.course import eval_e2e
    seen = []
    monkeypatch.setattr(eval_e2e, "in_window", lambda now: True)
    monkeypatch.setattr(eval_e2e, "run_one", lambda svc, g, sh: (seen.append(g["id"]), {"id": g["id"], "error": ""})[1])
    mk = lambda: type("C", (), {"_client": type("X", (), {"with_options": lambda s, **k: s})()})()
    monkeypatch.setattr(eval_e2e, "OpenAICompatClient", mk)
    monkeypatch.setattr(eval_e2e, "build_router_client", mk)
    monkeypatch.setattr(eval_e2e, "CourseContext", type("CC", (), {"load": staticmethod(lambda: type("K", (), {"chunkset": type("S", (), {"chunks": []})()})())}))
    monkeypatch.setattr(eval_e2e, "key_chunks", lambda ctx, key: [])
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--ids", "r13,r14,r16,r17", "--out", str(out)])
    eval_e2e.main()
    d = json.load(open(out, encoding="utf-8"))
    assert seen == ["r13", "r14", "r16", "r17"] and [r["id"] for r in d["rows"]] == seen and d["config"]["ids"] == "r13,r14,r16,r17"
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    eval_e2e.main()
    assert len(seen) == 10 and json.load(open(out, encoding="utf-8"))["config"]["ids"] == "all"      # 不给 --ids：行为不变，跑全部


def test_eval_full_accepts_ids_argument():
    src = open("rag/course/eval_full.py", encoding="utf-8").read()
    assert '"--ids"' in src and "select_ids(" in src


# ---------------------------------------------------------------- 去掉标记后不留多余空格

@pytest.mark.parametrize("raw,expected", [
    ("治理区别于管理 [本体]。", "治理区别于管理。"),
    ("可按无形资产确认 [onto1]。入表需登记[S1]。", "可按无形资产确认。入表需登记[1]。"),
    ("甲  [S99]乙", "甲乙"),                       # 无效编号同样吃掉前面的空格
    ("甲 [S1] 乙", "甲 [1] 乙"),                    # 有效引用不动空格
    ("甲 [第一章] 乙", "甲 [第一章] 乙"),            # 普通方括号不动
])
def test_dropping_a_mark_also_drops_the_space_before_it(raw, expected):
    assert render(raw)[0] == expected


def test_space_handling_identical_in_streaming_for_every_split_point():
    raw = "治理区别于管理 [本体]。可按存货确认  [onto1]，见[S1] 与 [S99] 结束。"
    expected = render(raw)[0]
    assert expected == "治理区别于管理。可按存货确认，见[1] 与结束。" or "  " not in expected
    for k in range(len(raw) + 1):
        shown, _ = _stream([raw[:k], raw[k:]])
        assert "".join(shown) == expected, k
