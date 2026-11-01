"""课程问答管线（M3）测试：全部离线，用 ScriptedLLM 按脚本返回固定的 tool call / JSON，不依赖 openai。"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from xml.sax.saxutils import escape

import pytest
from openpyxl import Workbook

from rag.course import config, agent_select, answer, bm25, chunker, evidence, llm, ontology, passage_filter, pipeline, prompts, recall
from rag.course.aliases import build_alias_table
from rag.course.llm import LLMNotConfigured, LLMResult, ToolCall, Trace, TracingLLM
from rag.course.textbook import parse_docx
from rag.course.vocab_tools import VocabIndex

SENTINEL_KEY = "kb-SENTINEL-KEY-DO-NOT-LEAK-0123456789"

_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_STYLES = (
    f'<w:styles {_NS}>'
    + "".join(f'<w:style w:type="paragraph" w:styleId="Heading{i}"><w:name w:val="heading {i}"/></w:style>' for i in range(1, 5))
    + "</w:styles>"
)

DEF_ELEMENT = "数据要素是指根据特定生产需求经过汇聚整理加工形成的能够参与价值创造的数据资源"
DEF_ASSET = "数据资产是指由特定主体合法拥有或控制且能够带来经济利益的数据资源"

# 段落序号（含空段）：
#  1 书名  2 第一章  3 第一节(H2)  4 一、(H3)  5 要素段  6 资产段  7 空  8 权属段  9 引用文献  10 [1]
# 11 目录行  12 第二章  13 第一节(H2)  14 一、(H3 同名)  15 价值段  16 引用文献  17 [2]
PARAS = [
    ("数据要素导论", "Heading1"),
    ("第一章 数据要素", "Heading1"),
    ("第一节 数据要素的概念", "Heading2"),
    ("一、数据要素的内涵", "Heading3"),
    (DEF_ELEMENT + "。该概念强调数据在生产中的投入作用。", None),
    (DEF_ASSET + "。数据资产需要入表。", None),
    ("", None),
    ("数据权属决定数据归谁所有，是数据要素流通的前提。", None),
    ("引用文献", None),
    ("[1]某某.数据的本质[J].2017.", None),
    ("一、数据价值的衡量", None),
    ("第二章 数据价值", "Heading1"),
    ("第一节 价值概述", "Heading2"),
    ("一、数据价值的衡量", "Heading3"),
    ("数据价值是数据在使用中带来的效用，数据要素的价值需要通过流通实现。", None),
    ("引用文献", None),
    ("[2]某某.价值[J].2020.", None),
]


def _docx(tmp_path):
    body = ""
    for text, style in PARAS:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        run = f"<w:r><w:t>{escape(text)}</w:t></w:r>" if text else ""
        body += f"<w:p>{ppr}{run}</w:p>"
    path = tmp_path / "book.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document {_NS}><w:body>{body}</w:body></w:document>')
        z.writestr("word/styles.xml", _STYLES)
    return path


HEADER = list(ontology._RECORD_STR_FIELDS[:12]) + [
    "source_file", "source_anchor", "retrieval_anchor", "source_section", "evidence_quote", "evidence_requirement",
    "logic_level", "flow_chain", "driving_mechanism", "confidence", "review_status", "normalization_note",
    "expert_relevance_weight"]


def _rec(rid, s, p, o, anchor, quote="", conf="0.9", w="1", olabel="", view="onto2"):
    d = dict.fromkeys(HEADER, "")
    d.update(record_id=rid, chapter_id="Ch01", profile=view, subject_id=s, subject_label=s, predicate_id=p,
             object_id=o, object_label=olabel or o, source_anchor=anchor, evidence_quote=quote,
             confidence=conf, expert_relevance_weight=w)
    return d


def _concept_row(cid, ch, name, expert="", issue="通过核验", ctx=""):
    row = [""] * 16
    row[0], row[1], row[3], row[6], row[10], row[14] = cid, ch, name, ctx, expert, issue
    return row


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    # 夹具段落都很短，默认阈值下 M1 会把 P0005/P0006/P0008 合并成一个子块；调小阈值让每段独立成块
    # 既有测试脚本化多轮选词；快速路径另有专门测试（tests/test_router.py），这里默认关闭
    monkeypatch.setattr(config, "COURSE_SELECT_FAST_PATH", False)
    # 既有测试验证旧的多轮 agent 选词 + 分批筛选；single 模式另有专门测试（tests/test_single_mode.py）
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "agent")
    # 既有测试验证本体召回本身；BM25 兜底召回另有专门测试（tests/test_router.py），这里默认关闭
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(chunker, "CHILD_MIN", 10)
    monkeypatch.setattr(chunker, "CHILD_MAX", 30)
    cx = tmp_path / "c.xlsx"
    wb = Workbook(); ws = wb.active; ws.title = "concept总表"
    ws.append(["概念ID", "章号", "章节主题", "概念名称", "概念类型", "上位或关联项", "释义或知识语境", "收录依据",
               "教材定位", "教材原文摘录", "专家原词", "专家定位", "归并及版本说明", "收录分组", "问题类型", "问题说明"])
    ws.append(["通过核验"] + [""] * 15)
    for r in [
        _concept_row("C01-0001", 1, "数据要素（Data Element）", "数据要素；数据要素的定义", ctx="参与价值创造的数据资源"),
        _concept_row("C01-0002", 1, "数据资产", ctx="可带来经济利益的数据资源"),
        _concept_row("C01-0003", 1, "数据权属", ctx="数据归属问题"),
        _concept_row("C01-0004", 1, "数据"),                                   # 泛词，不可召回
        _concept_row("C01-0005", 1, "噪声词", issue="D 非概念噪声"),
        _concept_row("C01-0006", 1, "孤立概念"),                               # 本章零字面提及
        _concept_row("C02-0001", 2, "数据价值", ctx="数据使用中带来的效用"),
        _concept_row("C02-0002", 2, "数据资产", ctx="第二章同名概念"),
    ]:
        ws.append(r)
    wb.save(cx)
    d1, d2 = tmp_path / "onto1", tmp_path / "onto2"
    d1.mkdir(); d2.mkdir()
    def write(path, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER); w.writeheader(); w.writerows(rows)
    write(d1 / "Ch01_x_onto1_9.11.csv", [
        # 定义锚在标题行 P0004：应下沉到匹配定义文本的段落 P0005
        _rec("D1", "C01-0001", "hasDefinition", "Ch01_DEF_A", "P0004", olabel=DEF_ELEMENT, w="1", view="onto1"),
        # 定义锚在参考文献 P0010：排除
        _rec("D2", "C01-0002", "hasDefinition", "Ch01_DEF_B", "P0010", olabel=DEF_ASSET, view="onto1"),
        # 定义锚在 toc 行 P0011：先解析到 P0014 标题，再下沉到 P0015
        _rec("D3", "C02-0001", "hasDefinition", "Ch02_DEF_C", "P0011", olabel="数据价值是数据在使用中带来的效用", view="onto1"),
        # 定义锚在标题行但定义文本与正文完全不相关：落到 section_first
        _rec("D4", "C01-0003", "hasDefinition", "Ch01_DEF_D", "P0003", olabel="qqq completely unrelated zzz", view="onto1"),
    ])
    write(d2 / "Ch01_x_onto2_9.11.csv", [
        _rec("E1", "C01-0001", "relatedTo", "C01-0002", "P0006", quote="数据资产需要入表", conf="0.95", w="1"),
        _rec("E2", "C01-0001", "dependsOn", "C01-0003", "P0008", quote="数据权属决定数据归谁所有", conf="0.8", w="0.8"),
        _rec("E3", "C01-0003", "supports", "C01-0001", "P0008", quote="是数据要素流通的前提", conf="0.7", w="0.6"),
        _rec("E4", "C01-0002", "relatedTo", "C01-0001", "onto2!A10", quote="", conf="0.99", w="1"),
        _rec("E5", "C01-0001", "relatedTo", "C01-0004", "P0005", quote="x", conf="0.9", w="1"),
    ])
    concepts = ontology.load_concepts(cx)
    records = ontology.load_records(d1, "onto1") + ontology.load_records(d2, "onto2")
    return pipeline.CourseContext.build(concepts, records, parse_docx(_docx(tmp_path)), use_jieba=False)


# ---------------------------------------------------------------- Scripted LLM

class ScriptedLLM:
    """按顺序返回预设的 LLMResult；记录每次调用收到的 messages，便于断言 prompt 内容。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": json.loads(json.dumps(messages)), "model": model,
                           "tools": tools, "json_mode": json_mode})
        if not self.script:
            raise AssertionError(f"脚本用尽：多余的 {step} 调用")
        item = self.script.pop(0)
        return item(step, messages) if callable(item) else item


def tc(name, **args):
    return LLMResult(tool_calls=[ToolCall(f"id{abs(hash((name, json.dumps(args, sort_keys=True)))) % 10**6}", name, args)],
                     prompt_tokens=10, completion_tokens=5)


def finish_native(*items):
    return tc("finish", selected=[{"concept_id": c, "reason": "r", "role": role} for c, role in items])


def jtxt(obj) -> LLMResult:
    return LLMResult(content=json.dumps(obj, ensure_ascii=False), prompt_tokens=10, completion_tokens=5)


def verdicts(*pairs) -> LLMResult:
    """pairs: (S# 编号, useful)。编号由 filter_passages 按送入顺序全局分配（S1、S2…）。"""
    return jtxt({"verdicts": [{"id": r, "useful": u, "reason": "r"} for r, u in pairs]})


def refs_for(items, start=1):
    """与 passage_filter.assign_refs 一致：按顺序给 items 编号 S{start}…"""
    return [f"S{start + i}" for i in range(len(items))]


def ordered(ctx, selected, strategy="A", max_chars=10**6):
    """按管线实际顺序（分批后展平）返回召回的子块，便于脚本里按位置写 S#。"""
    rr = ctx.recaller.recall(selected, strategy)
    return [i for b in recall.batch_by_budget(rr.items, max_chars) for i in b]


def chunk_of(ctx, anchor):
    return next(c for c in ctx.chunkset.chunks if anchor in c.anchors)


# ---------------------------------------------------------------- BM25 / 词表工具

def test_bm25_bigram_fallback_and_ranking():
    assert bm25.tokenize("数据权属", use_jieba=False) == ["数据", "据权", "权属"]
    assert bm25.tokenize("Data 要", use_jieba=False) == ["data", "要"]
    docs = [bm25.tokenize(t, False) for t in ("数据权属与归属", "数据价值的衡量", "无关文本")]
    top = bm25.BM25(docs).top(bm25.tokenize("数据权属", False), 3)
    assert top[0][0] == 0 and 2 not in dict(top)


def test_vocab_excludes_noise_and_generic(ctx):
    idx = ctx.index
    assert "C01-0004" not in idx.selectable and "C01-0005" not in idx.selectable
    assert idx.reject_reason("C01-0004") == "blocked_generic" and idx.reject_reason("C01-0005") == "excluded_issue_type"
    assert idx.reject_reason("C99-0001") == "unknown_concept_id" and idx.reject_reason("C01-0001") is None


def test_search_vocab_alias_bm25_and_chapter_filter(ctx):
    idx = ctx.index
    hits = idx.search_vocab("什么是数据要素？")
    assert hits[0].concept_id == "C01-0001" and hits[0].via == "alias_in_query"
    assert [h.concept_id for h in idx.search_vocab("data element")][:1] == ["C01-0001"]    # 英文括注别名
    hits = idx.search_vocab("数据资产")
    assert {h.concept_id for h in hits[:2]} == {"C01-0002", "C02-0002"} and hits[0].via == "alias_exact"
    assert all(h.via == "bm25" for h in hits[2:])                                  # bigram 下「数据」会命中其他概念，但排在精确命中之后
    ch2 = idx.search_vocab("数据资产", chapter=2)
    assert ch2[0].concept_id == "C02-0002" and {h.chapter for h in ch2} == {2}
    assert idx.search_vocab("") == [] and all(h.concept_id != "C01-0004" for h in idx.search_vocab("数据"))
    assert len(idx.search_vocab("数据", limit=1)) == 1


def test_get_concept_same_name_across_chapters(ctx):
    card = ctx.index.get_concept("C01-0002")
    assert card.same_name_ids == ["C02-0002"] and card.chapter == 1 and card.selectable
    assert ctx.index.get_concept("C01-0001").aliases and ctx.index.get_concept("nope") is None


def test_get_neighbors_is_directed_and_filters_view(ctx):
    out = ctx.index.get_neighbors("C01-0001", "onto2")
    assert {(n.concept_id, n.direction, n.predicate_id) for n in out} == {
        ("C01-0002", "out", "relatedTo"), ("C01-0003", "out", "dependsOn"),
        ("C01-0003", "in", "supports"), ("C01-0002", "in", "relatedTo")}
    assert ctx.index.get_neighbors("C01-0001", "onto1") == []
    assert all(n.concept_id != "C01-0004" for n in ctx.index.get_neighbors("C01-0001"))    # 泛词概念不出现
    with pytest.raises(ValueError):
        ctx.index.get_neighbors("C01-0001", "bad")


def test_relations_among_not_symmetrized(ctx):
    rel = ctx.index.relations_among(["C01-0001", "C01-0003"])
    assert {(r["subject_id"], r["predicate_id"], r["object_id"]) for r in rel} == {
        ("C01-0001", "dependsOn", "C01-0003"), ("C01-0003", "supports", "C01-0001")}


# ---------------------------------------------------------------- Step 1：工具调用循环

def test_select_native_loop_validates_and_drops_invalid(ctx):
    llm_ = ScriptedLLM([
        tc("search_vocab", query="数据要素"),
        tc("get_neighbors", concept_id="C01-0001", view="both"),
        finish_native(("C01-0001", "question_concept"), ("C99-9999", "related_concept"),
                      ("C01-0005", "related_concept"), ("C01-0004", "related_concept"),
                      ("C01-0003", "bad_role"), ("C01-0001", "related_concept"), ("C01-0003", "related_concept")),
    ])
    r = agent_select.select_concepts("什么是数据要素？", [], ctx.index, llm_, max_steps=6, native_tools=True)
    assert [(s.concept_id, s.role) for s in r.selected] == [("C01-0001", "question_concept"), ("C01-0003", "related_concept")]
    assert {(d.get("concept_id"), d["reason"]) for d in r.dropped} == {
        ("C99-9999", "unknown_concept_id"), ("C01-0005", "excluded_issue_type"), ("C01-0004", "blocked_generic"),
        ("C01-0003", "invalid_role"), ("C01-0001", "duplicate")}
    assert r.steps == 2 and r.finished and not r.truncated
    assert llm_.calls[0]["tools"] == prompts.TOOL_SPECS and not llm_.calls[0]["json_mode"]
    # 工具结果以 role=tool 回传，且 assistant 的 tool_calls 在其前面
    roles = [m["role"] for m in llm_.calls[1]["messages"]]
    assert roles[-2:] == ["assistant", "tool"]


def test_select_step_limit_truncates(ctx):
    llm_ = ScriptedLLM([tc("search_vocab", query="数据要素")] * 5)
    r = agent_select.select_concepts("q", [], ctx.index, llm_, max_steps=3, native_tools=True)
    assert r.steps == 2 and r.truncated and not r.finished and r.selected == [] and r.end_reason == "truncated"
    assert len(llm_.calls) == 3          # 上限 3 = 最多 3 次调用；第 3 次只提供 finish，LLM 仍不 finish → 结束
    assert [t["function"]["name"] for t in llm_.calls[2]["tools"]] == ["finish"]
    assert "最后一次回复机会（共 3 次）" in llm_.calls[2]["messages"][-1]["content"]


def _multi(*calls) -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(f"m{i}", n, a) for i, (n, a) in enumerate(calls)])


def test_select_counts_rounds_not_parallel_tool_calls_and_forced_finish_keeps_selection(ctx):
    llm_ = ScriptedLLM([
        _multi(("search_vocab", {"query": "数据要素"}), ("search_vocab", {"query": "数据资产"})),
        _multi(("get_concept", {"concept_id": "C01-0001"}), ("get_concept", {"concept_id": "C01-0002"}),
               ("get_neighbors", {"concept_id": "C01-0001"})),
        finish_native(("C01-0001", "question_concept"))])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, max_steps=3, native_tools=True)
    assert r.steps == 2 and len(r.tool_log) == 5                     # 5 个工具调用、2 轮（第 3 次调用是强制 finish）
    assert r.truncated and r.finished and r.end_reason == "finish"
    assert [s.concept_id for s in r.selected] == ["C01-0001"]
    tool_msgs = [m for m in llm_.calls[2]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 5                                        # 每个 tool_call 都有对应的 tool 结果


def test_select_finish_in_parallel_batch_is_taken(ctx):
    llm_ = ScriptedLLM([_multi(("search_vocab", {"query": "x"}),
                               ("finish", {"selected": [{"concept_id": "C01-0001", "reason": "r", "role": "question_concept"}]}))])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, max_steps=6, native_tools=True)
    assert [s.concept_id for s in r.selected] == ["C01-0001"] and r.steps == 0 and len(llm_.calls) == 1


def test_select_json_path_forced_finish(ctx):
    llm_ = ScriptedLLM([jtxt({"tool": "search_vocab", "args": {"query": "数据要素"}}),
                        jtxt({"tool": "finish", "args": {"selected": [
                            {"concept_id": "C01-0001", "reason": "r", "role": "question_concept"}]}})])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, max_steps=2, native_tools=False)
    assert r.truncated and r.end_reason == "finish" and [s.concept_id for s in r.selected] == ["C01-0001"]
    assert "最后一次回复机会（共 2 次）" in llm_.calls[1]["messages"][-1]["content"]


def test_select_active_concepts_rendered_in_prompt(ctx):
    llm_ = ScriptedLLM([finish_native()])
    agent_select.select_concepts("q", ["C01-0003"], ctx.index, llm_, native_tools=True)
    assert "C01-0003《数据权属》" in llm_.calls[0]["messages"][0]["content"]


def test_select_tool_errors_are_returned_to_llm_not_raised(ctx):
    out = agent_select.run_tool(ctx.index, "get_concept", {"concept_id": "nope"})
    assert "error" in out and "error" in agent_select.run_tool(ctx.index, "get_neighbors", {"concept_id": "C01-0001", "view": "x"})
    assert "error" in agent_select.run_tool(ctx.index, "rm_rf", {})


def test_select_json_fallback_path_without_native_tools(ctx):
    llm_ = ScriptedLLM([
        jtxt({"tool": "search_vocab", "args": {"query": "数据权属"}}),
        jtxt({"tool": "finish", "args": {"selected": [{"concept_id": "C01-0003", "reason": "r", "role": "question_concept"}]}}),
    ])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, native_tools=False)
    assert [s.concept_id for s in r.selected] == ["C01-0003"] and r.finished
    assert all(c["tools"] is None for c in llm_.calls)                       # 回退路径不传 tools
    assert '"tool": "search_vocab"' in llm_.calls[0]["messages"][0]["content"]    # 协议说明已追加且花括号已还原
    assert "工具结果：" in llm_.calls[1]["messages"][-1]["content"]


def test_select_json_fallback_retries_once_then_gives_up(ctx):
    llm_ = ScriptedLLM([LLMResult(content="我觉得应该选数据要素"), LLMResult(content="还是不是 JSON")])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, native_tools=False)
    assert r.selected == [] and r.parse_failures == 2 and len(llm_.calls) == 2
    assert "无法解析" in llm_.calls[1]["messages"][-1]["content"]
    ok = ScriptedLLM([LLMResult(content="```json\n{bad"), jtxt({"tool": "finish", "args": {"selected": []}})])
    r2 = agent_select.select_concepts("q", [], ctx.index, ok, native_tools=False)
    assert r2.finished and r2.parse_failures == 1


def test_select_json_rejects_unknown_tool_name(ctx):
    llm_ = ScriptedLLM([jtxt({"tool": "drop_table", "args": {}}), jtxt({"tool": "finish", "args": {"selected": []}})])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, native_tools=False)
    assert r.parse_failures == 1 and r.finished


# ---------------------------------------------------------------- Step 2：召回

def _recall(ctx, ids, strategy="A"):
    return ctx.recaller.recall(ids, strategy)


def test_recall_dedups_chunks_and_records_reasons(ctx):
    rr = _recall(ctx, [("C01-0001", "question_concept"), ("C01-0003", "related_concept")])
    ids = [i.chunk_id for i in rr.items]
    assert len(ids) == len(set(ids))
    c8 = next(i for i in rr.items if "P0008" in i.anchors)
    labels = {(r.concept_id, r.via, r.predicate_id, r.direction) for r in c8.reasons}
    assert ("C01-0001", "edge_evidence", "dependsOn", "out") in labels         # 同一子块同时因两个概念、两条边被召回
    assert ("C01-0003", "edge_evidence", "supports", "out") in labels
    assert ("C01-0003", "edge_evidence", "dependsOn", "in") in labels          # 方向保留：C01-0003 是 dependsOn 的宾语
    assert ("C01-0003", "mention", None, None) in labels                       # 名称出现在 P0008 正文


def test_recall_excludes_title_references_and_non_paragraph_anchors(ctx):
    rr = _recall(ctx, [("C01-0002", "question_concept")])
    why = {(e["record_id"], e["reason"]) for e in rr.excluded}
    assert ("D2", "references") in why and ("E4", "non_paragraph_anchor") in why
    assert all("P0010" not in i.anchors and "P0009" not in i.anchors for i in rr.items)


def test_title_row_sinks_to_matched_paragraph(ctx):
    rr = _recall(ctx, [("C01-0001", "related_concept")], "B")
    (d1,) = [r for i in rr.items for r in i.reasons if r.record_id == "D1"]
    assert d1.sunk == evidence.SINK_MATCHED and d1.sink_score >= 0.6 and d1.anchor == "P0004"
    item = next(i for i in rr.items if "P0005" in i.anchors)
    assert d1 in item.reasons and d1.evidence == "verified"
    (rec,) = [s for s in rr.sunk if s["record_id"] == "D1"]
    assert (rec["from"], rec["to"], rec["via"]) == ("P0004", "P0005", evidence.SINK_MATCHED)


def test_title_row_low_coverage_becomes_unmatched_and_is_not_recalled_as_evidence(ctx):
    rr = _recall(ctx, [("C01-0003", "related_concept")], "B")
    assert not [r for i in rr.items for r in i.reasons if r.record_id == "D4"]           # 不作为教材证据召回
    (rec,) = [s for s in rr.sunk if s["record_id"] == "D4"]
    assert rec["via"] == evidence.SINK_UNMATCHED and rec["score"] == 0.0 and rec["to"] == "P0005"
    assert [o["record_id"] for o in rr.ontology_only] == ["D4"]                            # 只作 [本体] 信息
    assert rr.ontology_only[0]["via"] == evidence.SINK_UNMATCHED


def test_title_row_mid_coverage_falls_back_to_section_first_and_demotes(ctx):
    # 定义的前半句出自 P0005、后半句在书中不存在：小节内覆盖率约 0.5（>=0.4 且 <0.6），也不满足全章 0.8
    half = DEF_ELEMENT[:20] + "这一段后半截的文字教材里完全没有出现过的内容啊"
    s = ctx.anchors.sink_heading(ctx.anchors.unit(3), half)
    assert s.via == evidence.SINK_FIRST and 0.4 <= s.score < 0.6 and s.to_anchor == "P0005"
    rec = ontology.Record(**{**{f: "" for f in ontology._RECORD_STR_FIELDS}, "record_id": "M1", "view": "onto1",
                             "subject_id": "C01-0003", "object_id": "Ch01_DEF_M", "predicate_id": "hasDefinition",
                             "object_label": half, "source_anchor": "P0003", "confidence": 0.9,
                             "expert_relevance_weight": 1.0})
    rc = recall.Recaller([rec], ctx.chunkset, ctx.anchors, ctx.aliases, ctx.concepts)
    rr = rc.recall([("C01-0003", "related_concept")], "B")
    (reason,) = [r for i in rr.items for r in i.reasons if r.record_id == "M1"]
    assert reason.sunk == evidence.SINK_FIRST and reason.evidence == "unverified"
    assert next(i for i in rr.items if "P0005" in i.anchors).tier == 2 and not rr.ontology_only


def test_no_relocation_across_chapters(ctx):
    # 第二章的标题 P0013 范围内只有 P0015；DEF_ASSET 出自第一章 P0006，全章重定位只看本章，不跨章
    s = ctx.anchors.sink_heading(ctx.anchors.unit(13), DEF_ASSET)
    assert s.via == evidence.SINK_UNMATCHED and s.alt_anchor == "P0015"                    # 备选只来自本章，绝不是第一章的 P0006


def test_matching_inside_subsections_is_not_relocation(ctx):
    h3 = ctx.anchors.unit(4)                                  # 「一、数据要素的内涵」，范围 P0005、P0006、P0008
    s = ctx.anchors.sink_heading(h3, DEF_ELEMENT)
    assert s.via == evidence.SINK_MATCHED and s.to_anchor == "P0005"


def test_relocated_via_chapter_pool(tmp_path):
    import zipfile
    from xml.sax.saxutils import escape

    paras = [
        ("第一章 甲", "Heading1"),                                                   # P1
        ("第一节 小节甲", "Heading2"),                                               # P2
        ("这里只有一段与定义毫无关系的文字，讲的是别的事情。", None),               # P3
        ("第二节 小节乙", "Heading2"),                                               # P4
        ("数据要素安全流通是指在保障数据安全的前提下实现数据在主体之间的合规流动与价值释放。", None),  # P5 完整定义在另一小节
        ("引用文献", None), ("[1]某。", None),
    ]
    body = ""
    for text, style in paras:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
    p = tmp_path / "r.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document {_NS}><w:body>{body}</w:body></w:document>')
        z.writestr("word/styles.xml", _STYLES)
    ix = evidence.AnchorIndex(parse_docx(p))
    defin = "数据要素安全流通是指在保障数据安全的前提下，实现数据在主体之间的合规流动与价值释放"
    s = ix.sink_heading(ix.unit(2), defin)                                      # 锚在 P2，小节内只有 P3
    assert s.via == evidence.SINK_RELOCATED and s.to_anchor == "P0005" and s.score >= 0.8
    # 全章也只有 0.8 以下的近似：不 relocate，且作为 alt 给人工核对
    near = "数据要素安全流通是指" + "完全不同的另外一些内容" * 4
    s2 = ix.sink_heading(ix.unit(2), near)
    assert s2.via in (evidence.SINK_FIRST, evidence.SINK_UNMATCHED)


def test_definition_with_semicolons_requires_every_segment(ctx):
    h = ctx.anchors.unit(3)
    both = DEF_ELEMENT + "；" + DEF_ASSET                                          # 两段分别出自 P0005、P0006
    s = ctx.anchors.sink_heading(h, both)
    assert s.via == evidence.SINK_MATCHED and s.to_anchor == "P0005" and s.extra_anchors == ("P0006",)
    assert s.score >= 0.9                                                           # 取各段最小覆盖率，而不是被拼接串稀释
    one_bad = DEF_ELEMENT + "；qqq nothing like it anywhere zzz"
    s2 = ctx.anchors.sink_heading(h, one_bad)
    assert s2.via != evidence.SINK_MATCHED and s2.via != evidence.SINK_RELOCATED     # 一段不过关，整体不算匹配
    # 整串在一个段落里的老用法不受影响
    assert ctx.anchors.sink_heading(h, DEF_ASSET).extra_anchors == ()


def test_semicolon_definition_recalls_all_segment_chunks(ctx):
    rec = ontology.Record(**{**{f: "" for f in ontology._RECORD_STR_FIELDS}, "record_id": "S1", "view": "onto1",
                             "subject_id": "C01-0003", "object_id": "Ch01_DEF_S", "predicate_id": "hasDefinition",
                             "object_label": DEF_ELEMENT + "；" + DEF_ASSET, "source_anchor": "P0003",
                             "confidence": 0.9, "expert_relevance_weight": 1.0})
    rc = recall.Recaller([rec], ctx.chunkset, ctx.anchors, ctx.aliases, ctx.concepts)
    rr = rc.recall([("C01-0003", "related_concept")], "B")
    got = {a for i in rr.items for a in i.anchors}
    assert {"P0005", "P0006"} <= got
    assert all(r.sunk == evidence.SINK_MATCHED for i in rr.items for r in i.reasons)


def test_toc_anchor_resolves_to_real_heading_then_sinks(ctx):
    rr = _recall(ctx, [("C02-0001", "related_concept")], "B")
    (rec,) = [s for s in rr.sunk if s.get("via") == "toc→heading"]
    assert (rec["from"], rec["to"]) == ("P0011", "P0014")
    item = next(i for i in rr.items if "P0015" in i.anchors)
    assert any(r.record_id == "D3" and r.sunk == evidence.SINK_MATCHED for r in item.reasons)


def test_sink_scope_includes_subsections_in_document_order(ctx):
    h2 = ctx.anchors.unit(3)
    units = ctx.anchors.section_units(h2)
    assert [u.anchor for u in units] == ["P0005", "P0006", "P0008"]           # 含下级 H3，按文档顺序，不含 references
    s = ctx.anchors.sink_heading(h2, DEF_ASSET)
    assert (s.to_anchor, s.via) == ("P0006", evidence.SINK_MATCHED)
    assert ctx.anchors.sink_heading(h2, "qqq", fallback_first=False) is None


def test_coverage_beats_ratio_for_short_text_in_long_paragraph(ctx):
    from difflib import SequenceMatcher

    short, para = "数据价值是数据在使用中带来的效用", PARAS[14][0] * 3
    assert SequenceMatcher(None, para, short).ratio() < 0.6                    # ratio 会被长段落稀释
    s = ctx.anchors.sink_heading(ctx.anchors.unit(14), short)
    assert s.via == evidence.SINK_MATCHED and s.score >= 0.9


def test_mention_only_within_concept_own_chapter(ctx):
    assert ctx.recaller.mention_anchors("C01-0001") == ["P0005", "P0008"]       # 第二章 P0015 也含「数据要素」，但不跨章
    assert ctx.recaller.mention_anchors("C02-0001") == ["P0015"]
    assert ctx.recaller.mention_anchors("C01-0002") == ["P0006"]                # P0005 里没有「数据资产」；第二章同名概念不在本章


def test_strategy_b_skips_mention_for_related_concepts_and_reports_zero_mention(ctx):
    sel = [("C01-0001", "question_concept"), ("C01-0006", "related_concept")]
    a, b = _recall(ctx, sel, "A"), _recall(ctx, sel, "B")
    assert "C01-0006" in a.zero_mention and "C01-0006" in b.zero_mention
    only_mention = lambda rr, cid: [i for i in rr.items if all(r.via == "mention" and r.concept_id == cid for r in i.reasons)]  # noqa: E731
    assert only_mention(a, "C01-0006") == [] and len(b.items) <= len(a.items)
    rb = _recall(ctx, [("C01-0003", "related_concept")], "B")
    assert not any(r.via == "mention" for i in rb.items for r in i.reasons)


def test_tier_ordering_and_batching_never_drops(ctx):
    rr = _recall(ctx, [("C01-0001", "question_concept"), ("C01-0003", "related_concept")])
    assert [i.tier for i in rr.items] == sorted(i.tier for i in rr.items)
    one = recall.batch_by_budget(rr.items, 10**6)
    assert len(one) == 1 and len(one[0]) == len(rr.items)
    tiny = recall.batch_by_budget(rr.items, 1)                                  # 每个子块都超限：各自成批，不丢
    assert [len(b) for b in tiny] == [1] * len(rr.items)
    assert {i.chunk_id for b in tiny for i in b} == {i.chunk_id for i in rr.items}
    mid = recall.batch_by_budget(rr.items, sum(len(i.text) for i in rr.items) // 2)
    assert len(mid) >= 2 and [i.tier for b in mid for i in b] == sorted(i.tier for b in mid for i in b)


def test_split_paragraph_recalls_only_matching_piece_else_all(ctx):
    from rag.course.chunker import Chunk, ChunkSet

    mk = lambda cid, part, text: Chunk(cid, "p", "s", "b", 1, ["h"], "body", text, text, ["P0099"], part)  # noqa: E731
    cs = ChunkSet("t", "x", chunks=[mk("c0", 0, "第一句话。第二句话。"), mk("c1", 1, "引文所在的第三句。")])
    loc = recall.ChunkLocator(cs)
    assert [c.chunk_id for c in loc.find("P0099", "引文所在的第三句")] == ["c1"]
    assert [c.chunk_id for c in loc.find("P0099", "查无此句")] == ["c0", "c1"]
    assert [c.chunk_id for c in loc.find("P0099")] == ["c0", "c1"]


def test_sink_report_distribution(ctx):
    recs = [r for r in ctx.records if r.view == "onto1"]
    rep = recall.sink_report(recs, ctx.anchors)
    assert rep["total"] == 3 and rep["by_via"] == {evidence.SINK_MATCHED: 2, evidence.SINK_UNMATCHED: 1}   # D2 在参考文献，不计；D4 覆盖率 0
    assert sum(rep["score_histogram"].values()) == 3
    d4 = next(r for r in rep["rows"] if r["record_id"] == "D4")
    assert d4["via"] == evidence.SINK_UNMATCHED and d4["to"] == "P0005" and d4["extra"] == []


def test_default_strategy_is_b_and_a_is_opt_in(ctx):
    assert recall.DEFAULT_STRATEGY == recall.STRATEGY_B
    sel = [("C01-0001", "question_concept"), ("C01-0003", "related_concept")]
    assert ctx.recaller.recall(sel).strategy == "B" and pipeline.DEFAULT_STRATEGY == "B"
    default, a = ctx.recaller.recall(sel), ctx.recaller.recall(sel, "A")
    assert [i.chunk_id for i in default.items] == [i.chunk_id for i in ctx.recaller.recall(sel, "B").items]
    assert len(a.items) >= len(default.items)


def test_dry_run_report_compares_strategies(ctx):
    sel = [("C01-0001", "question_concept"), ("C01-0003", "related_concept")]
    rep = recall.dry_run_report(ctx.recaller, sel, max_chars=60, max_steps=6)
    a, b = rep["strategies"]["A"], rep["strategies"]["B"]
    assert a["chunks"] >= b["chunks"] and a["batches"] >= 1
    assert a["est_llm_calls"]["total_max"] == 6 + a["batches"] + 1 and a["est_llm_calls"]["step4"] == 1


# ---------------------------------------------------------------- Step 3：筛选

def _items(ctx, *anchors):
    rr = _recall(ctx, [("C01-0001", "question_concept"), ("C01-0003", "related_concept")])
    return [i for i in rr.items if any(a in i.anchors for a in anchors)]


def test_filter_drops_fabricated_refs_and_records_unjudged(ctx):
    items = _items(ctx, "P0005", "P0008")
    a, b = items[0].chunk_id, items[1].chunk_id
    llm_ = ScriptedLLM([verdicts(("S1", True), ("S99", True), ("S1", False))])
    fr = pipeline.filter_passages("q", [items], llm_, native_tools=True)
    assert fr.useful_ids == [a] and fr.dropped_unknown_ids == ["S99"] and fr.duplicates == ["S1"]
    assert fr.unjudged == [b] and llm_.calls[0]["json_mode"] is True
    assert fr.batches[0]["unjudged"] == 1 and fr.refs == {"S1": a, "S2": b}


def test_filter_refs_are_global_across_batches_and_checked_per_batch(ctx):
    items = _items(ctx, "P0005", "P0008")
    a, b = items[0].chunk_id, items[1].chunk_id
    # 第一批只有 S1；第二批只有 S2。第二批里出现 S1 属于别的批 → 视为编造
    llm_ = ScriptedLLM([verdicts(("S1", True)), verdicts(("S1", True), ("S2", False))])
    fr = pipeline.filter_passages("q", [[items[0]], [items[1]]], llm_, native_tools=True)
    assert fr.useful_ids == [a] and fr.dropped_unknown_ids == ["S1"] and [v.chunk_id for v in fr.verdicts] == [a, b]
    assert len(fr.batches) == 2 and len(llm_.calls) == 2
    assert "[S2]" in llm_.calls[1]["messages"][1]["content"] and "[S1]" not in llm_.calls[1]["messages"][1]["content"]


def test_filter_accepts_ref_spellings(ctx):
    (it,) = _items(ctx, "P0008")[:1]
    for raw in ("S1", "s1", "[S1]", " S1 "):
        llm_ = ScriptedLLM([jtxt({"verdicts": [{"id": raw, "useful": True, "reason": "r"}]})])
        assert pipeline.filter_passages("q", [[it]], llm_, native_tools=True).useful_ids == [it.chunk_id]


def test_filter_json_fallback_retry_and_bare_array(ctx):
    (it,) = _items(ctx, "P0008")[:1]
    llm_ = ScriptedLLM([LLMResult(content="好的，我来判断"), LLMResult(content=json.dumps(
        [{"id": "S1", "useful": True, "reason": "r"}]))])
    fr = pipeline.filter_passages("q", [[it]], llm_, native_tools=False)
    assert fr.useful_ids == [it.chunk_id] and fr.batches[0]["parse_failures"] == 1
    assert all(c["json_mode"] is False for c in llm_.calls) and "不要加 Markdown" in llm_.calls[0]["messages"][0]["content"]
    bad = ScriptedLLM([LLMResult(content="no"), LLMResult(content="still no")])
    fr2 = pipeline.filter_passages("q", [[it]], bad, native_tools=False)
    assert fr2.verdicts == [] and fr2.unjudged == [it.chunk_id] and fr2.batches[0]["parse_failures"] == 2


def test_filter_prompt_has_short_refs_and_no_recall_reason_or_chunk_id(ctx):
    (it,) = _items(ctx, "P0008")[:1]
    assert it.reasons                                            # 该子块确实有召回原因，下面断言它们没有泄漏给 LLM
    llm_ = ScriptedLLM([verdicts(("S1", True))])
    pipeline.filter_passages("为什么需要确权？", [[it]], llm_, native_tools=True)
    sysm, user = (m["content"] for m in llm_.calls[0]["messages"])
    assert "为什么需要确权？" in user and "[S1]" in user and it.text in user and " › ".join(it.path) in user
    assert it.chunk_id not in user and "de:" not in user
    for leaked in ("召回原因", "edge_evidence", "mention", "dependsOn", "supports", "C01-"):
        assert leaked not in user
    assert "较低的一档" in sysm and "拿不准时判 true" not in sysm


# ---------------------------------------------------------------- Step 4：作答与引用

def test_answer_uses_short_refs_and_maps_back_to_chunk_ids(ctx):
    kept = _items(ctx, "P0005", "P0008")[:2]
    a, b = kept[0].chunk_id, kept[1].chunk_id
    text = f"数据要素是…[S1]。权属见 [S2]；另有 [S9] 与 [{a}] 与 [de:fake:000]，本体中记录为依赖 [本体]。参见 [第一章 › 第一节]。"
    llm_ = ScriptedLLM([LLMResult(content=text)])
    ar = answer.answer_question("q", kept, [], llm_)
    sysm, user = (m["content"] for m in llm_.calls[0]["messages"])
    assert "[S1]" in user and "[S2]" in user and a not in user and b not in user       # LLM 只看到短编号
    assert ar.citations == [a, b] and ar.refs == {"S1": a, "S2": b}
    assert set(ar.invalid_citations) == {"S9", a, "de:fake:000"}                        # 越界编号、绕过编号直接写 chunk_id、编造 ID 都不信任
    assert f"[{a}]" in ar.text and f"[{b}]" in ar.text                                   # 有效 [S#] 换回 [chunk_id]
    assert "[S9]" not in ar.text and "de:fake:000" not in ar.text
    assert "[本体]" in ar.text and ar.uses_ontology and "[第一章 › 第一节]" in ar.text


def test_answer_citations_only_from_kept_chunks(ctx):
    kept = _items(ctx, "P0005")[:1]
    other = _items(ctx, "P0008")[0]
    assert other.chunk_id != kept[0].chunk_id
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="只有 [S1]，没有 [S2]")]))
    assert ar.citations == [kept[0].chunk_id] and ar.invalid_citations == ["S2"] and "[S2]" not in ar.text


def test_answer_prompt_separates_textbook_from_ontology(ctx):
    kept = _items(ctx, "P0005")[:1]
    rel = ctx.index.relations_among(["C01-0001", "C01-0003"])
    llm_ = ScriptedLLM([LLMResult(content="材料不足。教材材料不足以回答")])
    ar = answer.answer_question("q", kept, rel, llm_)
    sysm, user = (m["content"] for m in llm_.calls[0]["messages"])
    # 新提示词不再要求声明来源性质（师兄反馈：防御性写作）：系统提示里没有「非教材原文」，本体段仍与教材段分开
    assert "非教材原文" not in sysm and "【教材原文】" in user and "【本体结构信息】" in user
    assert user.index("【教材原文】") < user.index("[S1]") < user.index("【本体结构信息】")
    assert "—dependsOn→" in user
    assert ar.insufficient


# ---------------------------------------------------------------- 管线：回退 / 全流程 / 保密

SEL2 = [("C01-0001", "question_concept"), ("C01-0003", "related_concept")]


def test_pipeline_full_run_native(ctx):
    items = ordered(ctx, SEL2)
    keep, drop = items[0], items[-1]
    refs = refs_for(items)
    script = [
        tc("search_vocab", query="数据要素"),
        finish_native(*SEL2),
        verdicts(*[(r, i.chunk_id == keep.chunk_id) for r, i in zip(refs, items)]),
        LLMResult(content="数据要素是参与价值创造的数据资源[S1]；本体记录其依赖数据权属[本体]。",
                  prompt_tokens=100, completion_tokens=20),
    ]
    llm_ = ScriptedLLM(script)
    res = pipeline.ask("什么是数据要素？", ctx=ctx, llm=llm_)
    assert res["fallback"] is None and res["answer"]["citations"] == [keep.chunk_id]
    assert [c["chunk_id"] for c in res["citations"]] == [keep.chunk_id]
    assert f"[{keep.chunk_id}]" in res["answer"]["text"] and "[S1]" not in res["answer"]["text"]   # 输出时换回 chunk_id
    assert [s["concept_id"] for s in res["selected_concepts"]] == ["C01-0001", "C01-0003"]
    assert {r["chunk_id"] for r in res["recalled"]} == {i.chunk_id for i in items}
    assert [c["step"] for c in res["trace"].calls] == ["select", "select", "filter", "answer"]
    assert res["trace"].totals()["calls"] == 4 and res["trace"].totals()["prompt_tokens"] == 130
    ans_user = llm_.calls[-1]["messages"][1]["content"]
    assert keep.text in ans_user and drop.text not in ans_user                             # 作答只看筛选后的子块
    assert keep.chunk_id not in ans_user                                                   # 且只以短编号呈现
    assert res["trace"].stages["filter"]["refs"]["S1"] == items[0].chunk_id
    json.dumps(pipeline.result_to_json(res), ensure_ascii=False)                           # 全部可序列化


def test_pipeline_no_chunk_id_ever_sent_to_llm(ctx):
    items = ordered(ctx, SEL2)
    llm_ = ScriptedLLM([finish_native(*SEL2), verdicts(*[(r, True) for r in refs_for(items)]),
                        LLMResult(content="见[S1]")])
    pipeline.ask("q", ctx=ctx, llm=llm_)
    for call in llm_.calls[1:]:
        blob = json.dumps(call["messages"], ensure_ascii=False)
        assert "de:0906" not in blob and "de:" not in blob


def test_pipeline_fallback_no_valid_concepts(ctx):
    llm_ = ScriptedLLM([finish_native(("C99-0000", "question_concept"), ("C01-0004", "question_concept"))])
    res = pipeline.ask("q", ctx=ctx, llm=llm_)
    assert res["fallback"]["reason"] == "no_valid_concepts" and res["answer"] is None and len(llm_.calls) == 1
    assert res["fallback"]["status"] == "insufficient_evidence" and res["fallback"]["next"] == "hybrid_retriever(stub)"
    assert {d["reason"] for d in res["trace"].stages["select"]["dropped"]} == {"unknown_concept_id", "blocked_generic"}


def test_pipeline_fallback_all_filtered_out(ctx):
    sel = [("C01-0003", "question_concept")]
    items = ordered(ctx, sel)
    llm_ = ScriptedLLM([finish_native(*sel), verdicts(*[(r, False) for r in refs_for(items)])])
    res = pipeline.ask("q", ctx=ctx, llm=llm_)
    assert res["fallback"]["reason"] == "all_filtered_out" and res["answer"] is None and res["citations"] == []
    assert len(llm_.calls) == 2                                                     # 没有调用 Step 4


def test_pipeline_fallback_no_recall(ctx):
    llm_ = ScriptedLLM([finish_native(("C01-0006", "related_concept"))])           # 孤立概念：无边无提及
    res = pipeline.ask("q", ctx=ctx, llm=llm_, strategy="B")
    assert res["fallback"]["reason"] == "no_recall" and res["trace"].stages["recall"]["zero_mention"] == ["C01-0006"]


def test_pipeline_oversized_recall_is_batched_and_traced(ctx):
    items = ordered(ctx, SEL2, max_chars=1)
    n = len(items)
    script = [finish_native(*SEL2)]
    script += [verdicts((r, True)) for r in refs_for(items)]                        # max_chars=1 → 每块一批，S# 跨批全局递增
    script += [LLMResult(content="ok")]
    llm_ = ScriptedLLM(script)
    res = pipeline.ask("q", ctx=ctx, llm=llm_, max_chars=1)
    st = res["trace"].stages
    assert len(st["recall"]["batches"]) == n and len(st["filter"]["batches"]) == n
    assert [b["tier_min"] for b in st["recall"]["batches"]] == sorted(b["tier_min"] for b in st["recall"]["batches"])
    assert st["filter"]["unjudged"] == [] and st["filter"]["dropped_unknown_ids"] == [] and res["answer"] is not None
    assert [f"[S{i + 1}]" in llm_.calls[1 + i]["messages"][1]["content"] for i in range(n)] == [True] * n


def test_pipeline_dry_run_never_touches_llm(ctx):
    res = pipeline.ask("q", ctx=ctx, llm=None, dry_run=True,
                       select=[("C01-0001", "question_concept"), ("C01-0003", "related_concept"), ("C99-0000", "x")])
    assert [s["concept_id"] for s in res["selected_concepts"]] == ["C01-0001", "C01-0003"]   # 非法 ID 被丢
    assert set(res["dry_run"]["strategies"]) == {"A", "B"} and res["answer"] is None
    with pytest.raises(ValueError):
        pipeline.ask("q", ctx=ctx, llm=None)


# ---------------------------------------------------------------- reasoning_content / LLM 客户端 / 密钥

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_response(content="最终答案", reasoning="很长很长的思考过程" * 20, calls=None):
    msg = _Obj(content=content, reasoning_content=reasoning, tool_calls=calls)
    usage = _Obj(prompt_tokens=11, completion_tokens=22, completion_tokens_details=_Obj(reasoning_tokens=7))
    return _Obj(choices=[_Obj(message=msg)], usage=usage)


def test_reasoning_content_dropped_only_length_traced():
    r = llm.parse_response(_fake_response(), 1.5)
    assert r.content == "最终答案" and r.reasoning_chars == len("很长很长的思考过程" * 20) and r.reasoning_tokens == 7
    assert "思考" not in repr(r)
    trace = Trace()
    t = TracingLLM(ScriptedLLM([r]), trace)
    t.chat("answer", [{"role": "user", "content": "x"}], model="m")
    call = trace.calls[0]
    assert call["reasoning_chars"] == r.reasoning_chars and call["prompt_tokens"] == 11 and call["model"] == "m"
    assert "思考" not in json.dumps(trace.to_dict(), ensure_ascii=False)


def test_parse_response_tool_calls_and_bad_arguments():
    fn = lambda name, args: _Obj(id="c1", function=_Obj(name=name, arguments=args))  # noqa: E731
    r = llm.parse_response(_fake_response(content=None, calls=[fn("search_vocab", '{"query": "数据"}'), fn("finish", "{oops")]))
    assert r.tool_calls[0].args == {"query": "数据"} and r.tool_calls[1].args == {"_parse_error": True}


def test_parse_json_value_tolerates_fences_and_prose():
    assert llm.parse_json_value('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json_value('好的：{"a": [1]} 以上') == {"a": [1]}
    assert llm.parse_json_value('结果 [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        llm.parse_json_value("没有 json")
    with pytest.raises(ValueError):
        llm.parse_json_object("[1]")


def test_client_requires_base_and_key_without_leaking(monkeypatch):
    with pytest.raises(LLMNotConfigured) as e:
        llm.OpenAICompatClient(base="", key=SENTINEL_KEY)
    assert "COURSE_LLM_API_BASE" in str(e.value) and SENTINEL_KEY not in str(e.value)
    with pytest.raises(LLMNotConfigured) as e2:
        llm.OpenAICompatClient(base="http://example.invalid/v1", key="")
    assert "COURSE_LLM_API_KEY" in str(e2.value)


def test_client_without_openai_package_is_clear_error_and_mocks_do_not_import_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)                    # 模拟未安装
    with pytest.raises(LLMNotConfigured) as e:
        llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY)
    assert "openai" in str(e.value) and SENTINEL_KEY not in str(e.value)
    ScriptedLLM([])                                                       # mock 路径不需要 openai


def test_client_repr_and_errors_never_contain_key(monkeypatch):
    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            outer = self
            self.chat = _Obj(completions=_Obj(create=lambda **kw: (_ for _ in ()).throw(
                RuntimeError(f"401 bad key {api_key} for {kw['model']}"))))

    monkeypatch.setitem(sys.modules, "openai", _Obj(OpenAI=FakeOpenAI))
    c = llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY)
    assert SENTINEL_KEY not in repr(c) and SENTINEL_KEY not in str(vars(c).get("_base", ""))
    with pytest.raises(llm.LLMError) as e:
        c.chat("select", [{"role": "user", "content": "x"}], model="m")
    assert SENTINEL_KEY not in str(e.value) and "***" in str(e.value)


def test_client_sends_temperature_zero_and_json_mode(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            def create(**kw):
                seen.clear()
                seen.update(kw)
                return _fake_response()
            self.chat = _Obj(completions=_Obj(create=create))

    monkeypatch.setitem(sys.modules, "openai", _Obj(OpenAI=FakeOpenAI))
    c = llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY)
    c.chat("filter", [{"role": "user", "content": "x"}], model="deepseek/deepseek-v4-pro", json_mode=True)
    assert seen["temperature"] == 0 and seen["response_format"] == {"type": "json_object"} and "tools" not in seen
    c.chat("select", [{"role": "user", "content": "x"}], model="m", tools=prompts.TOOL_SPECS)
    assert seen["tools"] == prompts.TOOL_SPECS and "response_format" not in seen


def test_key_absent_from_result_trace_and_outputs(ctx, monkeypatch):
    monkeypatch.setattr("rag.course.config.COURSE_LLM_API_KEY", SENTINEL_KEY)
    sel = [("C01-0003", "question_concept")]
    items = ordered(ctx, sel)
    llm_ = ScriptedLLM([finish_native(*sel), verdicts(*[(r, True) for r in refs_for(items)]),
                        LLMResult(content="见[S1]")])
    res = pipeline.ask("q", ctx=ctx, llm=llm_)
    assert SENTINEL_KEY not in json.dumps(pipeline.result_to_json(res), ensure_ascii=False)


def test_config_defaults_do_not_hardcode_api_base():
    from rag.course import config

    assert config.COURSE_LLM_MODEL
    src = open(config.__file__, encoding="utf-8").read()
    assert 'COURSE_LLM_API_BASE = os.environ.get("COURSE_LLM_API_BASE", "")' in src      # 不写死默认地址
    assert "kb-" not in src


def test_every_prompt_template_formats_without_error():
    """所有字符串模板都过一遍 str.format：JSON 示例的花括号没双写时会在这里报错。"""
    values = dict(max_steps=6, active_concepts="（无）", question="q", items="x", error="e", ref="S1",
                  path="a › b", text="t", passages="p", ontology="o", subject="甲", subject_id="C01-0001",
                  predicate="p", object="乙", object_id="C01-0002", view="onto2", candidates="C01-0001｜甲｜第1章｜定义")
    names = [n for n, v in vars(prompts).items() if n.isupper() and isinstance(v, str)]
    assert {"SELECT_SYSTEM", "SELECT_JSON_PROTOCOL", "SELECT_USER", "FILTER_SYSTEM", "FILTER_JSON_FALLBACK",
            "RETRY_JSON", "FILTER_USER", "FILTER_ITEM", "ANSWER_SYSTEM", "ANSWER_USER", "ANSWER_PASSAGE",
            "SINGLE_SELECT_SYSTEM", "SINGLE_SELECT_USER",
            "ANSWER_RELATION"} <= set(names)
    for n in names:
        getattr(prompts, n).format(**values)               # 任何一个模板花括号写错都会在这里抛 KeyError / ValueError
    # JSON 示例经 format 后必须是合法 JSON 片段（单花括号）
    assert '{"verdicts":[{"id":"S1","grade":"direct"' in prompts.FILTER_SYSTEM.format()
    assert '{"tool": "finish"' in prompts.SELECT_JSON_PROTOCOL.format()
    assert json.loads(prompts.FILTER_SYSTEM.format().split("输出严格 JSON：")[1].split("\n")[0])["verdicts"][0]["id"] == "S1"


def test_prompt_contents_follow_review():
    sel = prompts.SELECT_SYSTEM.format(max_steps=6, active_concepts="（无）")
    assert "最多有 6 次回复机会" in sel and "换说法" in sel and "泛词不要选" in sel
    assert "召回原因" not in prompts.FILTER_ITEM and "{reasons}" not in prompts.FILTER_ITEM
    assert "chunk_id" not in prompts.FILTER_SYSTEM and "chunk_id" not in prompts.ANSWER_SYSTEM
    assert "[S#]" in prompts.ANSWER_SYSTEM and "较低的一档" in prompts.FILTER_SYSTEM and "拿不准时判 true" not in prompts.FILTER_SYSTEM
    assert all(g in prompts.FILTER_SYSTEM for g in ("direct", "background", "irrelevant"))


def grades(*pairs) -> LLMResult:
    """pairs: (S# 编号, grade)。"""
    return jtxt({"verdicts": [{"id": r, "grade": g, "reason": "r"} for r, g in pairs]})


def test_filter_grades_legacy_useful_and_invalid_grade(ctx):
    items = ordered(ctx, SEL2)[:3]
    assert len(items) == 3
    llm_ = ScriptedLLM([jtxt({"verdicts": [
        {"id": "S1", "grade": "Background", "reason": "r"}, {"id": "S2", "useful": True, "reason": "r"},
        {"id": "S3", "grade": "maybe", "reason": "r"}]})])
    fr = pipeline.filter_passages("q", [items], llm_, native_tools=True)
    assert [v.grade for v in fr.verdicts] == ["background", "direct"]                  # 旧格式 useful=true 视为 direct
    assert fr.useful_ids == [i.chunk_id for i in items[:2]]
    assert fr.unjudged == [items[2].chunk_id] and fr.batches[0]["invalid_grade"] == 1  # 不合法等级不静默当作无用
    assert fr.verdicts[1].to_dict()["grade"] == "direct"


def test_pick_for_answer_direct_first_then_background_with_cap(ctx):
    items = ordered(ctx, SEL2)
    assert len(items) >= 3
    gs = ["background", "direct"] + ["irrelevant"] * (len(items) - 3) + ["direct"]
    fr = pipeline.filter_passages("q", [items], ScriptedLLM([grades(*zip(refs_for(items), gs))]), native_tools=True)
    kept, capped = passage_filter.pick_for_answer(items, fr, max_chunks=10)
    assert [k.chunk_id for k in kept] == [items[1].chunk_id, items[-1].chunk_id, items[0].chunk_id]
    assert capped == []
    kept2, capped2 = passage_filter.pick_for_answer(items, fr, max_chunks=2)
    assert [k.chunk_id for k in kept2] == [items[1].chunk_id, items[-1].chunk_id] and capped2 == [items[0].chunk_id]


def test_pipeline_answer_uses_capped_graded_material(ctx, monkeypatch):
    monkeypatch.setattr("rag.course.config.COURSE_ANSWER_MAX_CHUNKS", 1)
    items = ordered(ctx, SEL2)
    gs = ["background"] + ["irrelevant"] * (len(items) - 2) + ["direct"]
    llm_ = ScriptedLLM([finish_native(*SEL2), grades(*zip(refs_for(items), gs)), LLMResult(content="见[S1]")])
    res = pipeline.ask("q", ctx=ctx, llm=llm_)
    st = res["trace"].stages["answer"]
    assert st["kept"] == 1 and st["kept_grades"] == {"direct": 1, "background": 0} and st["capped"] == [items[0].chunk_id]
    ans_user = llm_.calls[-1]["messages"][1]["content"]
    assert items[-1].text in ans_user and items[0].text not in ans_user                  # 只给 direct，background 被上限截掉
    assert res["citations"][0]["chunk_id"] == items[-1].chunk_id


def test_all_background_or_irrelevant_still_answers_only_when_some_background(ctx):
    sel = [("C01-0003", "question_concept")]
    items = ordered(ctx, sel)
    llm_ = ScriptedLLM([finish_native(*sel), grades(*[(r, "irrelevant") for r in refs_for(items)])])
    assert pipeline.ask("q", ctx=ctx, llm=llm_)["fallback"]["reason"] == "all_filtered_out"
    llm2 = ScriptedLLM([finish_native(*sel), grades(*[(r, "background") for r in refs_for(items)]),
                        LLMResult(content="见[S1]")])
    assert pipeline.ask("q", ctx=ctx, llm=llm2)["fallback"] is None


def test_reasoning_param_mapping_and_validation():
    assert llm.reasoning_param("") is None and llm.reasoning_param("  ") is None
    assert llm.reasoning_param("disabled") == {"type": "disabled"}
    assert llm.reasoning_param("LOW") == {"type": "enabled", "effort": "low"}
    with pytest.raises(ValueError):
        llm.reasoning_param("off")


def test_client_sends_reasoning_per_step_and_trace_records_it(monkeypatch):
    seen = []

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            self.chat = _Obj(completions=_Obj(create=lambda **kw: (seen.append(kw), _fake_response())[1]))

    monkeypatch.setitem(sys.modules, "openai", _Obj(OpenAI=FakeOpenAI))
    monkeypatch.setattr("rag.course.config.COURSE_REASONING", {"select": "", "filter": "disabled", "answer": "low"})
    c = llm.OpenAICompatClient(base="http://example.invalid/v1", key=SENTINEL_KEY)
    tr = Trace()
    t = TracingLLM(c, tr)
    for step in ("select", "filter", "answer"):
        t.chat(step, [{"role": "user", "content": "x"}], model="m")
    assert "extra_body" not in seen[0]
    assert seen[1]["extra_body"] == {"reasoning": {"type": "disabled"}}
    assert seen[2]["extra_body"] == {"reasoning": {"type": "enabled", "effort": "low"}}
    assert [x["reasoning"] for x in tr.calls] == ["", "disabled", "low"]
    assert [x["reasoning"] for x in Trace(calls=[]).calls] == []
    t2 = Trace()
    TracingLLM(ScriptedLLM([LLMResult(content="x")]), t2).chat("answer", [], model="m")
    assert t2.calls[0]["reasoning"] is None                                           # mock 不声明 reasoning_for


def test_tool_specs_are_plain_json_schema():
    names = [t["function"]["name"] for t in prompts.TOOL_SPECS]
    assert names == ["search_vocab", "get_concept", "get_neighbors", "finish"]
    sv = prompts.TOOL_SPECS[0]["function"]["parameters"]
    assert sv["properties"]["chapter"]["type"] == "integer" and "chapter" not in sv["required"]
    assert sv["required"] == ["query"]

    def no_type_arrays(node):
        if isinstance(node, dict):
            assert not isinstance(node.get("type"), list)       # 部分兼容层不支持类型数组
            for v in node.values():
                no_type_arrays(v)
        elif isinstance(node, list):
            for v in node:
                no_type_arrays(v)

    no_type_arrays(prompts.TOOL_SPECS)


def test_select_records_end_reason_for_text_only_reply_and_bad_finish(ctx):
    sr = agent_select.select_concepts("q", (), ctx.index, ScriptedLLM([
        tc("search_vocab", query="数据要素"), LLMResult(content="我选 C01-0001 数据要素")]))
    assert sr.selected == [] and not sr.finished
    assert sr.end_reason == "no_tool_call" and sr.final_text == "我选 C01-0001 数据要素"
    sr = agent_select.select_concepts("q", (), ctx.index, ScriptedLLM([
        tc("finish", concepts=[{"concept_id": "C01-0001", "reason": "r", "role": "question_concept"}])]))
    assert sr.selected == [] and sr.finished and sr.end_reason == "finish"
    assert sr.dropped == [{"reason": "finish_missing_selected", "keys": ["concepts"]}]
    sr = agent_select.select_concepts("q", (), ctx.index, ScriptedLLM([finish_native(("C01-0001", "question_concept"))]))
    assert sr.end_reason == "finish" and sr.to_dict()["end_reason"] == "finish" and sr.final_text == ""
