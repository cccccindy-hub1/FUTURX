"""判例检索（agentic）：点名直达、BM25 初筛 ≤12 案、章节优先 + 跨章补足、模型选案、改写只重查一次、列表模式。
全部 mock：不读 docx、不用向量库、不用 bge-m3、不调用真实模型。"""
from __future__ import annotations

import json

import pytest

from rag.cases import config as cfg
from rag.cases.retrieval import (CaseIndex, CaseRetriever, NOT_FOUND_TEXT, build_aliases, find_named, list_chapter, load_case_refs,
                                  render_case_list)
from rag.course.llm import LLMResult

ROWS = [
    {"case_id": "case:01", "seq": 1, "chapter": "第一章 数据要素", "chapter_no": 1, "title": "甲公司诉乙公司不正当竞争纠纷案",
     "meta": {"case_number": "（2021）浙8601民初1号"}},
    {"case_id": "case:02", "seq": 2, "chapter": "第二章 数据权属", "chapter_no": 2, "title": "某网络信息技术有限公司诉某信息科技有限公司不正当竞争纠纷案（指导性案例263号）",
     "meta": {"case_number": "（2019）沪73民终263号"}},
    {"case_id": "case:03", "seq": 3, "chapter": "第六章 数据市场", "chapter_no": 6, "title": "北京微梦创科网络技术有限公司诉云智联网络科技（北京）有限公司不正当竞争纠纷案",
     "meta": {"case_number": "（2020）京73民终2980号", "court": "北京知识产权法院"}},
    {"case_id": "case:04", "seq": 4, "chapter": "第六章 数据市场", "chapter_no": 6, "title": "公开数据权益归属及不正当数据利用行为的认定（大数据搬家软件案）", "meta": {}},
    {"case_id": "case:05", "seq": 5, "chapter": "第六章 数据市场", "chapter_no": 6, "title": "北京字节跳动科技有限公司与北京微梦创科网络技术有限公司不正当竞争纠纷案", "meta": {}},
    {"case_id": "case:06", "seq": 6, "chapter": "第八章 数据安全", "chapter_no": 8, "title": "翟某等与某有限公司等侵害技术秘密纠纷上诉案（DAKS数据库）", "meta": {}},
    {"case_id": "case:07", "seq": 7, "chapter": "第九章 隐私保护", "chapter_no": 9, "title": "罗某诉某科技有限公司隐私权、个人信息保护纠纷案（指导性案例265号）", "meta": {}},
]
REFS = load_case_refs(ROWS)


KEYWORDS = {1: "数据要素 市场化", 2: "数据权属 确权", 3: "爬取 抓取 平台数据 竞争", 4: "爬取 公开数据 搬家软件", 5: "短视频 平台 竞争", 6: "数据库 技术秘密", 7: "隐私 个人信息"}


def chunk(case_no, kind, text=None, heading="", part=0):
    row = next(r for r in ROWS if r["case_id"] == f"case:{case_no:02d}")
    body = text or f"{KEYWORDS[case_no]} {kind}"
    return {"chunk_id": f"case:{case_no:02d}:{kind}{part}", "case_id": f"case:{case_no:02d}", "seq": case_no, "chapter": row["chapter"],
            "chapter_no": row["chapter_no"], "title": row["title"], "kind": kind, "part": part, "heading": heading,
            "text": f"【{row['chapter']}】{row['title']}\n{body}"}


CHUNKS = [chunk(n, "card") for n in range(1, 8)] + [
    chunk(3, "body", "基本案情 爬取 平台数据 抓取", "基本案情", 1), chunk(3, "body", "裁判理由 竞争 秩序", "裁判理由", 2),
    chunk(3, "body", "无关的附带说明 其他内容", "附注", 3), chunk(3, "body", "另一段 爬取 抓取 技术手段", "技术", 4)]
INDEX = CaseIndex(REFS, CHUNKS)


class PickLLM:
    """脚本化选案模型：按调用次序返回 {"picks": [...], "rewrite": ...}；记录每次的消息。"""

    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "model": model, "json_mode": json_mode})
        out = self.outs.pop(0)
        return LLMResult(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))


def retr(*outs, index=INDEX):
    llm = PickLLM(*outs)
    return CaseRetriever(index, llm, model="flash"), llm


# ---------------------------------------------------------------- 别名与点名

def test_aliases_include_nicknames_numbers_and_unique_party_short_names():
    a = {r.case_id: r.aliases for r in REFS}
    assert "指导性案例263号" in a["case:02"] and "指导案例263号" in a["case:02"] and "（2019）沪73民终263号" in a["case:02"]
    assert "大数据搬家软件案" in a["case:04"] and "大数据搬家软件" in a["case:04"] and "DAKS数据库" in a["case:06"]
    assert "云智联" in a["case:03"] and "（2020）京73民终2980号" in a["case:03"]


def test_alias_shared_by_several_cases_is_dropped_so_naming_is_unambiguous():
    shared = [a for r in REFS for a in r.aliases if sum(a in x.aliases for x in REFS) > 1]
    assert shared == []                                                       # 「北京微梦创科」同时在案 3、5，不能用来点名
    assert find_named("北京微梦创科的案子", REFS) == []


@pytest.mark.parametrize("q,seqs", [
    ("微博诉云智联案是怎么判的", [3]), ("指导性案例263号讲了什么", [2]), ("指导案例265号", [7]), ("大数据搬家软件案的争议焦点", [4]),
    ("DAKS数据库案怎么判的", [6]), ("（2019）沪73民终263号", [2]), ("比较大数据搬家软件案和云智联案", [4, 3]),
    ("什么是数据确权", []), ("举三个判例", []), ("法院怎么认定不正当竞争", [])])
def test_find_named_direct_hits_and_no_false_triggers(q, seqs):
    assert [r.seq for r in find_named(q, REFS)] == seqs                       # 按问题里出现的先后；普通问题不触发


def test_boilerplate_and_truncated_fragments_never_become_aliases():
    al = build_aliases("某某公司等不正当竞争纠纷上诉案（房源数据搬运）", {"case_number": "官方来源未披露"})
    assert "房源数据搬运" in al and not any(a.endswith(("纠纷上", "纠纷", "上诉")) for a in al) and "官方来源未披露" not in al


# ---------------------------------------------------------------- 点名直达：不检索、不调用选案模型

def test_named_case_is_fetched_directly_without_any_model_call():
    r, llm = retr()
    out = r.retrieve("微博诉云智联案是怎么判的", count=1)
    assert out["mode"] == "named" and llm.calls == [] and out["trace"]["llm_calls"] == 0
    h = out["hits"][0]
    assert h.ref.seq == 3 and h.via == "named" and "爬取" in h.card and len(h.bodies) == 3 and out["trace"]["named"] == ["case:03"]


def test_named_respects_count_and_order():
    r, llm = retr()
    assert [h.ref.seq for h in r.retrieve("比较大数据搬家软件案和云智联案", count=5)["hits"]] == [4, 3] and llm.calls == []
    assert [h.ref.seq for h in r.retrieve("比较大数据搬家软件案和云智联案", count=1)["hits"]] == [4]


# ---------------------------------------------------------------- BM25 初筛 + 模型选案

def test_search_runs_bm25_then_exactly_one_pick_call_and_returns_the_picked_cases():
    r, llm = retr({"picks": ["3"], "rewrite": ""})
    out = r.retrieve("爬取平台数据的判例", count=1)
    assert out["mode"] == "search" and [h.ref.seq for h in out["hits"]] == [3] and out["trace"]["llm_calls"] == 1
    assert len(llm.calls) == 1 and llm.calls[0]["step"] == "case_pick" and llm.calls[0]["model"] == "flash" and llm.calls[0]["json_mode"] is True
    assert out["hits"][0].via == "search" and out["hits"][0].card.startswith("【第六章")


def test_pick_prompt_lists_only_bm25_candidates_with_their_cards():
    r, llm = retr({"picks": ["3"], "rewrite": ""})
    r.retrieve("爬取抓取的判例", count=2)
    user = llm.calls[0]["messages"][1]["content"]
    assert "[3] " in user and "[4] " in user and "爬取" in user                  # 含爬取的案子是候选
    assert "[7] " not in user and "[1] " not in user                             # 与问题无关的案子不进候选
    assert "最多 2 个" in llm.calls[0]["messages"][0]["content"]


def test_candidates_are_capped_by_config(monkeypatch):
    monkeypatch.setattr(cfg, "CASE_CANDIDATES", 2)
    assert len(INDEX.candidates("爬取 数据 竞争 平台 权属 隐私 技术秘密")) == 2


def test_picks_are_capped_at_count_and_unknown_or_duplicate_ids_are_dropped():
    r, llm = retr({"picks": ["99", "3", "3", "4", "5"], "rewrite": ""})
    out = r.retrieve("爬取平台数据竞争的判例", count=2)
    assert [h.ref.seq for h in out["hits"]] == [3, 4]
    assert len(out["trace"]["rounds"][0]["dropped"]) == 2


def test_chapter_candidates_come_first_and_other_chapters_top_up_and_are_marked():
    r, _ = retr({"picks": ["4", "3", "5"], "rewrite": ""})
    ids, via = r._candidates("爬取 竞争 平台", 6)
    assert set(ids) >= {"case:03", "case:04", "case:05"} and ids[0] in ("case:03", "case:04", "case:05")
    out = r.retrieve("爬取 竞争 平台", count=3, chapter=6)
    assert {h.ref.seq for h in out["hits"]} == {3, 4, 5} and all(h.via == "search" for h in out["hits"])      # 都在本章
    r2, _ = retr({"picks": ["3", "4"], "rewrite": ""})
    out2 = r2.retrieve("爬取 竞争 数据权属", count=3, chapter=2)                                              # 第 2 章只有案 2；案 3 / 4 靠跨章补足
    assert {h.ref.seq: h.via for h in out2["hits"]} == {3: "search_other_chapter", 4: "search_other_chapter"}


def test_chapter_hits_sort_before_cross_chapter_ones():
    r, _ = retr({"picks": ["3", "2"], "rewrite": ""})
    out = r.retrieve("爬取 数据权属 确权", count=2, chapter=2)
    assert [(h.ref.seq, h.via) for h in out["hits"]] == [(2, "search"), (3, "search_other_chapter")]


def test_bodies_are_the_most_relevant_chunks_limited_by_config(monkeypatch):
    monkeypatch.setattr(cfg, "CASE_MAX_BODY_CHUNKS", 2)
    r, _ = retr({"picks": ["3"], "rewrite": ""})
    h = r.retrieve("爬取 抓取 平台数据", count=1)["hits"][0]
    assert len(h.bodies) == 2 and all("爬取" in b for b in h.bodies) and not any("无关的附带说明" in b for b in h.bodies)


# ---------------------------------------------------------------- 都不相关：改写后只重查一次

def test_none_relevant_rewrites_once_and_searches_again_then_succeeds():
    r, llm = retr({"picks": [], "rewrite": "隐私 个人信息"}, {"picks": ["7"], "rewrite": ""})
    out = r.retrieve("爬取平台数据的判例", count=1)
    assert out["mode"] == "search" and [h.ref.seq for h in out["hits"]] == [7] and len(llm.calls) == 2
    assert out["trace"]["rewritten"] == "隐私 个人信息" and [x["query"] for x in out["trace"]["rounds"]] == ["爬取平台数据的判例", "隐私 个人信息"]
    assert "隐私" in llm.calls[1]["messages"][1]["content"] and "[7] " in llm.calls[1]["messages"][1]["content"]
    assert "爬取平台数据的判例" in llm.calls[1]["messages"][1]["content"]           # 第二次仍以学生原问题为准


def test_still_none_after_rewrite_is_not_found_and_never_rewrites_twice():
    r, llm = retr({"picks": [], "rewrite": "隐私 个人信息"}, {"picks": [], "rewrite": "再改写 抓取"}, {"picks": ["3"], "rewrite": ""})
    out = r.retrieve("爬取平台数据的判例", count=1)
    assert out["mode"] == "not_found" and out["hits"] == [] and len(llm.calls) == 2 and out["trace"]["llm_calls"] == 2
    assert len(llm.outs) == 1                                                       # 第三个脚本没被用掉：只重查一次


def test_unchanged_or_empty_rewrite_is_not_searched_again():
    r, llm = retr({"picks": [], "rewrite": "爬取平台数据的判例"})
    assert r.retrieve("爬取平台数据的判例", count=1)["mode"] == "not_found" and len(llm.calls) == 1
    r, llm = retr({"picks": [], "rewrite": ""})
    assert r.retrieve("爬取平台数据的判例", count=1)["mode"] == "not_found" and len(llm.calls) == 1


def test_no_bm25_candidates_means_not_found_without_calling_the_model():
    r, llm = retr()
    out = r.retrieve("ZZZ QQQ", count=1)
    assert out["mode"] == "not_found" and llm.calls == [] and out["trace"]["llm_calls"] == 0


@pytest.mark.parametrize("bad", ["不是JSON", "[]", '{"picks": "3"}', '{"picks": [null, {}]}'])
def test_unusable_model_output_is_not_found_not_a_crash_and_not_retried(bad):
    r, llm = retr(bad)
    out = r.retrieve("爬取平台数据的判例", count=1)
    assert out["mode"] == "not_found" and len(llm.calls) == 1


def test_model_error_is_recorded_by_type_name_only():
    class Down:
        def chat(self, *a, **k):
            raise TimeoutError("key=SECRET")
    out = CaseRetriever(INDEX, Down(), model="m").retrieve("爬取平台数据的判例", count=1)
    assert out["mode"] == "not_found" and out["trace"]["rounds"][0]["error"] == "TimeoutError" and "SECRET" not in json.dumps(out["trace"])


def test_empty_index_is_safe():
    r, llm = retr(index=CaseIndex(REFS, []))
    assert r.retrieve("爬取", count=1)["mode"] == "not_found" and llm.calls == []


# ---------------------------------------------------------------- 加载：docx 缺失跳过；解析结果走 JSON 缓存

def test_load_returns_none_when_the_docx_is_missing(tmp_path):
    assert CaseIndex.load(tmp_path / "不存在.docx") is None


def test_load_parses_once_then_reuses_the_cache_and_reparses_when_the_source_changes(tmp_path, monkeypatch):
    import rag.cases.retrieval as mod
    from rag import docscache
    monkeypatch.setattr(docscache, "cache_path", lambda doc, kind: tmp_path / f"{kind}.json")
    doc = tmp_path / "cases.docx"
    doc.write_bytes(b"v1")
    built = []
    monkeypatch.setattr(mod, "_chunk_rows", lambda d: built.append(1) or {"refs": [{k: v for k, v in r.items()} for r in ROWS], "chunks": CHUNKS})
    a = CaseIndex.load(doc)
    b = CaseIndex.load(doc)
    assert len(built) == 1 and a.info["source"] == "parsed" and b.info["source"] == "cache" and len(b.refs) == 7 and len(b.chunks) == len(CHUNKS)
    doc.write_bytes(b"version-2")                                                      # 大小变了 -> 缓存作废
    assert CaseIndex.load(doc).info["source"] == "parsed" and len(built) == 2


# ---------------------------------------------------------------- 列表模式：不检索、不调模型

def test_list_mode_lists_cases_of_the_chapter_without_any_retrieval():
    items = list_chapter(REFS, 6)
    assert [r.seq for r in items] == [3, 4, 5]
    text = render_case_list(REFS, 6)
    assert text.startswith("第六章 数据市场共有 3 个判例：") and "1. 北京微梦创科" in text and "（2020）京73民终2980号" in text
    assert "2. 公开数据权益归属" in text and "3. 北京字节跳动" in text


def test_list_mode_all_cases_and_empty_chapter():
    assert render_case_list(REFS, None).startswith("判例库共有 7 个判例：")
    assert render_case_list(REFS, 3) == NOT_FOUND_TEXT                           # 第三章没有判例
