"""判例检索：点名直达、章节优先、跨章补足、按案聚合、低分改写重查、列表模式。假服务，不依赖真实向量库 / 模型。"""
from __future__ import annotations

import pytest

from rag.cases import config as cfg
from rag.cases.retrieval import (CaseRetriever, NOT_FOUND_TEXT, build_aliases, find_named, list_chapter, load_case_refs,
                                  render_case_list)
from rag.retrieval import RetrievalResult

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


def chunk(case_no, kind, text=None, heading="", chapter_no=None, score=0.9, part=1):
    row = next(r for r in ROWS if r["case_id"] == f"case:{case_no:02d}")
    return {"id": f"case:{case_no:02d}:{kind}{part}", "doc": text or f"{row['title']}的{kind}文本", "score": score,
            "meta": {"case_id": f"case:{case_no:02d}", "seq": case_no, "kind": kind, "heading": heading,
                     "chapter_no": chapter_no or row["chapter_no"], "title": row["title"]}}


class FakeCollection:
    """get(where=...) 取某案全部块。"""

    def __init__(self, chunks):
        self.chunks = chunks

    def get(self, where=None, include=None):
        sel = [c for c in self.chunks if all(c["meta"].get(k) == v for k, v in (where or {}).items())]
        return {"documents": [c["doc"] for c in sel], "metadatas": [c["meta"] for c in sel], "ids": [c["id"] for c in sel]}


class FakeSvc:
    """retrieve_result：按 where 过滤后，返回预设的（已按分数排序的）块；记录每次调用的 query / where。"""
    mode = "hybrid"
    llm = None

    def __init__(self, chunks, scripts=None):
        self.chunks, self.where, self.calls = chunks, None, []
        self.collection = FakeCollection(chunks)
        self.scripts = list(scripts or [])         # 每次调用的“覆盖结果”：list[chunk]；用完后按默认行为

    def retrieve_result(self, query, top_k):
        self.calls.append({"query": query, "where": self.where, "top_k": top_k})
        pool = self.scripts.pop(0) if self.scripts else [c for c in self.chunks if all(c["meta"].get(k) == v for k, v in (self.where or {}).items())]
        pool = sorted(pool, key=lambda c: -c["score"])[:top_k]
        rs = {c["id"]: c["score"] for c in pool}
        return RetrievalResult(ids=[c["id"] for c in pool], documents=[c["doc"] for c in pool], metadatas=[c["meta"] for c in pool],
                               distances=[1 - c["score"] for c in pool], scores=[c["score"] for c in pool],
                               debug={"rerank_scores": rs, "rerank_s": 0.0})


def retriever(chunks, scripts=None, min_score=0.3):
    svc = FakeSvc(chunks, scripts)
    return CaseRetriever(svc, REFS, min_rerank_score=min_score), svc


CHUNKS = [
    chunk(1, "card", score=0.9), chunk(1, "body", score=0.8, heading="裁判理由"),
    chunk(2, "card", score=0.7), chunk(2, "body", score=0.6),
    chunk(3, "card", score=0.95), chunk(3, "body", score=0.9, part=1, heading="基本案情"), chunk(3, "body", score=0.4, part=2, heading="裁判理由"),
    chunk(4, "card", score=0.85), chunk(4, "body", score=0.5),
    chunk(5, "card", score=0.5),
    chunk(6, "card", score=0.45), chunk(6, "body", score=0.4),
    chunk(7, "card", score=0.35),
]


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


# ---------------------------------------------------------------- 点名直达：不做向量检索

def test_named_case_is_fetched_directly_without_vector_search():
    r, svc = retriever(CHUNKS)
    out = r.retrieve("微博诉云智联案是怎么判的", count=1)
    assert out["mode"] == "named" and svc.calls == []                          # 一次向量检索都没有
    h = out["hits"][0]
    assert h.ref.seq == 3 and h.via == "named" and h.card.endswith("的card文本")                # 取到的是该案的卡片
    assert len(h.bodies) == 2 and out["trace"]["named"] == ["case:03"]


def test_named_respects_count_and_order():
    r, svc = retriever(CHUNKS)
    out = r.retrieve("比较大数据搬家软件案和云智联案", count=5)
    assert [h.ref.seq for h in out["hits"]] == [4, 3] and svc.calls == []
    assert [h.ref.seq for h in r.retrieve("比较大数据搬家软件案和云智联案", count=1)["hits"]] == [4]


# ---------------------------------------------------------------- 章节优先 / 跨章补足 / 按案聚合

def test_chapter_filter_applied_first_and_cases_aggregated_by_max_score():
    r, svc = retriever(CHUNKS)
    out = r.retrieve("平台抓取数据的判例", count=2, chapter=6)
    assert svc.calls[0]["where"] == {"chapter_no": 6}                          # 先在本章里检索
    assert [h.ref.seq for h in out["hits"]] == [3, 4] and all(h.via == "search" for h in out["hits"])
    assert out["hits"][0].score == 0.95                                        # 取该案块的最高分（卡片 0.95，而不是正文块的 0.9 / 0.4）
    assert len(svc.calls) == 1                                                  # 本章够 2 个，不需要跨章


def test_each_case_appears_once_even_if_many_chunks_hit():
    r, _ = retriever(CHUNKS)
    out = r.retrieve("平台抓取数据", count=5, chapter=6)
    seqs = [h.ref.seq for h in out["hits"]]
    assert len(seqs) == len(set(seqs))
    h3 = [h for h in out["hits"] if h.ref.seq == 3][0]
    assert len(h3.bodies) == 2 and h3.body_headings == ["基本案情", "裁判理由"]                              # 正文块按分数排序，最相关的在前


def test_max_body_chunks_per_case_is_limited(monkeypatch):
    monkeypatch.setattr(cfg, "CASE_MAX_BODY_CHUNKS", 1)
    r, _ = retriever(CHUNKS)
    h = [x for x in r.retrieve("x", count=5, chapter=6)["hits"] if x.ref.seq == 3][0]
    assert len(h.bodies) == 1 and h.body_headings == ["基本案情"]


def test_not_enough_in_chapter_fills_from_other_chapters_and_marks_them():
    r, svc = retriever(CHUNKS)
    out = r.retrieve("判例", count=4, chapter=2)                                # 第二章只有 1 个案
    assert [c["where"] for c in svc.calls] == [{"chapter_no": 2}, None]        # 本章不够 -> 再不限章检索一次
    assert out["hits"][0].ref.seq == 2 and out["hits"][0].via == "search"
    assert all(h.via == "search_other_chapter" for h in out["hits"][1:]) and len(out["hits"]) == 4
    assert [h.ref.seq for h in out["hits"]][0] == 2                              # 本章结果排在前面


def test_no_chapter_means_single_unfiltered_search():
    r, svc = retriever(CHUNKS)
    out = r.retrieve("判例", count=2)
    assert len(svc.calls) == 1 and svc.calls[0]["where"] is None
    assert [h.ref.seq for h in out["hits"]] == [3, 1]


def test_card_is_fetched_when_only_body_chunks_hit():
    only_body = [chunk(4, "body", score=0.9, heading="裁判理由")]
    r, _ = retriever(CHUNKS, scripts=[only_body])
    h = r.retrieve("x", count=1)["hits"][0]
    assert h.ref.seq == 4 and "card" in h.card and h.bodies                    # 命中的只有正文块：补取该案卡片


# ---------------------------------------------------------------- 纠错：低分改写重查，仍低则 not_found（不调作答）

def test_low_confidence_triggers_one_rewrite_and_retry_then_succeeds():
    low = [chunk(5, "card", score=0.1)]
    good = [chunk(3, "card", score=0.9)]
    r, svc = retriever(CHUNKS, scripts=[low, good])
    rewrites = []
    out = r.retrieve("模糊问题", count=1, rewrite=lambda q: (rewrites.append(q), "改写后的查询")[1])
    assert rewrites == ["模糊问题"] and [c["query"] for c in svc.calls] == ["模糊问题", "改写后的查询"]
    assert out["mode"] == "search" and out["hits"][0].ref.seq == 3 and out["trace"]["rewritten"] == "改写后的查询"


def test_still_low_after_rewrite_returns_not_found_and_only_retries_once():
    low = [chunk(5, "card", score=0.05)]
    r, svc = retriever(CHUNKS, scripts=[low, low, low, low])
    out = r.retrieve("无关问题", count=1, rewrite=lambda q: "改写")
    assert out["mode"] == "not_found" and out["hits"] == []
    assert len(svc.calls) == 2                                                  # 原查询 + 改写后各一次，不再多
    assert NOT_FOUND_TEXT.startswith("判例库里没有找到")


def test_no_rewrite_available_means_not_found_without_extra_calls():
    r, svc = retriever(CHUNKS, scripts=[[chunk(5, "card", score=0.05)]])
    out = r.retrieve("无关问题", count=1, rewrite=None)
    assert out["mode"] == "not_found" and len(svc.calls) == 1


def test_unchanged_rewrite_is_not_searched_again():
    r, svc = retriever(CHUNKS, scripts=[[chunk(5, "card", score=0.05)]])
    r.retrieve("问题", count=1, rewrite=lambda q: q)
    assert len(svc.calls) == 1


def test_reliable_first_search_never_calls_rewrite():
    r, svc = retriever(CHUNKS)
    r.retrieve("平台抓取数据", count=1, rewrite=lambda q: pytest.fail("不应改写"))
    assert len(svc.calls) == 1


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
