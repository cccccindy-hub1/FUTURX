"""名言检索与作答：按讲话检索、纠错、模型只挑段落编号、程序原文照录。假服务 + 假 LLM，不依赖真实向量库 / 模型。"""
from __future__ import annotations

import json

import pytest

from rag.course.llm import LLMResult
from rag.quotes.retrieval import (NOT_FOUND_TEXT, PICK_SYSTEM, QuotePick, QuoteRetriever, SpeechHit, pick_paragraphs,
                                  render_for_pick, render_quote_answer, source_line)
from rag.retrieval import RetrievalResult

SPEECHES = {
    "quote:2018:01": ("全国网络安全和信息化工作会议上的讲话", "讲话", 2018, ["第一段：没有网络安全就没有国家安全。", "第二段：要加快数据资源整合。", "第三段：要强化关键信息基础设施保护。"]),
    "quote:2022:01": ("习近平.不断做强做优做大我国数字经济", "文章", 2022, ["数字经济是继农业经济、工业经济之后的主要经济形态。"]),
    "quote:2026:03": ("向世界数据组织成立致贺信", "贺信", 2026, ["数据是新型生产要素。", "愿同各方一道推动数据开放共享。"]),
}


def mk_chunk(sid, score):
    title, kind, year, paras = SPEECHES[sid]
    return {"id": f"{sid}:abc", "doc": "\n".join(f"[{i}] {p}" for i, p in enumerate(paras, 1)), "score": score,
            "meta": {"speech_id": sid, "year": year, "title": title, "kind": kind, "paragraphs_json": json.dumps(paras, ensure_ascii=False)}}


class FakeSvc:
    mode = "hybrid"
    llm = None

    def __init__(self, scripts):
        self.scripts, self.calls = list(scripts), []

    def retrieve_result(self, query, top_k):
        self.calls.append((query, top_k))
        pool = sorted(self.scripts.pop(0), key=lambda c: -c["score"])
        return RetrievalResult(ids=[c["id"] for c in pool], documents=[c["doc"] for c in pool], metadatas=[c["meta"] for c in pool],
                               distances=[1 - c["score"] for c in pool], scores=[c["score"] for c in pool],
                               debug={"rerank_scores": {c["id"]: c["score"] for c in pool}, "rerank_s": 0.0})


class LLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "json_mode": json_mode})
        if self.exc:
            raise self.exc
        return LLMResult(content=self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False))


def hits_for(*sids):
    out = []
    for sid in sids:
        t, k, y, p = SPEECHES[sid]
        out.append(SpeechHit(sid, y, t, k, p, 0.9))
    return out


# ---------------------------------------------------------------- 检索：按整次讲话，取前 count 次

def test_retrieves_whole_speeches_ranked_and_limited_to_count():
    svc = FakeSvc([[mk_chunk("quote:2018:01", 0.9), mk_chunk("quote:2022:01", 0.7), mk_chunk("quote:2026:03", 0.5)]])
    out = QuoteRetriever(svc, min_rerank_score=0.3).retrieve("总书记谈数据安全", count=2)
    assert out["mode"] == "search" and [h.speech_id for h in out["hits"]] == ["quote:2018:01", "quote:2022:01"]
    assert out["hits"][0].paragraphs == SPEECHES["quote:2018:01"][3] and out["hits"][0].year == 2018 and len(svc.calls) == 1


def test_same_speech_returned_once_and_paragraphs_recovered_from_numbered_text_when_no_json():
    c = mk_chunk("quote:2018:01", 0.9)
    del c["meta"]["paragraphs_json"]
    dup = dict(c, id="dup")
    svc = FakeSvc([[c, dup]])
    hits = QuoteRetriever(svc, min_rerank_score=0.3).retrieve("q", count=5)["hits"]
    assert len(hits) == 1 and hits[0].paragraphs == SPEECHES["quote:2018:01"][3]          # 从 [N] 标注的文本切回原段落


def test_low_confidence_rewrites_once_then_not_found_without_picking():
    low = [mk_chunk("quote:2026:03", 0.05)]
    svc = FakeSvc([low, low, low])
    out = QuoteRetriever(svc, min_rerank_score=0.3).retrieve("无关问题", count=1, rewrite=lambda q: "改写后")
    assert out["mode"] == "not_found" and out["hits"] == [] and [c[0] for c in svc.calls] == ["无关问题", "改写后"]
    assert NOT_FOUND_TEXT == "合集中没有找到相关表述。"


def test_rewrite_recovers_when_second_search_is_reliable():
    svc = FakeSvc([[mk_chunk("quote:2026:03", 0.05)], [mk_chunk("quote:2018:01", 0.9)]])
    out = QuoteRetriever(svc, min_rerank_score=0.3).retrieve("模糊", count=1, rewrite=lambda q: "清晰的查询")
    assert out["mode"] == "search" and out["hits"][0].speech_id == "quote:2018:01" and out["trace"]["rewritten"] == "清晰的查询"


def test_reliable_search_never_rewrites():
    svc = FakeSvc([[mk_chunk("quote:2018:01", 0.9)]])
    QuoteRetriever(svc, min_rerank_score=0.3).retrieve("q", rewrite=lambda q: pytest.fail("不应改写"))
    assert len(svc.calls) == 1


# ---------------------------------------------------------------- 挑段落：1 次调用，只出编号

def test_pick_prompt_shows_numbered_speeches_and_paragraphs_and_asks_for_ids_only():
    msgs_text = render_for_pick(hits_for("quote:2018:01", "quote:2026:03"))
    assert "[Q1] 2018年，全国网络安全和信息化工作会议上的讲话（讲话）" in msgs_text and "(2) 第二段：要加快数据资源整合。" in msgs_text and "[Q2] 2026年" in msgs_text
    assert "不要改写" in PICK_SYSTEM and "只挑编号" in PICK_SYSTEM and "只输出一个 JSON" in PICK_SYSTEM


def test_pick_makes_exactly_one_call_and_returns_chosen_paragraphs():
    llm = LLM({"picks": [{"id": "Q1", "paragraphs": [2, 3]}, {"id": "Q2", "paragraphs": [1]}]})
    picks, info = pick_paragraphs("数据安全", hits_for("quote:2018:01", "quote:2026:03"), llm, model="m")
    assert len(llm.calls) == 1 and llm.calls[0]["step"] == "quote_pick" and llm.calls[0]["json_mode"] is True
    assert [(p.n, p.indices) for p in picks] == [(1, [2, 3]), (2, [1])] and info["fallback"] == ""
    assert picks[0].texts() == ["第二段：要加快数据资源整合。", "第三段：要强化关键信息基础设施保护。"]


def test_out_of_range_ids_and_paragraphs_are_dropped_and_recorded():
    llm = LLM({"picks": [{"id": "Q1", "paragraphs": [2, 9, 0, "x", 2]}, {"id": "Q7", "paragraphs": [1]}, {"id": "Q1", "paragraphs": [1]}]})
    picks, info = pick_paragraphs("q", hits_for("quote:2018:01"), llm, model="m")
    assert [(p.n, p.indices) for p in picks] == [(1, [2])]
    reasons = {(d["id"], d["reason"]) for d in info["dropped"]}
    assert ("Q7", "unknown_or_duplicate_speech") in reasons and ("Q1", "unknown_or_duplicate_speech") in reasons and ("Q1", "out_of_range_or_duplicate") in reasons


@pytest.mark.parametrize("bad", [LLM("这不是 JSON"), LLM({"picks": []}), LLM({"picks": [{"id": "Q1", "paragraphs": [99]}]}), LLM(exc=TimeoutError("t")), LLM({"nope": 1})])
def test_unusable_model_output_falls_back_to_first_paragraph_without_retry(bad):
    picks, info = pick_paragraphs("q", hits_for("quote:2018:01", "quote:2022:01"), bad, model="m")
    assert len(bad.calls) == 1 and info["fallback"]                                       # 不重试
    assert [(p.n, p.indices) for p in picks] == [(1, [1]), (2, [1])]                       # 退回每次讲话第一段（程序决定，模型不生成任何字）


def test_pick_at_most_three_paragraphs_per_speech_and_sorted():
    llm = LLM({"picks": [{"id": "Q1", "paragraphs": [3, 1, 2, 2]}]})
    picks, _ = pick_paragraphs("q", hits_for("quote:2018:01"), llm, model="m")
    assert picks[0].indices == [1, 2, 3]


def test_more_than_three_picked_paragraphs_are_capped_at_three_keeping_the_earliest():
    long = SpeechHit("quote:x", 2020, "长讲话", "讲话", [f"第{i}段原文。" for i in range(1, 7)], 0.9)
    picks, _ = pick_paragraphs("q", [long], LLM({"picks": [{"id": "Q1", "paragraphs": [6, 5, 4, 3, 2, 1]}]}), model="m")
    assert picks[0].indices == [1, 2, 3]                                                  # 最多 3 段，按原文顺序保留靠前的


def test_first_n_offsets_q_numbers_for_combined_questions():
    llm = LLM({"picks": [{"id": "Q3", "paragraphs": [1]}]})
    picks, _ = pick_paragraphs("q", hits_for("quote:2022:01"), llm, model="m", first_n=3)
    assert picks[0].n == 3 and "[Q3]" in llm.calls[0]["messages"][1]["content"]


# ---------------------------------------------------------------- 程序照录：原文逐字，带出处

def test_answer_is_verbatim_with_source_and_no_model_text():
    picks = [QuotePick(1, hits_for("quote:2018:01")[0], [2, 3]), QuotePick(2, hits_for("quote:2026:03")[0], [1])]
    text = render_quote_answer(picks)
    assert text == ("「第二段：要加快数据资源整合。\n第三段：要强化关键信息基础设施保护。」[1]\n——2018年，全国网络安全和信息化工作会议上的讲话\n\n"
                    "「数据是新型生产要素。」[2]\n——2026年，向世界数据组织成立致贺信")
    for p in picks:
        for para in p.texts():
            assert para in text                                                           # 每个入选段落逐字出现
    assert source_line(picks[0].speech) == "2018年，全国网络安全和信息化工作会议上的讲话"


def test_unnumbered_rendering_has_no_markers():
    text = render_quote_answer([QuotePick(1, hits_for("quote:2022:01")[0], [1])], numbered=False)
    assert text == "「数字经济是继农业经济、工业经济之后的主要经济形态。」\n——2022年，习近平.不断做强做优做大我国数字经济"
