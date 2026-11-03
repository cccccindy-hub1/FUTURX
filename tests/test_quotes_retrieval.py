"""名言检索（agentic）：BM25 初筛段落、模型只挑段落编号、改写只重查一次、程序原文照录。mock，不读 docx、不用向量库、不调用真实模型。"""
from __future__ import annotations

import json

import pytest

from rag.course.llm import LLMResult
from rag.quotes import config as cfg
from rag.quotes.retrieval import (NOT_FOUND_TEXT, PICK_SYSTEM, QuoteIndex, QuotePick, QuoteRetriever, SpeechHit, group_picks,
                                  pick_paragraphs, render_quote_answer, source_line)

SPEECHES = {
    "quote:2018:01": ("全国网络安全和信息化工作会议上的讲话", "讲话", 2018, ["第一段：没有网络安全就没有国家安全。", "第二段：要加快数据资源整合。", "第三段：要强化关键信息基础设施保护。"]),
    "quote:2022:01": ("习近平.不断做强做优做大我国数字经济", "文章", 2022, ["数字经济是继农业经济、工业经济之后的主要经济形态。"]),
    "quote:2026:03": ("向世界数据组织成立致贺信", "贺信", 2026, ["数据是新型生产要素。", "愿同各方一道推动数据开放共享。"]),
}
INDEX = QuoteIndex([{"speech_id": sid, "year": y, "n": 1, "title": t, "kind": k, "paragraphs": p} for sid, (t, k, y, p) in SPEECHES.items()])


class LLM:
    """脚本化挑段模型：每次调用按次序返回一个 {"picks": [...], "rewrite": ...}。"""

    def __init__(self, *outs, exc=None):
        self.outs, self.exc, self.calls = list(outs), exc, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "model": model, "json_mode": json_mode})
        if self.exc:
            raise self.exc
        out = self.outs.pop(0)
        return LLMResult(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))


def hits_for(*sids):
    return [INDEX.speech_hit(sid) for sid in sids]


def retr(*outs, **kw):
    llm = LLM(*outs, **kw)
    return QuoteRetriever(INDEX, llm, model="flash"), llm


# ---------------------------------------------------------------- 初筛 + 挑段

def test_bm25_candidates_are_paragraph_level_and_capped(monkeypatch):
    c = INDEX.candidates("数据 开放 共享")
    assert c[0][:2] == ("quote:2026:03", 2) and all(isinstance(x[1], int) for x in c)
    monkeypatch.setattr(cfg, "QUOTE_CANDIDATES", 1)
    assert len(INDEX.candidates("数据 开放 共享 网络安全")) == 1


def test_search_makes_exactly_one_pick_call_and_returns_verbatim_paragraphs():
    class Peek(LLM):
        def chat(self, step, messages, *, model, tools=None, json_mode=False):
            self.outs = [{"picks": [number_of_msgs(messages, "开放共享")], "rewrite": ""}]
            return super().chat(step, messages, model=model, json_mode=json_mode)

    def number_of_msgs(messages, text):
        import re
        return next(int(k) for k, t in re.findall(r"^\[(\d+)\] .*\n(.*)$", messages[1]["content"], flags=re.M) if text in t)
    llm = Peek()
    out = QuoteRetriever(INDEX, llm, model="flash").retrieve("数据开放共享", count=1)
    assert out["mode"] == "search" and len(llm.calls) == 1 and out["trace"]["llm_calls"] == 1
    assert llm.calls[0]["step"] == "quote_pick" and llm.calls[0]["model"] == "flash" and llm.calls[0]["json_mode"] is True
    p = out["picks"][0]
    assert (p.n, p.speech.speech_id, p.indices) == (1, "quote:2026:03", [2]) and p.texts() == ["愿同各方一道推动数据开放共享。"]


def test_pick_prompt_has_numbered_candidates_with_source_and_the_count_limit():
    r, llm = retr({"picks": [], "rewrite": ""})
    r.retrieve("数据开放共享", count=3)
    sysm, user = (m["content"] for m in llm.calls[0]["messages"])
    assert "最多 3 个" in sysm and "rewrite" in sysm and "数据开放共享" in user
    assert "[1] 2026年，向世界数据组织成立致贺信\n愿同各方一道推动数据开放共享。" in user


def test_picks_capped_at_count_and_unknown_or_duplicate_ids_dropped():
    r, llm = retr({"picks": [1, 1, 99, 2, 3], "rewrite": ""})
    got = r.retrieve("数据 开放 共享 资源 整合", count=2)
    assert len(got["picks"]) >= 1 and sum(len(p.indices) for p in got["picks"]) == 2
    assert len(got["trace"]["rounds"][0]["dropped"]) == 2


def test_same_speech_paragraphs_are_grouped_in_original_order_and_q_numbers_start_at_first_n():
    picks = group_picks(INDEX, [("quote:2018:01", 3), ("quote:2026:03", 1), ("quote:2018:01", 1)], first_n=3)
    assert [(p.n, p.speech.speech_id, p.indices) for p in picks] == [(3, "quote:2018:01", [1, 3]), (4, "quote:2026:03", [1])]


# ---------------------------------------------------------------- 都不相关：改写后只重查一次

def test_none_relevant_rewrites_once_then_picks_from_the_new_search():
    r, llm = retr({"picks": [], "rewrite": "数字经济 主要经济形态"}, {"picks": [1], "rewrite": ""})
    out = r.retrieve("数据开放共享", count=1)
    assert out["mode"] == "search" and len(llm.calls) == 2 and out["trace"]["rewritten"] == "数字经济 主要经济形态"
    assert [x["query"] for x in out["trace"]["rounds"]] == ["数据开放共享", "数字经济 主要经济形态"] and out["picks"][0].speech.speech_id == "quote:2022:01"
    assert "数据开放共享" in llm.calls[1]["messages"][1]["content"]


def test_still_none_after_rewrite_is_not_found_and_never_rewrites_twice():
    r, llm = retr({"picks": [], "rewrite": "数字经济"}, {"picks": [], "rewrite": "再改"}, {"picks": [1], "rewrite": ""})
    out = r.retrieve("数据开放共享", count=1)
    assert out["mode"] == "not_found" and out["picks"] == [] and len(llm.calls) == 2 and len(llm.outs) == 1


def test_unchanged_or_empty_rewrite_is_not_searched_again():
    for rw in ("数据开放共享", ""):
        r, llm = retr({"picks": [], "rewrite": rw})
        assert r.retrieve("数据开放共享", count=1)["mode"] == "not_found" and len(llm.calls) == 1


def test_no_candidates_means_not_found_without_calling_the_model():
    r, llm = retr()
    assert r.retrieve("ZZZ QQQ", count=1)["mode"] == "not_found" and llm.calls == []


@pytest.mark.parametrize("bad", ["不是JSON", "[]", '{"picks": "1"}', '{"picks": [null, {}]}'])
def test_unusable_model_output_is_not_found_and_not_retried(bad):
    r, llm = retr(bad)
    assert r.retrieve("数据开放共享", count=1)["mode"] == "not_found" and len(llm.calls) == 1


def test_model_error_is_recorded_by_type_name_only():
    r, _ = retr(exc=TimeoutError("key=SECRET"))
    out = r.retrieve("数据开放共享", count=1)
    assert out["mode"] == "not_found" and out["trace"]["rounds"][0]["error"] == "TimeoutError" and "SECRET" not in json.dumps(out["trace"])


def test_empty_index_and_missing_docx_are_safe(tmp_path):
    assert QuoteIndex([]).candidates("数据") == [] and QuoteIndex.load(tmp_path / "没有.docx") is None


def test_pick_prompt_text_forbids_rewriting():
    assert "不要改写" in PICK_SYSTEM and "不要概括" in PICK_SYSTEM


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
