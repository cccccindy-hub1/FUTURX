"""跨库组合问答与统一入口接入：各库并行取材、只调一次作答、材料分区、来源分区、讲话原文照录、调用总数。全部 mock。"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag import combined
from rag.cases.retrieval import CaseHit, CaseIndex, CaseRef, load_case_refs
from rag.combined import (Materials, answer_combined, build_refs, quote_appendix, render_materials, scrub_fabricated_quotes,
                          tidy_after_removal)
from rag.course import config
from rag.course.llm import LLMResult
from rag.quotes.retrieval import QuoteIndex, QuotePick, SpeechHit
from rag.retrieval import RetrievalResult
from rag.router import RouteDecision
from rag.unified import UnifiedService, router as unified_router
from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: F401

SPEECH = SpeechHit("quote:2018:01", 2018, "全国网络安全和信息化工作会议上的讲话", "讲话",
                   ["没有网络安全就没有国家安全，就没有经济社会稳定运行。", "要加快数据资源整合和开放共享。", "这是与问题无关、没有被挑中的第三段讲话原文。"], 0.9)
ROWS = [{"case_id": "case:03", "seq": 3, "chapter": "第六章 数据市场", "chapter_no": 6,
         "title": "北京微梦创科诉云智联不正当竞争纠纷案", "meta": {"case_number": "（2020）京73民终2980号", "court": "北京知识产权法院"}},
        {"case_id": "case:07", "seq": 7, "chapter": "第二章 数据权属", "chapter_no": 2, "title": "大数据搬家软件案", "meta": {}}]
REFS = load_case_refs(ROWS)


@pytest.fixture(autouse=True)
def single_mode(request, monkeypatch):
    if "ctx" in request.fixturenames:
        request.getfixturevalue("ctx")
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(config, "COURSE_SELECT_FAST_PATH", False)
    monkeypatch.setattr(config, "COURSE_FILTER_SKIP_MAX", 0)


def route_llm(routes, count=1, chapter=None):
    class R:
        calls = 0

        def chat(self, step, messages, *, model, **k):
            R.calls += 1
            return LLMResult(content=json.dumps({"routes": routes, "complexity": "simple", "count": count, "chapter": chapter, "reason": "t"}))
    R.calls = 0
    return R()


def case_index():
    """内存里的小判例索引（不读 docx、不用向量库）：2 个案子，各有卡片，案 3 另有一个正文块。"""
    from rag.cases.retrieval import CaseIndex

    def ch(n, kind, text, heading="", part=0):
        row = next(r for r in ROWS if r["seq"] == n)
        return {"chunk_id": f"case:{n:02d}:{kind}{part}", "case_id": f"case:{n:02d}", "seq": n, "chapter": row["chapter"], "chapter_no": row["chapter_no"],
                "title": row["title"], "kind": kind, "part": part, "heading": heading, "text": text}
    return CaseIndex(REFS, [ch(3, "card", "【第六章 数据市场】北京微梦创科诉云智联\n裁判要旨：平台对其数据享有经营利益。爬取数据构成不正当竞争，是数据确权的典型判例。"),
                            ch(3, "body", "基本案情：云智联抓取了微博平台数据。", "基本案情", 1),
                            ch(7, "card", "【第二章 数据权属】大数据搬家软件案\n裁判要旨：公开数据也受保护。")])


def quote_index():
    from rag.quotes.retrieval import QuoteIndex
    return QuoteIndex([{"speech_id": SPEECH.speech_id, "year": 2018, "n": 1, "title": SPEECH.title, "kind": "讲话", "paragraphs": SPEECH.paragraphs}])


class Recorder(ScriptedLLM):
    """脚本化 LLM：记录每一步的名称（select / filter / answer / case_answer / quote_pick）。"""

    def steps(self):
        return [c["step"] for c in self.calls]


def pick(): return jtxt({"question_concepts": ["C01-0001"], "related_concepts": ["C01-0003"], "reason": "r"})


def filt(step, msgs):
    refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
    return jtxt({"direct": refs[:1], "background": refs[1:]})


def case_answer_json(): return jtxt({"cases": [{"id": "C1", "facts": "云智联抓取数据[C1]。", "holding": "构成不正当竞争[C1]。", "relation": "说明确权难[C1]。"}]})


def case_pick_json(*seqs): return jtxt({"picks": [str(s) for s in seqs], "rewrite": ""})


def quote_pick_json(*phrases):
    """按内容挑候选段落：候选里含任一短语的都选（默认第一段）。候选编号由检索顺序决定，测试不依赖它。"""
    phrases = phrases or ("没有网络安全",)

    def run(step, msgs):
        cands = re.findall(r"^\[(\d+)\] .*\n(.*)$", msgs[1]["content"], flags=re.M)
        return jtxt({"picks": [int(k) for k, text in cands if any(ph in text for ph in phrases)], "rewrite": ""})
    return run


def svc_with(llm, routes, *, case=True, quote=True, count=1, chapter=None):
    return UnifiedService(router_client=route_llm(routes, count, chapter), course_ctx=_CTX["ctx"], course_llm=llm,
                          case_index=case_index() if case else None, quote_index=quote_index() if quote else None)


_CTX: dict = {}


@pytest.fixture
def cctx(ctx):
    _CTX["ctx"] = ctx
    return ctx


# ---------------------------------------------------------------- 引用校验：讲话引号文字必须逐字出自入选原文

ORIG = SPEECH.paragraphs[:2]


def test_verbatim_quotes_are_kept_and_paraphrased_quotes_removed():
    ok, rm = scrub_fabricated_quotes("总书记指出「没有网络安全就没有国家安全」，见[1]。", ORIG)
    assert ok == "总书记指出「没有网络安全就没有国家安全」，见[1]。" and rm == []
    bad, rm = scrub_fabricated_quotes("总书记强调“网络安全是国家安全的重要基石，必须高度重视”，见[1]。", ORIG)
    assert "重要基石" not in bad and rm == ["网络安全是国家安全的重要基石，必须高度重视"] and "见[1]" in bad


@pytest.mark.parametrize("quote", ["「要加快推进数据资源整合和开放共享」", "“没有网络安全就没有国家安全，也就没有经济社会稳定运行”", '"要加快数据资源整合与开放共享之举措"'])
def test_near_miss_rewrites_are_removed(quote):
    out, rm = scrub_fabricated_quotes(f"总书记说{quote}，如[1]。", ORIG)
    assert rm and quote[1:-1] not in out


def test_short_quoted_words_are_not_treated_as_speech_quotes():
    out, rm = scrub_fabricated_quotes("所谓「数据要素」是指…，也叫“新型生产要素”。", ORIG)
    assert rm == [] and "「数据要素」" in out


def test_punctuation_and_whitespace_differences_do_not_cause_false_removal():
    out, rm = scrub_fabricated_quotes("总书记说“没有网络安全就没有国家安全 就没有经济社会稳定运行”。", ORIG)
    assert rm == []


def test_quote_can_come_from_any_picked_paragraph_but_not_from_unpicked_ones():
    unpicked = SPEECH.paragraphs[2]
    out, rm = scrub_fabricated_quotes(f"他说「{unpicked}」。", ORIG)                    # 第三段没被挑中 -> 不是入选原文
    assert rm and unpicked[:6] not in out


def test_tidy_after_removal_fixes_dangling_punctuation():
    assert tidy_after_removal("总书记说：。见[1]。") == "总书记说。见[1]。"
    assert tidy_after_removal("要点（）。") == "要点。"


# ---------------------------------------------------------------- 材料分区与引用表

def test_materials_are_partitioned_and_ordered_course_case_quote(cctx):
    from test_course_pipeline import _items
    kept = _items(cctx, "P0005")[:1]
    m = Materials(course_kept=kept, course_refs={"S1": kept[0].chunk_id}, course_text=f"[S1] 路径\n{kept[0].text}",
                  case_hits=[CaseHit(ref=REFS[0], score=.9, card="卡片文本", bodies=["正文文本"])],
                  quote_picks=[QuotePick(1, SPEECH, [1])])
    text = render_materials(m)
    assert text.index("【教材材料】") < text.index("【判例材料】") < text.index("【讲话原文】")
    assert "[S1] 路径" in text and "[C1] 卡片文本\n正文文本" in text and "[Q1] 2018年，全国网络安全和信息化工作会议上的讲话（只供引用，不得改写）" in text
    assert SPEECH.paragraphs[0] in text and SPEECH.paragraphs[2] not in text             # 讲话区只放挑出的段落
    assert m.sections == ["course", "case", "quote"]


def test_sections_only_include_nonempty_libraries():
    m = Materials(case_hits=[CaseHit(ref=REFS[0], score=.9, card="c", bodies=[])])
    assert render_materials(m).startswith("【判例材料】") and "【教材材料】" not in render_materials(m) and m.sections == ["case"]


def test_build_refs_gives_unique_markers_per_library_in_order(cctx):
    from test_course_pipeline import _items
    kept = _items(cctx, "P0005", "P0008")[:2]
    m = Materials(course_kept=kept, course_refs={"S1": kept[0].chunk_id, "S2": kept[1].chunk_id},
                  case_hits=[CaseHit(ref=REFS[0], score=.9, card="c", bodies=[])], quote_picks=[QuotePick(1, SPEECH, [1])])
    refs, info = build_refs(m)
    assert list(refs) == ["S1", "S2", "S3", "S4"] and [info[v][0] for v in refs.values()] == ["course", "course", "case", "quote"]


# ---------------------------------------------------------------- 一次作答：编号统一、来源分区、讲话原文贴在末尾

def _materials(cctx):
    from test_course_pipeline import _items
    kept = _items(cctx, "P0005")[:1]
    return Materials(course_kept=kept, course_refs={"S1": kept[0].chunk_id}, course_text=f"[S1] 路径\n{kept[0].text}",
                     case_hits=[CaseHit(ref=REFS[0], score=.9, card="卡片", bodies=["正文"])], quote_picks=[QuotePick(1, SPEECH, [1])])


class AnsLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages})
        return LLMResult(content=self.text)


def test_one_call_unified_numbers_and_partitioned_sources(cctx):
    m = _materials(cctx)
    llm = AnsLLM("数据确权难在权属不清[S1]。云智联案中法院认定平台对数据享有经营利益[C1]。总书记强调网络安全是国家安全的前提[Q1]。")
    ans = answer_combined("结合判例和讲话谈数据确权", m, llm, model="m")
    assert len(llm.calls) == 1 and llm.calls[0]["step"] == "answer"
    assert ans.body == "数据确权难在权属不清[1]。云智联案中法院认定平台对数据享有经营利益[2]。总书记强调网络安全是国家安全的前提[3]。"
    assert [(s["n"], s["kind"]) for s in ans.sources] == [(1, "course"), (2, "case"), (3, "quote")]          # 教材 / 判例 / 讲话 三区，序号连续
    assert ans.sources[1]["case_number"] == "（2020）京73民终2980号" and ans.sources[2]["title"].startswith("全国网络安全") and ans.sources[2]["year"] == 2018
    assert re.search(r"\[[SCQ]\d", ans.text) is None                                      # 正文与附录里都没有残留的 [S#]/[C#]/[Q#]
    assert "case:" not in ans.body and "quote:" not in ans.body and "course:" not in ans.body


def test_quote_original_is_pasted_by_program_at_the_end_only_for_cited_speeches(cctx):
    m = _materials(cctx)
    cited = answer_combined("问", m, AnsLLM("论断[S1]。总书记强调[Q1]。"), model="m")
    assert cited.text.endswith("【讲话原文】\n「" + SPEECH.paragraphs[0] + "」\n——2018年，全国网络安全和信息化工作会议上的讲话")
    not_cited = answer_combined("问", m, AnsLLM("论断[S1]。判例[C1]。"), model="m")
    assert "【讲话原文】" not in not_cited.text and [s["kind"] for s in not_cited.sources] == ["course", "case"]
    assert quote_appendix([]) == ""


def test_paraphrased_speech_in_quotes_is_removed_but_marker_kept(cctx):
    m = _materials(cctx)
    llm = AnsLLM("总书记指出“网络安全是国家发展的基础，要高度重视”[Q1]；教材认为权属要清晰[S1]。")
    ans = answer_combined("问", m, llm, model="m")
    assert "国家发展的基础" not in ans.body and ans.removed_quotes == ["网络安全是国家发展的基础，要高度重视"]
    assert ans.body == "总书记指出[1]；教材认为权属要清晰[2]。"                              # 引号文字被删，[Q1] 标记保留（讲话先出现 -> [1]）
    assert [(s["n"], s["kind"]) for s in ans.sources] == [(1, "quote"), (2, "course")]
    assert "【讲话原文】" in ans.text and SPEECH.paragraphs[0] in ans.text                  # 原文由程序贴出，学生看到的是真实原话


def test_verbatim_excerpt_in_quotes_survives_and_original_is_still_appended(cctx):
    llm = AnsLLM("总书记指出「没有网络安全就没有国家安全」[Q1]。")
    ans = answer_combined("问", _materials(cctx), llm, model="m")
    assert "「没有网络安全就没有国家安全」" in ans.body and ans.removed_quotes == []


def test_markers_out_of_range_are_dropped_and_not_listed_as_sources(cctx):
    ans = answer_combined("问", _materials(cctx), AnsLLM("甲[S9]乙[C9]丙[Q9]丁[S1]。"), model="m")
    assert ans.body == "甲乙丙丁[1]。" and [s["n"] for s in ans.sources] == [1]


def test_stream_delta_matches_final_body_with_markers_split_across_chunks(cctx):
    class Streamer(AnsLLM):
        def chat_stream(self, step, messages, *, model, on_delta):
            for piece in ("数据确权难[", "S1]。判例[C", "1]说明。讲话[Q", "1]。"):
                on_delta(piece)
            return LLMResult(content="")
    got, mats = [], []
    ans = answer_combined("问", _materials(cctx), Streamer(""), model="m", on_delta=got.append, on_source=mats.append)
    assert "".join(got) == ans.body == "数据确权难[1]。判例[2]说明。讲话[3]。"
    assert [(e["n"], e["kind"]) for e in mats] == [(1, "course"), (2, "case"), (3, "quote")] and all("path" in e for e in mats)


def test_combined_prompt_requires_single_answer_and_forbids_writing_speech_content():
    s = combined.COMBINED_SYSTEM
    for w in ("只写一个回答", "先把知识点讲清楚", "不要分成", "[Q#]", "不要自己写出讲话里的话", "不要把讲话文字放进引号"):
        assert w in s


# ---------------------------------------------------------------- 统一入口：组合问答只调一次作答；课程 + 判例 = 4 次

def test_course_plus_case_is_exactly_four_llm_calls_with_one_answer(cctx):
    llm = StepLLM({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [LLMResult(content="知识点讲解[S1]。判例印证[C1]。")]})
    svc = svc_with(llm, ["course", "case"])
    out = svc.ask("结合判例说明数据确权为什么难")
    assert sorted(llm.steps()) == ["answer", "case_pick", "filter", "select"] and svc.router_client.calls == 1
    assert svc.router_client.calls + len(llm.calls) == 5                                  # 路由 1 + 选概念 1 + 筛选 1 + 选案 1 + 作答 1
    assert out["routes"] == ["course", "case"] and out["route"] == "course" and out["status"] == "ok"
    assert [s["kind"] for s in out["sources"]] == ["course", "case"] and "[1]" in out["answer"] and "[2]" in out["answer"]
    assert "case:03" not in out["answer"] and "【讲话原文】" not in out["answer"]
    assert out["trace"]["combined"]["sections"] == ["course", "case"] and "course" in out["trace"]["combined"] and "case" in out["trace"]["combined"]


def test_course_plus_quote_adds_exactly_one_pick_call(cctx):
    llm = Recorder([pick(), filt, jtxt({"picks": [{"id": "Q1", "paragraphs": [1]}]}), LLMResult(content="知识点[S1]。总书记指出[Q1]。")])
    # 并行：挑段落与课程的选概念 / 筛选并发，调用顺序不固定，所以用按 step 取脚本的实现
    llm = StepLLM({"select": [pick()], "filter": [filt], "quote_pick": [quote_pick_json()], "answer": [LLMResult(content="知识点[S1]。总书记指出[Q1]。")]})
    svc = svc_with(llm, ["course", "quote"])
    out = svc.ask("用总书记的讲话谈谈数据要素与网络安全的作用")
    assert sorted(llm.steps()) == ["answer", "filter", "quote_pick", "select"] and svc.router_client.calls + len(llm.calls) == 5      # 课程 + 名言 = 路由 1 + 4
    assert llm.steps().count("answer") == 1                                               # 只有一次作答
    assert out["answer"].endswith("——2018年，全国网络安全和信息化工作会议上的讲话") and [s["kind"] for s in out["sources"]] == ["course", "quote"]


class StepLLM:
    """按 step 名取脚本，线程安全（并行取材时各库的调用顺序不固定）。"""

    def __init__(self, scripts):
        import threading
        self.scripts, self.calls, self.lock = {k: list(v) for k, v in scripts.items()}, [], threading.Lock()

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        with self.lock:
            self.calls.append({"step": step, "messages": messages, "model": model})
            item = self.scripts[step].pop(0)
        return item(step, messages) if callable(item) else item

    def steps(self):
        return [c["step"] for c in self.calls]


def test_case_plus_quote_makes_pick_and_one_answer_without_course(cctx):
    llm = StepLLM({"case_pick": [case_pick_json(3)], "quote_pick": [quote_pick_json()], "answer": [LLMResult(content="判例讲了数据权益[C1]。总书记强调[Q1]。")]})
    svc = svc_with(llm, ["case", "quote"])
    out = svc.ask("结合判例和总书记的讲话谈谈网络安全")
    assert sorted(llm.steps()) == ["answer", "case_pick", "quote_pick"] and [s["kind"] for s in out["sources"]] == ["case", "quote"]
    assert out["trace"]["combined"]["sections"] == ["case", "quote"]


def test_materials_given_to_the_answer_model_are_partitioned(cctx):
    llm = StepLLM({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [LLMResult(content="x[S1][C1]")]})
    svc = svc_with(llm, ["course", "case"])
    svc.ask("结合判例说明数据确权为什么难")
    user = next(c for c in llm.calls if c["step"] == "answer")["messages"][1]["content"]
    assert user.index("【教材材料】") < user.index("【判例材料】") and "[S1]" in user and "[C1]" in user
    assert "云智联" in user and "【讲话原文】" not in user                               # 判例材料确实被带入，没选讲话就没有讲话区


def test_each_library_is_filtered_independently_and_timed(cctx):
    llm = StepLLM({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [LLMResult(content="x[S1][C1]")]})
    out = svc_with(llm, ["course", "case"]).ask("结合判例说明数据确权为什么难")
    c = out["trace"]["combined"]
    assert c["course"]["kept"] >= 1 and c["course"]["trace"]["totals"]["calls"] == 2        # 课程库自己的 select + filter，不含作答
    assert c["case"]["mode"] == "search" and "retrieve_s" in c["case"] and "wall_s" in c and "answer_s" in c


def test_libraries_run_in_parallel_threads(cctx, monkeypatch):
    """两个库的取材必须同时在跑：用一个 2 方屏障——串行执行时第一个库永远等不到第二个，会超时失败。"""
    import threading
    barrier = threading.Barrier(2, timeout=3)
    orig_case, orig_course = combined._case_material, combined._course_material

    def wrap(fn):
        def run(*a, **k):
            barrier.wait()                                   # 两个库都进入取材后才继续
            return fn(*a, **k)
        return run
    monkeypatch.setattr(combined, "_case_material", wrap(orig_case))
    monkeypatch.setattr(combined, "_course_material", wrap(orig_course))
    llm = StepLLM({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [LLMResult(content="x[S1][C1]")]})
    out = svc_with(llm, ["course", "case"]).ask("结合判例说明数据确权为什么难")
    assert out["status"] == "ok" and not barrier.broken


def test_one_library_empty_still_answers_from_the_other_and_reports_missing(cctx):
    empty_case = CaseIndex(REFS, [])
    llm = StepLLM({"select": [pick()], "filter": [filt], "answer": [LLMResult(content="只有教材[S1]。")]})
    svc = UnifiedService(router_client=route_llm(["course", "case"]), course_ctx=cctx, course_llm=llm, case_index=empty_case)
    out = svc.ask("结合判例说明数据确权为什么难")
    assert out["status"] == "ok" and [s["kind"] for s in out["sources"]] == ["course"] and out["trace"]["combined"]["missing"] == ["case"]


def test_all_libraries_empty_means_insufficient_and_no_answer_call(cctx):
    llm = StepLLM({"select": [LLMResult(content="乱码")], "filter": []})
    svc = UnifiedService(router_client=route_llm(["case", "quote"]), course_ctx=cctx, course_llm=llm,
                         case_index=CaseIndex(REFS, []), quote_index=QuoteIndex([]))
    out = svc.ask("结合判例和总书记的讲话谈谈网络安全")
    assert out["status"] == "insufficient" and "answer" not in llm.steps() and out["sources"] == []


def test_combined_with_unloaded_library_returns_unavailable_without_any_call(cctx):
    llm = StepLLM({})
    out = svc_with(llm, ["course", "case"], case=False).ask("结合判例说明数据确权为什么难")
    assert out["status"] == "unavailable" and "判例库" in out["answer"] and llm.calls == []


# ================================================================ 单库判例 / 名言（统一入口）

def test_single_case_route_one_answer_call_no_stream(cctx):
    llm = StepLLM({"case_pick": [case_pick_json(3)], "case_answer": [case_answer_json()]})
    svc = svc_with(llm, ["case"], count=1, chapter=6)
    out = svc.ask("举一个关于数据爬取的判例")
    assert llm.steps() == ["case_pick", "case_answer"] and svc.router_client.calls == 1 and svc.router_client.calls + len(llm.calls) == 3    # 路由 1 + 选案 1 + 作答 1
    assert out["route"] == "case" and out["routes"] == ["case"] and out["count"] == 1 and out["chapter"] == 6 and out["status"] == "ok"
    assert out["answer"].startswith("【第六章 数据市场】北京微梦创科诉云智联不正当竞争纠纷案\n案号：（2020）京73民终2980号") and "[1]" in out["answer"]
    assert out["sources"][0]["kind"] == "case" and out["sources"][0]["case_number"] == "（2020）京73民终2980号"
    assert out["trace"]["case"]["answer_calls"] == 1 and out["trace"]["case"]["llm_calls"] == 1 and out["trace"]["case"]["mode"] == "search"


def test_single_case_named_goes_straight_to_the_case_without_vector_search(cctx):
    llm = StepLLM({"case_answer": [case_answer_json()]})
    svc = svc_with(llm, ["case"])
    out = svc.ask("微博诉云智联案是怎么判的")
    assert out["trace"]["case"]["mode"] == "named" and llm.steps() == ["case_answer"]        # 点名直达：不检索、不调用选案模型


def test_case_list_mode_calls_no_model_and_no_retrieval(cctx):
    llm = StepLLM({})
    svc = svc_with(llm, ["case"], chapter=6)
    out = svc.ask("第六章有哪些判例？")
    assert llm.calls == [] and svc.router_client.calls == 1
    assert out["status"] == "ok" and out["answer"].startswith("第六章 数据市场共有 1 个判例：") and "1. 北京微梦创科" in out["answer"]
    assert out["trace"]["case"]["mode"] == "list" and out["trace"]["case"]["llm_calls"] == 0


def test_case_not_found_does_not_call_the_answer_model(cctx):
    llm = StepLLM({})
    svc = UnifiedService(router_client=route_llm(["case"]), course_ctx=cctx, course_llm=llm, case_index=CaseIndex(REFS, []))
    out = svc.ask("一个完全无关的判例问题")
    assert llm.calls == [] and out["status"] == "insufficient" and out["answer"].startswith("判例库里没有找到") and out["sources"] == []


def test_single_quote_route_is_verbatim_with_one_pick_call_and_no_answer_model(cctx):
    llm = StepLLM({"quote_pick": [quote_pick_json("没有网络安全", "资源整合")]})
    svc = svc_with(llm, ["quote"], count=2)
    out = svc.ask("总书记关于网络安全与数据资源整合是怎么说的")
    assert llm.steps() == ["quote_pick"] and svc.router_client.calls + len(llm.calls) == 2         # 路由 1 + 挑段落 1，不调用作答模型
    assert out["answer"] == ("「" + SPEECH.paragraphs[0] + "\n" + SPEECH.paragraphs[1] + "」[1]\n——2018年，全国网络安全和信息化工作会议上的讲话")
    assert out["sources"] == [{"n": 1, "kind": "quote", "chunk_id": "quote:2018:01", "path": "2018年，全国网络安全和信息化工作会议上的讲话",
                               "title": "全国网络安全和信息化工作会议上的讲话", "year": 2018}]


def test_quote_not_found_says_so_and_skips_the_pick_call(cctx):
    llm = StepLLM({})
    svc = UnifiedService(router_client=route_llm(["quote"]), course_ctx=cctx, course_llm=llm, quote_index=QuoteIndex([]))
    out = svc.ask("一个合集里没有的话题")
    assert llm.calls == [] and out["status"] == "insufficient" and out["answer"] == "合集中没有找到相关表述。"


def test_case_and_quote_unavailable_when_libraries_not_loaded(cctx):
    llm = StepLLM({})
    a = svc_with(llm, ["case"], case=False).ask("举个判例")
    b = svc_with(llm, ["quote"], quote=False).ask("总书记怎么说")
    assert a["status"] == "unavailable" and "判例库" in a["answer"] and b["status"] == "unavailable" and "名言库" in b["answer"] and llm.calls == []


# ================================================================ 现有路由的行为不变

def test_existing_routes_unchanged_by_the_new_dispatch(cctx):
    llm = StepLLM({"select": [pick()], "filter": [filt], "answer": [LLMResult(content="课程答案[S1]。")]})
    out = svc_with(llm, ["course"], case=False, quote=False).ask("什么是数据要素")
    assert out["route"] == "course" and out["routes"] == ["course"] and out["status"] == "ok" and out["answer"] == "课程答案[1]。"
    assert llm.steps() == ["select", "filter", "answer"] and "combined" not in out["trace"] and "case" not in out["trace"]
    off = svc_with(StepLLM({}), ["off_topic"]).ask("今天天气怎么样")
    pol = svc_with(StepLLM({}), ["policy"]).ask("数据二十条第几条")
    assert off["status"] == "refused" and off["answer"] == config_text("refusal") and pol["status"] == "not_available"


def config_text(kind):
    from rag import config as rc
    return rc.ROUTER_REFUSAL_TEXT if kind == "refusal" else rc.ROUTER_POLICY_TEXT


def test_thesis_route_still_uses_adaptive(cctx):
    class Adaptive:
        def run(self, q, top_k, cx):
            return {"status": "ok", "answer": "论文答案", "sources": [{"source": "a"}], "contexts": ["c"], "trace": {"path": "simple"}}
    svc = svc_with(StepLLM({}), ["thesis"], case=False, quote=False)
    svc.adaptive = Adaptive()
    out = svc.ask("怎么写开题报告")
    assert out["route"] == "thesis" and out["answer"] == "论文答案" and out["trace"]["adaptive"] == {"path": "simple"}


def test_old_route_only_json_from_the_router_still_dispatches_course(cctx):
    class OldRouter:
        def chat(self, step, messages, *, model, **k):
            return LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"}))
    llm = StepLLM({"select": [pick()], "filter": [filt], "answer": [LLMResult(content="a[S1]")]})
    svc = UnifiedService(router_client=OldRouter(), course_ctx=cctx, course_llm=llm)
    assert svc.ask("什么是数据要素")["routes"] == ["course"]


# ================================================================ HTTP / SSE

def _app(svc):
    app = FastAPI()
    app.state.unified = svc
    app.include_router(unified_router)
    return TestClient(app)


def _events(resp):
    return [json.loads(b[6:]) for b in resp.text.split("\n\n") if b.startswith("data: ")]


class StreamStep(StepLLM):
    """StepLLM + 作答流式：分片里把 [C1]、[Q1] 拆在边界上。"""

    def __init__(self, scripts, pieces):
        super().__init__(scripts)
        self.pieces = pieces

    def chat_stream(self, step, messages, *, model, on_delta):
        with self.lock:
            self.calls.append({"step": step, "messages": messages, "model": model, "stream": True})
            self.scripts[step].pop(0)
        for p in self.pieces:
            on_delta(p)
        return LLMResult(content="".join(self.pieces))


def test_auto_ask_http_returns_routes_count_chapter_and_sources(cctx):
    llm = StepLLM({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [LLMResult(content="知识点[S1]。判例[C1]。")]})
    r = _app(svc_with(llm, ["course", "case"], count=2, chapter=2)).post("/auto/ask", json={"question": "结合判例说明数据确权为什么难"})
    body = r.json()
    assert r.status_code == 200 and body["routes"] == ["course", "case"] and body["route"] == "course" and body["count"] == 2 and body["chapter"] == 2
    assert [s["kind"] for s in body["sources"]] == ["course", "case"]


def test_sse_single_case_is_not_streamed_result_in_done(cctx):
    llm = StepLLM({"case_pick": [case_pick_json(3)], "case_answer": [case_answer_json()]})
    ev = _events(_app(svc_with(llm, ["case"])).post("/auto/ask/stream", json={"question": "举一个关于数据爬取的判例"}))
    kinds = [e["type"] for e in ev]
    assert kinds == ["route", "done"] and ev[0]["routes"] == ["case"] and "delta" not in kinds and "materials" not in kinds
    assert ev[-1]["answer"].startswith("【第六章 数据市场】") and ev[-1]["sources"][0]["kind"] == "case" and ev[-1]["routes"] == ["case"]


def test_sse_single_quote_is_not_streamed_result_in_done(cctx):
    llm = StepLLM({"quote_pick": [quote_pick_json()]})
    ev = _events(_app(svc_with(llm, ["quote"])).post("/auto/ask/stream", json={"question": "总书记关于网络安全与数据资源整合是怎么说的"}))
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["answer"].startswith("「" + SPEECH.paragraphs[0] + "」")
    assert ev[-1]["sources"][0]["kind"] == "quote"


def test_sse_case_list_mode_has_no_model_calls(cctx):
    llm = StepLLM({})
    ev = _events(_app(svc_with(llm, ["case"], chapter=6)).post("/auto/ask/stream", json={"question": "第六章有哪些判例"}))
    assert [e["type"] for e in ev] == ["route", "done"] and "共有 1 个判例" in ev[-1]["answer"] and llm.calls == []


def test_sse_combined_streams_the_single_answer_with_kind_in_materials(cctx):
    pieces = ["数据确权难在权属不清[S", "1]。判例印证[C", "1]。总书记强调[Q", "1]。"]
    llm = StreamStep({"select": [pick()], "filter": [filt], "case_pick": [case_pick_json(3)], "answer": [None]}, pieces)
    svc = svc_with(llm, ["course", "case"])
    ev = _events(_app(svc).post("/auto/ask/stream", json={"question": "结合判例说明数据确权为什么难"}))
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "route" and ev[0]["routes"] == ["course", "case"] and kinds[-1] == "done" and kinds.count("done") == 1 and "error" not in kinds
    assert "delta" in kinds and llm.steps().count("answer") == 1                                    # 只有一次作答，且流式
    mats = [e for e in ev if e["type"] == "materials"]
    assert [(e["n"], e["kind"]) for e in mats] == [(1, "course"), (2, "case")] and all(e.get("path") and e.get("chunk_id") for e in mats)
    text = "".join(e["text"] for e in ev if e["type"] == "delta")
    assert "[S" not in text and "[C" not in text and "[Q" not in text and "S1" not in text and "C1" not in text      # 分片边界上的标记已缓冲，没有半截泄露
    done = ev[-1]
    assert done["status"] == "ok" and done["routes"] == ["course", "case"] and [s["kind"] for s in done["sources"]] == ["course", "case"]
    first_marker = next(i for i, e in enumerate(ev) if e["type"] == "delta" and "[1]" in e["text"])
    assert ev.index(mats[0]) < first_marker                                                         # materials 先于第一个带 [1] 的 delta


def test_sse_combined_with_quote_streams_and_done_carries_verbatim_appendix(cctx):
    pieces = ["知识点[S1]。", "总书记指出「没有网络安全就没有国家安全」[Q", "1]。"]
    llm = StreamStep({"select": [pick()], "filter": [filt], "quote_pick": [quote_pick_json()], "answer": [None]}, pieces)
    ev = _events(_app(svc_with(llm, ["course", "quote"])).post("/auto/ask/stream", json={"question": "用总书记的讲话谈谈数据要素与网络安全的作用"}))
    done = ev[-1]
    assert done["type"] == "done" and done["answer"].endswith("——2018年，全国网络安全和信息化工作会议上的讲话")
    assert "【讲话原文】\n「" + SPEECH.paragraphs[0] + "」" in done["answer"] and [s["kind"] for s in done["sources"]] == ["course", "quote"]
    assert "quote:" not in "".join(e.get("text", "") for e in ev if e["type"] == "delta")


def test_sse_combined_error_in_a_library_is_an_error_event_with_type_name_only(cctx):
    class Boom(StreamStep):
        def chat(self, step, messages, *, model, **k):
            if step == "select":
                raise RuntimeError("upstream key=SECRET-KEY detail")
            return super().chat(step, messages, model=model, **k)
    llm = Boom({"filter": [filt], "case_pick": [case_pick_json(3)], "answer": [None]}, ["x"])
    r = _app(svc_with(llm, ["course", "case"])).post("/auto/ask/stream", json={"question": "结合判例说明数据确权为什么难"})
    ev = _events(r)
    assert ev[-1] == {"type": "error", "error": "RuntimeError"} and "done" not in [e["type"] for e in ev]
    assert "SECRET-KEY" not in r.text and "upstream" not in r.text


def test_sse_combined_unavailable_library_gives_done_unavailable(cctx):
    ev = _events(_app(svc_with(StepLLM({}), ["course", "case"], case=False)).post("/auto/ask/stream", json={"question": "结合判例说明数据确权为什么难"}))
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["status"] == "unavailable"


def test_sse_course_only_route_still_streams_as_before(cctx):
    llm = StreamStep({"select": [pick()], "filter": [filt], "answer": [None]}, ["课程答案[S1]。"])
    ev = _events(_app(svc_with(llm, ["course"], case=False, quote=False)).post("/auto/ask/stream", json={"question": "什么是数据要素"}))
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "route" and ev[0]["routes"] == ["course"] and "delta" in kinds and ev[-1]["answer"] == "课程答案[1]。"


def test_combined_stream_slot_is_in_concurrency_paths():
    from rag import concurrency
    assert "/auto/ask/stream" in concurrency.RequestConcurrencyMiddleware._PATHS            # 组合 / 单库都走这两个已计入并发的路径


# ---------------------------------------------------------------- 组合回答：总—分—总 + 结尾问句

def test_combined_prompt_requires_total_part_total_and_a_closing_question_without_citation():
    from rag.combined import COMBINED_SYSTEM as S
    for w in ("总—分—总", "总：开头用一两句话直接回答", "分：先把知识点讲清楚，再用判例或讲话来印证", "总：用一句话收束要点",
              "最后单独一行，写一个问句", "这个问句不加引用编号", "不要分成「教材部分」「判例部分」两个独立回答"):
        assert w in S
    assert "不写引言和总结段" not in S                           # 旧的与“总”相矛盾的指令已去掉


def test_combined_prompt_keeps_quote_rules_and_matches_the_course_prompts_structure_wording():
    from rag.combined import COMBINED_SYSTEM as S
    from rag.course import prompts
    for w in ("[Q#] 标记来指明位置", "不要把讲话文字放进引号", "[S2]、[C1]、[Q1]"):
        assert w in S
    shared = "最后单独一行，写一个问句：结合本题内容引导学生进一步思考，或询问学生还想了解哪方面"
    assert shared in S and shared in prompts.ANSWER_SYSTEM                  # 与线上课程单库的问句要求同一措辞


def test_closing_question_survives_citation_rewriting_and_quote_scrubbing():
    """问句不带编号、可能含引号 / 问号：数字编号改写和“引号内讲话文字核对”都不能把它删掉或改动。"""
    from rag.combined import scrub_fabricated_quotes
    body = "数据确权的难点在于边界不清[1]，判例也体现了这一点[2]。\n你想进一步了解数据确权的权利束，还是平台数据的保护边界？"
    out, removed = scrub_fabricated_quotes(body, ["数据是新型生产要素。"])
    assert out == body and removed == []


def test_closing_question_with_short_quoted_terms_is_not_mistaken_for_a_fabricated_speech():
    from rag.combined import scrub_fabricated_quotes
    q = "你想进一步了解“数据要素”的内涵，还是“三权分置”？"
    assert scrub_fabricated_quotes(q, ["数据是新型生产要素。"]) == (q, [])
    out, removed = scrub_fabricated_quotes("总书记指出“数据是最重要的生产要素，必须无条件开放”[Q1]。\n" + q, ["数据是新型生产要素。"])
    assert removed == ["数据是最重要的生产要素，必须无条件开放"] and out.endswith(q) and "必须无条件开放" not in out      # 编造的删，问句原样留下
