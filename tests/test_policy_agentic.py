"""政策库（agentic）：按条切块与元数据、点名直达（文件 / 条）、地区限定的 BM25、模型选块、改写只重查一次、
作答的引文逐字校验 / 征求意见稿标注 / 附注 / 来源格式、统一入口与组合问答接入。全部 mock，不用向量库，不调用真实模型。"""
from __future__ import annotations

import json
import re

import pytest

from rag.course.llm import LLMResult
from rag.policy import config
from rag.policy import retrieval as pr
from rag.policy.corpus import article_number, build_corpus, cn_to_int, corpus_stats, short_name
from rag.policy.retrieval import PolicyIndex, PolicyRetriever, answer_policy, pick_chunks
from tests.test_course_chunk import _p, _tbl, _write

ART21 = "国家建立数据分类分级保护制度，对数据实行分类分级保护。国家数据安全工作协调机制统筹协调有关部门制定重要数据目录。"


def _law() -> list[str]:
    return [
        _p("1.2 《中华人民共和国数据安全法》", "Heading2"),
        _p("中华人民共和国数据安全法"),
        _p("（2021年6月10日第十三届全国人民代表大会常务委员会第二十九次会议通过）"),
        _p("第一章　总　　则"),
        _p("第一条　为了规范数据处理活动，保障数据安全，促进数据开发利用，制定本法。"),
        _p("第二条　在中华人民共和国境内开展数据处理活动及其安全监管，适用本法。"),
        _p("第三条　本法所称数据，是指任何以电子或者其他方式对信息的记录。"),
        _p("第二章　数据安全与发展"),
        _p("第四条　维护数据安全，应当坚持总体国家安全观。"),
        _p("第二十一条　" + ART21),
        _p("关系国家安全、国民经济命脉的数据，实行更加严格的管理制度。"),
    ]


def _opinion() -> list[str]:
    return [
        _p("2.50 《中共中央 国务院关于构建数据基础制度更好发挥数据要素作用的意见》", "Heading2"),
        _p("中共中央 国务院关于构建数据基础制度更好发挥数据要素作用的意见"),
        _p("（2022年12月2日）"),
        _p("数据作为新型生产要素，是数字化、网络化、智能化的基础。" * 30),
        _p("一、建立数据产权制度"),
        _p("探索数据产权结构性分置制度，建立公共数据、企业数据、个人数据的分类分级确权授权制度。" * 12),
        _tbl([["事项", "要求"], ["确权授权", "分类分级"]]),
        _p("二、建立合规高效的数据要素流通和交易制度。" + "完善数据全流程合规与监管规则体系。" * 30),
    ]


def _local(no: str, title: str, article: str, pian: str) -> list[str]:
    arts = [f"第{n}条　{article}" for n in ("一", "二", "三", "四", "五")]
    return [_p(f"{no} {title}", "Heading2"), _p(title.replace("（征求意见稿）", "")), *map(_p, arts)]


def _sample(tmp_path):
    body = [_p("第1篇 国家立法", "Heading1"), *_law(),
            _p("第2篇 中共中央、国务院及国家部委相关政策", "Heading1"), *_opinion(),
            _p("第4篇 上海市数据相关政策", "Heading1"), *_local("4.2", "上海市公共数据开放暂行办法", "公共数据开放应当遵循需求导向、分类分级的原则。", "上海"),
            _p("第6篇 深圳市数据相关政策", "Heading1"), *_local("6.6", "深圳市公共数据开放管理办法（征求意见稿）", "公共数据开放应当遵循安全可控的原则。", "深圳")]
    return _write(tmp_path, body, "汇编.docx")


@pytest.fixture
def corpus(tmp_path):
    return build_corpus(_sample(tmp_path))


@pytest.fixture
def index(corpus):
    return PolicyIndex(corpus)


# ---------------------------------------------------------------- 切块与元数据

def test_article_files_are_split_per_article_with_chapter_and_number_in_the_prefix(corpus):
    arts = [c for c in corpus["chunks"] if c["doc_no"] == "1.2" and c["kind"] == "article"]
    assert [c["article"] for c in arts] == [1, 2, 3, 4, 21]
    a21 = arts[-1]
    assert a21["body"].startswith("第二十一条") and "更加严格的管理制度" in a21["body"]               # 条下的各款并入本条
    assert a21["text"].startswith("第1篇 国家立法｜中华人民共和国数据安全法｜第二章 数据安全与发展\n第二十一条")
    assert a21["pian"] == "第1篇 国家立法" and a21["policy"] == "中华人民共和国数据安全法" and a21["article_label"] == "第二十一条"


def test_every_chunk_is_prefixed_with_pian_and_file_name(corpus):
    assert corpus["chunks"] and all(c["text"].startswith(f"{c['pian']}｜{c['policy']}") for c in corpus["chunks"])


def test_non_article_file_accumulates_paragraphs_into_800_to_1200_char_chunks_with_tables_inline(corpus):
    ch = [c for c in corpus["chunks"] if c["doc_no"] == "2.50"]
    assert all(c["kind"] != "article" and c["article"] is None for c in ch)
    assert all(len(c["body"]) <= config.POLICY_CHUNK_CAP for c in ch)
    assert all(len(c["body"]) >= config.POLICY_CHUNK_MIN for c in ch[:-1])                                   # 只有最后一块可以短
    assert any("确权授权 | 分类分级" in c["body"] for c in ch)                                                # 表格并入所在文件的块
    assert "".join(c["body"] for c in ch).count("建立数据产权制度") == 1


def test_metadata_region_short_name_date_and_draft_flag(corpus):
    f = {x["doc_no"]: x for x in corpus["files"]}
    assert f["1.2"]["region"] == "国家" and f["2.50"]["region"] == "国家" and f["4.2"]["region"] == "上海" and f["6.6"]["region"] == "深圳"
    assert f["1.2"]["date"] == "2021-06-10" and f["2.50"]["date"] == "2022-12-02"                           # 只取开头括号注里的日期
    assert f["4.2"]["date"] == "" and f["6.6"]["date"] == ""                                                   # 没有就留空，绝不猜
    assert f["6.6"]["draft"] is True and f["4.2"]["draft"] is False and f["1.2"]["draft"] is False
    assert all(c["draft"] == f[c["doc_no"]]["draft"] and c["region"] == f[c["doc_no"]]["region"] for c in corpus["chunks"])


def test_corpus_is_reproducible_and_chunk_ids_unique(tmp_path):
    a, b = build_corpus(_sample(tmp_path)), build_corpus(_sample(tmp_path))
    assert a == b and len({c["chunk_id"] for c in a["chunks"]}) == len(a["chunks"])


def test_stats_report_counts_and_region_distribution(corpus):
    s = corpus_stats(corpus)
    assert s["files"] == 4 and s["chunks"] == len(corpus["chunks"]) and s["article_chunks"] == 5 + 5 + 5
    assert s["chunks_by_region"]["上海"] == 5 and s["chunks_by_region"]["深圳"] == 5 and s["draft_files"] == 1 and s["duplicate_chunk_ids"] == 0


@pytest.mark.parametrize("title,short", [
    ("4.2 上海市公共数据开放暂行办法", "上海市公共数据开放暂行办法"),
    ("1.2 《中华人民共和国数据安全法》", "中华人民共和国数据安全法"),
    ("2.51 工业和信息化部关于印发《工业和信息化领域数据安全管理办法（试行）》的通知", "工业和信息化领域数据安全管理办法（试行）"),
    ("2.54 中共中央、国务院印发了《数字中国建设整体布局规划》", "数字中国建设整体布局规划")])
def test_short_name_strips_numbers_book_marks_and_notice_shell(title, short):
    assert short_name(title) == short


@pytest.mark.parametrize("s,n", [("二十一", 21), ("一百零五", 105), ("十", 10), ("十二", 12), ("21", 21), ("三百", 300), ("abc", None)])
def test_chinese_numbers(s, n):
    assert cn_to_int(s) == n
    assert article_number(f"第{s}条") == n or n is None


# ---------------------------------------------------------------- 点名直达

def test_alias_data_twenty_articles_locks_file_2_50(index):
    assert index.find_files("数据二十条讲了什么") == ["2.50"]
    assert index.aliases["数据二十条"] == "2.50"


def test_file_name_and_short_name_lock_the_file_and_longest_alias_wins(index):
    assert index.find_files("上海市公共数据开放暂行办法有哪些规定") == ["4.2"]
    assert index.find_files("数据安全法怎么规定") == ["1.2"]                         # 「数据安全法」是自动别名（去掉「中华人民共和国」）
    assert index.find_files("今天天气怎么样") == []


def test_ambiguous_alias_is_dropped_so_naming_is_unique(corpus):
    other = {"doc_no": "9.9", "title": "9.9 《中华人民共和国数据安全法实施细则》", "policy": "中华人民共和国数据安全法实施细则",
             "short": "中华人民共和国数据安全法实施细则", "pian": "第2篇", "region": "国家", "date": "", "draft": False, "doc_type": "other",
             "n_chunks": 0, "n_articles": 0}
    ix = PolicyIndex({**corpus, "files": corpus["files"] + [other]})
    assert ix.find_files("数据安全法实施细则") == ["9.9"] and ix.find_files("中华人民共和国数据安全法") == ["1.2"]


class NoLLM:
    def chat(self, *a, **k):
        raise AssertionError("点名到具体条时不应调用选块模型")


def test_named_file_plus_article_fetches_that_article_without_any_model_call(index):
    out = PolicyRetriever(index, NoLLM(), model="flash").retrieve("数据安全法第二十一条规定了什么")
    assert out["mode"] == "named_article" and out["trace"]["llm_calls"] == 0 and out["trace"]["article"] == 21
    assert [h.chunk["article_label"] for h in out["hits"]] == ["第二十一条"] and out["hits"][0].n == 1 and "分类分级" in out["hits"][0].chunk["body"]


def test_named_article_that_does_not_exist_falls_back_to_search_inside_the_file(index):
    class Pick:
        calls = []

        def chat(self, step, messages, *, model, **k):
            Pick.calls.append(messages[1]["content"])
            return LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))
    out = PolicyRetriever(index, Pick(), model="m").retrieve("数据安全法第九十九条规定了什么")
    assert out["mode"] == "search" and out["trace"]["article_missing"] == 99 and out["trace"]["files"] == ["1.2"]
    assert all("中华人民共和国数据安全法" in c for c in Pick.calls)                              # 候选只来自被点名的文件


# ---------------------------------------------------------------- BM25 初筛（地区）与模型选块

def test_candidates_are_capped_and_region_filter_applies(index, monkeypatch):
    assert all(index.chunks[i]["region"] == "上海" for i in index.candidates("公共数据开放", regions={"上海"}))
    assert {index.chunks[i]["region"] for i in index.candidates("公共数据开放原则")} >= {"上海", "深圳"}
    monkeypatch.setattr(config, "POLICY_CANDIDATES", 2)
    assert len(index.candidates("公共数据开放原则")) == 2


class PickLLM:
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "model": model, "json_mode": json_mode})
        out = self.outs.pop(0)
        return LLMResult(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))


def test_search_mentions_region_limits_candidates_and_makes_one_pick_call(index):
    llm = PickLLM({"picks": [1], "rewrite": ""})
    out = PolicyRetriever(index, llm, model="flash").retrieve("上海有哪些公共数据开放的规定")
    assert out["mode"] == "search" and out["trace"]["llm_calls"] == 1 and len(llm.calls) == 1
    assert llm.calls[0]["step"] == "policy_pick" and llm.calls[0]["model"] == "flash" and llm.calls[0]["json_mode"] is True
    assert out["hits"][0].chunk["region"] == "上海" and out["trace"]["rounds"][0]["regions"] == ["上海"]
    assert "深圳市公共数据开放管理办法" not in llm.calls[0]["messages"][1]["content"]                       # 地区限定生效


def test_region_filter_is_loosened_when_too_few_candidates(index):
    llm = PickLLM({"picks": [1], "rewrite": ""})
    PolicyRetriever(index, llm, model="m").retrieve("贵州的数据安全分类分级保护制度")                       # 库里没有贵州：放宽到全库
    assert "分类分级" in llm.calls[0]["messages"][1]["content"]


def test_picks_are_capped_at_six_and_unknown_or_duplicate_ids_dropped(index, monkeypatch):
    monkeypatch.setattr(config, "POLICY_PICK_MAX", 2)
    llm = PickLLM({"picks": [99, 1, 1, 2, 3], "rewrite": ""})
    out = PolicyRetriever(index, llm, model="m").retrieve("公共数据开放的原则")
    assert len(out["hits"]) == 2 and [h.n for h in out["hits"]] == [1, 2] and len(out["trace"]["rounds"][0]["dropped"]) == 2
    assert "最多 2 个" in llm.calls[0]["messages"][0]["content"]


def test_default_pick_limit_is_six():
    assert config.POLICY_PICK_MAX == 6 and config.POLICY_CANDIDATES == 15


def test_none_relevant_rewrites_once_then_not_found_without_a_third_call(index):
    llm = PickLLM({"picks": [], "rewrite": "数据分类分级 保护"}, {"picks": [], "rewrite": "再改写"}, {"picks": [1], "rewrite": ""})
    out = PolicyRetriever(index, llm, model="m").retrieve("公共数据开放的原则")
    assert out["mode"] == "not_found" and out["hits"] == [] and len(llm.calls) == 2 and len(llm.outs) == 1
    assert out["trace"]["rewritten"] == "数据分类分级 保护" and [r["query"] for r in out["trace"]["rounds"]] == ["公共数据开放的原则", "数据分类分级 保护"]


def test_rewrite_then_success(index):
    llm = PickLLM({"picks": [], "rewrite": "数据分类分级 保护"}, {"picks": [1], "rewrite": ""})
    out = PolicyRetriever(index, llm, model="m").retrieve("公共数据开放的原则")
    assert out["mode"] == "search" and len(llm.calls) == 2 and "公共数据开放的原则" in llm.calls[1]["messages"][1]["content"]


@pytest.mark.parametrize("bad", ["不是JSON", "[]", '{"picks": "1"}', '{"picks": [null, {}]}'])
def test_unusable_model_output_is_not_found_and_not_retried(index, bad):
    llm = PickLLM(bad)
    assert PolicyRetriever(index, llm, model="m").retrieve("公共数据开放的原则")["mode"] == "not_found" and len(llm.calls) == 1


def test_no_candidates_or_empty_index_means_not_found_without_a_call(index):
    llm = PickLLM()
    assert PolicyRetriever(index, llm, model="m").retrieve("ZZZ QQQ")["mode"] == "not_found" and llm.calls == []
    assert PolicyRetriever(PolicyIndex({"files": [], "chunks": []}), llm, model="m").retrieve("数据")["mode"] == "not_found"


def test_model_error_recorded_by_type_name_only(index):
    class Down:
        def chat(self, *a, **k):
            raise TimeoutError("key=SECRET")
    out = PolicyRetriever(index, Down(), model="m").retrieve("公共数据开放的原则")
    assert out["mode"] == "not_found" and out["trace"]["rounds"][0]["error"] == "TimeoutError" and "SECRET" not in json.dumps(out["trace"])


def test_load_skips_missing_docx_and_caches_by_mtime_and_size(tmp_path, monkeypatch):
    assert PolicyIndex.load(tmp_path / "没有.docx") is None
    doc = _sample(tmp_path)
    a, b = PolicyIndex.load(doc), PolicyIndex.load(doc)
    assert a.info["source"] == "parsed" and b.info["source"] == "cache" and len(a.chunks) == len(b.chunks)
    doc.write_bytes(doc.read_bytes() + b"\0")
    assert PolicyIndex.load(doc).info["source"] == "parsed"


# ---------------------------------------------------------------- 作答

def _hits(index, *ids):
    """ids：数据安全法的条号（如 21）；按条号取块，不依赖块序号。"""
    by = {c["article"]: c for c in index.chunks if c["doc_no"] == "1.2" and c["kind"] == "article"}
    return [pr.PolicyHit(k, by[i]) for k, i in enumerate(ids, 1)]


class Ans:
    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages, "model": model})
        return LLMResult(content=self.text)


def test_answer_numbers_citations_lists_sources_and_appends_the_closing_note(index):
    hits = _hits(index, 21)                                  # 第二十一条
    llm = Ans("《数据安全法》第二十一条规定了数据分类分级保护制度[P1]。\n你想进一步了解重要数据目录吗？")
    ans = answer_policy("数据安全法第二十一条规定了什么", hits, llm, model="pro")
    assert llm.calls[0]["step"] == "policy_answer" and llm.calls[0]["model"] == "pro" and "[P1] 第1篇 国家立法 › 中华人民共和国数据安全法 › 第二十一条" in llm.calls[0]["messages"][1]["content"]
    assert ans.text.startswith("《数据安全法》第二十一条规定了数据分类分级保护制度[1]。") and "[P1]" not in ans.text
    assert ans.text.endswith("\n\n" + pr.CLOSING_NOTE) and pr.CLOSING_NOTE == "以汇编收录版本为准，现行有效性请以官方发布为准。"
    assert [(s["n"], s["kind"], s["path"]) for s in ans.sources] == [(1, "policy", "第1篇 国家立法 › 中华人民共和国数据安全法 › 第二十一条")]


def test_verbatim_statute_quote_is_kept_and_paraphrased_quote_is_removed_but_marker_kept(index):
    hits = _hits(index, 21)
    ok = answer_policy("q", hits, Ans("条文规定“国家建立数据分类分级保护制度，对数据实行分类分级保护”[P1]。"), model="m")
    assert "“国家建立数据分类分级保护制度，对数据实行分类分级保护”[1]" in ok.text and ok.removed_quotes == []
    bad = answer_policy("q", hits, Ans("条文规定“国家要求所有企业必须对全部数据实行统一的三级分类保护”[P1]。"), model="m")
    assert "国家要求所有企业" not in bad.text and "[1]" in bad.text and bad.removed_quotes == ["国家要求所有企业必须对全部数据实行统一的三级分类保护"]


def test_draft_is_labelled_in_the_materials_the_answer_the_note_and_the_source(index):
    draft = next(c for c in index.chunks if c["draft"])
    hits = [pr.PolicyHit(1, draft)]
    llm = Ans("深圳的办法（征求意见稿）规定了安全可控原则[P1]。")
    ans = answer_policy("深圳公共数据开放", hits, llm, model="m")
    assert "征求意见稿，非正式生效文本" in llm.calls[0]["messages"][1]["content"]
    assert "《深圳市公共数据开放管理办法（征求意见稿）》为征求意见稿，非正式生效文本。" in ans.text and ans.text.endswith(pr.CLOSING_NOTE)
    assert ans.sources[0]["draft"] is True and ans.sources[0]["path"].endswith("（征求意见稿，非正式生效文本）")


def test_out_of_range_markers_are_dropped_and_only_cited_chunks_are_sources(index):
    hits = _hits(index, 21, 1)
    ans = answer_policy("q", hits, Ans("要点[P2][P9]。"), model="m")
    assert [s["chunk_id"] for s in ans.sources] == [hits[1].chunk["chunk_id"]] and hits[1].chunk["article"] == 1 and "[P9]" not in ans.text and "[2]" not in ans.text and "[1]" in ans.text


def test_answer_prompt_requires_total_part_total_closing_question_and_forbids_rewritten_quotes():
    from rag import prompt_rules as rules
    s = pr.ANSWER_SYSTEM
    assert rules.ANSWER_STRUCTURE in s and rules.COURSE_SAMPLES in s      # 结构与样例引用同一常量
    for w in ("总—分—总", "逐字摘录", "征求意见稿", "文件名和条号"):
        assert w in s
    # 结尾问句已移交 followups 管道
    assert "最后单独一行，写一个问句" not in s and "不加引用编号" not in s


def test_streaming_answer_matches_final_body_with_split_markers(index):
    hits = _hits(index, 21)

    class Stream(Ans):
        def chat_stream(self, step, messages, *, model, on_delta):
            for p in ["规定了分类分级[P", "1]。你想了解更多吗？"]:
                on_delta(p)
            return LLMResult(content="")
    got = []
    ans = answer_policy("q", hits, Stream(""), model="m", on_delta=got.append)
    assert "".join(got) == ans.body == "规定了分类分级[1]。你想了解更多吗？" and "[P" not in "".join(got)


# ---------------------------------------------------------------- 统一入口：单库与组合

from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: E402,F401
from rag.unified import UnifiedService  # noqa: E402


class Steps:
    """按 step 取脚本（线程安全）。"""

    def __init__(self, scripts):
        import threading
        self.scripts, self.calls, self.lock = {k: list(v) for k, v in scripts.items()}, [], threading.Lock()

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        with self.lock:
            self.calls.append({"step": step, "model": model, "messages": messages})
            item = self.scripts[step].pop(0)
        return item(step, messages) if callable(item) else item

    def steps(self):
        return [c["step"] for c in self.calls]


def route(routes):
    class R:
        calls = 0

        def chat(self, step, messages, *, model, **k):
            R.calls += 1
            return LLMResult(content=json.dumps({"routes": routes, "complexity": "simple", "count": 1, "chapter": None, "reason": "t"}))
    R.calls = 0
    return R()


def svc_for(llm, routes, index, **kw):
    return UnifiedService(router_client=route(routes), course_ctx=kw.pop("ctx", None), course_llm=llm, policy_index=index, **kw)


def test_single_policy_search_costs_route_plus_pick_plus_answer(index):
    llm = Steps({"policy_pick": [LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))], "policy_answer": [LLMResult(content="办法规定了开放原则[P1]。")],
                 "followup": [jtxt({"followups": ["还有哪些相关规定？", "这条规定怎么落地？"]})]})
    svc = svc_for(llm, ["policy"], index)
    out = svc.ask("上海有哪些公共数据开放的规定")
    core = [x for x in llm.steps() if x != "followup"]
    assert core == ["policy_pick", "policy_answer"] and svc.router_client.calls + len(core) == 3          # 普通问题 3 次
    assert out["route"] == "policy" and out["status"] == "ok" and out["answer"].endswith(pr.CLOSING_NOTE) and out["sources"][0]["kind"] == "policy"
    assert llm.calls[0]["model"] == svc.pick_model and llm.calls[1]["model"] == svc.answer_model and svc.pick_model != svc.answer_model


def test_named_article_costs_route_plus_answer_only(index):
    llm = Steps({"policy_answer": [LLMResult(content="第二十一条建立分类分级制度[P1]。")], "followup": [jtxt({"followups": ["还有哪些相关规定？", "这条规定怎么落地？"]})]})
    svc = svc_for(llm, ["policy"], index)
    out = svc.ask("数据安全法第二十一条规定了什么")
    core = [x for x in llm.steps() if x != "followup"]
    assert core == ["policy_answer"] and svc.router_client.calls + len(core) == 2 and out["trace"]["policy"]["mode"] == "named_article"


def test_rewrite_retry_costs_four_calls(index):
    llm = Steps({"policy_pick": [LLMResult(content=json.dumps({"picks": [], "rewrite": "数据分类分级 保护"})), LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))],
                 "policy_answer": [LLMResult(content="规定了[P1]。")], "followup": [jtxt({"followups": ["还有哪些相关规定？", "这条规定怎么落地？"]})]})
    svc = svc_for(llm, ["policy"], index)
    svc.ask("公共数据开放的原则")
    core = [x for x in llm.steps() if x != "followup"]
    assert svc.router_client.calls + len(core) == 4                      # 路由 1 + 选块 2 + 作答 1


def test_not_found_says_so_and_never_calls_the_answer_model(index):
    llm = Steps({"policy_pick": [LLMResult(content=json.dumps({"picks": [], "rewrite": ""}))]})
    out = svc_for(llm, ["policy"], index).ask("公共数据开放的原则")
    assert out["status"] == "insufficient" and out["answer"] == "政策汇编中没有找到相关规定。" and llm.steps() == ["policy_pick"] and out["sources"] == []


def test_policy_library_missing_is_unavailable_without_any_call():
    llm = Steps({})
    out = svc_for(llm, ["policy"], None).ask("数据二十条讲了什么")
    assert out["status"] == "unavailable" and llm.calls == []


def test_sse_single_policy_is_not_streamed_and_result_is_in_done(index):
    import json as _j
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from rag.unified import router as unified_router
    llm = Steps({"policy_answer": [LLMResult(content="第二十一条[P1]。")]})
    app = FastAPI()
    app.state.unified = svc_for(llm, ["policy"], index)
    app.include_router(unified_router)
    ev = [_j.loads(b[6:]) for b in TestClient(app).post("/auto/ask/stream", json={"question": "数据安全法第二十一条规定了什么"}).text.split("\n\n") if b.startswith("data: ")]
    assert [e["type"] for e in ev] == ["route", "done"] and ev[-1]["sources"][0]["kind"] == "policy" and ev[-1]["answer"].endswith(pr.CLOSING_NOTE)


@pytest.fixture
def cctx(ctx, monkeypatch):
    from rag.course import config as cc
    monkeypatch.setattr(cc, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(cc, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(cc, "COURSE_SELECT_FAST_PATH", False)
    monkeypatch.setattr(cc, "COURSE_FILTER_SKIP_MAX", 0)
    return ctx


def _filt(step, msgs):
    refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
    return jtxt({"direct": refs[:1], "background": refs[1:]})


def test_course_plus_policy_combined_has_a_policy_zone_and_partitioned_sources(cctx, index):
    llm = Steps({"select": [jtxt({"question_concepts": ["C01-0001"], "related_concepts": ["C01-0003"], "reason": "r"})], "filter": [_filt],
                 "policy_pick": [LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))],
                 "answer": [LLMResult(content="知识点[S1]。条文印证[P1]。")], "followup": [jtxt({"followups": ["还有哪些相关规定？", "这条规定怎么落地？"]})]})
    svc = svc_for(llm, ["course", "policy"], index, ctx=cctx)
    out = svc.ask("结合政策法规说明数据确权为什么难")
    core = [x for x in llm.steps() if x != "followup"]
    assert sorted(core) == ["answer", "filter", "policy_pick", "select"] and svc.router_client.calls + len(core) == 5
    assert [s["kind"] for s in out["sources"]] == ["course", "policy"] and "[1]" in out["answer"] and "[2]" in out["answer"] and "[P1]" not in out["answer"]
    user = next(c for c in llm.calls if c["step"] == "answer")["messages"][1]["content"]
    assert user.index("【教材材料】") < user.index("【政策条文】") and "[P1]" in user and out["answer"].endswith(pr.CLOSING_NOTE)
    assert out["trace"]["combined"]["sections"] == ["course", "policy"]


def test_combined_policy_quotes_are_verified_against_the_chosen_chunks(cctx, index):
    llm = Steps({"select": [jtxt({"question_concepts": ["C01-0001"], "related_concepts": [], "reason": "r"})], "filter": [_filt],
                 "policy_pick": [LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))],
                 "answer": [LLMResult(content="知识点[S1]。条文规定“国家要求所有企业必须对全部数据实行统一的三级分类保护”[P1]。")]})
    out = svc_for(llm, ["course", "policy"], index, ctx=cctx).ask("结合政策法规说明数据确权为什么难")
    assert "国家要求所有企业" not in out["answer"] and "[2]" in out["answer"]


def test_combined_policy_not_found_still_answers_from_the_other_library(cctx, index):
    llm = Steps({"select": [jtxt({"question_concepts": ["C01-0001"], "related_concepts": [], "reason": "r"})], "filter": [_filt],
                 "policy_pick": [LLMResult(content=json.dumps({"picks": [], "rewrite": ""}))], "answer": [LLMResult(content="只有教材[S1]。")]})
    out = svc_for(llm, ["course", "policy"], index, ctx=cctx).ask("结合政策法规说明数据确权为什么难")
    assert out["status"] == "ok" and [s["kind"] for s in out["sources"]] == ["course"] and out["trace"]["combined"]["missing"] == ["policy"]
    assert not out["answer"].endswith(pr.CLOSING_NOTE)                                              # 没引用政策就不附政策附注


def test_combined_with_unloaded_policy_library_is_unavailable(cctx):
    llm = Steps({})
    out = svc_for(llm, ["course", "policy"], None, ctx=cctx).ask("结合政策法规说明数据确权为什么难")
    assert out["status"] == "unavailable" and "政策库" in out["answer"] and llm.calls == []


def test_combined_policy_verbatim_quote_survives_but_only_because_the_policy_text_is_in_the_originals(cctx, index):
    """逐字摘录政策条文的引号要保留——引文校验池必须包含入选的政策块（否则正确的引文会被误删）。"""
    quote = "国家建立数据分类分级保护制度，对数据实行分类分级保护"
    llm = Steps({"select": [jtxt({"question_concepts": ["C01-0001"], "related_concepts": [], "reason": "r"})], "filter": [_filt],
                 "policy_pick": [LLMResult(content=json.dumps({"picks": [1], "rewrite": ""}))],
                 "answer": [LLMResult(content=f"知识点[S1]。条文规定“{quote}”[P1]。")]})
    out = svc_for(llm, ["course", "policy"], index, ctx=cctx).ask("数据安全法第二十一条与数据确权")
    hit = [s for s in out["sources"] if s["kind"] == "policy"]
    assert hit and f"“{quote}”" in out["answer"]


def test_article_number_alone_is_a_policy_signal_without_any_policy_keyword():
    from rag.router import _looks_policy
    assert _looks_policy("第十二条怎么规定") and _looks_policy("第 21 条的内容") and not _looks_policy("今天天气怎么样")
