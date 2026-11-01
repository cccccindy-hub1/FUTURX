"""single 模式（候选 + 一次选概念、一次筛选、直接作答）的离线测试：全部 mock，不调用真实模型。"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from rag.course import answer, config, passage_filter, pipeline, prompts, single_select
from rag.course.llm import LLMResult
from rag.course.recall import BM25_VIA, Reason, Recalled
from rag.unified import UnifiedService
from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: F401  (ctx 为课程夹具)


@pytest.fixture(autouse=True)
def single_mode(request, monkeypatch):
    # ctx 夹具会把模式固定为 agent（给旧测试用）；这里必须在 ctx 之后再设回 single
    if "ctx" in request.fixturenames:
        request.getfixturevalue("ctx")
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "single")
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    monkeypatch.setattr(config, "COURSE_SELECT_FAST_PATH", False)


def pick(q=("C01-0001",), r=("C01-0003",), reason="r") -> LLMResult:
    return jtxt({"question_concepts": list(q), "related_concepts": list(r), "reason": reason})


# ---------------------------------------------------------------- 候选生成

def test_candidates_deterministic_deduped_and_alias_hits_first(ctx):
    a = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    b = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    assert [c.concept_id for c in a] == [c.concept_id for c in b] and a                  # 确定性
    ids = [c.concept_id for c in a]
    assert len(ids) == len(set(ids))                                                      # 去重
    assert a[0].concept_id == "C01-0001" and a[0].via.startswith("alias")                # 别名命中排最前
    rank = {"alias_exact": 0, "alias_in_query": 1, "query_in_alias": 2, "neighbor": 3, "bm25": 4}
    assert [rank[c.via] for c in a] == sorted(rank[c.via] for c in a)                    # 别名 > 一跳邻居 > BM25


def test_candidates_limit_and_definition_truncated_and_fields(ctx):
    assert len(single_select.build_candidates("数据要素", ctx.index, 1)) == 1
    c = single_select.build_candidates("数据要素", ctx.index, 30)[0]
    assert c.concept_id in c.line() and c.name in c.line() and f"第{c.chapter}章" in c.line()
    assert len(c.definition) <= single_select.DEFINITION_CHARS


def test_candidates_include_one_hop_neighbors_of_literal_hits(ctx):
    cands = single_select.build_candidates("数据要素", ctx.index, 30)
    nbrs = {n.concept_id for n in ctx.index.get_neighbors("C01-0001")} & ctx.index.selectable
    assert nbrs and any(c.via == "neighbor" and c.concept_id in nbrs for c in cands) or all(
        c.concept_id in {x.concept_id for x in cands} for c in cands if c.concept_id in nbrs)


# ---------------------------------------------------------------- 单轮选概念

def test_single_select_normal_json_one_call(ctx):
    llm_ = ScriptedLLM([pick(("C01-0001",), ("C01-0003",))])
    r = single_select.select_single("什么是数据要素", ctx.index, llm_)
    assert [(s.concept_id, s.role) for s in r.selected] == [("C01-0001", "question_concept"), ("C01-0003", "related_concept")]
    assert len(llm_.calls) == 1 and llm_.calls[0]["json_mode"] and r.fallback == "" and r.end_reason == "single"
    assert "C01-0001" in llm_.calls[0]["messages"][1]["content"] and r.candidates and r.reason == "r"
    assert len(r.rounds) == 1


def test_single_select_drops_out_of_candidate_ids_and_records_them(ctx):
    llm_ = ScriptedLLM([pick(("C01-0001", "C99-9999"), ("C88-0001",))])
    r = single_select.select_single("什么是数据要素", ctx.index, llm_)
    assert [s.concept_id for s in r.selected] == ["C01-0001"]
    assert {(d["concept_id"], d["reason"]) for d in r.dropped} == {("C99-9999", "not_in_candidates"), ("C88-0001", "not_in_candidates")}


def test_single_select_limits_counts(ctx):
    many = [c.concept_id for c in single_select.build_candidates("数据要素", ctx.index, 30)]
    llm_ = ScriptedLLM([pick(many[:5], many[5:15])])
    r = single_select.select_single("数据要素", ctx.index, llm_)
    assert sum(s.role == "question_concept" for s in r.selected) <= 3 and sum(s.role == "related_concept" for s in r.selected) <= 5
    assert any(d["reason"] == "over_limit" for d in r.dropped) or len(many) < 6


@pytest.mark.parametrize("bad", [LLMResult(content="这不是 JSON"), pick((), ()), pick(("C99-9999",), ())])
def test_single_select_fallback_uses_top3_without_retry(ctx, bad):
    llm_ = ScriptedLLM([bad, bad, bad])
    r = single_select.select_single("什么是数据要素", ctx.index, llm_)
    top3 = [c["concept_id"] for c in r.candidates[:3]]
    assert len(llm_.calls) == 1                                                           # 不重试
    assert [s.concept_id for s in r.selected] == top3 and all(s.role == "question_concept" for s in r.selected)
    assert r.fallback in ("parse_failed", "empty_selection") and r.end_reason == "single_fallback"


def test_single_select_no_candidates_makes_no_llm_call(ctx):
    llm_ = ScriptedLLM([])
    r = single_select.select_single("zzzzqqq", ctx.index, llm_)
    assert llm_.calls == [] and r.selected == [] and r.fallback == "no_candidates"


def test_select_mode_agent_still_dispatches_to_old_loop(ctx, monkeypatch):
    from rag.course import agent_select
    from test_course_pipeline import finish_native
    monkeypatch.setattr(config, "COURSE_SELECT_MODE", "agent")
    llm_ = ScriptedLLM([finish_native(("C01-0001", "question_concept"))])
    r = agent_select.select_concepts("q", [], ctx.index, llm_, native_tools=True, fast_path=False)
    assert [s.concept_id for s in r.selected] == ["C01-0001"] and llm_.calls[0]["tools"]    # 带 tools 说明走的是旧 agent


# ---------------------------------------------------------------- 筛选：一次调用与截断

def _item(cid, tier_reason, score=None):
    r = Reason(concept_id="", via=BM25_VIA, anchor="", score=score) if tier_reason == "bm25" else Reason(
        concept_id="C", via="edge_evidence", anchor="P1", evidence="verified", source_tier="expert")
    return Recalled(cid, "p", ["章"], f"文本{cid}", "body", ["P1"], [r])


def test_cap_for_filter_noop_under_limit():
    items = [_item("a", "edge"), _item("b", "bm25", 1.0)]
    kept, st = passage_filter.cap_for_filter(items, 5)
    assert kept == items and st["dropped"] == 0 and st["kept"] == 2


def test_cap_for_filter_keeps_ontology_evidence_first_then_top_bm25():
    onto = [_item(f"o{i}", "edge") for i in range(3)]
    bm = [_item("b_low", "bm25", 1.0), _item("b_high", "bm25", 9.0), _item("b_mid", "bm25", 5.0)]
    items = [bm[0], onto[0], bm[1], onto[1], bm[2], onto[2]]
    kept, st = passage_filter.cap_for_filter(items, 5)                                    # 3 个本体 + 2 个 BM25 名额
    assert {i.chunk_id for i in kept} == {"o0", "o1", "o2", "b_high", "b_mid"}
    assert [i.chunk_id for i in kept] == [i.chunk_id for i in items if i.chunk_id in {"o0", "o1", "o2", "b_high", "b_mid"}]   # 保持原顺序
    assert st["dropped"] == 1 and st["dropped_ids"] == ["b_low"] and st["dropped_by_source"] == {"bm25": 1}
    kept2, st2 = passage_filter.cap_for_filter(items, 2)                                  # 本体证据块本身超上限：只留前 2 个
    assert [i.chunk_id for i in kept2] == ["o0", "o1"] and st2["dropped"] == 4


def test_filter_called_once_even_when_recall_is_large(ctx, monkeypatch):
    monkeypatch.setattr(config, "COURSE_RECALL_MAX_CHARS", 1)                             # 旧逻辑下每块一批
    monkeypatch.setattr(config, "COURSE_BM25_TOPK", 0)
    llm_ = ScriptedLLM([pick(), lambda s, m: jtxt({"verdicts": []})])
    pipeline.ask("什么是数据要素", ctx=ctx, llm=llm_, max_chars=1)
    assert [c["step"] for c in llm_.calls].count("filter") == 1


# ---------------------------------------------------------------- 整条管线：固定 4 次（含路由）

def _full_script(ctx):
    def verdict_all(step, msgs):
        user = msgs[1]["content"]
        import re
        refs = re.findall(r"^\[(S\d+)\]", user, flags=re.M)
        return jtxt({"verdicts": [{"id": r, "grade": "direct" if i == 0 else "background", "reason": "r"} for i, r in enumerate(refs)]})
    return [pick(), verdict_all, LLMResult(content="数据要素是参与价值创造的数据资源[S1]；其依赖数据权属[本体]。",
                                            prompt_tokens=100, completion_tokens=20)]


def test_single_mode_course_path_is_exactly_four_llm_calls_including_router(ctx):
    course_llm = ScriptedLLM(_full_script(ctx))
    router = type("R", (), {"calls": 0, "chat": lambda self, *a, **k: (setattr(self, "calls", self.calls + 1),
                                                                         LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"})))[1]})()
    out = UnifiedService(router_client=router, course_ctx=ctx, course_llm=course_llm).ask("什么是数据要素")
    assert out["route"] == "course" and out["status"] == "ok" and out["answer"]
    steps = [c["step"] for c in course_llm.calls]
    assert steps == ["select", "filter", "answer"] and router.calls == 1
    assert router.calls + len(course_llm.calls) == 4                                      # 路由 1 + 选概念 1 + 筛选 1 + 作答 1
    tr = out["trace"]["course"]
    assert tr["totals"]["calls"] == 3 and set(tr["totals"]["by_step"]) == {"select", "filter", "answer"}
    assert all(v["calls"] == 1 for v in tr["totals"]["by_step"].values())               # 每环节恰好一次
    assert tr["stages"]["select"]["fallback"] == "" and tr["stages"]["filter"]["cap"]["dropped"] == 0
    assert out["trace"]["router"]["source"] == "llm"


def test_single_mode_select_fallback_still_four_calls(ctx):
    script = _full_script(ctx)
    script[0] = LLMResult(content="乱码")                                                 # 选概念解析失败：回退候选前 3，不重试
    course_llm = ScriptedLLM(script)
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=course_llm)
    assert [c["step"] for c in course_llm.calls] == ["select", "filter", "answer"]
    assert r["trace"].stages["select"]["fallback"] == "parse_failed"


# ---------------------------------------------------------------- 作答提示词与免责短语

@pytest.mark.parametrize("phrase", ["非教材原文", "不能当作教材原文", "教材材料不足以回答", "本体关系材料中未提供"])
def test_answer_system_prompt_has_no_defensive_instructions(phrase):
    assert phrase not in prompts.ANSWER_SYSTEM
    assert "套话式免责" in prompts.ANSWER_SYSTEM and "[本体]" in prompts.ANSWER_SYSTEM and "[S2]" in prompts.ANSWER_SYSTEM
    assert "结尾不要附加说明引用方式的话" in prompts.ANSWER_SYSTEM


def test_answer_prompt_distinguishes_boilerplate_from_specific_reason_notes():
    """禁止的是没有具体原因的套话；教材只给相近/上位概念、或只是学界观点时，要一句话说明并给出最接近的内容。"""
    sysm = prompts.ANSWER_SYSTEM
    assert "没有具体原因的套话式免责" in sysm and "有具体原因时" in sysm
    assert "相近或上位概念" in sysm and "数据市场" in sysm and "数据要素市场" in sysm       # 例：只讲“数据市场”没讲“数据要素市场”
    assert "学界观点" in sysm and "不要当成教材自己的结论" in sysm
    assert "最接近的内容" in sysm and "不要因此拒答" in sysm
    assert "优先采用教材正面给出的界定或构成" in sysm                                        # r14：学界看法不应压过教材正面给出的构成


def test_disclaimer_phrases_produce_warnings_without_editing_text(ctx):
    from test_course_pipeline import _items
    kept = _items(ctx, "P0005")[:1]
    text = "数据要素是…[S1]。这部分不是教材原文，只是本体中记录的关系；教材并没有列出更多内容。"
    ar = answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content=text)]))
    assert {"这部分不是教材原文", "不是教材原文", "只是本体中记录", "教材并没有列出"} <= set(ar.warnings)
    assert "这部分不是教材原文" in ar.text                                                 # 不自动删改正文
    assert answer.answer_question("q", kept, [], ScriptedLLM([LLMResult(content="直接作答[S1]")])).warnings == []
    assert ar.to_dict()["warnings"] == ar.warnings


def test_pipeline_trace_carries_answer_warnings(ctx):
    script = _full_script(ctx)
    script[2] = LLMResult(content="数据要素是…[S1]。以上不能作为教材依据，材料中未提供更多。")
    r = pipeline.ask("什么是数据要素", ctx=ctx, llm=ScriptedLLM(script))
    assert r["trace"].stages["answer"]["warnings"] and r["answer"]["warnings"]


def test_defaults():
    import os
    if "COURSE_SELECT_MODE" not in os.environ:
        assert config.COURSE_SELECT_MODE == "single"
    if "COURSE_SELECT_CANDIDATES" not in os.environ:
        assert config.COURSE_SELECT_CANDIDATES == 30
    if "COURSE_FILTER_MAX_CHUNKS" not in os.environ:
        assert config.COURSE_FILTER_MAX_CHUNKS == 40


# ---------------------------------------------------------------- 同名概念合并候选

@pytest.fixture
def ctx_same_name(ctx, monkeypatch):
    """夹具里没有同名跨章概念：把 C01-0003 并入「数据要素」的同名组（只改索引，不改数据文件）。"""
    key = ctx.index.name_key("C01-0001")
    monkeypatch.setitem(ctx.index._name_groups, key, ["C01-0001", "C01-0003"])
    return ctx


def test_same_name_concepts_merge_into_one_candidate_with_chapter_list(ctx_same_name):
    ctx = ctx_same_name
    cands = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    merged = [c for c in cands if c.concept_id == "C01-0001"]                # 夹具里另有同名的 C01-0002/C02-0002，也会被合并
    assert len(merged) == 1 and len(merged[0].members) == 2
    assert {m for m, _ in merged[0].members} == {"C01-0001", "C01-0003"}
    ids = [c.concept_id for c in cands]
    assert "C01-0003" not in ids and len(ids) == len(set(ids))             # 同名成员不再单独占名额
    assert "第" in merged[0].line() and "章" in merged[0].line() and "—" in merged[0].line()
    assert merged[0].to_dict()["members"] and merged[0].expand() == ["C01-0001", "C01-0003"]


def test_no_concept_appears_in_two_candidates(ctx_same_name):
    cands = single_select.build_candidates("什么是数据要素", ctx_same_name.index, 30)
    all_ids = [m for c in cands for m, _ in c.members]
    assert len(all_ids) == len(set(all_ids))


def test_merged_candidate_saves_slots_for_other_concepts(ctx_same_name):
    ctx = ctx_same_name
    n_merged = len(single_select.build_candidates("什么是数据要素", ctx.index, 30))
    monkey_total = len({c.concept_id for c in single_select.build_candidates("什么是数据要素", ctx.index, 30)})
    assert n_merged == monkey_total                                          # 每条候选对应一个不同名称


def test_selecting_merged_candidate_expands_to_chapters_up_to_cap(ctx_same_name):
    ctx = ctx_same_name
    llm_ = ScriptedLLM([pick(("C01-0001",), ())])
    r = single_select.select_single("什么是数据要素", ctx.index, llm_)
    assert [(s.concept_id, s.role) for s in r.selected] == [("C01-0001", "question_concept"), ("C01-0003", "question_concept")]
    assert r.expanded == [{"candidate": "C01-0001", "name": r.expanded[0]["name"], "to": ["C01-0001", "C01-0003"], "dropped": []}]
    assert "（同名章展开）" in r.selected[1].reason


def test_selecting_a_member_id_directly_also_expands_without_duplicates(ctx_same_name):
    ctx = ctx_same_name
    r = single_select.select_single("什么是数据要素", ctx.index, ScriptedLLM([pick(("C01-0003",), ("C01-0001",))]))
    ids = [s.concept_id for s in r.selected]
    assert ids[0] == "C01-0003" and set(ids) == {"C01-0003", "C01-0001"} and len(ids) == len(set(ids))   # 模型写的那一章必留且排第一，不重复
    assert all(s.role == "question_concept" for s in r.selected)               # 同一概念不重复，question 优先于 related
    assert r.dropped == []


def test_unmerged_candidates_unchanged_and_prompt_explains_merged_format(ctx):
    cands = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    single = [c for c in cands if len(c.members) == 1]
    assert single and all("—" not in c.line() and f"第{c.chapter}章" in c.line() for c in single)
    assert "同名概念" in prompts.SINGLE_SELECT_SYSTEM and "无需再逐章选" in prompts.SINGLE_SELECT_SYSTEM


def test_merged_candidates_deterministic(ctx_same_name):
    a = single_select.build_candidates("什么是数据要素", ctx_same_name.index, 30)
    b = single_select.build_candidates("什么是数据要素", ctx_same_name.index, 30)
    assert [(c.concept_id, c.members) for c in a] == [(c.concept_id, c.members) for c in b]


def test_candidate_eval_counts_merged_members_as_covered(ctx_same_name):
    from rag.course import candidate_eval
    gold = [{"id": "g", "question": "什么是数据要素", "must_recall": [{"key": "数据要素"}]}]
    res = candidate_eval.evaluate(ctx_same_name, gold, 30)
    assert res["rows"][0]["n_concepts_covered"] >= res["rows"][0]["n_candidates"]


# ---------------------------------------------------------------- 端到端评测脚本（离线）

def test_unified_course_output_carries_recalled_with_source_and_grade(ctx):
    router = type("R", (), {"chat": lambda self, *a, **k: LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"}))})()
    out = UnifiedService(router_client=router, course_ctx=ctx, course_llm=ScriptedLLM(_full_script(ctx))).ask("什么是数据要素")
    rec = out["course"]["recalled"]
    assert rec and {"chunk_id", "source", "grade", "path"} <= set(rec[0]) and rec[0]["source"] in ("ontology", "bm25", "both")
    assert any(r["grade"] == "direct" for r in rec)


def test_e2e_run_one_records_stage_timings_and_never_leaks_key(ctx, monkeypatch):
    from rag.course import eval_e2e
    router = type("R", (), {"chat": lambda self, *a, **k: LLMResult(content=json.dumps({"route": "course", "complexity": "simple", "reason": "t"}), latency_s=0.5)})()
    svc = UnifiedService(router_client=router, course_ctx=ctx, course_llm=ScriptedLLM(_full_script(ctx)))
    row = eval_e2e.run_one(svc, {"id": "r11", "question": "什么是数据要素"}, {"calls": 0, "cap": 99})
    assert row["error"] == "" and row["route"] == "course" and row["e2e_s"] >= row["total_s"] - 0.01
    assert set(row["step_calls"]) == {"select", "filter", "answer"} and all(v == 1 for v in row["step_calls"].values())
    assert row["router_s"] >= 0 and set(row["step_s"]) == {"select", "filter", "answer"} and row["course_calls"] == 3
    assert "warnings" in row and row["recalled"]


def test_e2e_stops_after_two_llm_errors_and_checks_window(monkeypatch, tmp_path):
    import sys
    from rag.course import eval_e2e
    monkeypatch.setattr(eval_e2e, "in_window", lambda now: True)
    seq = iter(["", "LLMError", "LLMError"] + [""] * 20)
    monkeypatch.setattr(eval_e2e, "run_one", lambda svc, g, sh: {"id": g["id"], "error": next(seq)})
    mk = lambda: type("C", (), {"_client": type("X", (), {"with_options": lambda s, **k: s})()})()
    monkeypatch.setattr(eval_e2e, "OpenAICompatClient", mk)
    monkeypatch.setattr(eval_e2e, "build_router_client", mk)
    monkeypatch.setattr(eval_e2e, "CourseContext", type("CC", (), {"load": staticmethod(lambda: type("K", (), {"chunkset": type("S", (), {"chunks": []})()})())}))
    monkeypatch.setattr(eval_e2e, "key_chunks", lambda ctx, key: [])
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    eval_e2e.main()
    d = json.load(open(out, encoding="utf-8"))
    assert [r["error"] for r in d["rows"]] == ["", "LLMError", "LLMError"] and "LLMError x2" in d["stopped"]
    assert d["started_bj"] <= d["finished_bj"]


# ---------------------------------------------------------------- 截断按相关性 / 展开限章 / 候选防挤占

def _cand_item(cid, text, tier_kind="edge", bm=None, path=("章",)):
    r = Reason(concept_id="", via=BM25_VIA, anchor="", score=bm) if tier_kind == "bm25" else Reason(
        concept_id="C", via="edge_evidence", anchor="P1", evidence="verified", source_tier="expert")
    return Recalled(cid, "p", list(path), text, "body", ["P1"], [r])


def test_cap_ranks_by_relevance_not_recall_order_even_within_ontology_blocks():
    q = "数据资产入表的基本思路"
    filler = [_cand_item(f"o{i}", f"完全无关的内容第{i}段关于别的话题") for i in range(6)]       # 召回顺序在前，但不相关
    good = _cand_item("good", "数据资产入表是将符合条件的数据资产纳入财务报表的基本思路", path=("入表",))
    items = filler + [good]                                                                         # 相关块排在最后：旧逻辑（前 N 个）会把它丢掉
    kept, st = passage_filter.cap_for_filter(items, 3, question=q, concept_names=["数据资产入表"])
    assert "good" in {i.chunk_id for i in kept} and st["ranking"] == "relevance" and st["dropped"] == 4
    assert [i.chunk_id for i in kept] == [i.chunk_id for i in items if i.chunk_id in {k.chunk_id for k in kept}]   # kept 保持召回顺序
    assert st["dropped_max_score"] <= st["kept_min_score"] + 1e-9                                  # 被丢块的最高分不高于保留块的最低分


def test_cap_prefers_ontology_evidence_over_equally_relevant_bm25_only():
    q = "数据资产入表"
    onto = _cand_item("onto", "数据资产入表的内容", "edge")
    bm = _cand_item("bm", "数据资产入表的内容", "bm25", bm=5.0)
    kept, _ = passage_filter.cap_for_filter([bm, onto], 1, question=q)
    assert [i.chunk_id for i in kept] == ["onto"]


def test_cap_is_deterministic():
    items = [_cand_item(f"x{i}", f"文本{i % 3}数据要素") for i in range(10)]
    a = passage_filter.cap_for_filter(items, 4, question="数据要素")[0]
    b = passage_filter.cap_for_filter(items, 4, question="数据要素")[0]
    assert [i.chunk_id for i in a] == [i.chunk_id for i in b]


def test_expansion_keeps_selected_chapter_and_at_most_two_chapters(ctx, monkeypatch):
    key = ctx.index.name_key("C01-0001")
    monkeypatch.setitem(ctx.index._name_groups, key, ["C01-0001", "C01-0003", "C01-0005", "C01-0006"])
    monkeypatch.setattr(ctx.index, "selectable", ctx.index.selectable | {"C01-0003", "C01-0005", "C01-0006"})
    cands = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    cand = next(c for c in cands if c.concept_id == "C01-0001")
    assert len(cand.members) >= 3
    r = single_select.select_single("什么是数据要素", ctx.index, ScriptedLLM([pick(("C01-0001",), ())]))
    picked = [s.concept_id for s in r.selected if s.concept_id in {m for m, _ in cand.members}]
    assert picked[0] == "C01-0001" and len(picked) == single_select.MAX_CHAPTERS_PER_CANDIDATE
    assert r.expanded[0]["dropped"] and set(r.expanded[0]["to"]) | set(r.expanded[0]["dropped"]) == {m for m, _ in cand.members}


def test_expansion_prefers_chapter_whose_definition_matches_question(ctx, monkeypatch):
    members = ["C01-0001", "C01-0002", "C02-0001"]
    monkeypatch.setattr(ctx.index, "definition_scores", lambda q, ids: {"C01-0001": 0.0, "C01-0002": 1.0, "C02-0001": 9.0})
    out = single_select._expand(["C01-0001"], [], {"C01-0001": single_select.Candidate(
        "C01-0001", "名", 1, "", "alias_in_query", 1.0, tuple((m, int(m[1:3])) for m in members))},
        {m: "C01-0001" for m in members}, type("R", (), {"expanded": []})(), "q", ctx.index)
    assert [s.concept_id for s in out] == ["C01-0001", "C02-0001"]               # 选中的一章必留，另取定义分最高的一章


def test_candidates_neighbors_do_not_crowd_out_bm25_hits(ctx):
    cands = single_select.build_candidates("什么是数据要素", ctx.index, 30)
    nb = [c for c in cands if c.via == "neighbor"]
    assert len(nb) <= max(30 // 3, 1) and len(nb) <= len(cands)
    full = single_select.build_candidates("什么是数据要素", ctx.index, 3)
    assert len(full) == 3 and len([c for c in full if c.via == "neighbor"]) <= 1


def test_select_prompt_tells_model_not_to_waste_slots_on_similar_names():
    assert "名称相近" in prompts.SINGLE_SELECT_SYSTEM and "把名额留给" in prompts.SINGLE_SELECT_SYSTEM


def test_replay_eval_runs_offline_and_reports_counts(ctx):
    from rag.course import replay_eval
    row = {"id": "g", "question": "什么是数据要素", "error": "",
           "selected_concepts": [{"concept_id": "C01-0001", "role": "question_concept"}],
           "select_stage": {"expanded": []}, "recalled": [{"chunk_id": "x", "source": "ontology", "grade": "direct"}],
           "filter_cap": {"kept": 1, "dropped": 0}}
    gold = {"must_recall": [{"key": "数据要素"}]}
    out = replay_eval.replay_question(ctx, row, gold)
    assert out["new"]["concepts"] >= 1 and out["new"]["recalled"] >= 1 and out["new"]["kept"] <= config.COURSE_FILTER_MAX_CHUNKS
    assert out["gold"][0]["recalled"] and "in_top_kept" in out["gold"][0] and out["missing_old_picks"] == []
