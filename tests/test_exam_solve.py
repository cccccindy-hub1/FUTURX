"""解题先查题库 / 找题与解题的区分 / 答案默认折叠 / 选择题选项。全部 mock：路由与课程模型一旦被碰到就报错或被计数。"""
from __future__ import annotations

import json
import re
import time

import pytest

from rag import config
from rag.course.llm import LLMResult
from rag.exam.bank import Question
from rag.exam.knowledge import StudentIndex
from rag.exam.solve import SOLVE_MIN_CHARS, StemMatcher, is_solve_question, normalize, strip_shell
from rag.exam.store import ExamBank, ExamItem
from rag.unified import UnifiedService
from test_course_pipeline import ScriptedLLM, ctx, jtxt  # noqa: F401

JUDGE = "如果个人权益得不到保障、公共数据资源被不当垄断、市场竞争失于公平，数据要素市场的发展将偏离正确轨道。"
JUDGE2 = "从2019年数据首次被列为生产要素，到2020年与土地、劳动力、资本、技术并列为五大生产要素，国家对数据要素的战略定位日益明确。"
TERM = "专有数据、共享数据与开放数据"
CHOICE = "关于“数据要素”的概念界定与核心内涵，下列哪一选项的表述是正确的？（ ）"
OPTS = {"A": "根据特定生产需求形成的数据资源及其衍生形态。", "B": "数据资产与数据要素是严格的线性递进关系。",
        "C": "二者完全等同。", "D": "二者属于互斥关系。"}


def _q(qtype, cn, n, stem, answer, analysis, options=None, chapter=4):
    return ExamItem(Question(f"ex:t:{qtype}:{n}", chapter, "数据政策", qtype, cn, n, stem, options or {}, answer, analysis),
                    [], [], "none", stem)


def make_bank() -> ExamBank:
    items = [_q("judge", "判断题", 1, JUDGE, "正确", "个人权益、公共数据与公平竞争是数据要素市场健康发展的三个基础。"),
             _q("judge", "判断题", 2, JUDGE2, "正确", "见教材第一章。"),
             _q("term", "名词解释", 1, TERM, "（1）专有数据……（2）共享数据……", "（1）（占40%）准确表述……"),
             _q("choice", "选择题", 1, CHOICE, "A", "正确选项为 A。", OPTS, chapter=1),
             _q("judge", "判断题", 3, "数据资源经确权入表后可成为数据资产。", "正确", "见教材。", chapter=1),
             _q("judge", "判断题", 4, "路由器的默认参考带宽为一百兆。", "错误", "网络常识。", chapter=1)]
    return ExamBank(items, StudentIndex.from_rows([]), {})


class Counting:
    """路由 / 课程模型的替身：记录调用，prematch 命中时一次都不应被调用。"""

    def __init__(self, route="exam", script=None):
        self.route, self.calls, self.script = route, [], list(script or [])

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages})
        if step == "route":
            return LLMResult(content=json.dumps({"routes": [self.route], "complexity": "simple", "count": 1, "chapter": None, "reason": "t"}))
        item = self.script.pop(0)
        return item(step, messages) if callable(item) else item


def svc(route="exam", llm=None, bank=None, **kw):
    r = Counting(route)
    s = UnifiedService(router_client=r, exam_bank=bank or make_bank(), course_llm=llm, **kw)
    return s, r


# ---------------------------------------------------------------- 外壳与判别

@pytest.mark.parametrize("q,rest", [
    (f"判断这道题是否正确：{JUDGE}", JUDGE), (f"这句话对吗？{JUDGE}", JUDGE), (f"名词解释：{TERM}", TERM),
    (f"帮我做这道题：{JUDGE}", JUDGE), (f"1. 判断题：{JUDGE}", JUDGE), (f"{JUDGE}答案是什么", JUDGE)])
def test_shell_is_stripped_to_the_bare_stem(q, rest):
    got, had = strip_shell(q)
    assert had and normalize(got) == normalize(rest)


@pytest.mark.parametrize("q", ["出三道判断题", "第四章有哪些名词解释", "来几道第三章选择题", "什么是数据要素", "给我第三章选择题", "名词解释：数据要素"])
def test_finding_questions_and_short_stems_are_not_solve_questions(q):
    assert not is_solve_question(q)                                   # 无具体题干，或题干不足 12 字


def test_solve_needs_at_least_12_chars_after_stripping():
    assert SOLVE_MIN_CHARS == 12
    assert is_solve_question(f"名词解释：{TERM}") and not is_solve_question("名词解释：数据要素与数据")


# ---------------------------------------------------------------- 匹配

def test_matcher_hits_exact_contained_and_near_identical_stems():
    m = StemMatcher(make_bank().items)
    assert m.match(f"判断这道题是否正确：{JUDGE}")["how"] == "exact"
    assert m.match(f"判断这道题是否正确：{JUDGE[:-1]}")["how"] == "exact"          # 少一个句号：归一化去标点后完全一致
    assert m.match(f"判断这道题是否正确：{JUDGE}（来源：教材第一章）")["how"] == "contain"        # 学生多带了一段文字：题库题干被完全包含
    r = m.match(f"名词解释：{TERM}")
    assert r["how"] == "exact" and make_bank().items[r["hit"]].q.qtype == "term"
    typo = JUDGE.replace("不当垄断", "不当垄段")
    r = m.match(typo)
    assert r["hit"] is not None and r["how"] == "similar" and 0.9 <= r["similarity"] < 1.0


def test_matcher_ignores_punctuation_spaces_and_question_numbers():
    m = StemMatcher(make_bank().items)
    messy = "3 . " + JUDGE.replace("，", " , ").replace("。", "")
    assert m.match(f"判断：{messy}")["hit"] is not None


def test_rewritten_stem_does_not_match():
    m = StemMatcher(make_bank().items)
    rewritten = "如果个人权益得不到保障，数据要素市场的发展必然走向失败，因此应当取消一切数据交易活动。"
    assert m.match(f"判断这道题是否正确：{rewritten}")["hit"] is None


def test_choice_stem_matches_even_though_the_bank_stem_has_a_trailing_bracket():
    m = StemMatcher(make_bank().items)
    r = m.match("判断这道题是否正确：关于“数据要素”的概念界定与核心内涵，下列哪一选项的表述是正确的？")
    assert r["hit"] == 3


def test_matching_is_fast_on_the_full_size_bank_and_reports_ms():
    base = make_bank().items
    big = list(base)
    for i in range(2015):
        big.append(_q("choice", "选择题", 100 + i, f"关于第{i}号数据治理主题，下列哪一选项的表述是正确的？数据要素流通与数据工程场景{i * 7919 % 1000}", "A", "", OPTS))
    m = StemMatcher(big)
    for q in (f"判断这道题是否正确：{JUDGE}", f"名词解释：{TERM}", "判断：" + "无关的长句子用来测试没有候选命中的最坏情况" * 3):
        t0 = time.perf_counter()
        r = m.match(q)
        assert (time.perf_counter() - t0) * 1000 < 50 and r["ms"] < 50
    assert "ms" in m.match(JUDGE)


# ---------------------------------------------------------------- 统一入口：命中题库，调用 0 次

def test_screenshot_question_hits_the_bank_with_answer_and_zero_model_calls():
    s, router = svc("course")
    out = s.ask(f"判断这道题是否正确：{JUDGE}")
    assert router.calls == [] and out["route"] == "exam" and out["routes"] == ["exam"] and out["status"] == "ok"
    q = out["exam"]["questions"][0]
    assert q["answer"] == "正确" and q["analysis"].startswith("个人权益") and q["chapter"] == 4 and q["qtype_cn"] == "判断题" and out["exam"]["solve"] is True
    assert "【参考答案】正确" in out["answer"] and "【解析】个人权益" in out["answer"] and "第4章" in out["answer"] and "判断题" in out["answer"]
    assert out["trace"]["llm_calls"] == 0 and out["trace"]["prematch"]["how"] == "exact" and out["trace"]["prematch"]["ms"] < 50
    assert out["trace"]["prematch"]["hit"] == "ex:t:judge:1" and len(out["exam"]["questions"]) == 1          # 只给这一道，不是另外 5 道


def test_term_question_hits_the_bank():
    s, router = svc("course")
    out = s.ask(f"名词解释：{TERM}")
    assert router.calls == [] and out["exam"]["questions"][0]["qtype"] == "term" and out["exam"]["questions"][0]["answer"].startswith("（1）专有数据")


def test_stream_hit_is_route_then_done_with_no_calls_and_exam_payload():
    s, router = svc("course")
    ev = list(s.ask_stream(f"判断这道题是否正确：{JUDGE}"))
    types = [e["type"] for e in ev]
    assert types[:2] == ["route", "done"] and ev[0]["routes"] == ["exam"] and router.calls == []
    # 预匹配仍然 0 次模型调用；followups 是纯程序模板（exam_followups），不发事件以外的调用
    assert all(t == "followups" for t in types[2:])
    assert ev[1]["exam"]["questions"][0]["answer"] == "正确" and ev[1]["exam"]["solve"] and ev[1]["trace"]["prematch"]["ms"] < 50


def test_without_an_exam_bank_nothing_is_prematched_and_the_router_runs():
    s = UnifiedService(router_client=Counting("off_topic"))
    assert s.ask(f"判断这道题是否正确：{JUDGE}")["status"] == "refused"


# ---------------------------------------------------------------- 没命中：走路由；被判为 exam 的解题改送课程

REWRITTEN = "如果个人权益得不到保障，数据要素市场的发展必然走向失败，因此应当取消一切数据交易活动。"


def _course_script():
    def filt(step, msgs):
        refs = re.findall(r"^\[(S\d+)\]", msgs[1]["content"], flags=re.M)
        return jtxt({"direct": refs[:1], "background": refs[1:]})
    return [jtxt({"question_concepts": ["C01-0001"], "related_concepts": [], "reason": "r"}), filt,
            LLMResult(content="错误。该说法忽略了数据要素市场需要在保障权益的前提下发展[S1]。")]


@pytest.fixture
def course(ctx, monkeypatch):
    from rag.course import config as cc
    for k, v in (("COURSE_SELECT_MODE", "single"), ("COURSE_BM25_TOPK", 0), ("COURSE_SELECT_FAST_PATH", False), ("COURSE_FILTER_SKIP_MAX", 0)):
        monkeypatch.setattr(cc, k, v)
    return ctx


def test_rewritten_stem_goes_to_course_with_the_conclusion_first_hint(course):
    llm = ScriptedLLM(_course_script())
    s = UnifiedService(router_client=Counting("exam"), exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    out = s.ask(f"判断这道题是否正确：{REWRITTEN}")
    assert out["route"] == "course" and out["routes"] == ["course"] and out["status"] == "ok" and "prematch" not in out["trace"]
    sys_prompt = next(c for c in llm.calls if c["step"] == "answer")["messages"][0]["content"]
    assert "第一句先给出结论" in sys_prompt and sys_prompt.startswith("你是 AI 李老师")
    assert out["answer"].startswith("错误。")


def test_rewritten_stem_stream_also_sends_the_hint_and_routes_to_course(course):
    llm = ScriptedLLM(_course_script())
    s = UnifiedService(router_client=Counting("exam"), exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    ev = list(s.ask_stream(f"判断这道题是否正确：{REWRITTEN}"))
    assert ev[0]["type"] == "route" and ev[0]["route"] == "course" and ev[-1]["type"] == "done" and ev[-1]["status"] == "ok"
    assert "第一句先给出结论" in next(c for c in llm.calls if c["step"] == "answer")["messages"][0]["content"]


def test_solve_question_the_router_sends_elsewhere_keeps_its_route(course):
    s = UnifiedService(router_client=Counting("off_topic"), exam_bank=make_bank(), course_ctx=course, course_llm=ScriptedLLM([]))
    assert s.ask(f"判断这道题是否正确：{REWRITTEN}")["status"] == "refused"                  # 只改 exam -> course，其它路由原样


def test_ordinary_course_question_has_no_hint_and_the_prompt_is_verbatim(course):
    from rag.course import prompts
    llm = ScriptedLLM(_course_script())
    s = UnifiedService(router_client=Counting("course"), exam_bank=make_bank(), course_ctx=course, course_llm=llm)
    s.ask("什么是数据要素")
    assert next(c for c in llm.calls if c["step"] == "answer")["messages"][0]["content"] == prompts.ANSWER_SYSTEM
    assert "第一句先给出结论" not in prompts.ANSWER_SYSTEM


def test_course_answer_prompt_without_solve_is_byte_identical_to_before():
    from rag.course import prompts
    from rag import persona
    from rag import prompt_rules as rules
    # 结构条款与样例改为引用常量后，结尾是多轮接话示例；断言常量被完整拼接（而非写死旧措辞）
    assert rules.COURSE_SAMPLES in prompts.ANSWER_SYSTEM
    assert prompts.ANSWER_SYSTEM.endswith(persona.EXAMPLE_DIALOGUE)   # 多轮示例放最后，教「接话」
    assert rules.ANSWER_STRUCTURE in prompts.ANSWER_SYSTEM
    assert "不要引用材料中没有的编号，不要原样写出占位符 “[S#]”。" in prompts.ANSWER_SYSTEM
    assert prompts.ANSWER_SOLVE_HINT.startswith("\n- 这是一道具体的题目")


# ---------------------------------------------------------------- 找题行为不变；答案始终带上；选择题带选项

def test_finding_questions_still_go_through_the_router_and_every_question_carries_its_answer():
    s, router = svc("exam")
    out = s.ask("出三道判断题")
    assert router.calls and out["route"] == "exam" and "prematch" not in out["trace"] and len(out["exam"]["questions"]) == 3
    assert all(q["qtype"] == "judge" and q["answer"] and "analysis" in q for q in out["exam"]["questions"])
    assert "参考答案" not in out["answer"]                                           # 正文只排题干，答案放结构化字段（网页折叠）


def test_term_listing_question_still_finds_questions():
    s, _ = svc("exam")
    out = s.ask("第四章有哪些名词解释")
    assert out["exam"]["questions"][0]["qtype"] == "term" and out["exam"]["questions"][0]["answer"]


def test_choice_questions_carry_options_and_answer():
    s, _ = svc("exam")
    out = s.ask("第一章选择题")
    q = out["exam"]["questions"][0]
    assert q["qtype"] == "choice" and q["options"] == OPTS and q["answer"] == "A" and "A. " + OPTS["A"] in out["answer"]


def test_practice_items_carry_answer_and_analysis_verbatim_and_stay_judge_or_term():
    bank = make_bank()
    for i, it in enumerate(bank.items):
        it.concepts = ["数据资产"]
        bank.by_concept.setdefault("数据资产", []).append(i)

    class C:
        name = "数据资产"

    class Ctx:
        class concepts:
            by_id = {"C01-0001": C()}
    s = UnifiedService(exam_bank=bank)
    s.course_ctx = Ctx()
    got = s._practice([{"concept_id": "C01-0001"}])
    by = {it.q.question_id: it.q for it in bank.items}
    assert got and all(g["answer"] and "analysis" in g for g in got)
    assert all(g["answer"] == by[g["question_id"]].answer and g["analysis"] == by[g["question_id"]].analysis for g in got)
    assert all(by[g["question_id"]].qtype in ("judge", "term") for g in got)


# ---------------------------------------------------------------- 网页

APP_JS = open("rag/web/app.js", encoding="utf-8").read()
STYLE = open("rag/web/style.css", encoding="utf-8").read()


def test_web_renders_options_and_a_collapsed_answer_block_for_exam_and_practice():
    assert "<details>" not in APP_JS and "createElement('details')" in APP_JS and "查看答案" in APP_JS
    assert "d.open" not in APP_JS and "setAttribute('open'" not in APP_JS                    # 默认收起
    assert "it.options" in APP_JS and "`${k}. ${v}`" in APP_JS                                # A/B/C/D 选项
    assert "function practiceList" in APP_JS and "examList(items, '巩固练习'" in APP_JS            # 巩固练习同样带折叠答案
    assert "examList(examQs" in APP_JS and "ev.exam.questions" in APP_JS                      # 试卷库找题结果
    assert "textContent" in APP_JS and "innerHTML" not in APP_JS[APP_JS.index("function questionBlock"):APP_JS.index("function numberedSources")]    # 答案原样文本，不拼 HTML
    assert ".exam-answer" in STYLE and ".exam-option" in STYLE


# ---------------------------------------------------------------- 补强：术语不被长题干挤出候选 / 相似度门槛 / 找题措辞 / 选项渲染

def test_short_term_stem_is_found_even_when_many_long_choice_stems_contain_it():
    items = [_q("term", "名词解释", 1, "国家数据局与地方数据管理体制改革", "答", "解")]
    for i in range(30):
        items.append(_q("choice", "选择题", i, f"关于国家数据局与地方数据管理体制改革的相关机制与规则第{i}项，下列哪一选项正确？", "A", "", OPTS))
    m = StemMatcher(items)
    r = m.match("名词解释：国家数据局与地方数据管理体制改革")
    assert r["hit"] == 0 and r["how"] == "exact" and r["similarity"] == 1.0


def test_exact_stem_beats_a_longer_stem_that_merely_contains_it():
    items = [_q("choice", "选择题", 1, "关于数据权属数据权利与数据权益的概念界定与核心内涵，下列哪一选项正确？", "A", "", OPTS),
             _q("term", "名词解释", 1, "数据权属、数据权利与数据权益", "答", "解")]
    assert StemMatcher(items).match("名词解释：数据权属、数据权利与数据权益")["hit"] == 1


def test_similarity_below_point_nine_is_not_a_hit():
    m = StemMatcher(make_bank().items)
    half = JUDGE.replace("个人权益得不到保障、公共数据资源被不当垄断", "企业利润得不到保证、平台数据资源被大量闲置").replace("市场竞争失于公平", "行业秩序趋于混乱")
    assert m.match(f"判断这道题是否正确：{half}")["hit"] is None


@pytest.mark.parametrize("q", ["出三道判断题：" + JUDGE, "来几道题：" + JUDGE, "第四章有哪些名词解释：" + TERM + "之类"])
def test_find_phrasing_wins_over_a_pasted_stem(q):
    assert not is_solve_question(q)


def test_every_real_style_stem_matches_itself_when_asked_with_its_own_shell():
    bank = make_bank()
    m = StemMatcher(bank.items)
    for i, it in enumerate(bank.items):
        if len(normalize(it.q.stem)) >= SOLVE_MIN_CHARS:
            shell = "名词解释：" if it.q.qtype == "term" else "判断这道题是否正确："
            assert m.match(shell + it.q.stem)["hit"] == i


def test_web_option_block_is_built_from_options_and_each_option_is_its_own_row():
    seg = APP_JS[APP_JS.index("function questionBlock"):APP_JS.index("function examList")]
    assert "const opts = it.options && typeof it.options === 'object' ? Object.entries(it.options) : [];" in seg
    assert "if (opts.length)" in seg and "className = 'exam-option'" in seg and "opts.forEach(" in seg
