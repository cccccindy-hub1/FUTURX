"""试卷库测试：全部离线，用标准库现造题库 docx（结构仿照 10.31 修订版教师版 / 学生版）与小型学生知识集合。"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import json

from rag.course.llm import LLMResult
from rag.exam import config
from rag.exam.api import router as exam_router
from rag.exam.bank import Question, compare_student, parse_bank, split_options
from rag.exam.knowledge import StudentIndex, link_question
from rag.exam.store import ExamBank, ExamItem, format_items
from rag.router import RouteDecision, classify, rule_route
from rag.unified import UnifiedService

from tests.test_course_chunk import _p, _tbl, _write

OPTS = ("A. 特定主体合法拥有或者控制的、能进行货币计量的数据资源。B. 数据资产与数据要素是线性递进关系"
        "C. “数据资产”与“公共数据”完全等同。D. 数据资产只存在于政府部门。")
ANS1 = ("【参考答案】A【评分要点与解析】正确选项为 A。“数据资产”的核心界定为：特定主体合法拥有或者控制的数据资源。")
TERM_ANS = ("【参考答案】（1）定义界定：数据资源是具有价值创造潜力的数据的总称。【评分要点与解析】"
            "（1）（占40%）准确表述核心定义；（2）（占35%）阐述核心内涵；（3）（占25%）说明适用边界。")
CASE = ("【案例标题】某超市小票挖掘案【案例材料】某超市积累了大量小票，“数据很重要”。"
        "【分析问题】（1）围绕“数据资产”相关理论，分析小票如何转化？（2）体现了哪一属性？")


def _teacher(answers: bool = True) -> list[str]:
    a = (lambda s: [_p(s)]) if answers else (lambda s: [])
    return [
        _p("数据要素全题型题库（教师版·共7题）"),
        _tbl([["章节模块", "选择题"], ["第一章 数据要素", "2"]]),
        _p("第一章 数据要素（共 5 题）"),
        _p("一、选择题（共 2 题）"),
        _p("1. 关于“数据资产”的概念界定与核心内涵，下列哪一选项的表述是正确的？（ ）"),
        _p(OPTS), *a(ANS1),
        _p("2. 关于“数据要素、数据资源与数据资产”的区别，下列哪一选项正确？（ ）"),
        _p("A. 甲。B. 乙。C. 丙。D. 丁。"), *a("【参考答案】C【评分要点与解析】正确选项为 C。"),
        _p("二、判断题（共 2 题）"),
        _p("1. 数据资源经确权入表后可成为数据资产。"), *a("【参考答案】正确【评分要点与解析】见教材。"),
        _p("2. 路由器的默认参考带宽为 100Mbps。"), *a("【参考答案】错误【评分要点与解析】网络常识。"),
        _p("三、名词解释（共 1 题）"),
        _p("1. 数据资源"), *a(TERM_ANS),
        _p("第二章 数据权属（共 2 题）"),
        _p("一、简答题（共 1 题）"),
        _p("1. 简述“数据资产”与“数据资源”的核心内涵及二者联系。"),
        *a("【参考答案】（1）数据资产……（2）数据资源……【评分要点与解析】（1）（占50%）资产；（2）（占50%）资源。"),
        _p("二、案例分析题（共 1 题）"),
        _p("1. " + CASE),
        *a("【参考答案】【问题（1）参考答案】转化过程……【问题（2）参考答案】价值不确定性。"),
    ]


def _student_rows() -> list[dict[str, str]]:
    def r(i, ch, s, rel, o):
        return {"集合ID": f"Student_{i:06d}", "章节": ch, "主体": s, "关系": rel, "客体": o, "公理": f"{s}{rel}{o}"}
    return [
        r(1, "Ch01", "数据资产", "hasDefinition", "特定主体合法拥有或者控制的数据资源"),
        r(2, "Ch01", "数据资产", "instanceOf", "核心概念"),                       # 结构性关系：不挂
        r(3, "Ch01", "数据资产", "hasFeature", "可货币计量"),
        r(4, "Ch01", "数据资源", "hasDefinition", "具有价值创造潜力的数据的总称"),
        r(5, "Ch01", "数据资源", "relatedTo", "经确权入表成为数据资产"),           # 两概念之间的直接三元组
        r(6, "Ch01", "数据要素", "hasDefinition", "投入生产经营的数据资源"),
        r(7, "Ch02", "数据资产", "hasDefinition", "第二章里的数据资产"),
        r(8, "Ch01", "第一节 数据要素概述", "hasSection", "x"),                   # 节标题：不当概念
    ]


@pytest.fixture
def docs(tmp_path):
    return _write(tmp_path, _teacher(), "teacher.docx"), _write(tmp_path, _teacher(answers=False), "student.docx")


@pytest.fixture
def idx():
    return StudentIndex.from_rows(_student_rows())


@pytest.fixture
def bank(docs, idx):
    questions, stats = parse_bank(docs[0])
    items = []
    for q in questions:
        link = link_question(q, idx)
        items.append(ExamItem(q, link.concepts, link.kp_ids, link.method,
                              q.stem + " " + " ".join(link.concepts)))
    return ExamBank(items, idx, stats)


# ---------------------------------------------------------------- 解析

def test_parse_counts_types_and_ids(docs):
    qs, stats = parse_bank(docs[0])
    assert len(qs) == 7 and stats["declared_total"] == 7 and not stats["count_mismatch"]
    assert [q.qtype for q in qs] == ["choice", "choice", "judge", "judge", "term", "short", "case"]
    assert stats["duplicate_ids"] == 0 and stats["unparsed_count"] == 0
    assert qs[0].question_id.startswith(f"ex:{config.EXAM_BANK_VERSION}:ch01:choice:001:")
    assert (qs[5].chapter, qs[5].chapter_title, qs[5].number) == (2, "数据权属", 1)


def test_parse_choice_options_answer_and_analysis(docs):
    q = parse_bank(docs[0])[0][0]
    assert list(q.options) == ["A", "B", "C", "D"] and q.answer == "A"
    assert q.options["B"].startswith("数据资产与数据要素") and "“数据资产”与“公共数据”" in q.options["C"]
    assert q.analysis.startswith("正确选项为 A") and q.basic and not q.flags


def test_parse_judge_term_scoring_and_case(docs):
    qs = parse_bank(docs[0])[0]
    assert [qs[2].answer, qs[3].answer] == ["正确", "错误"]
    term = qs[4]
    assert term.concepts == ["数据资源"] and [s["weight"] for s in term.scoring] == [40, 35, 25]
    case = qs[6]
    assert len(case.sub_questions) == 2 and case.sub_questions[0].startswith("围绕")
    assert case.concepts == ["数据资产"]          # 案例材料里的引号（引语）不算概念
    assert not case.basic and not qs[5].basic


def test_count_mismatch_is_reported(tmp_path):
    body = _teacher()
    body[3] = _p("一、选择题（共 3 题）")
    _, stats = parse_bank(_write(tmp_path, body, "bad.docx"))
    assert stats["count_mismatch"] == {"ch01:choice": {"declared": 3, "actual": 2}}


def test_student_version_aligns_with_teacher(docs):
    qs = parse_bank(docs[0])[0]
    r = compare_student(qs, docs[1])
    assert r["student_questions"] == 7 and r["stem_mismatch"] == 0 and r["student_answer_paragraphs"] == 0


def test_split_options_ignores_letters_inside_words():
    assert split_options("A. 采用 API 接口B. 使用 GDPR. 规则C. 丙D. 丁") == {
        "A": "采用 API 接口", "B": "使用 GDPR. 规则", "C": "丙", "D": "丁"}


# ---------------------------------------------------------------- 知识点挂接（学生知识集合粒度）

def test_link_exact_same_chapter_and_structural_relations_skipped(docs, idx):
    q = parse_bank(docs[0])[0][0]
    link = link_question(q, idx)
    assert link.concepts == ["数据资产"] and link.method == "exact:same"
    assert link.kp_ids == ["Student_000001", "Student_000003"]     # 不含 instanceOf，不含第二章同名


def test_link_compound_concept_is_split(docs, idx):
    link = link_question(parse_bank(docs[0])[0][1], idx)
    assert link.method == "split:same" and link.concepts == ["数据要素", "数据资源", "数据资产"]
    # 客体里提到另一个所考概念的三元组（概念之间的关系）排在最前，其后才是各概念自己的定义 / 特征
    assert set(link.kp_ids[:2]) == {"Student_000005", "Student_000006"}


def test_link_by_mention_and_unlinked(docs, idx):
    qs = parse_bank(docs[0])[0]
    judge = link_question(qs[2], idx)
    assert judge.method == "mention:same" and set(judge.concepts) == {"数据资源", "数据资产"}
    assert link_question(qs[3], idx).kp_ids == []                  # 网络题：教材没有对应知识点


def test_link_falls_back_to_other_chapter(docs, idx):
    q = parse_bank(docs[0])[0][5]   # 第二章简答：数据资源只在 Ch01
    link = link_question(q, idx)
    assert "数据资源" in link.concepts and "Student_000007" in link.kp_ids


def test_section_titles_are_not_concepts(idx):
    assert "第一节数据要素概述" not in idx._all_names()


# ---------------------------------------------------------------- 检索

def test_search_by_concept_basic_only(bank):
    r = bank.search("给我出2道关于数据资产的基础题")
    assert r["status"] == "ok" and r["strategy"] == "concept" and r["intent"]["count"] == 2
    assert all(it.q.basic for it in r["items"]) and len(r["items"]) == 2
    assert {it.q.qtype for it in r["items"]} == {"choice", "judge"}     # 题型轮转


def test_search_by_number_and_filter(bank):
    r = bank.search("第一章判断题第2题的答案")
    assert r["strategy"] == "number" and r["items"][0].q.answer == "错误" and r["intent"]["want_answer"]
    r = bank.search("来两道第一章选择题")
    assert r["strategy"] == "filter" and [it.q.number for it in r["items"]] == [1, 2]


def test_search_irrelevant_returns_no_match(bank):
    r = bank.search("今天天气怎么样")
    assert r["status"] == "no_match" and r["items"] == []


def test_by_concepts_excludes_and_filters(bank):
    first = bank.by_concepts(["数据资产"], 1, basic_only=True)
    again = bank.by_concepts(["数据资产"], 3, basic_only=True, exclude_ids=[first[0].q.question_id])
    assert first[0].q.question_id not in {it.q.question_id for it in again}
    assert [it.q.qtype for it in bank.by_concepts(["数据资产"], 5, qtypes=("short",))] == ["short"]


def test_format_hides_answer_by_default(bank):
    items = bank.by_concepts(["数据资产"], 1, qtypes=("choice",))
    text = format_items(items)
    assert "【选择题·第1章 数据要素】" in text and "  A. " in text and "参考答案" not in text
    assert "【参考答案】A" in format_items(items, include_answer=True)


# ---------------------------------------------------------------- 路由与统一入口

def test_rule_route_exam_vs_course_and_thesis():
    assert rule_route("出3道数据资产的基础练习题").route == "exam"
    assert rule_route("第一章选择题第3题的参考答案").route == "exam"
    assert rule_route("什么是数据资产").route == "course"
    assert rule_route("论文选题怎么确定").route == "thesis"     # 「选题」不是要题目


class _RouterLLM:
    def __init__(self, route):
        self.route = route

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        return LLMResult(content=json.dumps({"route": self.route, "complexity": "simple", "reason": "t"}))


def test_llm_exam_route_is_accepted():
    d: RouteDecision = classify("来几道题", _RouterLLM("exam"))
    assert (d.route, d.source) == ("exam", "llm")


def test_unified_exam_route_without_and_with_answers(bank):
    svc = UnifiedService(router_client=_RouterLLM("exam"), exam_bank=bank)
    out = svc.ask("出1道数据资产的选择题")
    assert out["route"] == "exam" and out["status"] == "ok"
    q = out["exam"]["questions"][0]
    assert q["qtype"] == "choice" and q["answer"] and "参考答案" not in out["answer"]       # 答案始终随题带出（网页默认折叠），正文只排题干与选项
    out = svc.ask("第一章选择题第1题的答案是什么")
    assert out["exam"]["questions"][0]["answer"] == "A" and "参考答案" not in out["answer"]       # 答案始终在结构化字段里（网页折叠），不拼进正文
    assert "A. " in out["answer"] and out["exam"]["questions"][0]["options"]["A"]              # 选择题的选项在题目下
    assert out["trace"]["exam"]["strategy"] == "number"


def test_unified_exam_route_streams_into_done(bank):
    evs = list(UnifiedService(router_client=_RouterLLM("exam"), exam_bank=bank).ask_stream("出1道数据资产的选择题"))
    types = [e["type"] for e in evs]
    assert types[:2] == ["route", "done"] and all(t == "followups" for t in types[2:])   # 追问在 done 之后追加
    done = evs[1]
    assert done["route"] == "exam" and done["status"] == "ok"
    assert done["exam"]["questions"][0]["qtype"] == "choice" and done["answer"]


def test_unified_exam_unavailable_and_no_match(bank):
    out = UnifiedService(router_client=_RouterLLM("exam")).ask("出几道题")
    assert out["status"] == "unavailable"
    out = UnifiedService(router_client=_RouterLLM("exam"), exam_bank=bank).ask("今天天气怎么样")
    assert out["status"] == "no_match" and out["exam"]["questions"] == []


def test_course_practice_uses_selected_concepts(bank):
    class _C:
        def __init__(self, name):
            self.name = name

    class _Table:
        by_id = {"C01-0001": _C("数据资产"), "C01-0002": _C("数据资源")}

    class _Ctx:
        concepts = _Table()

    svc = UnifiedService(exam_bank=bank)
    svc.course_ctx = _Ctx()
    got = svc._practice([{"concept_id": "C01-0001"}, {"concept_id": "C01-0002"}, {"concept_id": "nope"}])
    assert 1 <= len(got) <= 3 and all(g["qtype_cn"] in ("选择题", "判断题", "名词解释") for g in got)
    assert UnifiedService()._practice([{"concept_id": "C01-0001"}]) == []


# ---------------------------------------------------------------- HTTP

@pytest.fixture
def client(bank):
    app = FastAPI()
    app.state.exam = bank
    app.include_router(exam_router)
    return TestClient(app)


def test_api_search_by_concepts_and_question(client, bank):
    r = client.post("/exam/search", json={"query": "数据资产的题", "k": 2, "include_knowledge": True}).json()
    assert r["status"] == "ok" and len(r["questions"]) == 2 and "answer" not in r["questions"][0]
    assert r["questions"][0]["knowledge_points"][0]["subject"] == "数据资产"
    r = client.post("/exam/by_concepts", json={"concepts": ["数据资源"], "k": 1}).json()
    assert r["questions"][0]["basic"]
    qid = bank.items[0].q.question_id
    assert client.get(f"/exam/question/{qid}", params={"include_answer": True}).json()["answer"] == "A"
    assert client.get("/exam/question/nope").status_code == 404


def test_book_chapter_map_by_title_and_unmatched_chapter_links_cross_only(tmp_path):
    from openpyxl import Workbook

    from rag.exam.knowledge import book_chapter_map
    wb = Workbook(); ws = wb.active; ws.title = "concept总表"
    ws.append(["概念ID", "章号", "章节主题", "概念名称", "概念类型", "上位或关联项", "释义或知识语境", "收录依据", "教材定位",
               "教材原文摘录", "专家原词", "专家定位", "归并及版本说明", "收录分组", "问题类型", "问题说明"])
    for cid, ch, topic in (("C04-0001", 4, "数据政策"), ("C11-0001", 11, "数据资产")):
        ws.append([cid, ch, topic, topic] + [""] * 12)
    wb.save(tmp_path / "t.xlsx")
    m = book_chapter_map({4: "数据政策", 11: "数据工程师"}, tmp_path / "t.xlsx")
    assert m == {4: 4, 11: None}                       # 题库第11章与教材第11章同号不同名：不当作同章
    assert book_chapter_map({3: "x"}, tmp_path / "missing.xlsx") == {3: 3}
    idx = StudentIndex.from_rows([
        {"集合ID": "S1", "章节": "Ch11", "主体": "数据资产", "关系": "hasDefinition", "客体": "数据资产是……"}])
    q = Question(question_id="q", chapter=11, chapter_title="数据工程师", qtype="choice", qtype_cn="单项选择题",
                 number=1, stem="关于“数据资产”的说法", concepts=["数据资产"])
    link = link_question(q, idx, None)
    assert link.kp_ids == ["S1"] and link.method == "exact:cross"


# ---------------------------------------------------------------- 巩固练习不含选择题（网页不显示选项，学生没法作答）

class _Ctx2:
    class concepts:                                                   # noqa: N801  最小的课程上下文替身
        class _C:
            def __init__(self, n):
                self.name = n
        by_id = {"C01-0001": _C("数据资产"), "C01-0002": _C("数据资源")}


def _svc(bank):
    svc = UnifiedService(exam_bank=bank)
    svc.course_ctx = _Ctx2()
    return svc


def test_practice_never_contains_choice_questions_even_when_they_match_best(bank):
    from rag.unified import PRACTICE_QTYPES
    # 前提：题库里这个概念确实挂着选择题（否则这条测试什么都没验证）
    names = ["数据资产", "数据资源"]
    all_matched = bank.by_concepts(names, 10)
    assert any(it.q.qtype == "choice" for it in all_matched)
    got = _svc(bank)._practice([{"concept_id": "C01-0001"}, {"concept_id": "C01-0002"}])
    assert got and all(g["qtype_cn"] in ("判断题", "名词解释") for g in got)
    assert not any("选择" in g["qtype_cn"] for g in got) and PRACTICE_QTYPES == ("judge", "term")
    ids = {g["question_id"] for g in got}
    assert not (ids & {it.q.question_id for it in all_matched if it.q.qtype == "choice"})       # 一道选择题的 ID 都不在里面


def test_practice_is_empty_when_no_judge_or_term_question_matches(bank):
    """符合的题一道都没有时返回空列表（前端据此整块不显示巩固练习），而不是退回选择题凑数。"""
    only_choice = ExamBank([it for it in bank.items if it.q.qtype == "choice"], bank.idx, bank.stats)
    assert any(it.q.qtype == "choice" and "数据资产" in it.concepts for it in only_choice.items)
    assert _svc(only_choice)._practice([{"concept_id": "C01-0001"}]) == []


def test_practice_respects_max_count_and_keeps_stem_and_id_fields(bank, monkeypatch):
    from rag import config as cfg
    monkeypatch.setattr(cfg, "EXAM_PRACTICE_MAX", 1)
    got = _svc(bank)._practice([{"concept_id": "C01-0001"}, {"concept_id": "C01-0002"}])
    assert len(got) == 1 and set(got[0]) == {"question_id", "qtype_cn", "stem", "answer", "analysis"}
    monkeypatch.setattr(cfg, "EXAM_PRACTICE_MAX", 0)
    assert _svc(bank)._practice([{"concept_id": "C01-0001"}]) == []


def test_asking_for_choice_questions_directly_still_returns_choice_questions(bank):
    """试卷库直接问题目不受影响：“给我选择题”照常返回选择题（含 /auto/ask 的 exam 路由与 /exam/* 接口）。"""
    r = bank.search("给我第一章的选择题")
    assert r["status"] == "ok" and r["items"] and all(it.q.qtype == "choice" for it in r["items"])
    svc = UnifiedService(router_client=type("R", (), {"chat": lambda self, *a, **k: LLMResult(content=json.dumps({"route": "exam", "reason": "t"}))})(),
                         exam_bank=bank)
    out = svc.ask("给我第一章的选择题")
    assert out["route"] == "exam" and out["status"] == "ok" and out["exam"]["questions"] and all(q["qtype"] == "choice" for q in out["exam"]["questions"])
    assert "A." in out["answer"]                                                             # 选项照常给出
    assert all(it.q.qtype == "choice" for it in bank.by_concepts(["数据资产"], 5, qtypes=("choice",)))


def test_bank_level_basic_only_semantics_unchanged(bank):
    """试卷库本身的“基础题”定义不动（含选择题）；只有课程回答附带的巩固练习另外排除选择题。"""
    from rag.exam.bank import BASIC_QTYPES
    assert BASIC_QTYPES == ("choice", "judge", "term")
    assert any(it.q.qtype == "choice" for it in bank.by_concepts(["数据资产"], 10, basic_only=True))


def test_practice_filters_choice_out_even_if_the_bank_ignores_the_qtypes_filter(bank):
    """再保险：即使试卷库那一层的题型过滤失效（返回了选择题），巩固练习这一层仍然把选择题拿掉。"""
    class Leaky:
        def by_concepts(self, names, k, *, chapter=None, qtypes=(), basic_only=False, exclude_ids=()):
            return bank.by_concepts(names, 10)                       # 无视 qtypes，原样返回含选择题的结果
    assert any(it.q.qtype == "choice" for it in bank.by_concepts(["数据资产", "数据资源"], 10))
    svc = UnifiedService(exam_bank=Leaky())
    svc.course_ctx = _Ctx2()
    got = svc._practice([{"concept_id": "C01-0001"}, {"concept_id": "C01-0002"}])
    assert got and all(g["qtype_cn"] in ("判断题", "名词解释") for g in got)
