"""题库解析：新格式（11.7 题干修订版：章 / 题型标题无括号、每选项一段、「参考答案  X」「解析  」「评分要点：」「书稿依据：」）
与旧格式（10.31 修订版，以及 11.7 版第十一章沿用的写法）。两种写法可在同一份文件里混用。用标准库现造 docx。"""
from __future__ import annotations

import pytest

from rag.exam.bank import compare_student, parse_bank
from rag.exam.solve import StemMatcher
from rag.exam.store import ExamItem
from tests.test_course_chunk import _p, _write

SRC = "书稿依据：第一章 数据要素 › 第一节 数据要素的概念界定 P0062、P0068（综合上述讨论，可以对数据要素的内涵作…）"


def _new_chapter() -> list[str]:
    return [
        _p("数据要素流通与数据工程参考答案与评分要点"), _p("教师版  2015题  内容优化版  保留第十一章"),
        _p("第十一章数据工程师保留原有209题及原题型数量，所附书稿无对应章节，未作书稿依据核验。"),   # 说明段，不是章标题
        _p("使用说明"),
        _p("第一章 数据要素"), _p("本章共7题"),
        _p("一 选择题  共2题"),
        _p("1. 数据资源成为数据要素，关键在于具备哪一项特征？（ ）"),
        _p("A. 已被纳入企业财务报表"), _p("B. 具有可长期保存的电子载体"), _p("C. 能够排除其他主体同时使用"), _p("D. 能够投入生产经营并参与价值创造"),
        _p("参考答案  D"), _p("解析  本题的判断依据是：能够投入生产经营并参与价值创造。"), _p(SRC),
        _p("2. 下列哪项最能体现“数据资源”的概念重心？（ ）"),
        _p("A. 已在特定场景取得经营收益"), _p("B. 已完成会计确认和货币计量"), _p("C. 具有价值创造潜力并可再利用"), _p("D. 已进入公开交易市场"),
        _p("参考答案  C"), _p("解析  概念重心在价值创造潜力。"), _p(SRC),
        _p("二 判断题  共2题"),
        _p("1. 2019年中央文件确认数据按贡献参与分配，体现了数据要素制度建设的推进。（ ）"),
        _p("参考答案  正确"), _p("解析  书稿以2019年党的十九届四中全会为制度确立节点。"), _p(SRC),
        _p("2. 数据要素的正外部性只体现在使用层面，不存在积累层面的效应。（ ）"),
        _p("参考答案  错误"), _p("解析  正外部性体现在使用与积累两个层面。"), _p(SRC),
        _p("三 名词解释  共1题"),
        _p("1. 数据要素"),
        _p("参考答案  根据特定生产需求形成的、能够投入生产经营并参与价值创造的数据资源及其衍生形态。"),
        _p("评分要点：（1）根据特定生产需求形成的数据资源（50%）；（2）区别重点在于生产功能与价值创造（50%）。允许同义表述。"), _p(SRC),
        _p("四 简答题  共1题"),
        _p("1. 如何区分数据资源与数据资产？"),
        _p("参考答案  资源强调潜力；资产强调合法拥有与货币计量。"),
        _p("评分要点：（1）数据资源强调价值创造潜力（50%）；（2）数据资产强调合法拥有与货币计量（50%）。"), _p(SRC),
        _p("五 论述题  共0题"),
        _p("六 案例分析题  共1题"),
        _p("1. 【教学情境】商店过去只用购物记录核算营业额，后来开展关联分析，将结果用于商品陈列。"),
        _p("【分析问题】"),
        _p("（1）商店数据成为数据要素的依据是什么？（2）能否据此认定已完成市场经营？"),
        _p("参考答案  （1）加工记录并用于经营；（2）不能。"),
        _p("评分要点：（1）加工记录并用于经营（50%）；（2）不能认定（50%）。"), _p(SRC),
    ]


def _old_chapter() -> list[str]:
    return [
        _p("第十一章 数据工程师（共 3 题）"),
        _p("一、选择题（共 1 题）"),
        _p("1. 以下关于RISC和CISC的叙述中，正确的是？（ ）"),
        _p("A. RISC不采用流水线。B. RISC使用复杂指令。C. RISC采用较多通用寄存器。D. RISC普遍采用微程序控制器。"),
        _p("【参考答案】C"), _p("【评分要点与解析】正确选项为 C。"),
        _p("二、判断题（共 1 题）"),
        _p("1. 闪存在断电后会丢失数据。"), _p("【参考答案】错误\n【评分要点与解析】闪存是非易失存储器。"),
        _p("六、案例分析题（共 1 题）"),
        _p("1. 【案例标题】某政务云排障案\n【案例材料】某省政务云出现故障。\n【分析问题】（1）诊断成因；（2）给出措施。"),
        _p("【参考答案】【问题（1）参考答案】"), _p("成因是环路。"),
        _p("【评分要点与解析】（1）（占50%）成因；（2）（占50%）措施。"),
    ]


def _tail() -> list[str]:
    return [_p("书稿待核说明"), _p("1. 这是说明里的编号段，不是题目。"), _p("第一章 数据要素"), _p("一 选择题  共9题"), _p("2. 也不是题目。")]


@pytest.fixture
def qs(tmp_path):
    return parse_bank(_write(tmp_path, [*_new_chapter(), *_old_chapter(), *_tail()], "t.docx"))


def by(qs_, ch, t, n):
    return next(q for q in qs_[0] if (q.chapter, q.qtype, q.number) == (ch, t, n))


def test_counts_types_and_no_flags_in_a_mixed_file(qs):
    questions, stats = qs
    assert len(questions) == 10 and stats["flags"] == {} and stats["unparsed_count"] == 0 and stats["duplicate_ids"] == 0
    assert [q.qtype for q in questions] == ["choice", "choice", "judge", "judge", "term", "short", "case", "choice", "judge", "case"]
    assert [q.chapter for q in questions] == [1] * 7 + [11] * 3
    assert stats["declared_total"] == 2 + 2 + 1 + 1 + 0 + 1 + 1 + 1 + 1 and stats["count_mismatch"] == {"ch01:essay": {"declared": 0, "actual": None}}


def test_cover_notes_and_statement_paragraphs_are_not_chapters_or_questions(qs):
    assert not any("第十一章数据工程师保留" in q.stem for q in qs[0])
    assert {q.chapter_title for q in qs[0]} == {"数据要素", "数据工程师"}


def test_new_format_choice_options_are_one_per_paragraph_and_answer_is_the_letter(qs):
    q = by(qs, 1, "choice", 1)
    assert list(q.options) == ["A", "B", "C", "D"] and q.options["D"] == "能够投入生产经营并参与价值创造" and q.answer == "D" and q.flags == []
    assert q.analysis.startswith("本题的判断依据是") and q.stem == "数据资源成为数据要素，关键在于具备哪一项特征？（ ）"
    assert q.source.startswith("第一章 数据要素 › 第一节") and "P0062" in q.source and "书稿依据" not in q.analysis       # 出处单独存放，不混进解析


def test_new_format_judge_answers(qs):
    assert by(qs, 1, "judge", 1).answer == "正确" and by(qs, 1, "judge", 2).answer == "错误"
    assert by(qs, 1, "judge", 1).analysis.startswith("书稿以2019年")


def test_new_format_scoring_points_and_weights_for_subjective_types(qs):
    t = by(qs, 1, "term", 1)
    assert t.answer.startswith("根据特定生产需求") and [s["weight"] for s in t.scoring] == [50, 50]
    assert t.scoring[0]["point"] == "根据特定生产需求形成的数据资源" and t.concepts == ["数据要素"]
    s = by(qs, 1, "short", 1)
    assert [x["weight"] for x in s.scoring] == [50, 50] and s.analysis.startswith("（1）数据资源强调")


def test_new_format_case_stem_is_rejoined_and_sub_questions_split(qs):
    q = by(qs, 1, "case", 1)
    assert q.stem.startswith("【教学情境】商店过去") and "【分析问题】" in q.stem and q.sub_questions == ["商店数据成为数据要素的依据是什么？", "能否据此认定已完成市场经营？"]
    assert q.answer.startswith("（1）加工记录") and len(q.scoring) == 2


def test_old_format_chapter_in_the_same_file_still_parses(qs):
    c = by(qs, 11, "choice", 1)
    assert list(c.options) == ["A", "B", "C", "D"] and c.answer == "C" and c.analysis.startswith("正确选项为 C") and c.flags == []
    j = by(qs, 11, "judge", 1)
    assert j.answer == "错误" and j.analysis == "闪存是非易失存储器。"
    k = by(qs, 11, "case", 1)
    assert k.sub_questions == ["诊断成因；", "给出措施。"] and "成因是环路" in k.answer and [s["weight"] for s in k.scoring] == [50, 50]


def test_everything_after_the_end_marker_is_ignored(qs):
    assert not any("说明里的编号段" in q.stem or "也不是题目" in q.stem for q in qs[0]) and len(qs[0]) == 10


def test_question_id_depends_on_chapter_type_number_and_the_final_joined_stem(tmp_path, qs):
    q = by(qs, 1, "case", 1)
    from rag.exam.bank import _qid
    assert q.question_id == _qid(1, "case", 1, q.stem)                  # 题干跨段接回之后才算 id
    again = parse_bank(_write(tmp_path, [*_new_chapter(), *_old_chapter()], "u.docx"))
    assert [x.question_id for x in again[0]] == [x.question_id for x in qs[0]]          # 重新解析 id 稳定


def test_numbered_lines_inside_an_answer_are_not_new_questions(tmp_path):
    """答案 / 解析里的「1. …」「3. …」（序号不是紧接的下一题）属于上一题的答案，不能被当成新题。"""
    body = [_p("第一章 数据要素"), _p("四 简答题  共2题"), _p("1. 如何区分数据资源与数据资产？"),
            _p("参考答案  要点如下："), _p("1. 资源强调潜力；"), _p("3. 资产强调计量。"),
            _p("评分要点：（1）潜力（50%）；（2）计量（50%）。"), _p("2. 第二题题干？"), _p("参考答案  乙"), _p("评分要点：（1）乙（100%）。")]
    got = parse_bank(_write(tmp_path, body, "c.docx"))[0]
    assert [q.number for q in got] == [1, 2] and got[0].answer == "要点如下：\n1. 资源强调潜力；\n3. 资产强调计量。"
    assert [s["weight"] for s in got[0].scoring] == [50, 50] and got[1].answer == "乙"


def test_next_numbered_item_after_an_answer_starts_a_new_question(tmp_path):
    body = [_p("第一章 数据要素"), _p("四 简答题  共2题"), _p("1. 甲题？"), _p("参考答案  甲"), _p("评分要点：（1）甲（100%）。"), _p("2. 乙题？"), _p("参考答案  乙")]
    assert [q.number for q in parse_bank(_write(tmp_path, body, "d.docx"))[0]] == [1, 2]


def test_missing_answer_and_missing_options_are_flagged(tmp_path):
    body = [_p("第一章 数据要素"), _p("一 选择题  共2题"), _p("1. 没有答案的选择题？（ ）"), _p("A. 甲"), _p("B. 乙"),
            _p("2. 只有一个选项的选择题？（ ）"), _p("A. 甲"), _p("参考答案  Z"), _p("解析  无")]
    qs_ = parse_bank(_write(tmp_path, body, "f.docx"))[0]
    assert qs_[0].flags == ["no_answer"] and set(qs_[1].flags) == {"options_missing", "answer_not_in_options"}


def test_student_version_new_format_matches_teacher_stems(tmp_path):
    teacher = parse_bank(_write(tmp_path, [*_new_chapter(), *_old_chapter()], "t.docx"))[0]
    student = [_p("第一章 数据要素"), _p("本章共7题"), _p("一 选择题  共2题"),
               _p("1. 数据资源成为数据要素，关键在于具备哪一项特征？（ ）"), _p("A. 已被纳入企业财务报表"), _p("B. x"),
               _p("2. 下列哪项最能体现“数据资源”的概念重心？（ ）"), _p("A. x"),
               _p("二 判断题  共2题"), _p("1. 2019年中央文件确认数据按贡献参与分配，体现了数据要素制度建设的推进。（ ）"),
               _p("2. 数据要素的正外部性只体现在使用层面，不存在积累层面的效应。（ ）"),
               _p("三 名词解释  共1题"), _p("1. 数据要素"), _p("四 简答题  共1题"), _p("1. 如何区分数据资源与数据资产？"),
               _p("六 案例分析题  共1题"), _p("1. 【教学情境】商店过去只用购物记录核算营业额，后来开展关联分析，将结果用于商品陈列。"), _p("【分析问题】"), _p("（1）商店数据成为数据要素的依据是什么？（2）能否据此认定已完成市场经营？"),
               _p("第十一章 数据工程师"), _p("一 选择题  共1题"), _p("1. 以下关于RISC和CISC的叙述中，正确的是？（ ）"), _p("A. x"),
               _p("二 判断题  共1题"), _p("1. 闪存在断电后会丢失数据。"),
               _p("六 案例分析题  共1题"), _p("1. 【案例标题】某政务云排障案\n【案例材料】某省政务云出现故障。\n【分析问题】（1）诊断成因；（2）给出措施。")]
    cmp = compare_student(teacher, _write(tmp_path, student, "s.docx"))
    assert cmp["stem_mismatch"] == 0 and cmp["only_in_student"] == 0 and cmp["student_questions"] == 10 and cmp["student_answer_paragraphs"] == 0


def test_stem_matcher_indexes_new_format_stems(qs):
    items = [ExamItem(q, [], [], "none", q.stem) for q in qs[0]]
    m = StemMatcher(items)
    r = m.match("判断这道题是否正确：2019年中央文件确认数据按贡献参与分配，体现了数据要素制度建设的推进。")
    assert r["hit"] is not None and items[r["hit"]].q.qtype == "judge" and items[r["hit"]].q.answer == "正确"


def test_default_bank_version_is_the_new_one():
    from rag.exam import config
    assert config.EXAM_BANK_VERSION == "1107"
