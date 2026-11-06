"""scripts/check_answer_format.py 的离线测试：字数口径、WARN/FAIL 分级、exam N/A、样例复用。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_answer_format", Path(__file__).resolve().parent.parent / "scripts" / "check_answer_format.py")
caf = importlib.util.module_from_spec(_SPEC)
sys.modules["check_answer_format"] = caf
_SPEC.loader.exec_module(caf)

GOOD = ("数据资产入表的核心是把符合确认条件的数据资源，按无形资产或存货计入财务报表。\n"
        "- 确认条件：来源合法、由企业控制、能带来预期经济利益、成本可可靠计量；\n"
        "- 落地路径：先盘点与合规审查，再归集成本、评估价值，最后作会计判断与披露。\n"
        "入表的意义不在报表数字本身，而在于倒逼企业补齐数据资产的管理基础。")
GOOD_FU = ["入表对中小企业有什么门槛？", "数据资产评估一般找谁做？"]


def test_citations_do_not_count_toward_length():
    """[1][2] / [来源N] / 误留的 [S1] 都在统计前剥离。"""
    body = "数据要素是生产要素。" * 20                     # 200 字
    cited = body + "[1][2][来源3][S4]"
    assert len(caf.strip_citations(cited)) == len(body)
    assert caf.analyze(cited)["chars"] == 200


def test_length_bands_pass_warn_fail():
    base = "数据要素是生产函数中的新要素，可参与价值创造。" * 30   # 720 字
    ok = caf.analyze(base[:400])
    assert ok["chars"] == 400 and "超字数" not in "、".join(ok["problems"])
    warn = caf.analyze(base[:450])
    assert warn["chars"] == 450 and any("字数偏高" in w for w in warn["warnings"])
    assert not any("超字数" in p for p in warn["problems"])
    bad = caf.analyze(base[:620])
    assert any("超字数" in p for p in bad["problems"]) and not bad["ok"]


def test_more_than_four_points_fails():
    """分点硬上限 4 条：与 prompt 的「归并为最多 4 条」一致，>4 判 FAIL。"""
    body = ("数据定价方法有七种，可按维度归并理解。\n"
            "- 成本加成：按投入加收益定价；\n- 市场比较：参照同类成交价；\n"
            "- 收益分成：按增量收益分配；\n- 基于价值：按边际贡献定价；\n"
            "- 动态定价：按时效调整；\n- 政府指导：公共数据适用。\n"
            "这些方法常组合使用。")
    m = caf.analyze(body)
    assert m["points"] == 6 and not m["ok"]
    assert any("分点过多" in p for p in m["problems"])
    assert not any("分点过多" in w for w in m["warnings"])     # 不再只是警告


def test_citation_overflow_fails():
    """引用编号超过 materials 数：指向不存在的材料，判 FAIL 并列出越界编号。"""
    body = ("数据要素市场由多类主体构成。\n"
            "- 供给与需求：供给方、需求方各自分工；\n"
            "- 中介与监管：数据商、平台运营方、监管机构 [3]。\n"
            "四类主体共同支撑交易达成。")
    ok = caf.analyze(body, ["a？", "b？"], "course", [{"n": 1}, {"n": 2}, {"n": 3}])
    assert not ok["citation_overflow"] and ok["ok"]
    bad = caf.analyze(body, ["a？", "b？"], "course", [{"n": 1}, {"n": 2}])
    assert bad["citation_overflow"] == [3] and not bad["ok"]
    assert any("引用越界" in p for p in bad["problems"])
    assert not caf.analyze(body, ["a？", "b？"], "course", None)["citation_overflow"]   # 无 sources 不误报


def test_good_answer_passes_with_qualified_followups():
    m = caf.analyze(GOOD, GOOD_FU, route="course")
    assert m["ok"] and not m["problems"]
    assert m["opening"] and m["closing"] and m["points"] == 2
    assert m["followup_ok"] is True


def test_exam_route_followups_are_na_but_other_checks_still_apply():
    m = caf.analyze(GOOD, [], route="exam")
    assert m["followup_ok"] is None and m["ok"]            # 追问不进分母
    assert m["followups"] == 0
    bad = caf.analyze("太短", [], route="exam")
    assert not bad["ok"] and bad["followup_ok"] is None


def test_unqualified_followups_fail():
    m = caf.analyze(GOOD, ["这条追问明显超过了二十五个字所以应该被判不合格的对吧？"], route="course")
    assert m["followup_ok"] is False and "追问不合格" in m["problems"]


def test_sample_reuse_detected_by_long_overlap():
    from rag import prompt_rules
    sample_body = prompt_rules.COURSE_SAMPLES.split("回答：", 1)[1].lstrip("\n")
    copied = sample_body[:20]                               # 与样例连续 20 字相同
    assert caf.sample_overlap(copied) >= 20
    assert any("疑似复用样例" in w for w in caf.analyze(GOOD + "\n" + copied)["warnings"])
    assert caf.sample_overlap(GOOD) < caf.SAMPLE_OVERLAP   # 正常回答不误报
