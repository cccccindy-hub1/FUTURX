"""政策库切块测试：全部离线，用标准库现造汇编 docx（结构仿照《数据政策汇编》）。"""
from __future__ import annotations

import json
import re

import pytest

from rag.policy import chunker as policy
from rag.policy import config
from rag.policy.chunker import build_policy_set, policy_stats

from tests.test_course_chunk import _p, _tbl, _write

LONG_SENT = "推进数据要素市场化配置改革，健全数据资源登记、流通交易和收益分配等制度规则。"  # 39 字


def _law() -> list[str]:
    return [
        _p("1.2 《中华人民共和国数据安全法》", "Heading2"),
        _p("中华人民共和国数据安全法"),
        _p("（2021年6月10日第十三届全国人民代表大会常务委员会第二十九次会议通过）"),
        _p("目　　录"),
        _p("第一章　总　　则"),
        _p("第二章　数据安全制度"),
        _p("第一章　总　　则"),
        _p("第一条　为了规范数据处理活动，保障数据安全，促进数据开发利用，制定本法。"),
        _p("第二条　在中华人民共和国境内开展数据处理活动及其安全监管，适用本法。"),
        _p("在中华人民共和国境外开展数据处理活动，损害国家安全的，依法追究法律责任。"),
        _p("第三条　本法所称数据，是指任何以电子或者其他方式对信息的记录。"),
        _p("第二章　数据安全制度"),
        _p("第四条　国家建立数据分类分级保护制度，对数据实行分类分级保护。"),
        _p("第五条　重要数据的处理者应当按照规定开展风险评估，评估报告应当包括："),
        _p("（一）处理的重要数据的种类、数量；"),
        _p("（二）面临的数据安全风险及其应对措施。"),
        _p("- 3 -"),
        _p("第六条　本法自2021年9月1日起施行。"),
    ]


def _opinion(no: str = "2.50", title: str = "《中共中央 国务院关于构建数据基础制度更好发挥数据要素作用的意见》"):
    return [
        _p(f"{no} {title}", "Heading2"),
        _p("中共中央 国务院关于构建数据基础制度更好发挥数据要素作用的意见"),
        _p("（2022年12月2日）"),
        _p("数据作为新型生产要素，是数字化、网络化、智能化的基础。现提出如下意见。"),
        _p("一、总体要求"),
        _p("（一）指导思想。以维护国家数据安全、保护个人信息和商业秘密为前提，以促进数据合规高效流通使用为主线。"),
        _p("（二）工作原则"),
        _p("——遵循发展规律，创新制度安排。充分认识和把握数据产权、流通、交易等基本规律。"),
        _p("——坚持共享共用，释放价值红利。合理降低市场主体获取数据的门槛。"),
        _p("二、建立数据产权制度"),
        _p("（三）探索数据产权结构性分置制度。建立公共数据、企业数据、个人数据的分类分级确权授权制度。"),
        _p("申请登记应当提交下列材料："),
        _p("（一）申请人基本情况；"),
        _p("（二）数据来源说明。"),
        _p("（四）推进实施公共数据确权授权机制。加强汇聚共享和开放开发，强化统筹授权使用和管理。"),
    ]


def _build(tmp_path, body, name="汇编.docx"):
    return build_policy_set(_write(tmp_path, [_p("第1篇 国家立法", "Heading1"), *body], name))


def _by(ps, doc_no):
    return [c for c in ps.chunks if c.doc_no == doc_no]


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- 条款式

def test_law_one_chunk_per_article_with_its_clauses(tmp_path):
    ps = _build(tmp_path, _law())
    arts = [c for c in _by(ps, "1.2") if c.kind == "article"]
    assert [c.title for c in arts] == ["第一条", "第二条", "第三条", "第四条", "第五条", "第六条"]
    second = arts[1]
    assert second.text.startswith("第二条 在中华人民共和国境内") and "境外开展数据处理活动" in second.text
    fifth = arts[4]
    assert "（一）处理的重要数据的种类、数量；" in fifth.text and "（二）面临的数据安全风险" in fifth.text
    assert "- 3 -" not in fifth.text                                  # 页码行去掉


def test_law_chapter_in_path_and_embed_text(tmp_path):
    ps = _build(tmp_path, _law())
    by_title = {c.title: c for c in _by(ps, "1.2") if c.kind == "article"}
    assert by_title["第一条"].path == ["第一章 总 则"]
    assert by_title["第四条"].path == ["第二章 数据安全制度"]
    assert by_title["第四条"].embed_text.startswith("《中华人民共和国数据安全法》 › 第二章 数据安全制度\n第四条")
    assert by_title["第一条"].anchors == ["P00009"]                    # Heading1=P1, Heading2=P2 … 第一条=P9


def test_toc_removed_and_body_chapters_kept(tmp_path):
    ps = _build(tmp_path, _law())
    pre = [c for c in _by(ps, "1.2") if c.kind == "preamble"]
    assert len(pre) == 1 and "目" not in pre[0].text and "第二章" not in pre[0].text
    assert pre[0].flags == ["title_only"]
    assert ps.dropped["toc"] == 3


def test_boilerplate_article_exported_but_not_indexed(tmp_path):
    ps = _build(tmp_path, _law())
    last = [c for c in _by(ps, "1.2") if c.title == "第六条"][0]
    assert last.flags == ["boilerplate"] and last.indexed is False
    assert all(c.indexed for c in _by(ps, "1.2") if c.title in ("第一条", "第四条"))


def test_doc_metadata(tmp_path):
    body = _law()
    body.insert(1, _p("发文机关：国家数据局　发文字号：国数政策〔2024〕12号　发布日期：2024年3月5日"))
    ps = _build(tmp_path, body)
    d = ps.docs[0]
    assert (d.doc_type, d.status, d.doc_number, d.date) == ("article", "现行", "国数政策〔2024〕12号", "2024-03-05")


def test_appendix_after_articles_is_chunked_by_items(tmp_path):
    body = _law()[:-1] + [
        _p("附件"),
        _p("个人信息保护合规审计指引"),
        _p("一、审计内容"),
        _p("（一）个人信息处理者处理个人信息的合法性基础，包括是否取得个人同意。"),
        _p("（二）个人信息处理规则的公开情况，包括处理目的、处理方式和保存期限。"),
    ]
    ps = _build(tmp_path, body)
    app = [c for c in _by(ps, "1.2") if c.kind == "appendix"]
    assert [c.title[:3] for c in app if c.indexed] == ["（一）", "（二）"]
    assert app[-1].path == ["一、审计内容"]
    assert not [c for c in _by(ps, "1.2") if c.kind == "article" and "附件" in c.text]


# ---------------------------------------------------------------- 事项式

def test_opinion_items_and_heading_paths(tmp_path):
    ps = _build(tmp_path, _opinion())
    items = [c for c in _by(ps, "2.50") if c.kind == "item"]
    assert [c.title[:4] for c in items] == ["（一）指", "（二）工", "（三）探", "（四）推"]
    assert items[0].path == ["一、总体要求"] and items[2].path == ["二、建立数据产权制度"]
    # 纯标题的（二）带着后面的「——」各条成一块
    assert items[1].text.startswith("（二）工作原则\n——遵循") and "——坚持共享共用" in items[1].text
    assert items[1].embed_text.startswith("《中共中央 国务院关于构建数据基础制度更好发挥数据要素作用的意见》 › 一、总体要求\n")
    assert ps.docs[0].doc_type == "opinion" and ps.docs[0].date == "2022-12-02"


def test_enumeration_after_colon_stays_in_item(tmp_path):
    ps = _build(tmp_path, _opinion())
    third = [c for c in _by(ps, "2.50") if c.title.startswith("（三）")][0]
    assert third.text.endswith("提交下列材料：\n（一）申请人基本情况；\n（二）数据来源说明。")
    assert [c.title[:3] for c in _by(ps, "2.50") if c.kind == "item"].count("（一）") == 1


def test_preamble_kept_title_only_preamble_not_indexed(tmp_path):
    ps = _build(tmp_path, _opinion())
    pre = [c for c in _by(ps, "2.50") if c.kind == "preamble"][0]
    assert "数据作为新型生产要素" in pre.text and pre.indexed
    ps2 = _build(tmp_path, [_p("2.56 《个人信息出境标准合同办法》", "Heading2"), _p("个人信息出境标准合同办法"),
                            *_law()[7:]], "b.docx")
    pre2 = [c for c in ps2.chunks if c.kind == "preamble"][0]
    assert pre2.flags == ["title_only"] and not pre2.indexed


def test_no_content_lost(tmp_path):
    ps = _build(tmp_path, _opinion())
    joined = _compact("".join(c.text for c in _by(ps, "2.50")))
    for line in ["数据作为新型生产要素", "以促进数据合规高效流通使用为主线", "合理降低市场主体获取数据的门槛",
                 "（二）数据来源说明。", "强化统筹授权使用和管理"]:
        assert _compact(line) in joined


# ---------------------------------------------------------------- 超长拆分 / 表格 / 清洗

def test_long_item_split_with_continuation_title(tmp_path):
    subs = []
    for k in range(1, 5):
        subs += [_p(f"{k}.推进第{k}项重点任务。"), _p(LONG_SENT * 10)]          # 每项约 400 字
    body = [_p("2.39 《“十四五”数字经济发展规划》", "Heading2"), _p("一、主要任务"),
            _p("（一）优化升级数字基础设施。"), *subs, _p("（二）充分发挥数据要素作用。"), _p(LONG_SENT)]
    ps = _build(tmp_path, body)
    first = [c for c in ps.chunks if c.title.startswith("（一）")]
    assert len(first) >= 2 and all(len(c.text) <= config.POLICY_CHUNK_MAX for c in first)
    assert [c.part for c in first] == list(range(1, len(first) + 1))
    assert first[0].text.startswith("（一）优化升级数字基础设施。")
    assert all(c.text.startswith("（一）优化升级数字基础设施。（续）\n") for c in first[1:])
    assert all(re.match(r"^\d\.", c.text.split("\n")[1]) for c in first[1:])   # 在「1.」「2.」处断开
    body_only = "".join(c.text.split("\n", 1)[1] for c in first[1:])
    assert _compact(first[0].text + body_only).count(_compact(LONG_SENT)) == 40


def test_tables_become_own_chunks(tmp_path):
    rows = [["序号", "违法行为", "裁量阶次"]] + [[str(k), "未建立数据安全管理制度" * 5, "从轻处罚"] for k in range(40)]
    body = [_p("2.37 《“十四五”大数据产业发展规划》", "Heading2"), _p("一、主要任务"),
            _p("（一）加快培育数据要素市场。建立数据要素价值体系，健全要素市场规则。"),
            _tbl([["专栏1 数据要素价值评估试点"]]), _tbl([["开展数据资产计价研究，建立数据要素价值评估框架。"]]),
            _p("（二）发挥大数据特性优势。推动数据要素与其他要素融合。"),
            _tbl(rows)]
    ps = _build(tmp_path, body)
    tables = [c for c in ps.chunks if c.kind == "table"]
    assert tables[0].text == "专栏1 数据要素价值评估试点\n开展数据资产计价研究，建立数据要素价值评估框架。"
    assert tables[0].path == ["一、主要任务", "（一）加快培育数据要素市场。建立数据要素价值体系，健全要素市场规则。"]
    big = tables[1:]
    assert len(big) > 1 and all(c.text.startswith("序号 | 违法行为 | 裁量阶次\n") for c in big)
    assert all(len(c.text) <= config.POLICY_CHUNK_MAX for c in big)
    # 表格块按原文位置排在所属事项之后、下一事项之前
    kinds = [(c.kind, c.title[:3]) for c in ps.chunks]
    assert kinds.index(("table", "（一）")) == kinds.index(("item", "（一）")) + 1


def test_hard_wrapped_pdf_text_is_rejoined_and_figure_text_dropped(tmp_path):
    wrapped = ["为贯彻落实中共中央、国务院《数字中国建设整体布局规", "划》文件精神，统筹做好民航大数据建设发展工作，指导行",
               "业全面推动数字化转型，现印发给你们，请认真贯彻落实。", "一、总体要求",
               "（一）指导思想。坚持以习近平新时代中国特色社会主义思想", "为指导，推动“一张图”", "建设，统筹发展和安全。"]
    body = [_p("2.57 《关于加快推动智慧民航建设发展的指导意见》", "Heading2"), *map(_p, wrapped),
            _p("二、"), _p("数字"), _p("产业"), _p("绿色")]
    ps = _build(tmp_path, body)
    pre = [c for c in ps.chunks if c.kind == "preamble"][0]
    assert pre.text == "为贯彻落实中共中央、国务院《数字中国建设整体布局规划》文件精神，统筹做好民航大数据建设发展工作，" \
                       "指导行业全面推动数字化转型，现印发给你们，请认真贯彻落实。"
    item = [c for c in ps.chunks if c.kind == "item"][0]
    assert item.text.endswith("推动“一张图”建设，统筹发展和安全。")         # 行尾的右引号不算句末
    assert ps.dropped["figure_text"] == 4 and not [c for c in ps.chunks if "产业" in c.text]


def test_unlabeled_toc_in_reports_is_dropped(tmp_path):
    body = [_p("2.92 《数字中国发展报告（2023年）》", "Heading2"),
            _p("2023年是深入实施数字中国战略的重要推动之年，数字中国建设取得重要进展。"),
            _p("一、数字中国发展基础更加夯实"), _p("（一）数据制度体系不断健全"), _p("（二）数字基础设施扩容提速"),
            _p("（三）网络安全法律法规和标准体系逐步健全.................29"),
            _p("（一）数据制度体系不断健全"), _p("数据基础制度建设加快推进，数据要素市场化改革步伐进一步加快。"),
            _p("（二）数字基础设施扩容提速"), _p("网络基础设施进入提速升级新阶段，5G网络覆盖面更广。")]
    ps = _build(tmp_path, body)
    items = [c for c in ps.chunks if c.kind == "item"]
    assert [c.text.split("\n")[0] for c in items] == ["（一）数据制度体系不断健全", "（二）数字基础设施扩容提速"]
    assert all("\n" in c.text for c in items) and ps.dropped["toc"] == 4
    assert ps.docs[0].doc_type == "reference"


# ---------------------------------------------------------------- 排除 / 去重

def test_excluded_titles_are_not_chunked(tmp_path):
    body = [*_law(), _p("2.32 《国土空间用途管制数据规范（试行）》", "Heading2"), _p("字段名称 | 字段类型")]
    ps = _build(tmp_path, body)
    d = [d for d in ps.docs if d.doc_no == "2.32"][0]
    assert d.excluded and d.n_chunks == 0 and not _by(ps, "2.32")


def test_duplicate_doc_keeps_better_structured_copy(tmp_path):
    # 同一文件的另一份拷贝：全文挤在一段里，没有分段结构（汇编 2.87 / 2.88 的实况）
    lines = re.findall(r"<w:t>(.*?)</w:t>", "".join(_opinion("2.88")[1:]))
    flat = [_p("2.87 《关于构建数据基础制度更好发挥数据要素作用的意见》", "Heading2"), _p("".join(lines))]
    ps = _build(tmp_path, [*flat, *_opinion("2.88")])
    d87 = [d for d in ps.docs if d.doc_no == "2.87"][0]
    assert d87.duplicate_of == "2.88" and d87.n_indexed == 0
    assert all("duplicate_of:2.88" in c.flags for c in _by(ps, "2.87"))
    assert [d for d in ps.docs if d.doc_no == "2.88"][0].n_indexed > 0


def _audit(no, title, extra=""):
    return [_p(f"{no} {title}", "Heading2")] + [
        _p(f"第{n}条　{t}{extra}") for n, t in zip("一二三四五六", [
            "为了规范个人信息保护合规审计活动，制定本办法。", "本办法所称合规审计，是指对个人信息处理活动进行审查评价。",
            "个人信息处理者应当定期开展合规审计。", "处理超过一千万人个人信息的，应当指定合规审计负责人。",
            "专业机构应当保持独立客观。", "保护部门应当加强对合规审计的监督检查。"])]


def test_draft_superseded_by_final_with_same_name(tmp_path):
    draft = _audit("2.60", "《个人信息保护合规审计管理办法（征求意见稿）》", "征求意见阶段表述有所不同，各方意见待汇总。" * 3)
    final = _audit("2.114", "个人信息保护合规审计管理办法")
    ps = _build(tmp_path, [*draft, *final])
    d60 = [d for d in ps.docs if d.doc_no == "2.60"][0]
    assert d60.status == "征求意见稿" and d60.superseded_by == "2.114" and d60.n_indexed == 0


def test_draft_not_superseded_by_different_policy_containing_its_name(tmp_path):
    draft = _audit("2.15", "数据安全管理办法（征求意见稿）")
    other = [_p("2.51 工业和信息化部关于印发《工业和信息化领域数据安全管理办法（试行）》的通知", "Heading2")] + [
        _p(f"第{n}条　工业和信息化领域数据处理者应当落实第{n}项数据安全保护要求，建立健全管理制度。")
        for n in "一二三四五六"]
    ps = _build(tmp_path, [*draft, *other])
    d15 = [d for d in ps.docs if d.doc_no == "2.15"][0]
    assert d15.superseded_by == "" and d15.n_indexed > 0


# ---------------------------------------------------------------- ID 与导出

def test_chunk_ids_unique_and_stable(tmp_path):
    a = _build(tmp_path, [*_law(), *_opinion()], "a.docx")
    b = _build(tmp_path, [*_law(), *_opinion()], "b.docx")
    ids = [c.chunk_id for c in a.chunks]
    assert len(ids) == len(set(ids)) and ids == [c.chunk_id for c in b.chunks]
    assert ids[0].startswith("pol:1.2:001:")


def test_stats_counts(tmp_path):
    st = policy_stats(_build(tmp_path, [*_law(), *_opinion()]))
    assert st["not_indexed"] == {"boilerplate": 1, "title_only": 1}   # 第六条施行日期；法律前言只有名称和通过日期
    assert st["indexed"] == st["chunks"] - 2 and st["over_max"] == 0


def test_cli_requires_existing_doc(tmp_path, capsys):
    from rag.policy import cli
    assert cli.main(["--dry-run", "--doc", str(tmp_path / "missing.docx")]) == 2
    assert "政策汇编文件不存在" in capsys.readouterr().err


def test_cli_dry_run_prints_chunk_count_article_count_and_region_distribution_and_writes_nothing(tmp_path, capsys):
    from rag.policy import cli
    src = _write(tmp_path, [_p("第1篇 国家立法", "Heading1"), *_law(), _p("第4篇 上海市数据相关政策", "Heading1"), *_opinion()], "汇编.docx")
    before = sorted(p.name for p in tmp_path.iterdir())
    assert cli.main(["--dry-run", "--doc", str(src)]) == 0
    out = capsys.readouterr().out
    assert "块总数：" in out and "条款块：" in out and "各地区块数：" in out and "国家" in out and "上海" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == before                 # 干跑不写任何文件


def test_cli_export_writes_jsonl_and_refuses_a_non_empty_directory(tmp_path):
    from rag.policy import cli
    src = _write(tmp_path, [_p("第1篇 国家立法", "Heading1"), *_law()], "汇编.docx")
    out = tmp_path / "out"
    assert cli.main(["export", "--doc", str(src), "--out", str(out)]) == 0
    chunks = [json.loads(x) for x in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks and {"chunk_id", "text", "region", "draft", "article"} <= set(chunks[0]) and (out / "stats.json").is_file()
    assert cli.main(["export", "--doc", str(src), "--out", str(out)]) == 2        # 非空目录：拒绝覆盖


def test_module_has_no_python_docx_dependency():
    src = open(policy.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(?:from|import)\s+docx\b", src, re.M)
