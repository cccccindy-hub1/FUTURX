"""课程本体加载 / 别名表 / 质量报告 / Top3 候选（M2）测试：全部离线，现造 xlsx、CSV、docx。"""
from __future__ import annotations

import csv
import zipfile
from xml.sax.saxutils import escape

import pytest
from openpyxl import Workbook

from rag.course import aliases as al
from rag.course import evidence, ontology, quality, relevance
from rag.course.textbook import parse_docx

_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_STYLES = (
    f'<w:styles {_NS}>'
    + "".join(f'<w:style w:type="paragraph" w:styleId="Heading{i}"><w:name w:val="heading {i}"/></w:style>' for i in range(1, 5))
    + "</w:styles>"
)

CONCEPT_HEADER = ["概念ID", "章号", "章节主题", "概念名称", "概念类型", "上位或关联项", "释义或知识语境", "收录依据",
                  "教材定位", "教材原文摘录", "专家原词", "专家定位", "归并及版本说明", "收录分组", "问题类型", "问题说明"]
CSV_HEADER = list(ontology._RECORD_STR_FIELDS[:12]) + ["source_file", "source_anchor", "retrieval_anchor", "source_section",
    "evidence_quote", "evidence_requirement", "logic_level", "flow_chain", "driving_mechanism", "confidence",
    "review_status", "normalization_note", "expert_relevance_weight"]


def _concept_row(cid, ch, name, expert="", issue="通过核验"):
    row = [""] * 16
    row[0], row[1], row[3], row[10], row[14] = cid, ch, name, expert, issue
    return row


def _write_concepts(path, rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "concept总表"
    ws.append(CONCEPT_HEADER)
    ws.append(["通过核验"] + [""] * 15)
    for r in rows:
        ws.append(r)
    wb.create_sheet("说明与来源")
    wb.save(path)


def _rec(rid, s, p, o, anchor="P0004", quote="", conf="0.9", w="1", ch="Ch01", slabel="", olabel=""):
    d = dict.fromkeys(CSV_HEADER, "")
    d.update(record_id=rid, chapter_id=ch, subject_id=s, subject_label=slabel or s, predicate_id=p,
             object_id=o, object_label=olabel or o, source_anchor=anchor, evidence_quote=quote,
             confidence=conf, expert_relevance_weight=w)
    return d


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)


def _docx(tmp_path):
    paras = [
        ("数据要素导论", "Heading1"),                      # P1
        ("第一章 数据要素", "Heading1"),                    # P2
        ("第一节 数据要素的概念", "Heading2"),              # P3 标题行
        ("数据要素是指参与价值创造的数据资源。数据权属决定归属。", None),  # P4
        ("数据资产是可计量的数据资源，需要入表。", None),   # P5
        ("", None),                                         # P6 空段
        ("本章讨论了若干问题。", None),                     # P7
        ("引用文献", None),                                 # P8 -> references
        ("[1]某某.数据的本质[J].2017.", None),              # P9
    ]
    body = ""
    for text, style in paras:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        run = f"<w:r><w:t>{escape(text)}</w:t></w:r>" if text else ""
        body += f"<w:p>{ppr}{run}</w:p>"
    path = tmp_path / "book.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document {_NS}><w:body>{body}</w:body></w:document>')
        z.writestr("word/styles.xml", _STYLES)
    return path


@pytest.fixture
def data(tmp_path):
    cx = tmp_path / "concepts.xlsx"
    _write_concepts(cx, [
        _concept_row("C01-0001", 1, "数据要素（Data Element）", "数据要素；要素化数据；数据要素的定义；Data Factor"),
        _concept_row("C01-0002", "1", "数据资产", "数据资产；数据资产的定义"),
        _concept_row("C01-0003", 1, "数据"),                     # 泛词
        _concept_row("C01-0004", 1, "数据权属"),
        _concept_row("C01-0005", 1, "孤立概念"),                  # 零提及
        _concept_row("C02-0001", 2, "数据资产"),                  # 跨章同名
        _concept_row("C02-0002", 2, "噪声词", issue="D 非概念噪声"),
    ])
    d1, d2 = tmp_path / "onto1", tmp_path / "onto2"
    d1.mkdir(); d2.mkdir()
    _write_csv(d1 / "Ch01_第1章_数据要素_onto1_9.11.csv", [
        _rec("R1", "C01-0001", "hasDefinition", "Ch01_DEF_AAAA", anchor="P0003", quote="数据要素是指参与价值创造"),
        _rec("R2", "C01-0002", "hasDefinition", "Ch01_DEF_BBBB", anchor="P0009", quote="某某"),
        _rec("R3", "C01-0001", "hasDefinition", "Ch01_DEF_CCCC", anchor="P0004", quote="数据要素是指参与价值创造的数据资源"),
        _rec("R4", "C01-9999", "relatedTo", "C01-0001", anchor="P0004"),   # 悬空 ID
        _rec("R5", "C01-0001", "relatedTo", "C01-0002", anchor="P0099"),   # 锚点越界
    ])
    _write_csv(d2 / "Ch01_第1章_数据要素_onto2_9.11.csv", [
        _rec("S1", "C01-0001", "dependsOn", "C01-0004", anchor="P0004", quote="数据权属决定归属", conf="0.8", w="0.8"),
        _rec("S2", "C01-0001", "relatedTo", "C01-0002", anchor="P0005", quote="数据资产是可计量的数据资源", conf="0.95", w="1"),
        _rec("S3", "C01-0001", "relatedTo", "C01-0003", anchor="onto2!A10", quote="", conf="0.99", w="1"),
        _rec("S4", "C01-0001", "relatedTo", "C01-0005", anchor="P0004", quote="完全不存在的引文内容", conf="0.7", w="0.6"),
        _rec("S5", "C01-0001", "supports", "C01-0002", anchor="P0004", quote="数据权属决定归属", conf="0.5", w="0.6"),  # 与 S2 同对
        _rec("S6", "C01-0001", "relatedTo", "C01-0001", anchor="P0004"),   # 自环
        _rec("S7", "C01-0001", "hasClass", "Class::Concept", anchor="P0004"),  # 非概念边
        _rec("S8", "C01-0002", "relatedTo", "C01-0001", anchor="P0005", conf="0.9", w="1"),  # 反向边
    ])
    return {"root": tmp_path, "concepts": cx, "onto1": d1, "onto2": d2, "book": parse_docx(_docx(tmp_path))}


# ---------------------------------------------------------------- 加载

def test_load_concepts_skips_placeholder_and_normalizes_chapter(data):
    t = ontology.load_concepts(data["concepts"])
    assert [c.id for c in t.concepts][:2] == ["C01-0001", "C01-0002"]
    assert len(t.concepts) == 7 and t.skipped == [(2, "通过核验")]
    assert {c.chapter for c in t.concepts} == {1, 2}  # "1" 与 1 都规整成 int
    assert t.by_id["C02-0002"].issue_type == "D 非概念噪声"


def test_load_concepts_requires_sheet_and_columns(tmp_path):
    wb = Workbook(); wb.active.title = "别的"; wb.save(tmp_path / "x.xlsx")
    with pytest.raises(ValueError, match="concept总表"):
        ontology.load_concepts(tmp_path / "x.xlsx")
    wb = Workbook(); wb.active.title = "concept总表"; wb.active.append(["概念ID"]); wb.save(tmp_path / "y.xlsx")
    with pytest.raises(ValueError, match="缺少列"):
        ontology.load_concepts(tmp_path / "y.xlsx")


def test_load_records_bom_and_types(data):
    rs = ontology.load_records(data["onto2"], "onto2")
    assert len(rs) == 8 and rs[0].view == "onto2"
    assert rs[0].subject_label == "C01-0001" and rs[0].confidence == 0.8 and rs[0].expert_relevance_weight == 0.8
    assert rs[0].is_concept_edge and not rs[6].is_concept_edge
    with pytest.raises(FileNotFoundError):
        ontology.load_records(data["onto2"], "onto1")


def test_load_records_rejects_missing_columns(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    (d / "Ch01_x_onto1_9.11.csv").write_text("record_id,chapter_id\nR,Ch01\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="缺少列"):
        ontology.load_records(d, "onto1")


def test_node_kind_and_anchor_classification():
    assert ontology.node_kind("C01-0001") == "concept"
    assert ontology.node_kind("Ch09_KP_abc") == "KP" and ontology.node_kind("Ch09_DEF_1F") == "DEF"
    assert ontology.node_kind("PolicyLibrary_Entry_01") == "PolicyEntry" and ontology.node_kind("Class::Concept") == "Class"
    assert ontology.node_kind("") == "empty" and ontology.node_kind("???") == "other"
    assert ontology.classify_anchor("P0062") == ("paragraph", 62)
    assert ontology.classify_anchor("onto1!A10")[0] == "expert_cell"
    assert ontology.classify_anchor("PolicyEntry_3|P0010")[0] == "policy"
    assert ontology.classify_anchor("C01-0001")[0] == "concept"


def test_load_student_set_finds_header_row(tmp_path):
    wb = Workbook(); ws = wb.active; ws.title = "学生知识集合"
    ws.append(["标题"]); ws.append(["说明"]); ws.append([])
    ws.append(["集合ID", "来源编号", "主体"]); ws.append(["S1", "R1", "甲"]); ws.append([None, None, None])
    wb.save(tmp_path / "s.xlsx")
    assert ontology.load_student_set(tmp_path / "s.xlsx") == [{"集合ID": "S1", "来源编号": "R1", "主体": "甲"}]


# ---------------------------------------------------------------- 别名

def test_split_expert_terms_excludes_definition_entries():
    assert al.split_expert_terms("数据要素；数据要素的定义；；要素化数据；数据要素") == ["数据要素", "要素化数据"]
    assert al.split_expert_terms("甲;乙的定义;数据定义") == ["甲", "数据定义"]  # 只排除「XX的定义」


def test_split_name_extracts_english_only():
    assert al.split_name("数据要素（Data Element）") == ("数据要素", ["Data Element"])
    assert al.split_name("数据确权（简称确权）") == ("数据确权", [])


def test_alias_table_sources_and_blocking(data):
    cs = ontology.load_concepts(data["concepts"]).concepts
    t = al.build_alias_table(cs)
    assert t.lookup("数据要素") == ["C01-0001"] and t.lookup("data element") == ["C01-0001"]
    assert t.lookup("Data Factor") == ["C01-0001"] and t.lookup("要素化数据") == ["C01-0001"]
    assert t.lookup("数据要素的定义") == []                       # 排除项
    assert t.lookup("数据") == [] and t.lookup("数据", include_blocked=True) == ["C01-0003"]
    assert t.by_key[al.norm("数据")].blocked == "generic"
    assert t.lookup("噪声词") == []                                # 噪声概念不建别名
    assert t.lookup("数据资产") == ["C01-0002", "C02-0001"] and t.by_key[al.norm("数据资产")].ambiguous


@pytest.mark.parametrize("s,reason", [("技术", "generic"), ("链", "single_char"), ("AI", "short_ascii"), ("数据要素", None)])
def test_blocked_reason(s, reason):
    assert al.blocked_reason(s) == reason


def test_count_mentions_lists_all_ambiguous_candidates(data):
    t = al.build_alias_table(ontology.load_concepts(data["concepts"]).concepts)
    m = t.count_mentions("数据，资产。数据资产很重要，数据要素也是；DATA ELEMENT。")
    assert m["C01-0002"] == m["C02-0001"] == 1 and m["C01-0001"] == 2 and "C01-0003" not in m


# ---------------------------------------------------------------- 证据

def test_anchor_status(data):
    ix = evidence.AnchorIndex(data["book"])
    assert ix.n_paragraphs == 9
    assert ix.info(3).status == evidence.TITLE_ROW and ix.info(2).status == evidence.TITLE_ROW
    assert ix.info(1).status == evidence.FRONT and ix.info(4).status == evidence.BODY
    assert ix.info(6).status == evidence.EMPTY and ix.info(9).status == evidence.REFERENCES
    assert ix.info(10).status == evidence.OUT_OF_RANGE and ix.info(0).status == evidence.OUT_OF_RANGE


def test_toc_anchor_status_and_resolution(tmp_path):
    from rag.course.textbook import parse_docx as parse

    def p(text, style=None):
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{ppr}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"

    # 目录行出现在上一章「引用文献」之后、下一章标题之前
    body = "".join([
        p("第一章 甲", "Heading1"), p("正文一。"), p("引用文献"), p("[1]某。"),             # P1-P4
        p("一、核心环节"),                                                                   # P5 toc
        p("第二章 乙", "Heading1"), p("第一节 概述", "Heading2"),                            # P6-P7
        p("一、核心环节", "Heading3"), p("定义正文。"),                                       # P8-P9
    ])
    path = tmp_path / "t.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document {_NS}><w:body>{body}</w:body></w:document>')
        z.writestr("word/styles.xml", _STYLES)
    ix = evidence.AnchorIndex(parse(path))
    assert ix.info(4).status == evidence.REFERENCES and ix.info(5).status == evidence.TOC
    assert ix.resolve_toc(5).anchor == "P0008"        # 同名正文标题
    assert ix.resolve_toc(4) is None and ix.resolve_toc(9) is None


def test_node_kind_generic_hash_nodes():
    assert ontology.node_kind("Ch09_N_3fa91c02") == "Node"
    assert ontology.node_kind("Ch09_KP_x") == "KP"      # 具名前缀优先
    assert ontology.node_kind("Chapter9") == "other"


def test_node_kind_structure_nodes():
    assert ontology.node_kind("Ch01_Chapter") == "Chapter"
    assert ontology.node_kind("Course_DataElementCirculationIntro") == "Course"
    assert ontology.node_kind("Ch01_Chapters") == "other"


def test_composite_quote_verified_per_segment(data):
    ix = evidence.AnchorIndex(data["book"])
    # precedes 边：两个相邻标题用「；」拼接，拼接串本身不在书中，但每段都能逐字找到
    q = "第一节 数据要素的概念；数据资产是可计量的数据资源"
    assert ix.check_quote("P0003", q) == evidence.VERIFIED_COMPOSITE
    assert ix.check_quote("P0003", "第一节 数据要素的概念；根本不存在的标题") == evidence.NOT_FOUND
    assert ix.check_quote("P0003", "不存在甲；") == evidence.NOT_FOUND     # 拆出的有效段不足 2 段，不按复合处理


def test_composite_does_not_override_plain_verified(data):
    ix = evidence.AnchorIndex(data["book"])
    assert ix.check_quote("P0004", "数据要素是指参与价值创造的数据资源。数据权属决定归属") == evidence.VERIFIED


def test_composite_counted_separately_and_not_listed_as_not_found(data):
    ix = evidence.AnchorIndex(data["book"])
    mk = lambda rid, quote: ontology.Record(  # noqa: E731
        **{**_blank_record(), "record_id": rid, "view": "onto2", "source_anchor": "P0003", "evidence_quote": quote})
    recs = [mk("Y1", "第一节 数据要素的概念；数据资产是可计量的数据资源"), mk("Y2", "没有的标题；也没有的标题")]
    assert [x["record_id"] for x in quality.not_found_samples(recs, ix, "onto2")] == ["Y2"]
    cs = ontology.load_concepts(data["concepts"])
    rep = quality.build_report(cs, recs, al.build_alias_table(cs.concepts), book=data["book"])
    assert rep.quote_check["onto2"] == {evidence.VERIFIED_COMPOSITE: 1, evidence.NOT_FOUND: 1}
    assert "verified_composite=1" in rep.to_markdown()


def test_not_found_samples_distinguish_shift_from_rewrite(data):
    ix = evidence.AnchorIndex(data["book"])
    recs = [
        # 引文在 P0005，锚点写成 P0007：偏移，全书能找到
        ontology.Record(**{**_blank_record(), "record_id": "X1", "view": "onto2", "source_anchor": "P0007",
                           "evidence_quote": "数据资产是可计量的数据资源"}),
        # 引文被改写：全书找不到
        ontology.Record(**{**_blank_record(), "record_id": "X2", "view": "onto2", "source_anchor": "P0004",
                           "evidence_quote": "这句话教材里根本没有"}),
    ]
    got = quality.not_found_samples(recs, ix, "onto2")
    assert [x["record_id"] for x in got] == ["X1", "X2"]
    assert got[0]["found_in_book"] and got[0]["found_at"] == ["P0005"] and got[0]["offset"] == -2
    assert not got[1]["found_in_book"] and got[1]["offset"] is None
    assert got[1]["quote_head"].startswith("这句话") and got[0]["anchor_text_head"] == "本章讨论了若干问题。"


def _blank_record() -> dict:
    d = {f: "" for f in ontology._RECORD_STR_FIELDS}
    d.update(confidence=None, expert_relevance_weight=None, view="")
    return d


def test_check_quote_variants(data):
    ix = evidence.AnchorIndex(data["book"])
    assert ix.check_quote("P0004", "数据要素 是指参与价值创造") == evidence.VERIFIED
    assert ix.check_quote("P0004", "数据要素是指……数据权属决定归属") == evidence.VERIFIED        # 省略号分片
    assert ix.check_quote("P0003", "数据资产是可计量的数据资源") == evidence.VERIFIED_NEARBY      # 锚点偏前
    assert ix.check_quote("P0004", "完全不存在的引文内容") == evidence.NOT_FOUND
    assert ix.check_quote("P0004", "") == evidence.NO_QUOTE
    assert ix.check_quote("onto2!A10", "数据") == evidence.NOT_PARAGRAPH


# ---------------------------------------------------------------- 候选

@pytest.fixture
def cands(data):
    recs = ontology.load_records(data["onto2"], "onto2")
    return quality.collect_candidates(recs, view="onto2")


def test_candidates_are_directed_concept_edges_only(cands):
    got = {c.object_id for c in cands["C01-0001"]}
    assert got == {"C01-0004", "C01-0002", "C01-0003", "C01-0005"}   # 无自环、无 Class 节点
    assert [c.object_id for c in cands["C01-0002"]] == ["C01-0001"]   # 反向边是独立候选
    assert "C01-0004" not in cands                                    # 被指向 ≠ 有出边


def test_candidates_merge_edges_and_keep_attributes(cands):
    c = next(c for c in cands["C01-0001"] if c.object_id == "C01-0002")
    assert {e.predicate_id for e in c.edges} == {"relatedTo", "supports"}
    e = next(e for e in c.edges if e.predicate_id == "relatedTo")
    assert (e.confidence, e.expert_weight, e.source_anchor) == (0.95, 1.0, "P0005")


def test_candidates_known_filter_drops_dangling(data):
    recs = ontology.load_records(data["onto1"], "onto1") + ontology.load_records(data["onto2"], "onto2")
    all_c = quality.collect_candidates(recs, view=None)
    assert "C01-9999" in all_c
    known = set(ontology.load_concepts(data["concepts"]).by_id)
    assert "C01-9999" not in quality.collect_candidates(recs, view=None, known=known)


# ---------------------------------------------------------------- 排序方案

def test_scheme_a_orders_by_weight_then_confidence(cands, data):
    top = relevance.rank(cands["C01-0001"], relevance.scheme_a, None, 3)
    assert [s.candidate.object_id for s in top] == ["C01-0003", "C01-0002", "C01-0004"]  # 0.99 > 0.95（均 w=1），再 w=0.8


def test_scheme_b_demotes_unverifiable_evidence(cands, data):
    ix = evidence.AnchorIndex(data["book"])
    top = relevance.rank(cands["C01-0001"], relevance.scheme_b, ix, 4)
    ids = [s.candidate.object_id for s in top]
    assert ids[:2] == ["C01-0002", "C01-0004"] and ids[-1] == "C01-0005"  # 引文可核实的在前；引文找不到的垫底
    assert relevance.rank(cands["C01-0001"], relevance.scheme_a, ix, 4)[0].candidate.object_id == "C01-0003"


def test_scheme_c_uses_predicate_prior(cands, data):
    ix = evidence.AnchorIndex(data["book"])
    neutral = relevance.make_scheme_c()
    boosted = relevance.make_scheme_c(predicate_prior={"dependsOn": 1.0, "relatedTo": 0.0, "supports": 0.0})
    pick = lambda fn: relevance.rank(cands["C01-0001"], fn, ix, 1)[0].candidate.object_id  # noqa: E731
    assert pick(boosted) == "C01-0004" and pick(neutral) != "C01-0004"


def test_rank_is_deterministic_on_ties():
    e = quality.Edge("r", "p", "p", 0.9, 1.0, "P1", "", "", "", "", "q")
    cs = [quality.Candidate("C01-0001", o, [e]) for o in ("C01-0009", "C01-0003", "C01-0005")]
    out = [s.candidate.object_id for s in relevance.rank(cs, relevance.scheme_a, None, 3)]
    assert out == ["C01-0003", "C01-0005", "C01-0009"]
    assert relevance.tie_at_cutoff(cs + [quality.Candidate("C01-0001", "C01-0011", [e])], relevance.scheme_a)


def test_compare_schemes_reports_overlap_and_ties(cands, data):
    ix = evidence.AnchorIndex(data["book"])
    out = relevance.compare_schemes(cands, ix, k=2)
    assert out["concepts_with_more_than_k"] == 1 and set(out["ties_at_cutoff"]) == {"A", "B", "C"}
    assert set(out["topk_overlap"]) == {"A-B", "A-C", "B-C"}


# ---------------------------------------------------------------- 质量报告

@pytest.fixture
def report(data):
    cs = ontology.load_concepts(data["concepts"])
    recs = ontology.load_records(data["onto1"], "onto1") + ontology.load_records(data["onto2"], "onto2")
    return quality.build_report(cs, recs, al.build_alias_table(cs.concepts), book=data["book"],
                                student_rows=[{"来源编号": "R1"}, {"来源编号": "Ch01_O2_911_R99999"}, {"来源编号": "Ch1-O2-001"}])


def test_reference_validation(report):
    iss = report.reference_issues
    assert iss[quality.UNKNOWN_CONCEPT] == 1 and iss[quality.ANCHOR_OUT_OF_RANGE] == 1
    assert quality.DUPLICATE_RECORD_ID not in iss
    assert report.reference_samples[quality.UNKNOWN_CONCEPT][0]["record_id"] == "R4"


def test_definition_anchor_audit(report):
    d = report.definition_anchor["onto1"]
    assert d["total"] == 3 and d[evidence.TITLE_ROW] == 1 and d[evidence.REFERENCES] == 1 and d[evidence.BODY] == 1
    assert [x["record_id"] for x in report.definition_title_row] == ["R1"]
    assert [x["record_id"] for x in report.definition_references] == ["R2"]


def test_cross_chapter_names_and_issue_types(report):
    (g,) = report.cross_chapter_names
    assert g["name"] == "数据资产" and g["ids"] == ["C01-0002", "C02-0001"] and g["chapters"] == [1, 2]
    assert report.issue_types["D 非概念噪声"] == 1


def test_zero_mention(report):
    z = report.zero_mention
    assert "C01-0005" in z["ids"] and "C01-0001" not in z["ids"] and "C02-0002" not in z["ids"]  # 噪声概念不计
    assert z["no_matchable_alias"] == ["C01-0003"]                                              # 唯一别名是泛词


def test_top3_coverage(report):
    t = report.top3_coverage
    assert t["concepts_considered"] == 6                   # 排除噪声概念
    assert t["out_degree_distribution"] == {"0": 4, "1": 1, "4+": 1}
    assert t["with_at_least_3"] == 1 and t["fewer_than_3"] == 5
    assert t["no_outgoing_but_has_incoming"] == 3          # 0003、0004、0005 只有入边
    assert t["reverse_pairs_present"] == {"directed_pairs": 5, "bidirectional_pairs": 1}


def test_student_set_check(report):
    assert report.student_set["matched_record_id"] == 1 and report.student_set["machine_id_not_found"] == 1
    assert report.student_set["expert_style_id"] == 1


def test_report_renders_markdown_and_json(report):
    md = report.to_markdown()
    for h in ("ID 引用校验", "hasDefinition 证据落点", "跨章同名概念", "零提及概念", "Top3 候选覆盖"):
        assert h in md
    assert "锚在标题行：1 条；锚在参考文献：1 条" in md
    import json
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_report_without_textbook_skips_anchor_checks(data):
    cs = ontology.load_concepts(data["concepts"])
    recs = ontology.load_records(data["onto2"], "onto2")
    rep = quality.build_report(cs, recs, al.build_alias_table(cs.concepts), book=None)
    assert rep.anchor_status == {} and rep.zero_mention == {} and rep.top3_coverage["candidate_pairs"] > 0


# ---------------------------------------------------------------- 隔离

def test_onto_root_default_inside_course_data_not_store():
    import rag.config as base
    from rag.course import config as course

    assert base.STORE_DIR not in course.COURSE_ONTO_ROOT.parents


# ---------------------------------------------------------------- 11.2 版目录结构

def test_paths_detect_11_2_layout(tmp_path):
    for d in ("onto1", "onto2", "三层集合"):
        (tmp_path / d).mkdir()
    for name in ("第1-10章_concept总表_问题标注版.xlsx", "第1-11章_concept总表_问题标注版.xlsx", "书稿（1102合成版）.docx"):
        (tmp_path / name).write_bytes(b"")
    p = ontology.OntologyPaths(tmp_path)
    assert p.layout == "11.2" and p.onto_dir("onto1") == tmp_path / "onto1"
    assert p.concept_xlsx.name == "第1-11章_concept总表_问题标注版.xlsx"        # 取章数最多的那份
    assert p.student_xlsx == tmp_path / "三层集合" / "学生知识集合.xlsx"
    assert p.find_textbook().name == "书稿（1102合成版）.docx"


def test_paths_keep_9_11_layout(tmp_path):
    (tmp_path / "9.11update" / "onto1").mkdir(parents=True)
    p = ontology.OntologyPaths(tmp_path)
    assert p.layout == "9.11" and p.onto_dir("onto2") == tmp_path / "9.11update" / "onto2"
    assert p.concept_xlsx.name == "第1-10章_concept总表_问题标注版.xlsx"         # 文件不存在时仍给出 9.11 文件名，由加载处报错
    assert p.student_xlsx == tmp_path / "9.11update" / "三层集合" / "学生知识集合.xlsx"


def test_load_records_reads_11_2_files_and_rejects_mixed_versions(data, tmp_path):
    d = tmp_path / "v112"; d.mkdir()
    src = next(data["onto1"].glob("*.csv"))
    (d / "Ch04_第4章_数据政策_onto1_11.2.csv").write_bytes(src.read_bytes())
    assert len(ontology.load_records(d, "onto1")) == len(ontology.load_records(data["onto1"], "onto1"))
    (d / "Ch01_第1章_数据要素_onto1_9.11.csv").write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="多个版本"):
        ontology.load_records(d, "onto1")
