"""判例库、名言库的解析与切块单元测试：全部用 python-docx 现场合成的小文档，不依赖真实资料，离线可复现。"""
from __future__ import annotations

import json

import pytest

docx = pytest.importorskip("docx")
from docx.enum.style import WD_STYLE_TYPE

from rag.cases import chunker as cc
from rag.cases import cli as case_cli
from rag.cases import config as case_cfg
from rag.cases.parser import chapter_number, parse_cases
from rag.quotes import cli as quote_cli
from rag.quotes.chunker import chunk_quotes
from rag.quotes.parser import classify_kind, parse_quotes


# ---------------------------------------------------------------- 合成判例库 docx

def _case_doc(path, spec):
    """spec: [(kind, style, text)]；kind 'p' 段落、't' 表格（text 为二维列表）。"""
    d = docx.Document()
    for name in ("CaseMetadata", "CaseSubheading", "CaseBody", "CaseTable", "CaseTOCChapter", "CaseTOCEntry"):
        if name not in [s.name for s in d.styles]:
            d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    for kind, style, text in spec:
        if kind == "p":
            d.add_paragraph(text, style=style)
        else:
            t = d.add_table(rows=len(text), cols=len(text[0]))
            for i, row in enumerate(text):
                for j, c in enumerate(row):
                    t.cell(i, j).text = c
    d.save(str(path))
    return path


def P(style, text):
    return ("p", style, text)


LONG = "这是一段用来凑字数的法院认定事实的内容，需要足够长才能触发切块。" * 30        # 约 900 字


def base_spec():
    return [
        P("Title", "数据法典型判例库"),
        P("Heading 1", "目录"),
        P("CaseTOCChapter", "第一章 数据要素"),
        P("CaseTOCEntry", "1. 目录里的条目不能被当成判例"),
        P("Heading 1", "第一章 数据要素"),
        # 案1：有要点（CaseSubheading「裁判要点」），元数据齐全
        P("Heading 2", "1. 甲公司诉乙公司不正当竞争纠纷案"),
        P("CaseMetadata", "文本类型：裁判文书"),
        P("CaseMetadata", "案号：（2021）浙8601民初309号"),
        P("CaseMetadata", "审理法院：杭州铁路运输法院"),
        P("CaseMetadata", "裁判日期：2021-09-15"),
        P("CaseSubheading", "裁判要点"),
        P("CaseBody", "数据权益应当受到保护，未经许可抓取他人数据构成不正当竞争。"),
        P("CaseSubheading", "基本案情"),
        P("CaseBody", LONG),
        P("CaseSubheading", "裁判理由"),
        P("CaseBody", LONG),
        P("CaseSubheading", "典型意义"),
        P("CaseBody", "本案明确了平台数据的保护边界，对类案有指引价值。"),
        P("Heading 1", "第二章 数据权属"),
        # 案2：无要点，「本院认为」在段首
        P("Heading 2", "2. 丙与丁合同纠纷二审民事判决书"),
        P("CaseMetadata", "案号：（2022）京01民终1号"),
        P("CaseBody", "原告诉称：请求判令被告支付货款。" + "背景。" * 50),
        P("CaseBody", "本院认为，合同合法有效，被告应当履行付款义务。" + "说理。" * 80),
        P("CaseBody", "判决如下：被告于判决生效之日起十日内支付货款。"),
        # 案3：无要点，也没有「本院认为」，有「判决如下」
        P("Heading 2", "3. 戊诉己侵权案"),
        P("CaseBody", "事实经过。" * 200),
        P("CaseBody", "判决如下：驳回原告全部诉讼请求。"),
        # 案4：只有一个很短的小节、含表格
        P("Heading 2", "4. 庚案"),
        P("CaseSubheading", "案件基本信息"),
        ("t", None, [["案号", "（2023）沪01民终9号"], ["审理法院", "上海一中院"]]),
        P("CaseBody", "简短的案情。"),
        # 附录：不入库
        P("Heading 1", "附录一 最高人民法院关于发布第47批指导性案例的通知"),
        P("Heading 2", "附录里的 Heading 2 也不能算判例"),
        P("CaseBody", "附录正文。"),
    ]


@pytest.fixture
def case_doc(tmp_path):
    return _case_doc(tmp_path / "cases.docx", base_spec())


# ---------------------------------------------------------------- 判例解析

def test_parse_cases_counts_chapters_and_skips_toc_and_appendix(case_doc):
    cs = parse_cases(case_doc)
    assert [c.seq for c in cs] == [1, 2, 3, 4]                       # 目录条目、附录都不是判例
    assert [c.chapter for c in cs] == ["第一章 数据要素", "第二章 数据权属", "第二章 数据权属", "第二章 数据权属"]
    assert [c.chapter_no for c in cs] == [1, 2, 2, 2]
    assert cs[0].title == "甲公司诉乙公司不正当竞争纠纷案"                # 序号被去掉
    assert all("附录" not in c.title for c in cs)


def test_parse_metadata_aliases_and_missing_fields(case_doc):
    cs = parse_cases(case_doc)
    assert cs[0].meta == {"text_type": "裁判文书", "case_number": "（2021）浙8601民初309号", "court": "杭州铁路运输法院",
                          "judgment_date": "2021-09-15"}
    assert cs[1].meta == {"case_number": "（2022）京01民终1号"}          # 各案字段不全
    assert cs[2].meta == {} and cs[0].meta_raw[0].startswith("文本类型")


def test_parse_keeps_block_kinds_in_document_order_and_tables(case_doc):
    c4 = parse_cases(case_doc)[3]
    assert [b.kind for b in c4.blocks] == ["subheading", "table", "body"]
    assert "案号｜（2023）沪01民终9号" in c4.blocks[1].text and "审理法院｜上海一中院" in c4.blocks[1].text


def test_chapter_number_handles_chinese_numerals():
    assert [chapter_number(f"第{n}章 x") for n in "一二三四五六七八九十"] == list(range(1, 11))
    assert chapter_number("第十一章 x") == 11 and chapter_number("附录一") == 0


# ---------------------------------------------------------------- 判例切块：卡片

def _chunks(case_doc):
    cs = parse_cases(case_doc)
    return cs, cc.chunk_cases(cs)


def test_each_case_gets_exactly_one_card_and_card_has_prefix_and_metadata(case_doc):
    cs, ch = _chunks(case_doc)
    cards = [k for k in ch if k.kind == "card"]
    assert [k.seq for k in cards] == [1, 2, 3, 4]
    c1 = cards[0]
    assert c1.text.startswith("【第一章 数据要素】甲公司诉乙公司不正当竞争纠纷案")
    assert "案号：（2021）浙8601民初309号" in c1.text and "审理法院：杭州铁路运输法院" in c1.text and "裁判日期：2021-09-15" in c1.text
    assert c1.case_id == "case:01" and c1.chunk_id.startswith("case:01:card:")


def test_card_uses_points_sections_when_present(case_doc):
    c1 = _chunks(case_doc)[1][0]
    assert c1.card_source == "points"
    assert "裁判要点：数据权益应当受到保护" in c1.text and "典型意义：本案明确了平台数据的保护边界" in c1.text
    assert "这是一段用来凑字数" not in c1.text                         # 基本案情、裁判理由不进卡片


def test_card_falls_back_to_court_opinion_paragraph_start(case_doc):
    cards = {k.seq: k for k in _chunks(case_doc)[1] if k.kind == "card"}
    c2 = cards[2]
    assert c2.card_source == "court_opinion"
    assert "本院认为，合同合法有效" in c2.text and "原告诉称" not in c2.text    # 从「本院认为」那一段开始，不含前面的诉称
    assert c2.body_chars <= case_cfg.CASE_CARD_FALLBACK_CHARS + 120


def test_card_falls_back_to_text_before_judgment_when_no_court_opinion(case_doc):
    c3 = {k.seq: k for k in _chunks(case_doc)[1] if k.kind == "card"}[3]
    assert c3.card_source == "before_judgment"
    assert c3.body_chars <= case_cfg.CASE_CARD_FALLBACK_CHARS and "判决如下" not in c3.text    # 取「判决如下」之前的约 800 字


def test_card_last_resort_is_lead_of_body():
    c = parse_cases.__globals__["ParsedCase"](seq=9, chapter="第一章 数据要素", chapter_no=1, title="无任何标记的案")
    c.blocks = [parse_cases.__globals__["Block"]("body", "只有一段普通事实。" * 5)]
    k = cc.build_card(c)
    assert k.card_source == "lead" and "只有一段普通事实" in k.text


@pytest.mark.parametrize("text,ok", [
    ("本院认为，被告构成不正当竞争", True), ("本院经审理认为：合同应当解除", True), ("二审法院认为，随着数据价值提升", True),
    ("一审法院审理后认为，原告主张成立", True), ("本院结合双方诉辩请求，认为争议焦点如下", True),
    ("原告认为被告侵权", False), ("二被告提交了证据证明原告违规", False), ("杭州市中级人民法院经审理认为", False)])
def test_court_opinion_regex(text, ok):
    assert bool(cc.COURT_OPINION_RE.match(text)) is ok


@pytest.mark.parametrize("text,expected", [
    ("裁判要点", ("裁判要点", "")), ("【裁判要旨】", ("裁判要旨", "")), ("[裁判要旨]", ("裁判要旨", "")), ("〖裁判要旨〗", ("裁判要旨", "")),
    ("【典型意义】本案为典型案例。", ("典型意义", "本案为典型案例。")),                  # 标题与正文同一段
    ("裁判要点：企业数据受保护。", ("裁判要点", "企业数据受保护。")),
    ("执行实施要点", ("执行实施要点", "")),
    ("典型意义重大，值得关注。", None),                                                   # 普通句子，不是标题
    ("本案的裁判要点如下", None)])
def test_point_heading_detection(text, expected):
    assert cc._point_heading(text) == expected


def test_points_section_ends_at_next_heading_or_bracket_marker():
    from rag.cases.parser import Block, ParsedCase
    c = ParsedCase(seq=1, chapter="第一章 数据要素", chapter_no=1, title="案")
    c.blocks = [Block("body", "[裁判要旨]"), Block("body", "要旨正文一。"), Block("body", "要旨正文二。"),
                Block("body", "[相关法条]"), Block("body", "法条内容不能进要点。")]
    assert cc._point_sections(c) == [("裁判要旨", "要旨正文一。\n要旨正文二。")]


# ---------------------------------------------------------------- 判例切块：正文块

def test_body_chunks_carry_case_and_chapter_prefix_and_respect_size(case_doc, monkeypatch):
    monkeypatch.setattr(case_cfg, "CASE_BODY_MIN", 800)
    monkeypatch.setattr(case_cfg, "CASE_BODY_MAX", 1200)
    cs, ch = _chunks(case_doc)
    bodies = [k for k in ch if k.kind == "body"]
    assert bodies and all(k.text.startswith(f"【{k.chapter}】{k.title}\n") for k in bodies)       # 每块都带「案名 + 章节标签」
    assert all(k.body_chars <= 1200 + 40 for k in bodies)                                          # 加上节名标记后允许略超
    assert [k.part for k in bodies if k.seq == 1] == list(range(1, len([k for k in bodies if k.seq == 1]) + 1))
    assert all(k.case_id == f"case:{k.seq:02d}" and k.chunk_id.startswith(f"case:{k.seq:02d}:body") for k in bodies)


def test_long_paragraph_is_split_by_sentences_not_dropped(monkeypatch):
    from rag.cases.parser import Block, ParsedCase
    monkeypatch.setattr(case_cfg, "CASE_BODY_MAX", 300)
    c = ParsedCase(seq=5, chapter="第六章 数据市场", chapter_no=6, title="长段案")
    c.blocks = [Block("body", "第一句话在这里。" * 100)]
    bodies = cc.build_bodies(c)
    assert len(bodies) >= 3 and all(k.body_chars <= 340 for k in bodies)
    assert "".join(k.text.split("\n", 1)[1].replace("\n", "") for k in bodies).count("第一句话在这里。") == 100   # 一句不丢


def test_short_sections_merge_with_neighbours():
    from rag.cases.parser import Block, ParsedCase
    c = ParsedCase(seq=6, chapter="第一章 数据要素", chapter_no=1, title="多小节案")
    c.blocks = [Block("subheading", "基本案情"), Block("body", "案情。" * 30),
                Block("subheading", "裁判结果"), Block("body", "结果。" * 30),
                Block("subheading", "相关法条"), Block("body", "法条。" * 10)]
    bodies = cc.build_bodies(c)
    assert len(bodies) == 1 and "【基本案情】" in bodies[0].text and "【裁判结果】" in bodies[0].text   # 三个短节合并成一块


def test_tables_are_included_in_body_text(case_doc):
    k4 = [k for k in _chunks(case_doc)[1] if k.seq == 4 and k.kind == "body"]
    assert any("审理法院｜上海一中院" in k.text for k in k4)


def test_chunk_ids_unique_and_deterministic(case_doc):
    a = [k.chunk_id for k in _chunks(case_doc)[1]]
    b = [k.chunk_id for k in cc.chunk_cases(parse_cases(case_doc))]
    assert a == b and len(a) == len(set(a))


# ---------------------------------------------------------------- 判例 CLI / 统计

def test_case_stats_report_totals_chapters_and_cards_without_points(case_doc, tmp_path):
    stats = case_cli.export_cases(case_doc, tmp_path / "out")
    assert stats["cases"] == 4 and stats["expected_cases"] == 47 and stats["per_chapter"] == {"第一章 数据要素": 1, "第二章 数据权属": 3}
    assert stats["cards"] == 4 and stats["card_source"] == {"points": 1, "court_opinion": 1, "before_judgment": 1, "lead": 1}
    np = {r["seq"]: r["card_source"] for r in stats["without_points"]}
    assert set(np) == {2, 3, 4} and np[2] == "court_opinion" and np[3] == "before_judgment"      # 没有要点的案子及其卡片取自哪里
    assert stats["appendix_excluded"] is True and stats["tables_in_cases"] == 1
    rows = [json.loads(l) for l in (tmp_path / "out" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == stats["chunks"] and {r["kind"] for r in rows} == {"card", "body"}
    assert len((tmp_path / "out" / "cases.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_case_cli_missing_doc_exits_with_message(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "chunk", "--doc", str(tmp_path / "nope.docx")])
    with pytest.raises(SystemExit) as e:
        case_cli.main()
    assert "判例库文件不存在" in str(e.value)


# ---------------------------------------------------------------- 合成名言库 docx

def _quote_doc(path, toc, body):
    """toc: [(年份, [标题…])]；body: [(年份, [段落…])]，段落以 N. 开头则开始新讲话。"""
    d = docx.Document()
    for name in ("toc 1", "toc 2"):
        if name not in [s.name for s in d.styles]:
            d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    d.add_paragraph("习近平总书记有关数智技术相关内容的表述合集")
    d.add_paragraph("一、检索范围", style="Heading 1")
    d.add_paragraph("目  录")
    for year, titles in toc:
        d.add_paragraph(f"{year}年\t3", style="toc 1")
        for n, t in enumerate(titles, 1):
            d.add_paragraph(f"{n}.{t}\t4", style="toc 2")
    d.add_paragraph("三、检索结果", style="Heading 1")
    for year, paras in body:
        d.add_paragraph(f"{year}年", style="Heading 1")
        for p in paras:
            d.add_paragraph(p)
    d.save(str(path))
    return path


@pytest.fixture
def quote_doc(tmp_path):
    toc = [(2016, ["网络安全和信息化工作座谈会上的讲话", "第三届世界互联网大会开幕式上的视频讲话"]),
           (2018, ["致首届数字中国建设峰会的贺信"]),
           (2020, ["考察杭州城市大脑运营指挥中心作出重要指示"]),
           (2022, ["习近平.不断做强做优做大我国数字经济"])]
    body = [(2016, ["1.第一次讲话第一段，讲网络安全。", "第一次讲话第二段，讲信息化。",
                    "2.第二次讲话第一段。", "互联网是我们这个时代最具发展活力的领域。", "第二次讲话中间一段。",
                    "互联网是我们这个时代最具发展活力的领域。"]),                        # 同一次讲话内重复
            (2018, ["1.贺信全文。"]), (2020, ["1.指示内容。"]), (2022, ["1.文章节选。"])]
    return _quote_doc(tmp_path / "q.docx", toc, body)


# ---------------------------------------------------------------- 名言解析

def test_quotes_split_into_one_speech_per_numbered_paragraph(quote_doc):
    sp = parse_quotes(quote_doc)
    assert [s.speech_id for s in sp] == ["quote:2016:01", "quote:2016:02", "quote:2018:01", "quote:2020:01", "quote:2022:01"]
    assert sp[0].paragraphs == ["第一次讲话第一段，讲网络安全。", "第一次讲话第二段，讲信息化。"]      # 「1.」前缀被去掉，不带编号的段落并入同一次讲话
    assert sp[1].title == "第三届世界互联网大会开幕式上的视频讲话"                                    # 标题来自目录


def test_quotes_dedupe_paragraphs_within_a_speech_and_record_removed(quote_doc):
    s = parse_quotes(quote_doc)[1]
    assert s.paragraphs == ["第二次讲话第一段。", "互联网是我们这个时代最具发展活力的领域。", "第二次讲话中间一段。"]
    assert len(s.removed_duplicates) == 1 and s.removed_duplicates[0].startswith("互联网是我们这个时代") and s.raw_paragraph_count == 4


def test_same_paragraph_in_different_speeches_is_kept(tmp_path):
    p = _quote_doc(tmp_path / "q.docx", [(2020, ["甲", "乙"])], [(2020, ["1.相同的一段话。", "2.相同的一段话。"])])
    sp = parse_quotes(p)
    assert [s.paragraphs for s in sp] == [["相同的一段话。"], ["相同的一段话。"]]                       # 只在同一次讲话内去重


@pytest.mark.parametrize("title,kind", [
    ("致首届数字中国建设峰会的贺信", "贺信"), ("向2024年世界互联网大会乌镇峰会开幕视频致贺", "贺信"), ("新年贺词", "贺信"),
    ("考察杭州城市大脑运营指挥中心作出重要指示", "指示"), ("在全国网络安全和信息化工作会议上对网络安全和信息化工作作出的重要指示", "指示"),
    ("习近平.不断做强做优做大我国数字经济", "文章"), ("央视网新闻联播：推进数字中国建设 打造高质量发展新引擎", "文章"),
    ("全国网络安全和信息化工作会议上的讲话", "讲话"), ("在中央政治局第二十次集体学习时的讲话", "讲话"), ("某个说不清的标题", "讲话")])
def test_speech_kind_from_title_keywords(title, kind):
    assert classify_kind(title) == kind


def test_decimal_number_at_paragraph_start_is_not_a_new_speech(tmp_path):
    p = _quote_doc(tmp_path / "q.docx", [(2020, ["甲"])], [(2020, ["1.第一段。", "5.5G网络建设加快，覆盖率持续提高。"])])
    sp = parse_quotes(p)
    assert len(sp) == 1 and len(sp[0].paragraphs) == 2                          # 「5.5G」里的 5. 后面是数字，不应被当成第 5 次讲话，而是同一次讲话的第 2 段


def test_missing_toc_title_or_orphan_paragraph_raises(tmp_path):
    with pytest.raises(ValueError, match="目录里找不到标题"):
        parse_quotes(_quote_doc(tmp_path / "a.docx", [(2020, ["甲"])], [(2020, ["1.甲", "2.目录里没有的第二条"])]))
    with pytest.raises(ValueError, match="不属于任何讲话"):
        parse_quotes(_quote_doc(tmp_path / "b.docx", [(2020, ["甲"])], [(2020, ["没有编号的开头段落", "1.甲"])]))
    d = docx.Document(); d.add_paragraph("没有检索结果标题"); d.save(str(tmp_path / "c.docx"))
    with pytest.raises(ValueError, match="找不到"):
        parse_quotes(tmp_path / "c.docx")


# ---------------------------------------------------------------- 名言切块与统计

def test_one_chunk_per_speech_with_year_title_and_numbered_paragraphs(quote_doc):
    ch = chunk_quotes(parse_quotes(quote_doc))
    assert len(ch) == 5
    c = ch[0]
    assert c.text.startswith("2016年　网络安全和信息化工作座谈会上的讲话（讲话）\n[1] 第一次讲话第一段") and "[2] 第一次讲话第二段" in c.text
    assert c.paragraphs == ["第一次讲话第一段，讲网络安全。", "第一次讲话第二段，讲信息化。"] and c.kind == "讲话" and c.year == 2016
    assert ch[1].removed_duplicates == 1 and ch[2].kind == "贺信" and ch[3].kind == "指示" and ch[4].kind == "文章"
    assert len({k.chunk_id for k in ch}) == 5 and ch[0].chunk_id.startswith("quote:2016:01:")


def test_quote_stats_and_outputs(quote_doc, tmp_path):
    stats = quote_cli.export_quotes(quote_doc, tmp_path / "o")
    assert stats["speeches"] == 5 and stats["per_year"] == {2016: 2, 2018: 1, 2020: 1, 2022: 1}
    assert stats["removed_duplicate_total"] == 1 and stats["removed_duplicates"][0]["speech_id"] == "quote:2016:02"
    assert [r["chars"] for r in stats["per_speech"]][0] == len("第一次讲话第一段，讲网络安全。第一次讲话第二段，讲信息化。")
    assert stats["per_kind"] == {"讲话": 2, "贺信": 1, "指示": 1, "文章": 1} and stats["same_title_in_one_year"] == []
    rows = [json.loads(l) for l in (tmp_path / "o" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and rows[0]["paragraphs"]
    on_disk = json.loads((tmp_path / "o" / "stats.json").read_text(encoding="utf-8"))
    assert on_disk["per_year"] == {"2016": 2, "2018": 1, "2020": 1, "2022": 1} and on_disk["speeches"] == 5


def test_same_title_in_one_year_is_reported(tmp_path):
    p = _quote_doc(tmp_path / "q.docx", [(2025, ["中央政治局第二十次集体学习", "中央政治局第二十次集体学习"])], [(2025, ["1.甲。", "2.乙。"])])
    stats = quote_cli.export_quotes(p, tmp_path / "o")
    assert stats["same_title_in_one_year"] == [{"year": 2025, "title": "中央政治局第二十次集体学习", "speech_ids": ["quote:2025:01", "quote:2025:02"]}]


# ---------------------------------------------------------------- 两个包的配置与输出目录互相独立

def test_packages_use_independent_cache_dirs_and_env_prefixes():
    from rag import config as base
    from rag.quotes import config as qc
    assert case_cfg.CASE_CACHE_DIR.name == "case_cache" and qc.QUOTE_CACHE_DIR.name == "quote_cache"
    assert case_cfg.CASE_CACHE_DIR != base.STORE_DIR and qc.QUOTE_CACHE_DIR != base.STORE_DIR       # 不与现有 Chroma 目录冲突
    for p in (case_cfg.CASE_CACHE_DIR, qc.QUOTE_CACHE_DIR):
        assert p.name not in ("chroma_db", "cache", "course_cache", "policy_cache")
    import os
    if "CASE_DOC" not in os.environ:
        assert case_cfg.CASE_DOC.suffix == ".docx" and qc.QUOTE_DOC.suffix == ".docx"
