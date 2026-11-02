"""判例库 docx 解析：按样式读出 章节标签 / 判例名 / 元数据 / 小标题 / 正文 / 表格。

样式约定（见设计文档）：
- Heading 1 = 章节标签（「第六章 数据市场」）；「目录」与「附录…」的 Heading 1 不是章，附录整体不入库；
- Heading 2 = 判例名（「19. 北京微梦创科…案」，前面的序号去掉）；
- CaseMetadata = 元数据行（「案号：…」「审理法院：…」…，各案字段不全）；
- CaseSubheading = 小标题；CaseBody = 正文；CaseTable = 表格（本文件里表格没有专用样式，按文档顺序里的 w:tbl 识别）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十]+章\s*\S+")
NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*[.．、]\s*")
META_RE = re.compile(r"^\s*([^：:]{1,12})\s*[：:]\s*(.*)$", re.S)

# 元数据字段名 -> 规范键（别名合并）
META_KEYS = {
    "案号": "case_number", "审理法院": "court", "裁判日期": "judgment_date", "发布日期": "publish_date",
    "来源链接": "source_url", "来源": "source", "文本类型": "text_type", "获取日期": "fetched_date",
    "发布机构": "publisher", "参考案例编号": "ref_case_no", "文本出处": "text_origin",
    "核验来源": "verified_source", "同案指导性案例来源": "related_guiding_source",
}


@dataclass
class Block:
    """案内的一个元素，按文档顺序。kind = subheading | body | table。"""
    kind: str
    text: str


@dataclass
class ParsedCase:
    seq: int                                  # 文档内序号（1..47，取自判例名前面的数字，没有则按出现顺序）
    chapter: str                              # 章节标签，如「第六章 数据市场」
    chapter_no: int
    title: str                                # 判例名（去掉序号）
    meta: dict[str, str] = field(default_factory=dict)       # 规范键 -> 值
    meta_raw: list[str] = field(default_factory=list)        # 元数据原行，保持顺序
    blocks: list[Block] = field(default_factory=list)

    @property
    def body_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def chapter_number(label: str) -> int:
    m = re.match(r"^第([一二三四五六七八九十]+)章", label)
    if not m:
        return 0
    s = m.group(1)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + _CN_NUM.get(s[1:], 0)
    return _CN_NUM.get(s[0], 0) if len(s) == 1 else _CN_NUM.get(s[0], 0) * 10 + _CN_NUM.get(s[1], 0)


def _ordered_elements(doc):
    """文档顺序的 (kind, obj)：段落 'p' 与表格 't' 混排。"""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield "p", Paragraph(el, doc)
        elif tag == "tbl":
            yield "t", Table(el, doc)


def _table_text(tbl) -> str:
    """表格转文本：一行一行，单元格用「｜」分隔；合并单元格造成的相邻重复只保留一次。"""
    rows = []
    for row in tbl.rows:
        cells, last = [], None
        for c in row.cells:
            t = " ".join(c.text.split())
            if t and t != last:
                cells.append(t)
            last = t
        if cells:
            rows.append("｜".join(cells))
    return "\n".join(rows)


def parse_cases(path: str | Path) -> list[ParsedCase]:
    """解析判例库 docx，返回按文档顺序的判例（附录不入库）。"""
    import docx
    doc = docx.Document(str(path))
    cases: list[ParsedCase] = []
    chapter = ""
    cur: ParsedCase | None = None
    in_appendix = False
    for kind, x in _ordered_elements(doc):
        if kind == "t":
            if cur is not None:
                txt = _table_text(x)
                if txt:
                    cur.blocks.append(Block("table", txt))
            continue
        style = x.style.name
        text = x.text.strip()
        if style == "Heading 1":
            if text.startswith("附录"):
                in_appendix, cur = True, None
            elif CHAPTER_RE.match(text):
                in_appendix, chapter, cur = False, re.sub(r"\s+", " ", text), None
            else:                                          # 「目录」等
                cur = None
            continue
        if in_appendix or not chapter:
            continue
        if style == "Heading 2":
            m = NUMBER_PREFIX_RE.match(text)
            seq = int(re.match(r"\s*(\d+)", text).group(1)) if m else len(cases) + 1
            cur = ParsedCase(seq=seq, chapter=chapter, chapter_no=chapter_number(chapter), title=NUMBER_PREFIX_RE.sub("", text).strip())
            cases.append(cur)
            continue
        if cur is None or not text:
            continue
        if style == "CaseMetadata":
            cur.meta_raw.append(text)
            m = META_RE.match(text)
            if m:
                key = META_KEYS.get(m.group(1).strip())
                if key and m.group(2).strip():
                    cur.meta.setdefault(key, m.group(2).strip())
        elif style == "CaseSubheading":
            cur.blocks.append(Block("subheading", text))
        elif style in ("CaseBody", "CaseTable"):
            cur.blocks.append(Block("table" if style == "CaseTable" else "body", text))
    return cases
