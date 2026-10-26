"""教材解析：直接读 docx 的 XML，保留标题层级、段落锚点与表格原位置。

与 rag.extract._extract_docx 的区别（后者把全文拼成纯文本，标题样式与表格位置都会丢失）：

- 标题层级取自 Word 样式（Heading N / 标题 N / outlineLvl），不靠正则猜；
- 正文中 "1.数据与数据要素" 这类短的编号条目作为下一级"子标题"（Normal 样式，但承担标题作用）；
- 段落锚点 P#### = body 下第 N 个直接 w:p（从 1 开始，含空段），与 python-docx 的
  `Document.paragraphs` 序号一致，即本体 CSV 中 source_anchor 的编号方式；
- 表格按在正文中的原位置输出（锚点 T##），不再追加到全文末尾；
- 文本框单独成单元，不混进所在段落；
- 按区域标注：front（书名/目录等首章之前内容）、body、summary（小结）、references（引用/参考文献）。

不依赖 python-docx，只用标准库。
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"

_HEADING_STYLE_RE = re.compile(r"^(?:heading|标题)\s*(\d)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]+)\s*章")
# 编号子标题：短、以 "1." "2、" 开头、不以句末标点结尾
_SUBHEADING_RE = re.compile(r"^\d{1,2}\s*[.．、]\s*\S")
_SUBHEADING_MAX = 40
_SENTENCE_END = "。！？；：!?;:"
_REFERENCES_RE = re.compile(r"^(引用文献|参考文献)\s*$")
_SUMMARY_TITLES = {"小结", "本章小结"}
_CAPTION_RE = re.compile(r"^[图表]\s*\d+")

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


@dataclass
class Unit:
    """教材中的一个最小单元（段落 / 标题 / 表格 / 文本框）。"""

    anchor: str                 # P0062 / T03 / P0062.tb1
    kind: str                   # heading | subheading | paragraph | caption | table | textbox
    text: str
    zone: str                   # front | body | summary | references
    chapter: int | None
    path: tuple[str, ...]       # 所在标题路径（含本单元自身，若本单元是标题）
    section_id: str             # 完整路径编号，如 ch09.3.1.2 或 ch01.1.1.2.i1（i=编号子标题）
    block_id: str               # 仅按样式标题计算的编号，父块以它为边界
    para_no: int | None = None  # 段落序号（表格、文本框为 None）
    level: int | None = None    # 标题层级（1=章所在的 heading 1）
    rows: list[list[str]] = field(default_factory=list)  # 仅表格


@dataclass
class Textbook:
    path: str
    sha256: str
    n_paragraphs: int
    units: list[Unit]


def _cn_number(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    if not s:
        return None
    if "百" in s:
        return None
    if "十" in s:
        head, _, tail = s.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collect_text(el, out: list[str], boxes: list[list[str]]) -> None:
    """取段落正文；文本框内容另收到 boxes，mc:Fallback 是重复渲染，跳过。"""
    for child in el:
        tag = child.tag
        if tag == _MC_FALLBACK:
            continue
        if tag == _W + "txbxContent":
            parts: list[str] = []
            for p in child.iter(_W + "p"):
                parts.append("".join(t.text or "" for t in p.iter(_W + "t")))
            boxes.append(parts)
            continue
        if tag == _W + "t":
            out.append(child.text or "")
        elif tag == _W + "tab":
            out.append(" ")
        elif tag in (_W + "br", _W + "cr"):
            out.append("\n")
        _collect_text(child, out, boxes)


def _style_map(z: zipfile.ZipFile) -> dict[str, tuple[str, int | None]]:
    """styleId -> (样式名, 样式自带的大纲级别 0-based)。"""
    try:
        root = ET.fromstring(z.read("word/styles.xml"))
    except KeyError:
        return {}
    out: dict[str, tuple[str, int | None]] = {}
    for s in root.iter(_W + "style"):
        sid = s.get(_W + "styleId")
        name_el = s.find(_W + "name")
        name = name_el.get(_W + "val") if name_el is not None else (sid or "")
        lvl = None
        ppr = s.find(_W + "pPr")
        if ppr is not None:
            o = ppr.find(_W + "outlineLvl")
            if o is not None and o.get(_W + "val", "").isdigit():
                lvl = int(o.get(_W + "val"))
        out[sid] = (name, lvl)
    return out


def _heading_level(p, styles) -> int | None:
    """优先级：标题样式名 > 段落直设 outlineLvl > 样式自带 outlineLvl。

    书稿中段落直设的 outlineLvl 与样式并不一致（如「第一节」用 heading 2 样式却直设
    outlineLvl=0），样式名才是排版者的真实意图，因此放在最前。
    """
    ppr = p.find(_W + "pPr")
    sid = None
    direct = None
    if ppr is not None:
        ps = ppr.find(_W + "pStyle")
        sid = ps.get(_W + "val") if ps is not None else None
        o = ppr.find(_W + "outlineLvl")
        if o is not None and o.get(_W + "val", "").isdigit() and int(o.get(_W + "val")) < 9:
            direct = int(o.get(_W + "val")) + 1
    name, lvl = styles.get(sid, (sid or "", None)) if sid else ("", None)
    m = _HEADING_STYLE_RE.match(name.strip())
    if m:
        return int(m.group(1))
    if direct is not None:
        return direct
    if lvl is not None and lvl < 9:
        return lvl + 1
    return None


def _table_rows(tbl) -> list[list[str]]:
    rows = []
    for tr in tbl.iter(_W + "tr"):
        cells = []
        for tc in tr.findall(_W + "tc"):
            cells.append(_clean("".join(t.text or "" for t in tc.iter(_W + "t"))))
        # 横向合并单元格在 XML 里只出现一次，但纵向合并/重复填充会产生相邻重复，去掉
        dedup = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
        if any(dedup):
            rows.append(dedup)
    return rows


class _Outline:
    """维护标题栈与各级序号，生成 path / section_id / block_id。"""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str, bool, int]] = []  # (level, text, styled, ordinal)
        self.counters: dict[tuple, int] = {}
        self.chapter: int | None = None
        self.chapter_seq = 0

    def push(self, level: int, text: str, styled: bool) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        # 序号在「同一父路径 + 同级 + 是否样式标题」内递增
        key = tuple(o for _lv, _t, _s, o in self.stack) + (level, styled)
        n = self.counters.get(key, 0) + 1
        self.counters[key] = n
        self.stack.append((level, text, styled, n))

    def start_chapter(self, number: int | None, text: str) -> None:
        self.chapter_seq += 1
        self.chapter = number if number is not None else self.chapter_seq
        self.stack = [(1, text, True, self.chapter)]
        self.counters = {}

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(t for _lv, t, _s, _o in self.stack)

    def _ids(self, styled_only: bool) -> str:
        if self.chapter is None:
            return "front"
        parts = [f"ch{self.chapter:02d}"]
        for lv, _t, styled, o in self.stack[1:]:
            if styled:
                parts.append(str(o))
            elif not styled_only:
                parts.append(f"i{o}")
        return ".".join(parts)

    @property
    def section_id(self) -> str:
        return self._ids(styled_only=False)

    @property
    def block_id(self) -> str:
        return self._ids(styled_only=True)

    @property
    def styled_level(self) -> int:
        for lv, _t, styled, _o in reversed(self.stack):
            if styled:
                return lv
        return 0


def _is_subheading(text: str) -> bool:
    return (
        len(text) <= _SUBHEADING_MAX
        and bool(_SUBHEADING_RE.match(text))
        and text[-1] not in _SENTENCE_END
    )


def parse_docx(path: str | Path) -> Textbook:
    path = Path(path)
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        styles = _style_map(z)
    body = doc.find(_W + "body")
    if body is None:
        raise ValueError(f"{path.name} 不是有效的 docx（缺少 w:body）")

    outline = _Outline()
    units: list[Unit] = []
    zone = "front"
    para_no = 0
    table_no = 0

    def emit(**kw) -> None:
        units.append(
            Unit(
                zone=zone,
                chapter=outline.chapter,
                path=outline.path,
                section_id=outline.section_id,
                block_id=outline.block_id,
                **kw,
            )
        )

    for el in body:
        if el.tag == _W + "p":
            para_no += 1
            anchor = f"P{para_no:04d}"
            parts: list[str] = []
            boxes: list[list[str]] = []
            _collect_text(el, parts, boxes)
            text = _clean("".join(parts))
            level = _heading_level(el, styles)

            if level is not None and text:
                m = _CHAPTER_RE.match(text) if level == 1 else None
                if m:
                    outline.start_chapter(_cn_number(m.group(1)), text)
                    zone = "body"
                elif level == 1 or outline.chapter is None:
                    zone = "front"  # 书名、附录等不编号的一级标题：不入正文
                else:
                    outline.push(level, text, styled=True)
                    zone = "summary" if text in _SUMMARY_TITLES else "body"
                emit(anchor=anchor, kind="heading", text=text, para_no=para_no, level=level)
            elif text:
                if zone in ("body", "summary") and _REFERENCES_RE.match(text):
                    zone = "references"
                    emit(anchor=anchor, kind="heading", text=text, para_no=para_no)
                elif zone in ("body", "summary") and _is_subheading(text):
                    outline.push(outline.styled_level + 1, text, styled=False)
                    emit(anchor=anchor, kind="subheading", text=text, para_no=para_no,
                         level=outline.styled_level + 1)
                else:
                    kind = "caption" if _CAPTION_RE.match(text) and len(text) <= 60 else "paragraph"
                    emit(anchor=anchor, kind=kind, text=text, para_no=para_no)
            for bi, box in enumerate(boxes, 1):
                bt = _clean(" ".join(box))
                if bt and bt != text and not bt.isdigit():
                    emit(anchor=f"{anchor}.tb{bi}", kind="textbox", text=bt, para_no=para_no)
        elif el.tag == _W + "tbl":
            table_no += 1
            rows = _table_rows(el)
            if rows:
                emit(
                    anchor=f"T{table_no:02d}",
                    kind="table",
                    text="\n".join(" | ".join(r) for r in rows),
                    rows=rows,
                )

    return Textbook(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        n_paragraphs=para_no,
        units=units,
    )
