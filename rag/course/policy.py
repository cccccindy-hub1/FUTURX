"""数据政策汇编 → 政策库：按「一条完整的意思」切块，供语义匹配（不是关键词匹配）。

汇编结构：Heading 1 = 篇（国家立法 / 部委政策 / 各省市），Heading 2 = 一部政策文件。
每部文件按写法选切法：

- 条款式（≥5 个「第X条」）：一条一块，含其下各款、各项；编/章/节只进路径。
  最后一条之后出现「附件 / 附：」时，其余部分改用事项式切；
- 事项式（意见、规划、通知）：「一、」「（一）」编号处断开，一个事项一块；
  只有标题、没有正文的编号行不单独成块，只进路径；没有编号的文件按段落打包；
- 超过 POLICY_CHUNK_MAX 的块在小标题 / 「1.」/ 段落处拆成约 POLICY_PART_TARGET 的片，
  后续片以「标题（续）」开头，保证每片单独拿出来也知道讲的是哪一条；
- 表格单独成块（如处罚裁量基准表），过长时按行分组并重复表头。

清洗：去掉目录、页码行；PDF 转来的硬换行（行尾无标点）接回一段。

不入检索（indexed=False，仍导出，供按条号精确引用）：
- 套话条：「本法自…施行」「由…负责解释」「从其规定」等；
- 重复文件：正文相似度 ≥ POLICY_DUP_JACCARD 的只留一份（优先现行稿、结构更完整的）；
- 已有正式稿的征求意见稿。
标题命中 POLICY_EXCLUDE_TITLES 的文件（数据字典类标准）整体不切。

锚点 P##### = body 下第 N 个直接 w:p（从 1 开始），与判例库「法条目录」的来源锚点一致。
chunk_id = pol:{文件编号}:{序号:03d}:{正文 sha1 前 8 位}。不依赖 python-docx，只用标准库。
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from . import config
from .chunker import split_sentences
from .textbook import _HEADING_STYLE_RE, _W, _collect_text, _style_map, _table_rows

PATH_SEP = " › "

_CN = "一二三四五六七八九十百零〇"
ART_RE = re.compile(rf"^第[{_CN}]+条(?:之[{_CN}]+)?")
CHAP_RE = re.compile(rf"^第[{_CN}]+[编章节](?![{_CN}])")
L1_RE = re.compile(r"^[一二三四五六七八九十]+[、．.](?!\d)")          # 一、
L2_RE = re.compile(r"^[（(][一二三四五六七八九十]+[）)]")              # （一）
L3_RE = re.compile(r"^(?:\d{1,2}\s*[.．、](?!\d)|[（(]\d{1,2}[）)]|——)")  # 1.  （1）  ——
APPENDIX_RE = re.compile(r"^附(?:件|录|表)?\s*[\d一二三四五六七八九十]*\s*(?:[：:]|$)")
TOC_RE = re.compile(r"^目\s*录$")
TOC_LEADER_RE = re.compile(r"[\u4e00-\u9fa5”）)][.．…·]+\s*\d{1,3}$")  # 目录行：「……标题.......12」
PAGE_NO_RE = re.compile(r"^(?:[-—–\s]*\d{1,3}[-—–\s]*|第\s*\d+\s*页(?:\s*共\s*\d+\s*页)?|-{3,})$")
DOC_NUMBER_RE = re.compile(r"[一-龥]{1,12}[〔\[［(（]\d{4}[〕\]］)）]\s*\d+\s*号")
DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SENT_END = "。！？；：:;!?…"
_CLOSERS = "”’」）)》"
_BOILERPLATE_RE = re.compile(
    r"(?:自.{0,20}(?:起)?(?:施行|执行)|有效期|负责解释|解释工作|参照(?:执行|本)|另行(?:规定|制定)"
    r"|从其规定|按照国家有关规定执行)"
)
_BOILERPLATE_MAX = 80
_REFERENCE_TITLES = ("报告", "白皮书")
_HEADING_MAX = 40  # 编号行不超过此长度且没有句中句号时，视为纯标题


@dataclass
class Line:
    text: str
    anchor: str                      # P00028 / T003
    pos: int = 0                     # 在 body 中的位置，表格与段落混排时排序用
    table: list[list[str]] | None = None


@dataclass
class PolicyChunk:
    chunk_id: str
    doc_no: str                      # 汇编编号，如 1.2 / 2.50
    policy: str                      # 文件名（去掉外层书名号）
    level: str                       # 篇，如「第1篇 国家立法」
    doc_type: str                    # article（条款式）| opinion（事项式）| reference（报告/白皮书）
    status: str                      # 现行 | 征求意见稿
    kind: str                        # article | item | preamble | appendix | table
    path: list[str]                  # 章 / 大标题等上级标题（不含本块自身）
    title: str                       # 本块标题：第五条 / （三）建立健全… / 空
    text: str                        # 原文（返回给 LLM、展示给学生）
    embed_text: str                  # 向量化用：《文件名》 › 路径 › 标题 + 原文
    anchors: list[str]
    part: int | None = None          # 超长块拆开后的片序号
    indexed: bool = True
    flags: list[str] = field(default_factory=list)  # boilerplate / duplicate_of:2.112 / superseded_by:2.105

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyDoc:
    doc_no: str
    title: str                       # Heading 2 原文首行
    policy: str
    level: str
    anchor: str
    doc_type: str = ""
    status: str = "现行"
    doc_number: str = ""             # 发文字号
    date: str = ""                   # 首个日期（发布/通过日期），YYYY-MM-DD
    chars: int = 0
    n_chunks: int = 0
    n_indexed: int = 0
    excluded: str = ""               # 不切的原因
    duplicate_of: str = ""
    superseded_by: str = ""
    lines: list[Line] = field(default_factory=list, repr=False)


@dataclass
class PolicySet:
    source_sha256: str
    docs: list[PolicyDoc]
    chunks: list[PolicyChunk]
    dropped: Counter


# ---------------------------------------------------------------- 解析

def _norm(text: str) -> str:
    return re.sub(r"[\s　\xa0]+", " ", text).strip()


def _para_text(p) -> list[str]:
    out: list[str] = []
    boxes: list[list[str]] = []
    _collect_text(p, out, boxes)
    lines = "".join(out).split("\n") + [t for box in boxes for t in box]
    return [_norm(t) for t in lines if _norm(t)]


def _style_level(p, styles) -> int | None:
    """只认样式名（Heading 1 / 2）。汇编正文段落上零散的直设大纲级别不可信，不采用。"""
    ppr = p.find(_W + "pPr")
    ps = ppr.find(_W + "pStyle") if ppr is not None else None
    if ps is None:
        return None
    m = _HEADING_STYLE_RE.match(styles.get(ps.get(_W + "val"), (ps.get(_W + "val") or "", None))[0].strip())
    return int(m.group(1)) if m else None


def parse_policy_docx(path: Path) -> tuple[str, list[PolicyDoc]]:
    data = Path(path).read_bytes()
    with zipfile.ZipFile(path) as z:
        styles = _style_map(z)
        body = ET.fromstring(z.read("word/document.xml")).find(_W + "body")
    docs: list[PolicyDoc] = []
    level, cur, p_no, t_no = "", None, 0, 0
    for pos, el in enumerate(body):
        if el.tag == _W + "p":
            p_no += 1
            lines = _para_text(el)
            lv = _style_level(el, styles)
            if lv == 1 and lines:
                level = lines[0]
            elif lv == 2 and lines:
                no, _, name = lines[0].partition(" ")
                name = name.strip()
                if name.startswith("《") and name.endswith("》") and name.count("《") == 1:
                    name = name[1:-1]
                cur = PolicyDoc(doc_no=no, title=lines[0], policy=name, level=level, anchor=f"P{p_no:05d}")
                docs.append(cur)
            elif cur is not None:
                cur.lines.extend(Line(t, f"P{p_no:05d}", pos) for t in lines)
        elif el.tag == _W + "tbl":
            t_no += 1
            rows = _table_rows(el)
            if cur is not None and rows:
                cur.lines.append(Line("", f"T{t_no:03d}", pos, table=rows))
    return hashlib.sha256(data).hexdigest(), docs


# ---------------------------------------------------------------- 清洗

def _ends_sentence(s: str) -> bool:
    """句末标点结尾；右引号/右括号只有紧跟在句末标点后才算（「……一张图”」是句中断行）。"""
    s = s.rstrip(_CLOSERS) if s and s[-1] in _CLOSERS else s
    return bool(s) and s[-1] in _SENT_END


def _is_marker(s: str) -> bool:
    return bool(ART_RE.match(s) or CHAP_RE.match(s) or L1_RE.match(s) or L2_RE.match(s) or L3_RE.match(s))


def clean_lines(lines: list[Line], dropped: Counter) -> list[Line]:
    """去页码、去目录、接回硬换行。"""
    out = []
    for ln in lines:
        if ln.table is None and PAGE_NO_RE.match(ln.text):
            dropped["page_no"] += 1
            continue
        out.append(ln)
    out = _drop_figure_text(_drop_unlabeled_toc(_drop_toc(out, dropped), dropped), dropped)
    # 硬换行：多数长行不以标点结尾时，判定为 PDF 转换稿，把这类行与下一行接回
    texts = [ln.text for ln in out if ln.table is None and len(ln.text) >= 20]
    if texts and sum(not _ends_sentence(t) for t in texts) / len(texts) > 0.3:
        merged: list[Line] = []
        for ln in out:
            prev = merged[-1] if merged else None
            if (prev is not None and prev.table is None and ln.table is None and len(prev.text) >= 20
                    and not _ends_sentence(prev.text) and not _is_marker(ln.text)):
                sep = " " if prev.text[-1].isascii() and ln.text[0].isascii() else ""
                prev.text += sep + ln.text
                dropped["rewrapped"] += 1
                continue
            merged.append(Line(ln.text, ln.anchor, ln.pos, ln.table))
        out = merged
    return out


def _drop_unlabeled_toc(lines: list[Line], dropped: Counter) -> list[Line]:
    """没有「目录」字样的目录：带页码引导线的行；或连续 ≥3 个纯标题编号行、其中多数（≥60%）在后文正文中再次出现
    （报告类正文常省略一级标题，只有目录里有）。"""
    lines = [ln for ln in lines if ln.table is not None or not TOC_LEADER_RE.search(ln.text)
             or dropped.update(["toc"])]
    seen = Counter(ln.text for ln in lines if ln.table is None)
    heading = [ln.table is None and bool(L1_RE.match(ln.text) or L2_RE.match(ln.text))
               and _is_heading_only(ln.text) for ln in lines]
    drop: set[int] = set()
    i = 0
    while i < len(lines):
        j, in_run = i, set()
        while j < len(lines) and heading[j] and lines[j].text not in in_run:  # 标题重复出现 = 正文开始
            in_run.add(lines[j].text)
            j += 1
        if j - i >= 3 and sum(seen[lines[k].text] >= 2 for k in range(i, j)) >= 0.6 * (j - i):
            drop.update(range(i, j))
            for k in range(i, j):  # 只去掉第一次出现（目录），正文里的标题保留
                seen[lines[k].text] -= 1
        i = max(j, i + 1)
    dropped["toc"] += len(drop)
    return [ln for k, ln in enumerate(lines) if k not in drop]


def _drop_figure_text(lines: list[Line], dropped: Counter) -> list[Line]:
    """图片里的文字（竖排、框图）转出来是连续多行 1–3 字的碎片，整串去掉。"""
    out: list[Line] = []
    run: list[Line] = []

    def flush():
        if len(run) >= 3:
            dropped["figure_text"] += len(run)
        else:
            out.extend(run)
        run.clear()

    for ln in lines:
        if ln.table is None and len(ln.text) <= 3 and not ART_RE.match(ln.text) and not _ends_sentence(ln.text):
            run.append(ln)
            continue
        flush()
        out.append(ln)
    flush()
    return out


def _drop_toc(lines: list[Line], dropped: Counter) -> list[Line]:
    """「目录」后的连续短行是目录；正文从目录首项第二次出现处重新开始（找不到则只去掉这串短行）。"""
    k = next((i for i, ln in enumerate(lines[:40]) if ln.table is None and TOC_RE.match(ln.text)), None)
    if k is None:
        return lines
    j = k + 1
    while j < len(lines) and lines[j].table is None and len(lines[j].text) <= 60 and not lines[j].text.endswith("。"):
        j += 1
    if j > k + 1:
        first = lines[k + 1].text
        restart = next((i for i in range(k + 2, min(len(lines), k + 300)) if lines[i].text == first), None)
        j = restart if restart is not None else j
    dropped["toc"] += j - k
    return lines[:k] + lines[j:]


# ---------------------------------------------------------------- 切块

@dataclass
class _Block:
    kind: str
    path: list[str]
    title: str
    lines: list[Line] = field(default_factory=list)


def _is_heading_only(s: str) -> bool:
    """编号行是否只是标题（短、句中没有句号）。"""
    body = s[:-1] if s.endswith("。") else s
    return len(s) <= _HEADING_MAX and "。" not in body


def _blocks_article(lines: list[Line]) -> tuple[list[_Block], list[Line]]:
    """条款式：返回 (各条的块, 附件起的剩余行)。"""
    blocks: list[_Block] = []
    path: list[str] = []
    cur: _Block | None = None
    seen_article = False
    for i, ln in enumerate(lines):
        t = ln.text
        if ln.table is None and seen_article and APPENDIX_RE.match(t):
            if cur:
                blocks.append(cur)
            return blocks, lines[i:]
        if ln.table is None and CHAP_RE.match(t) and len(t) <= _HEADING_MAX and not _ends_sentence(t):
            if cur:
                blocks.append(cur)
                cur = None
            path = path[:1] if t[1:].lstrip(_CN)[:1] == "节" and path else []
            path.append(t)
            continue
        m = ART_RE.match(t) if ln.table is None else None
        if m:
            if cur:
                blocks.append(cur)
            seen_article = True
            cur = _Block("article", list(path), m.group(0), [ln])
            continue
        if cur is None:
            cur = _Block("preamble", list(path), "", [])
        cur.lines.append(ln)
    if cur:
        blocks.append(cur)
    return blocks, []


_CN_VAL = {c: i for i, c in enumerate("一二三四五六七八九十", 1)}


def _marker_no(t: str) -> int | None:
    """编号行的序号：「（三）」→3，「十二、」→12；识别不了返回 None。"""
    m = re.match(r"^[（(]?([一二三四五六七八九十]+)", t)
    if not m:
        return None
    s = m.group(1)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + _CN_VAL.get(s[1:], 0)
    if "十" in s:
        head, _, tail = s.partition("十")
        return _CN_VAL.get(head, 0) * 10 + _CN_VAL.get(tail, 0)
    return _CN_VAL.get(s)


def _blocks_opinion(lines: list[Line], kind_default: str = "item") -> list[_Block]:
    """事项式：在「一、」「（一）」处断开。

    纯标题编号行（短、句中无句号）先进路径；若下一行是更深一级的编号，它只是上级标题，
    不单独成块；否则它本身就是一个短事项（如「（二）数据来源说明。」），照常成块。
    """
    blocks: list[_Block] = []
    path: list[str] = []          # [一、标题, （一）标题]
    levels: list[int] = []
    cur: _Block | None = None
    cur_lv, cur_heading = 0, False
    enum_next: int | None = None  # 列举中期望的下一个序号
    for ln in lines:
        t = ln.text
        lv = 0 if ln.table is not None else (1 if L1_RE.match(t) else 2 if L2_RE.match(t) else 0)
        # 「提交下列材料：（一）…；（二）…」：冒号后从（一）开始、序号连续的编号行是列举项，留在原块里
        if lv and cur and cur.lines and cur.lines[-1].table is None:
            no = _marker_no(t)
            if (cur.lines[-1].text.endswith(("：", ":")) and no == 1) or (enum_next is not None and no == enum_next):
                enum_next = (no or 0) + 1
                cur.lines.append(ln)
                continue
        enum_next = None
        if lv:
            if cur and cur.lines and not (cur_heading and len(cur.lines) == 1 and lv > cur_lv):
                blocks.append(cur)
            while levels and levels[-1] >= lv:
                levels.pop()
                path.pop()
            cur_lv, cur_heading = lv, _is_heading_only(t)
            cur = _Block(kind_default, list(path), t if cur_heading else t[:60], [ln])
            if cur_heading:
                path.append(t)
                levels.append(lv)
            continue
        if cur is None:
            cur, cur_lv, cur_heading = _Block("preamble", [], "", []), 0, False
        cur.lines.append(ln)
    if cur and cur.lines:
        blocks.append(cur)
    return blocks


def _block_text(lines: list[Line]) -> str:
    return "\n".join(ln.text for ln in lines if ln.table is None)


def _render_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


def _split_rows(rows: list[list[str]], target: int) -> list[list[list[str]]]:
    if len(_render_table(rows)) <= config.POLICY_CHUNK_MAX or len(rows) < 3:
        return [rows]
    head, groups, buf = rows[0], [], []
    for r in rows[1:]:
        if buf and len(_render_table([head] + buf + [r])) > target:
            groups.append([head] + buf)
            buf = []
        buf.append(r)
    if buf:
        groups.append([head] + buf)
    return groups


def _is_break(lines: list[Line], i: int) -> bool:
    """适合拆开的位置：编号子项、或后面跟着较长正文的短小标题。"""
    t = lines[i].text
    if L3_RE.match(t):
        return True
    nxt = lines[i + 1].text if i + 1 < len(lines) else ""
    return len(t) <= 30 and len(nxt) > len(t) and (t.endswith("。") or not _ends_sentence(t))


def _split_long(lines: list[Line]) -> list[list[Line]]:
    """把正文行拆成若干片：先在优选断点处断，超出上限时在任意行边界断，单行超长按句切。"""
    target, cap = config.POLICY_PART_TARGET, config.POLICY_CHUNK_MAX - 70  # 给后续片的「标题（续）」留位置
    pieces: list[Line] = []
    for ln in lines:
        if len(ln.text) > cap:
            pieces.extend(Line(s, ln.anchor, ln.pos) for s in split_sentences(ln.text, target))
        else:
            pieces.append(ln)
    parts: list[list[Line]] = []
    buf: list[Line] = []
    size = 0
    for i, ln in enumerate(pieces):
        if buf and ((size >= target // 2 and _is_break(pieces, i)) or size + len(ln.text) > cap):
            parts.append(buf)
            buf, size = [], 0
        buf.append(ln)
        size += len(ln.text) + 1
    if buf:
        parts.append(buf)
    # 末片过短、并回去也不超上限时，并回上一片
    if (len(parts) > 1 and sum(len(x.text) for x in parts[-1]) < target // 4
            and sum(len(x.text) + 1 for x in parts[-2] + parts[-1]) <= cap):
        parts[-2].extend(parts.pop())
    return parts


def _cont_title(block: _Block) -> str:
    head = block.title or (block.lines[0].text if block.lines else "")
    return (head if len(head) <= 60 else head[:60] + "…") + "（续）"


def _emit(doc: PolicyDoc, block: _Block, out: list[tuple]) -> None:
    """一个逻辑块 → 若干 (块, 正文, 锚点, 片序号, 原文位置)；表格拆出去单独成块。"""
    text_lines: list[Line] = []
    merged: list[Line] = []
    for ln in block.lines:  # 只有一行的标题表（如「专栏1 …」）与紧随其后的内容表合成一张
        if ln.table is not None and merged and merged[-1].table is not None and len(merged[-1].table) == 1:
            merged[-1] = Line("", merged[-1].anchor, merged[-1].pos, merged[-1].table + ln.table)
        else:
            merged.append(ln)
    for ln in merged:
        if ln.table is None:
            text_lines.append(ln)
            continue
        groups = _split_rows(ln.table, config.POLICY_PART_TARGET)
        for k, rows in enumerate(groups, 1):
            tb = _Block("table", block.path + ([block.title] if block.title else []), block.title)
            out.append((tb, _render_table(rows), [ln.anchor], k if len(groups) > 1 else None, ln.pos))
    if not text_lines:
        return
    text = _block_text(text_lines)
    anchors = sorted({ln.anchor for ln in text_lines})
    if len(text) <= config.POLICY_CHUNK_MAX:
        out.append((block, text, anchors, None, text_lines[0].pos))
        return
    for k, part in enumerate(_split_long(text_lines), 1):
        body = _block_text(part)
        if k > 1:
            body = _cont_title(block) + "\n" + body
        out.append((block, body, sorted({ln.anchor for ln in part}), k, part[0].pos))


def _is_boilerplate(kind: str, text: str) -> bool:
    return kind == "article" and len(text) <= _BOILERPLATE_MAX and bool(_BOILERPLATE_RE.search(text))


def _is_title_only(kind: str, text: str) -> bool:
    """只有文件名、日期等的前言（没有一句完整的话）。"""
    return kind in ("preamble", "appendix") and len(text) < 100 and "。" not in text


def _doc_meta(doc: PolicyDoc, lines: list[Line]) -> None:
    head = " ".join(ln.text for ln in lines[:8] if ln.table is None)
    m = DOC_NUMBER_RE.search(head)
    doc.doc_number = re.sub(r"\s+", "", m.group(0)) if m else ""
    m = re.search(r"(?:发布日期|印发日期|日期)[：:]\s*" + DATE_RE.pattern, head) or DATE_RE.search(head)
    doc.date = f"{int(m.group(m.lastindex - 2)):04d}-{int(m.group(m.lastindex - 1)):02d}-{int(m.group(m.lastindex)):02d}" if m else ""
    doc.status = "征求意见稿" if "征求意见稿" in doc.title else "现行"


def chunk_doc(doc: PolicyDoc, dropped: Counter) -> list[PolicyChunk]:
    lines = clean_lines(doc.lines, dropped)
    _doc_meta(doc, lines)
    doc.chars = sum(len(ln.text) for ln in lines)
    n_art = sum(bool(ART_RE.match(ln.text)) for ln in lines if ln.table is None)
    if n_art >= 5:
        doc.doc_type = "article"
        blocks, rest = _blocks_article(lines)
        blocks += [_Block("appendix", b.path, b.title, b.lines) for b in _blocks_opinion(rest, "appendix")]
    else:
        n_marked = sum(bool(L1_RE.match(ln.text) or L2_RE.match(ln.text)) for ln in lines if ln.table is None)
        doc.doc_type = "reference" if any(k in doc.title for k in _REFERENCE_TITLES) else (
            "opinion" if n_marked >= 2 else "reference")
        blocks = _blocks_opinion(lines)
    raw: list[tuple] = []
    for b in blocks:
        _emit(doc, b, raw)
    raw.sort(key=lambda r: r[4])  # 按原文位置（稳定排序，拆开的各片保持先后）
    chunks = []
    for seq, (b, text, anchors, part, _pos) in enumerate(raw, 1):
        own = [b.title] if b.kind in ("item", "appendix") and b.title and not text.startswith(b.title) else []
        prefix = PATH_SEP.join([f"《{doc.policy}》", *b.path, *own])
        flags = ["boilerplate"] if _is_boilerplate(b.kind, text) else (
            ["title_only"] if _is_title_only(b.kind, text) else [])
        chunks.append(PolicyChunk(
            chunk_id=f"pol:{doc.doc_no}:{seq:03d}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}",
            doc_no=doc.doc_no, policy=doc.policy, level=doc.level, doc_type=doc.doc_type, status=doc.status,
            kind=b.kind, path=b.path, title=ART_RE.match(text).group(0) if b.kind == "article" else b.title,
            text=text, embed_text=prefix + "\n" + text, anchors=anchors, part=part,
            indexed=not flags, flags=flags,
        ))
    return chunks


# ---------------------------------------------------------------- 去重

def _shingles(text: str, n: int = 4) -> set[str]:
    s = re.sub(r"[\s\W_]+", "", text)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


_DRAFT_RE = re.compile(r"[（(](?:公开)?征求意见稿[）)]")


def _base_name(name: str) -> str:
    return _DRAFT_RE.sub("", name).replace("《", "").replace("》", "").strip()


def mark_duplicates(docs: list[PolicyDoc], chunks: list[PolicyChunk]) -> None:
    """重复文件只留一份；有正式稿的征求意见稿退出检索。"""
    by_doc: dict[str, list[PolicyChunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_no, []).append(c)
    live = [d for d in docs if not d.excluded and by_doc.get(d.doc_no)]
    sh = {d.doc_no: _shingles("".join(c.text for c in by_doc[d.doc_no])) for d in live}
    rank = {d.doc_no: (d.status == "现行", len(by_doc[d.doc_no]), i) for i, d in enumerate(live)}
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            la, lb = len(sh[a.doc_no]), len(sh[b.doc_no])
            if min(la, lb) < 0.3 * max(la, lb):
                continue
            j = _jaccard(sh[a.doc_no], sh[b.doc_no])
            keep, drop = (a, b) if rank[a.doc_no] > rank[b.doc_no] else (b, a)
            if j >= config.POLICY_DUP_JACCARD:
                if not drop.duplicate_of and not drop.superseded_by:
                    drop.duplicate_of = keep.doc_no
            elif drop.status == "征求意见稿" and keep.status == "现行" and (
                    j >= config.POLICY_DRAFT_JACCARD or _base_name(drop.policy) == _base_name(keep.policy)
                    or (j >= 0.3 and len(_base_name(drop.policy)) >= 6
                        and _base_name(drop.policy) in _base_name(keep.title))):
                if not drop.duplicate_of and not drop.superseded_by:
                    drop.superseded_by = keep.doc_no
    for d in live:
        tag = f"duplicate_of:{d.duplicate_of}" if d.duplicate_of else (
            f"superseded_by:{d.superseded_by}" if d.superseded_by else "")
        if tag:
            for c in by_doc[d.doc_no]:
                c.flags.append(tag)
                c.indexed = False


# ---------------------------------------------------------------- 入口

def build_policy_set(path: Path) -> PolicySet:
    sha, docs = parse_policy_docx(path)
    dropped: Counter = Counter()
    chunks: list[PolicyChunk] = []
    for d in docs:
        hit = next((k for k in config.POLICY_EXCLUDE_TITLES if k in d.title), None)
        if hit:
            d.excluded = f"标题含「{hit}」"
            d.status = "征求意见稿" if "征求意见稿" in d.title else "现行"
            continue
        chunks.extend(chunk_doc(d, dropped))
    mark_duplicates(docs, chunks)
    counts = Counter(c.doc_no for c in chunks)
    indexed = Counter(c.doc_no for c in chunks if c.indexed)
    for d in docs:
        d.n_chunks, d.n_indexed = counts[d.doc_no], indexed[d.doc_no]
    return PolicySet(source_sha256=sha, docs=docs, chunks=chunks, dropped=dropped)


def doc_row(d: PolicyDoc) -> dict:
    row = asdict(d)
    row.pop("lines")
    return row


def _pct(xs: list[int], q: float) -> int:
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def policy_stats(ps: PolicySet) -> dict:
    idx = [c for c in ps.chunks if c.indexed]
    lens = sorted(len(c.text) for c in idx)
    return {
        "source_sha256": ps.source_sha256,
        "docs": len(ps.docs),
        "docs_by_type": dict(Counter(d.doc_type or "excluded" for d in ps.docs)),
        "docs_excluded": [f"{d.doc_no} {d.policy}（{d.excluded}）" for d in ps.docs if d.excluded],
        "docs_duplicate": [f"{d.doc_no} → {d.duplicate_of}" for d in ps.docs if d.duplicate_of],
        "docs_superseded": [f"{d.doc_no} → {d.superseded_by}" for d in ps.docs if d.superseded_by],
        "chunks": len(ps.chunks),
        "indexed": len(idx),
        "chunks_by_kind": dict(Counter(c.kind for c in idx)),
        "not_indexed": dict(Counter(f.split(":")[0] for c in ps.chunks if not c.indexed for f in c.flags)),
        "split_parts": sum(1 for c in idx if c.part),
        "chunk_len": {"min": lens[0] if lens else 0, "p10": _pct(lens, 0.1), "median": _pct(lens, 0.5),
                      "p90": _pct(lens, 0.9), "max": lens[-1] if lens else 0},
        "over_max": sum(1 for x in lens if x > config.POLICY_CHUNK_MAX),
        "under_40": sum(1 for x in lens if x < 40),
        "dropped_lines": dict(ps.dropped),
        "duplicate_chunk_ids": len(ps.chunks) - len({c.chunk_id for c in ps.chunks}),
    }
