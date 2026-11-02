"""判例切块：每案两类块，都带「案名 + 章节标签」前缀，检索命中后能归到判例。

- 判例卡片（card，每案 1 块）：案名、章节标签、案号/法院/日期等元数据，加「裁判要点 / 裁判要旨 / 典型意义」；
  没有这几节的案子（多为判决书全文），取以「本院认为」开头的段落约 CASE_CARD_FALLBACK_CHARS 字；
  找不到「本院认为」再取「判决如下」之前最后约这么多字；仍找不到就取正文开头。
  card_source 记录卡片取自哪里：points（要点类小节）/ court_opinion（本院认为）/ before_judgment（判决如下之前）/ lead（正文开头）。
- 正文块（body）：按小标题切，每块 CASE_BODY_MIN–CASE_BODY_MAX 字；太短的小节与相邻小节合并，太长的按段落（必要时按句）拆开。
  要点类小节已进卡片，正文块里仍保留（检索时卡片与正文互补，不重复也无妨）。表格作为正文一并切入，不单独成块。
chunk_id = case:{序号:02d}:{card|bodyNN}:{正文 sha1 前 8 位}，序号取自判例名前的编号，稳定且可读。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from ..course.chunker import split_sentences
from . import config
from .parser import Block, ParsedCase

PATH_SEP = " › "
POINT_KEYS = ("裁判要点", "裁判要旨", "要旨", "典型意义", "执行实施要点")      # 指导性案例里有「执行实施要点」这类变体
_STRIP_MARK = re.compile(r"[【】〖〗\[\]\s：:]")
# 「本院认为」多数写在段落开头而不是独立小标题；也有「本院经审理认为」「二审法院认为」「本院结合双方诉辩请求，认为」等写法
COURT_OPINION_RE = re.compile(r"^\s*(?:本院|[一二再]?审?法院|本院经[^，。：:]{0,6})[^。：:]{0,16}?认为[，,：:]?")
JUDGMENT_RE = re.compile(r"^\s*(?:判决如下|裁定如下|判决：|裁定：)")


@dataclass
class CaseChunk:
    chunk_id: str
    case_id: str                  # case:{序号:02d}
    seq: int
    chapter: str
    chapter_no: int
    title: str
    kind: str                     # card | body
    part: int                     # 正文块序号（卡片为 0）
    heading: str                  # 正文块所属小标题（可空）
    text: str                     # 带前缀的完整文本（入库、检索用）
    body_chars: int               # 不含前缀的字数
    card_source: str = ""         # 仅卡片：points / court_opinion / before_judgment / lead
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def case_prefix(c: ParsedCase) -> str:
    return f"【{c.chapter}】{c.title}"


_POINT_LEAD_RE = re.compile(r"^\s*[【〖\[]?\s*(裁判要点|裁判要旨|要旨|典型意义|执行实施要点)\s*[】〗\]]?\s*[：:]?\s*(.*)$", re.S)


def _point_heading(text: str) -> tuple[str, str] | None:
    """识别要点类标题，返回 (标题, 同一段里紧跟的正文)；不是要点类标题返回 None。

    两种写法都有：标题单独成段（CaseSubheading「裁判要点」、CaseBody「[裁判要旨]」，正文在后面的段落），
    以及标题和正文同在一段（「【裁判要旨】数据处理者对依法依规持有的数据享有……」）。
    标题和正文同段时，只在标题带括号或冒号时才认，避免把「典型意义重大，…」之类的普通句子当成标题。"""
    m = _POINT_LEAD_RE.match(text)
    if not m:
        return None
    key, rest = m.group(1), m.group(2).strip()
    if not rest:
        return key, ""
    marked = bool(re.match(r"^\s*[【〖\[]", text)) or bool(re.match(rf"^\s*{key}\s*[：:]", text))
    return (key, rest) if marked else None


def _is_point_heading(text: str) -> bool:
    return _point_heading(text) is not None


def _is_any_heading(b: Block) -> bool:
    """用来界定「要点类小节」在哪里结束：下一个小标题，或形如 [相关法条] 的方括号标记行。"""
    if b.kind == "subheading":
        return True
    return b.kind == "body" and bool(re.fullmatch(r"\s*[【〖\[][^】〗\]]{1,10}[】〗\]]\s*", b.text))


def _point_sections(c: ParsedCase) -> list[tuple[str, str]]:
    """要点类小节：[(标题, 正文)]。标题可能是 CaseSubheading，也可能是 CaseBody 里的标记行（正文在后面的段落，或就在同一段）。"""
    out: list[tuple[str, str]] = []
    bl = c.blocks
    i = 0
    while i < len(bl):
        b = bl[i]
        ph = _point_heading(b.text) if b.kind in ("subheading", "body") else None
        if ph:
            title, inline = ph
            body: list[str] = [inline] if inline else []
            j = i + 1
            while j < len(bl) and not _is_any_heading(bl[j]) and _point_heading(bl[j].text) is None:
                if bl[j].kind in ("body", "table"):
                    body.append(bl[j].text)
                j += 1
            if body:
                out.append((title, "\n".join(body)))
            i = j
            continue
        i += 1
    return out


def _meta_line(c: ParsedCase) -> str:
    m = c.meta
    parts = []
    for label, key in (("案号", "case_number"), ("审理法院", "court"), ("裁判日期", "judgment_date"),
                       ("发布日期", "publish_date"), ("发布机构", "publisher"), ("参考案例编号", "ref_case_no")):
        if m.get(key):
            parts.append(f"{label}：{m[key]}")
    return "　".join(parts)


def _fallback_excerpt(c: ParsedCase, n: int) -> tuple[str, str]:
    """没有要点类小节时的卡片正文：(文本, 来源标记)。"""
    paras = [b.text for b in c.blocks if b.kind in ("body", "table")]
    for k, p in enumerate(paras):                                  # 1. 「本院认为」多数写在段落开头而不是独立小标题
        if COURT_OPINION_RE.match(p):
            out, got = [], 0
            for q in paras[k:]:
                out.append(q)
                got += len(q)
                if got >= n:
                    break
            return _cut("\n".join(out), n), "court_opinion"
    for k, p in enumerate(paras):                                  # 2. 「判决如下」之前最后约 n 字
        if JUDGMENT_RE.match(p) and k > 0:
            before = "\n".join(paras[:k])
            return before[-n:], "before_judgment"
    return _cut("\n".join(paras), n), "lead"                       # 3. 实在找不到：正文开头


def _cut(text: str, n: int) -> str:
    """截到约 n 字，尽量落在句末。"""
    if len(text) <= n:
        return text
    cut = text[:n]
    m = max(cut.rfind("。"), cut.rfind("；"), cut.rfind("！"), cut.rfind("？"))
    return cut[: m + 1] if m >= int(n * 0.6) else cut


def build_card(c: ParsedCase) -> CaseChunk:
    secs = _point_sections(c)
    if secs:
        body = "\n".join(f"{t}：{x}" for t, x in secs)
        source = "points"
    else:
        body, source = _fallback_excerpt(c, config.CASE_CARD_FALLBACK_CHARS)
    head = [case_prefix(c)]
    ml = _meta_line(c)
    if ml:
        head.append(ml)
    text = "\n".join(head) + "\n" + body
    return CaseChunk(chunk_id=f"case:{c.seq:02d}:card:{_sha8(text)}", case_id=f"case:{c.seq:02d}", seq=c.seq,
                     chapter=c.chapter, chapter_no=c.chapter_no, title=c.title, kind="card", part=0, heading="",
                     text=text, body_chars=len(body), card_source=source, meta=dict(c.meta))


def _split_long(text: str, size: int) -> list[str]:
    """超长段落按句拆成不超过 size 的片。"""
    if len(text) <= size:
        return [text]
    out, buf = [], ""
    for s in split_sentences(text, size):
        if buf and len(buf) + len(s) > size:
            out.append(buf)
            buf = ""
        buf += s
    if buf:
        out.append(buf)
    return out


def _sections(c: ParsedCase) -> list[tuple[str, list[str]]]:
    """按小标题分节：[(小标题, [段落…])]。第一个小标题之前的内容归入「（开头）」节。"""
    secs: list[tuple[str, list[str]]] = [("", [])]
    for b in c.blocks:
        if b.kind == "subheading":
            secs.append((b.text, []))
        else:
            secs[-1][1].append(b.text)
    return [(h, ps) for h, ps in secs if ps]


def build_bodies(c: ParsedCase) -> list[CaseChunk]:
    lo, hi = config.CASE_BODY_MIN, config.CASE_BODY_MAX
    # 1. 每节先拆成不超过 hi 的片（节名随片带上）
    pieces: list[tuple[str, str]] = []
    for head, paras in _sections(c):
        buf = ""
        for p in paras:
            for frag in _split_long(p, hi):
                if buf and len(buf) + len(frag) + 1 > hi:
                    pieces.append((head, buf))
                    buf = ""
                buf = f"{buf}\n{frag}" if buf else frag
        if buf:
            pieces.append((head, buf))
    # 2. 太短的片与相邻片合并（合并后不超过 hi）；合并后的块标题取第一片的
    merged: list[tuple[str, str]] = []
    for head, txt in pieces:
        if merged and (len(merged[-1][1]) < lo) and len(merged[-1][1]) + len(txt) + 1 <= hi:
            ph, pt = merged[-1]
            label = ph if (not head or head == ph) else (f"{ph}；{head}" if ph else head)
            merged[-1] = (label, f"{pt}\n{('【' + head + '】') if head and head != ph else ''}{txt}")
        else:
            merged.append((head, f"{('【' + head + '】') if head else ''}{txt}"))
    # 3. 最后一块若仍很短，并入前一块（仍不超过 hi 才并，否则保持独立）
    if len(merged) >= 2 and len(merged[-1][1]) < lo // 2 and len(merged[-2][1]) + len(merged[-1][1]) + 1 <= hi:
        (ph, pt), (_, lt) = merged[-2], merged[-1]
        merged[-2:] = [(ph, f"{pt}\n{lt}")]
    out = []
    for n, (head, txt) in enumerate(merged, 1):
        text = f"{case_prefix(c)}\n{txt}"
        out.append(CaseChunk(chunk_id=f"case:{c.seq:02d}:body{n:02d}:{_sha8(text)}", case_id=f"case:{c.seq:02d}", seq=c.seq,
                             chapter=c.chapter, chapter_no=c.chapter_no, title=c.title, kind="body", part=n, heading=head,
                             text=text, body_chars=len(txt), meta=dict(c.meta)))
    return out


def chunk_cases(cases: list[ParsedCase]) -> list[CaseChunk]:
    out: list[CaseChunk] = []
    for c in cases:
        out.append(build_card(c))
        out.extend(build_bodies(c))
    return out
