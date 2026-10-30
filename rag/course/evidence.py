"""证据锚点核对：把本体记录的 source_anchor（P####）落到教材单元，并核实 evidence_quote 是否出自原文。

锚点编号与 textbook.parse_docx 一致（body 下第 N 个 w:p，含空段）。
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ontology import Record, classify_anchor
from .textbook import Textbook, Unit, toc_key

CONTENT_KINDS = {"paragraph", "caption", "table", "textbox"}
SINK_MATCHED = "heading→matched_para"
SINK_RELOCATED = "heading→relocated"      # 小节内找不到，在本章全章找到几乎逐字一致的段落，按已核实处理
SINK_FIRST = "heading→section_first"      # 没找到匹配，退而取范围内第一个正文单元（未核实）
SINK_UNMATCHED = "heading→unmatched"      # 连 section_first 也不可信（覆盖率太低）：不当作证据召回
DEFAULT_SINK_THRESHOLD = 0.6              # 小节内覆盖率 >= 此值算 matched
SINK_RELOCATE_MIN = 0.8                   # 全章重定位要求的覆盖率
SINK_UNMATCHED_BELOW = 0.4                # section_first 且覆盖率低于此值 → unmatched
DEFINITION_PREDICATE = "hasDefinition"

NEARBY = 2          # 证据引文允许落在锚点段落之后多少个段落内（跨段引文 / 锚点偏一两行）
MIN_FRAGMENT = 4    # 引文按省略号切片后，短于此（规范化后）的片段不参与核对

_ELLIPSIS_RE = re.compile(r"…+|⋯+|\.{3,}|。{3,}|\[\.\.\.\]|\(\.\.\.\)")
_NON_WORD_RE = re.compile(r"[\W_]+")

# 锚点状态（paragraph 类锚点）
TITLE_ROW = "title_row"
TOC = "toc"
REFERENCES = "references"
FRONT = "front"
BODY = "body"
EMPTY = "empty"
OUT_OF_RANGE = "out_of_range"

# 引文核对结果
VERIFIED = "verified"
VERIFIED_NEARBY = "verified_nearby"
VERIFIED_COMPOSITE = "verified_composite"   # 「；」拼接的多段引文（如 precedes 边的两个相邻标题），各段分别在全书中逐字找到
NOT_FOUND = "not_found"
NO_QUOTE = "no_quote"
NOT_PARAGRAPH = "not_paragraph_anchor"
COMPOSITE_SEP = "；"
MIN_COMPOSITE_SEGMENT = 2   # 拼接引文的单段规范化后至少这么长才参与（标题可以很短）


def squash(text: str) -> str:
    """NFKC + 小写 + 去掉所有空白与标点，用于引文与原文的宽松比对。"""
    return _NON_WORD_RE.sub("", unicodedata.normalize("NFKC", text or "").casefold())


def quote_fragments(quote: str) -> list[str]:
    parts = _ELLIPSIS_RE.split(quote or "")
    return [f for f in (squash(p) for p in parts) if len(f) >= MIN_FRAGMENT]


@dataclass(frozen=True)
class AnchorInfo:
    status: str                # TITLE_ROW | REFERENCES | FRONT | BODY | EMPTY | OUT_OF_RANGE
    kind: str | None = None    # 单元 kind：heading / subheading / paragraph ...
    zone: str | None = None


@dataclass(frozen=True)
class Sunk:
    """标题行证据下沉的结果。

    score：matched/relocated 时是各段覆盖率的最小值；first/unmatched 时是小节内的最高覆盖率（供阈值调参）。
    extra_anchors：定义含「；」且各段落在不同段落时，其余段落的锚点（to_anchor 是第一段的）。
    alt_anchor / alt_score：没匹配上时，全章里覆盖率最高的段落（供人工核对，不用于召回）。
    """

    from_anchor: str
    to_anchor: str
    via: str
    score: float
    candidates: int       # 小节范围内参与匹配的正文单元数
    extra_anchors: tuple[str, ...] = ()
    alt_anchor: str | None = None
    alt_score: float | None = None


def sink_segments(text: str) -> list[str]:
    """定义文本按「；」拆段（规范化后），过短的段丢弃；没有「；」或拆完为空时整体作一段。"""
    whole = squash(text)
    if COMPOSITE_SEP not in (text or ""):
        return [whole] if whole else []
    segs = [s for s in (squash(p) for p in text.split(COMPOSITE_SEP)) if len(s) >= MIN_FRAGMENT]
    return segs or ([whole] if whole else [])


class AnchorIndex:
    def __init__(self, book: Textbook):
        self.n_paragraphs = book.n_paragraphs
        self.ordered: list[Unit] = book.units
        self._pos: dict[str, int] = {u.anchor: i for i, u in enumerate(book.units)}
        self.units: dict[int, Unit] = {
            u.para_no: u for u in book.units if u.para_no is not None and u.anchor == f"P{u.para_no:04d}"
        }
        self._squashed: dict[int, str] = {}
        self._unit_sq: dict[str, str] = {}
        self._chapter_pool: dict[int, list[Unit]] | None = None
        self._book_squashed: str | None = None
        self._headings: dict[str, list[int]] = {}
        for n, u in sorted(self.units.items()):
            if u.kind in ("heading", "subheading") and u.zone in ("body", "summary"):
                self._headings.setdefault(toc_key(u.text), []).append(n)

    def resolve_toc(self, para_no: int) -> Unit | None:
        """toc 行 → 其后第一个同名的正文标题单元；找不到返回 None。"""
        u = self.units.get(para_no)
        if u is None or u.zone != "toc":
            return None
        for n in self._headings.get(toc_key(u.text), []):
            if n > para_no:
                return self.units[n]
        return None

    def unit(self, para_no: int) -> Unit | None:
        return self.units.get(para_no)

    def squashed(self, para_no: int) -> str:
        s = self._squashed.get(para_no)
        if s is None:
            u = self.units.get(para_no)
            s = self._squashed[para_no] = squash(u.text) if u else ""
        return s

    def window(self, para_no: int, after: int = NEARBY) -> str:
        return "".join(self.squashed(n) for n in range(para_no, para_no + after + 1))

    def info(self, para_no: int) -> AnchorInfo:
        if para_no < 1 or para_no > self.n_paragraphs:
            return AnchorInfo(OUT_OF_RANGE)
        u = self.units.get(para_no)
        if u is None:
            return AnchorInfo(EMPTY)
        if u.zone == "toc":
            return AnchorInfo(TOC, u.kind, u.zone)
        if u.zone == "references":
            return AnchorInfo(REFERENCES, u.kind, u.zone)
        if u.zone == "front":
            return AnchorInfo(FRONT, u.kind, u.zone)
        if u.kind in ("heading", "subheading"):
            return AnchorInfo(TITLE_ROW, u.kind, u.zone)
        return AnchorInfo(BODY, u.kind, u.zone)

    def anchor_info(self, anchor: str) -> AnchorInfo | None:
        """非 P#### 锚点返回 None。"""
        kind, n = classify_anchor(anchor)
        return self.info(n) if kind == "paragraph" and n is not None else None

    def check_quote(self, anchor: str, quote: str) -> str:
        kind, n = classify_anchor(anchor)
        if kind != "paragraph" or n is None:
            return NOT_PARAGRAPH
        frags = quote_fragments(quote)
        if not frags:
            return NO_QUOTE
        own = self.squashed(n)
        if own and all(f in own for f in frags):
            return VERIFIED
        win = self.window(n)
        if win and all(f in win for f in frags):
            return VERIFIED_NEARBY
        if self._composite_found(quote):
            return VERIFIED_COMPOSITE
        return NOT_FOUND

    def _book_text(self) -> str:
        if self._book_squashed is None:
            self._book_squashed = "\n".join(self.squashed(n) for n in sorted(self.units))
        return self._book_squashed

    def _composite_found(self, quote: str) -> bool:
        """引文含「；」时拆成各段，每段（规范化后）都在全书某一段落内逐字出现才算。"""
        if COMPOSITE_SEP not in (quote or ""):
            return False
        segs = [squash(s) for s in quote.split(COMPOSITE_SEP)]
        segs = [s for s in segs if s]
        if len(segs) < 2 or any(len(s) < MIN_COMPOSITE_SEGMENT for s in segs):
            return False
        text = self._book_text()   # 以 \n 分隔各段落，squash 后的片段不含 \n，不会跨段误配
        return all(s in text for s in segs)

    def find_quote(self, quote: str) -> list[int]:
        """引文在全书哪些段落中能完整找到（所有片段都落在同一段，或同一段 + 之后 NEARBY 段内）。"""
        frags = quote_fragments(quote)
        if not frags:
            return []
        return [n for n in sorted(self.units) if self.squashed(n) and all(f in self.window(n) for f in frags)
                and frags[0] in self.squashed(n)]

    # ---- 标题行下沉 ----
    def section_units(self, heading: Unit) -> list[Unit]:
        """标题自身及其所有下级小节的正文单元（section_id 以标题 section_id 为前缀），按文档顺序。"""
        i = self._pos.get(heading.anchor)
        if i is None:
            return []
        hid = heading.section_id
        out: list[Unit] = []
        for u in self.ordered[i + 1:]:
            if not (u.section_id == hid or u.section_id.startswith(hid + ".")):
                break
            if u.zone in ("body", "summary") and u.kind in CONTENT_KINDS:
                out.append(u)
        return out

    def chapter_units(self, chapter: int | None) -> list[Unit]:
        """本章全部正文单元（body + summary，不含表格），按文档顺序。"""
        if chapter is None:
            return []
        if self._chapter_pool is None:
            pool: dict[int, list[Unit]] = defaultdict(list)
            for u in self.ordered:
                if (u.chapter is not None and u.zone in ("body", "summary")
                        and u.kind in CONTENT_KINDS and u.kind != "table"):
                    pool[u.chapter].append(u)
            self._chapter_pool = dict(pool)
        return self._chapter_pool.get(chapter, [])

    def _unit_squash(self, u: Unit) -> str:
        s = self._unit_sq.get(u.anchor)
        if s is None:
            s = self._unit_sq[u.anchor] = squash(u.text)
        return s

    def _best_unit(self, key: str, units: list[Unit]) -> tuple[Unit | None, float]:
        """覆盖率 = 匹配块总长 / 定义文本长度。autojunk=False：中文长段落里高频字不能被当作 junk。"""
        if not key:
            return None, 0.0
        best: Unit | None = None
        best_score = 0.0
        sm = SequenceMatcher(None, "", key, autojunk=False)   # seq2 固定为定义文本，seq1 逐段替换
        for u in units:
            s = self._unit_squash(u)
            if not s:
                continue
            sm.set_seq1(s)
            score = sum(m.size for m in sm.get_matching_blocks()) / len(key)
            if score > best_score:
                best, best_score = u, score
        return best, best_score

    def sink_heading(self, heading: Unit, text: str, *, threshold: float = DEFAULT_SINK_THRESHOLD,
                     relocate_min: float = SINK_RELOCATE_MIN, unmatched_below: float = SINK_UNMATCHED_BELOW,
                     fallback_first: bool = True) -> "Sunk | None":
        """把锚在标题行的证据落到正文。

        定义文本含「；」时按段分别处理（每段都要过关才算匹配）；每一段：
        1. 在标题自身及其所有下级小节的正文单元里找覆盖率最高者，>= threshold → 用它；
        2. 否则在**本章全章**再找一次，>= relocate_min → 用它（relocated）；
        3. 都不行 → 该段失败。
        全部段都成功：via=heading→matched_para（有任一段靠全章找到则为 heading→relocated），
        to_anchor 为第一段的落点，其余落点在 extra_anchors。
        否则（fallback_first=True）取小节范围内第一个正文单元：小节内最高覆盖率 < unmatched_below
        记 heading→unmatched（调用方不应把它当作证据），否则记 heading→section_first。
        小节范围内没有正文单元且全章也没找到 → None。
        """
        units = [u for u in self.section_units(heading) if u.kind != "table"]
        segs = sink_segments(text)
        pool = self.chapter_units(heading.chapter)

        found: list[tuple[Unit, float, bool]] = []
        in_section_best: list[float] = []
        alt: tuple[Unit | None, float] = (None, 0.0)
        failed = False
        for seg in segs:
            u, sc = self._best_unit(seg, units)
            in_section_best.append(sc)
            if u is not None and sc >= threshold:
                found.append((u, sc, False))
                continue
            ru, rs = self._best_unit(seg, pool)
            if ru is not None and rs >= relocate_min:
                found.append((ru, rs, True))
                continue
            failed = True
            if ru is not None and rs > alt[1]:
                alt = (ru, rs)
        if segs and not failed:
            anchors: list[str] = []
            for u, _sc, _r in found:
                if u.anchor not in anchors:
                    anchors.append(u.anchor)
            via = SINK_RELOCATED if any(r for _u, _s, r in found) else SINK_MATCHED
            score = round(min(s for _u, s, _r in found), 3)
            return Sunk(heading.anchor, anchors[0], via, score, len(units), tuple(anchors[1:]))
        if not fallback_first or not units:
            return None
        best = max(in_section_best, default=0.0)
        via = SINK_UNMATCHED if best < unmatched_below else SINK_FIRST
        return Sunk(heading.anchor, units[0].anchor, via, round(best, 3), len(units), (),
                    alt[0].anchor if alt[0] else None, round(alt[1], 3) if alt[0] else None)

    def check_record(self, rec: Record) -> str:
        return self.check_quote(rec.source_anchor, rec.evidence_quote)
