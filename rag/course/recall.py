"""Step 2：确定性召回（概念 → 边证据 / 字面提及 → 教材段落 → M1 子块），不用 LLM。

召回两种来源（每个子块可同时有多条原因，去重后合并）：
- edge_evidence：选中概念作为 subject 或 object 的记录，其 source_anchor（P####）落到教材段落；
  - 标题行锚点：先下沉到正文（evidence.sink_heading，定义文本=hasDefinition 的 object_label）；
  - toc 行锚点：先按同名找到正文真实标题，再走下沉；
  - 参考文献 / 前言 / 空段 / 越界 / 非 P#### 锚点：排除并记录，绝不静默丢弃；
- mention：概念名称及非 blocked 别名在**该概念所属章**教材中的字面出现。

可信度分档 tier（越小越优先，分批送 Step 3 时先送小的，绝不丢弃）：
  0 = 教材证据可核实 + 专家来源（expert_relevance_weight >= EXPERT_MIN）
  1 = 教材证据可核实 + 机器来源
  2 = 未核实（含 section_first 下沉）
  3 = mention
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .aliases import AliasTable, norm
from .chunker import Chunk, ChunkSet
from .config import COURSE_RECALL_MAX_CHARS, SINK_THRESHOLD
from .evidence import (
    EMPTY, FRONT, OUT_OF_RANGE, REFERENCES, SINK_FIRST, SINK_MATCHED, SINK_RELOCATED, SINK_UNMATCHED, TITLE_ROW,
    TOC, VERIFIED, VERIFIED_COMPOSITE, VERIFIED_NEARBY, AnchorIndex, squash,
)
from .ontology import ConceptTable, Record, classify_anchor

DEFINITION_PREDICATE = "hasDefinition"
EXPERT_MIN = 0.99          # expert_relevance_weight 达到此值视为专家来源（数据中取值 1 / 0.8 / 0.6）
EDGE = "edge_evidence"
MENTION = "mention"
STRATEGY_A = "A"           # 所有选中概念都做 mention 召回（--strategy A）
STRATEGY_B = "B"           # 只有 question_concept 做 mention 召回；related_concept 只召回边证据（默认）
DEFAULT_STRATEGY = STRATEGY_B
VERIFIED_SET = {VERIFIED, VERIFIED_NEARBY, VERIFIED_COMPOSITE}

EXCLUDE_STATUS = {REFERENCES: "references", FRONT: "front", EMPTY: "empty", OUT_OF_RANGE: "out_of_range"}


@dataclass(frozen=True)
class Reason:
    concept_id: str
    via: str                       # edge_evidence | mention
    anchor: str                    # 触发召回的原始锚点（mention 时为命中的段落锚点）
    record_id: str | None = None
    predicate_id: str | None = None
    direction: str | None = None   # out：概念是 subject；in：概念是 object
    view: str | None = None
    evidence: str | None = None    # verified | nearby | composite | unverified | None（mention）
    source_tier: str | None = None # expert | machine | None
    sunk: str | None = None        # 下沉方式（SINK_MATCHED / SINK_FIRST），未下沉为 None
    sink_score: float | None = None
    confidence: float | None = None

    def label(self) -> str:
        if self.via == MENTION:
            return f"mention:{self.concept_id}"
        via = f"edge_evidence({self.sunk})" if self.sunk else "edge_evidence"
        return f"{via}:{self.concept_id}:{self.predicate_id}:{self.direction}"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Recalled:
    chunk_id: str
    parent_id: str
    path: list[str]
    text: str
    chunk_type: str
    anchors: list[str]
    reasons: list[Reason] = field(default_factory=list)

    @property
    def tier(self) -> int:
        return min(_reason_tier(r) for r in self.reasons) if self.reasons else 3

    def to_dict(self, with_text: bool = True) -> dict:
        d = {"chunk_id": self.chunk_id, "parent_id": self.parent_id, "path": self.path,
             "chunk_type": self.chunk_type, "anchors": self.anchors, "tier": self.tier,
             "reasons": [r.to_dict() for r in self.reasons]}
        if with_text:
            d["text"] = self.text
        return d


@dataclass
class RecallResult:
    items: list[Recalled] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)   # 被排除的锚点及原因
    sunk: list[dict] = field(default_factory=list)       # 标题行 / toc 行下沉记录
    zero_mention: list[str] = field(default_factory=list)  # 本章零字面提及的被选概念
    strategy: str = DEFAULT_STRATEGY
    ontology_only: list[dict] = field(default_factory=list)  # heading→unmatched：只作 [本体] 信息，不作教材证据

    def to_dict(self, with_text: bool = False) -> dict:
        return {"strategy": self.strategy, "items": [i.to_dict(with_text) for i in self.items],
                "excluded": self.excluded, "sunk": self.sunk, "zero_mention": self.zero_mention,
                "ontology_only": self.ontology_only}


def _reason_tier(r: Reason) -> int:
    if r.via == MENTION:
        return 3
    if r.evidence in ("verified", "nearby", "composite"):
        return 0 if r.source_tier == "expert" else 1
    return 2


def _evidence_label(status: str) -> str:
    return {VERIFIED: "verified", VERIFIED_NEARBY: "nearby", VERIFIED_COMPOSITE: "composite"}.get(status, "unverified")


class ChunkLocator:
    """锚点 → 子块。长段被按句切开时，只取含引文的分片；一个都没命中则召回全部分片。"""

    def __init__(self, chunkset: ChunkSet):
        self.chunks = chunkset.chunks
        self.by_id = {c.chunk_id: c for c in chunkset.chunks}
        self._by_anchor: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunkset.chunks:
            for a in c.anchors:
                self._by_anchor[a].append(c)

    def find(self, anchor: str, quote: str = "") -> list[Chunk]:
        cs = self._by_anchor.get(anchor, [])
        if len(cs) <= 1:
            return list(cs)
        pieces = [c for c in cs if c.split_part is not None]
        if len(pieces) != len(cs):
            return list(cs)
        key = squash(quote)
        if key:
            hit = [c for c in pieces if key in squash(c.text)]
            if hit:
                return hit
        return list(cs)


def _source_tier(rec: Record) -> str:
    w = rec.expert_relevance_weight
    return "expert" if w is not None and w >= EXPERT_MIN else "machine"


def _definition_text(rec: Record) -> str:
    return rec.object_label or ""


class Recaller:
    def __init__(self, records: list[Record], chunkset: ChunkSet, anchors: AnchorIndex,
                 aliases: AliasTable, concepts: ConceptTable, *, sink_threshold: float = SINK_THRESHOLD):
        self.anchors, self.aliases, self.concepts = anchors, aliases, concepts.by_id
        self.locator = ChunkLocator(chunkset)
        self.sink_threshold = sink_threshold
        self._by_concept: dict[str, list[tuple[Record, str]]] = defaultdict(list)
        for r in records:
            if r.is_concept_edge:
                self._by_concept[r.subject_id].append((r, "out"))
                if r.object_id != r.subject_id:
                    self._by_concept[r.object_id].append((r, "in"))
            elif r.subject_id in self.concepts:
                self._by_concept[r.subject_id].append((r, "out"))   # hasDefinition 等：宾语是非概念节点
        self._mention_cache: dict[str, list[str]] = {}
        self._chapter_units: dict[int, list] = defaultdict(list)
        for u in anchors.ordered:
            if u.zone in ("body", "summary") and u.kind in ("paragraph", "caption", "table", "textbox") and u.chapter:
                self._chapter_units[u.chapter].append(u)

    # ---- 边证据 ----
    def _edge_hits(self, cid: str, out: RecallResult, acc: dict[str, Recalled]) -> None:
        for rec, direction in self._by_concept.get(cid, []):
            kind, n = classify_anchor(rec.source_anchor)
            if kind != "paragraph" or n is None:
                out.excluded.append(self._excl(cid, rec, "non_paragraph_anchor"))
                continue
            info = self.anchors.info(n)
            target_anchor = rec.source_anchor
            sunk_via: str | None = None
            sunk_score: float | None = None
            status = info.status
            unit = self.anchors.unit(n)
            if status == TOC:
                real = self.anchors.resolve_toc(n)
                if real is None:
                    out.excluded.append(self._excl(cid, rec, "toc_unresolved"))
                    continue
                unit, status = real, TITLE_ROW
                out.sunk.append({"record_id": rec.record_id, "concept_id": cid, "from": rec.source_anchor,
                                 "via": "toc→heading", "to": real.anchor})
            if status == TITLE_ROW:
                text = _definition_text(rec) if rec.predicate_id == DEFINITION_PREDICATE else rec.evidence_quote
                s = self.anchors.sink_heading(unit, text, threshold=self.sink_threshold)
                if s is None:
                    out.excluded.append(self._excl(cid, rec, "title_row_no_body"))
                    continue
                out.sunk.append({"record_id": rec.record_id, "concept_id": cid, "from": rec.source_anchor,
                                 "via": s.via, "to": s.to_anchor, "extra": list(s.extra_anchors),
                                 "score": s.score, "candidates": s.candidates,
                                 "alt": s.alt_anchor, "alt_score": s.alt_score})
                if s.via == SINK_UNMATCHED:       # 落点不可信：不当作该定义的教材证据，只保留为本体信息
                    out.ontology_only.append({"concept_id": cid, "record_id": rec.record_id, "anchor": rec.source_anchor,
                                              "via": s.via, "score": s.score})
                    continue
                target_anchor, sunk_via, sunk_score = s.to_anchor, s.via, s.score
                extra = list(s.extra_anchors)
                evidence = "unverified" if s.via == SINK_FIRST else "verified"
            elif status in EXCLUDE_STATUS:
                out.excluded.append(self._excl(cid, rec, EXCLUDE_STATUS[status]))
                continue
            else:
                evidence = _evidence_label(self.anchors.check_record(rec))
                extra = []
            reason = Reason(
                concept_id=cid, via=EDGE, anchor=rec.source_anchor, record_id=rec.record_id,
                predicate_id=rec.predicate_id, direction=direction, view=rec.view, evidence=evidence,
                source_tier=_source_tier(rec), sunk=sunk_via, sink_score=sunk_score, confidence=rec.confidence,
            )
            quote = rec.evidence_quote if not sunk_via else ""
            chunks = self.locator.find(target_anchor, quote)
            for a in extra:
                chunks += [c for c in self.locator.find(a) if c not in chunks]
            if not chunks:
                out.excluded.append(self._excl(cid, rec, "no_chunk_for_anchor", target=target_anchor))
                continue
            for c in chunks:
                self._add(acc, c, reason)

    @staticmethod
    def _excl(cid: str, rec: Record, why: str, target: str | None = None) -> dict:
        d = {"concept_id": cid, "record_id": rec.record_id, "anchor": rec.source_anchor, "reason": why,
             "view": rec.view}
        if target:
            d["target"] = target
        return d

    # ---- 字面提及 ----
    def mention_anchors(self, cid: str) -> list[str]:
        """概念名称及非 blocked 别名在所属章正文（含小结）中的字面出现段落锚点。"""
        if cid in self._mention_cache:
            return self._mention_cache[cid]
        c = self.concepts.get(cid)
        keys = [self.aliases.by_key[k].key for k in self.aliases.by_concept.get(cid, [])
                if self.aliases.by_key[k].blocked is None]
        hits: list[str] = []
        if c is not None and keys:
            for u in self._chapter_units.get(c.chapter, []):
                hay = norm(u.text)
                if any(k in hay for k in keys):
                    hits.append(u.anchor)
        self._mention_cache[cid] = hits
        return hits

    def _mention_hits(self, cid: str, acc: dict[str, Recalled]) -> int:
        n = 0
        for anchor in self.mention_anchors(cid):
            for c in self.locator.find(anchor):
                self._add(acc, c, Reason(concept_id=cid, via=MENTION, anchor=anchor))
                n += 1
        return n

    def _add(self, acc: dict[str, Recalled], c: Chunk, reason: Reason) -> None:
        item = acc.get(c.chunk_id)
        if item is None:
            item = acc[c.chunk_id] = Recalled(c.chunk_id, c.parent_id, c.path, c.text, c.chunk_type, c.anchors)
        if reason not in item.reasons:
            item.reasons.append(reason)

    # ---- 入口 ----
    def recall(self, selected: list[tuple[str, str]], strategy: str = DEFAULT_STRATEGY) -> RecallResult:
        """selected: [(concept_id, role)]。strategy B（默认）=仅 question_concept 做 mention；A=全部做 mention。"""
        out = RecallResult(strategy=strategy)
        acc: dict[str, Recalled] = {}
        seen: set[str] = set()
        for cid, role in selected:
            if cid in seen or cid not in self.concepts:
                continue
            seen.add(cid)
            self._edge_hits(cid, out, acc)
            if strategy == STRATEGY_A or role == "question_concept":
                self._mention_hits(cid, acc)
            if not self.mention_anchors(cid):
                out.zero_mention.append(cid)
        order = {c.chunk_id: i for i, c in enumerate(self.locator.chunks)}
        out.items = sorted(acc.values(), key=lambda r: (r.tier, order.get(r.chunk_id, 0)))
        return out


def batch_by_budget(items: list[Recalled], max_chars: int = COURSE_RECALL_MAX_CHARS) -> list[list[Recalled]]:
    """按 tier 升序装箱成批；单个子块超限也单独成批，不丢弃。同一批内保持 tier、文档顺序。"""
    batches: list[list[Recalled]] = []
    cur: list[Recalled] = []
    size = 0
    for it in sorted(items, key=lambda r: r.tier):   # 稳定排序，tier 内保持文档顺序
        n = len(it.text)
        if cur and size + n > max_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += n
    if cur:
        batches.append(cur)
    return batches


def dry_run_report(recaller: Recaller, selected: list[tuple[str, str]], *,
                   max_chars: int = COURSE_RECALL_MAX_CHARS, max_steps: int = 6) -> dict:
    """不调 LLM：对比两种 mention 策略的子块数 / 批次数 / 预计调用次数，并统计零提及概念与下沉分布。"""
    out: dict = {"selected": len(selected), "strategies": {}}
    for strat in (STRATEGY_A, STRATEGY_B):
        res = recaller.recall(selected, strat)
        batches = batch_by_budget(res.items, max_chars)
        by_tier: dict[int, int] = defaultdict(int)
        for it in res.items:
            by_tier[it.tier] += 1
        out["strategies"][strat] = {
            "chunks": len(res.items), "chars": sum(len(i.text) for i in res.items),
            "batches": len(batches), "by_tier": dict(sorted(by_tier.items())),
            "excluded": len(res.excluded), "sunk": len(res.sunk),
            "zero_mention_selected": len(res.zero_mention), "zero_mention_ids": res.zero_mention,
            "est_llm_calls": {"step1_max": max_steps + 1, "step3": len(batches), "step4": 1,
                              "total_max": max_steps + 1 + len(batches) + 1},
        }
    return out


def sink_report(records: list[Record], anchors: AnchorIndex, *, threshold: float = SINK_THRESHOLD) -> dict:
    """全量体检：所有锚在标题行 / toc 行的 hasDefinition，下沉结果与覆盖率分布（供确定阈值）。"""
    rows: list[dict] = []
    for rec in records:
        if rec.predicate_id != DEFINITION_PREDICATE:
            continue
        kind, n = classify_anchor(rec.source_anchor)
        if kind != "paragraph" or n is None:
            continue
        info, unit = anchors.info(n), anchors.unit(n)
        if info.status == TOC:
            unit = anchors.resolve_toc(n)
            if unit is None:
                rows.append({"record_id": rec.record_id, "view": rec.view, "from": rec.source_anchor, "via": "toc_unresolved"})
                continue
        elif info.status != TITLE_ROW:
            continue
        s = anchors.sink_heading(unit, _definition_text(rec), threshold=threshold)
        rows.append({"record_id": rec.record_id, "view": rec.view, "from": rec.source_anchor,
                     "via": s.via if s else "no_body", "to": s.to_anchor if s else None,
                     "extra": list(s.extra_anchors) if s else [],
                     "score": s.score if s else None, "candidates": s.candidates if s else 0,
                     "alt": s.alt_anchor if s else None, "alt_score": s.alt_score if s else None})
    scores = sorted(r["score"] for r in rows if r.get("score") is not None)
    buckets: dict[str, int] = defaultdict(int)
    for sc in scores:
        buckets[f"{min(int(sc * 10), 9) / 10:.1f}-{min(int(sc * 10), 9) / 10 + 0.1:.1f}"] += 1
    by_via: dict[str, int] = defaultdict(int)
    for r in rows:
        by_via[r["via"]] += 1
    return {"threshold": threshold, "total": len(rows), "by_via": dict(by_via),
            "score_histogram": dict(sorted(buckets.items())), "rows": rows}
