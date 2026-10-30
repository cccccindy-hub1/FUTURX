"""数据质量报告：ID 引用校验、证据锚点体检、跨章同名、零提及、Top3 候选覆盖。

只做统计与核对，不对数据做修改，也不决定 Top3 的排序规则（见 relevance.py 的候选方案）。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .aliases import AliasTable, norm, split_name
from .evidence import NOT_FOUND, NOT_PARAGRAPH, OUT_OF_RANGE, REFERENCES, TITLE_ROW, AnchorIndex
from .ontology import Concept, ConceptTable, Record, classify_anchor, node_kind

DEFINITION_PREDICATE = "hasDefinition"
SAMPLE_LIMIT = 10
MENTION_ZONES = {"body", "summary"}

# ID 引用问题类型
UNKNOWN_CONCEPT = "unknown_concept_id"          # 形如 C##-#### 但不在概念总表
UNKNOWN_NODE = "unknown_node_kind"              # 既非概念ID也非已知节点前缀
EMPTY_ENDPOINT = "empty_endpoint"
CHAPTER_MISMATCH = "chapter_mismatch"           # 概念ID 的章号与所在文件章号不一致
DUPLICATE_RECORD_ID = "duplicate_record_id"
ANCHOR_OUT_OF_RANGE = "anchor_out_of_range"
EXCLUDED_CONCEPT_REF = "excluded_concept_ref"   # 引用了被标注为噪声/删去的概念


# ---------------------------------------------------------------- Top3 候选

@dataclass(frozen=True)
class Edge:
    """一条概念→概念的有向断言（保留全部原始属性）。"""

    record_id: str
    predicate_id: str
    predicate_label: str
    confidence: float | None
    expert_weight: float | None
    source_anchor: str
    retrieval_anchor: str
    assertion_type: str
    logic_level: str
    review_status: str
    evidence_quote: str


@dataclass
class Candidate:
    """subject → object 的候选关联；多条断言（不同谓词/证据）合并在同一候选下。"""

    subject_id: str
    object_id: str
    edges: list[Edge] = field(default_factory=list)


def to_edge(r: Record) -> Edge:
    return Edge(
        r.record_id, r.predicate_id, r.predicate_label, r.confidence, r.expert_relevance_weight,
        r.source_anchor, r.retrieval_anchor, r.assertion_type, r.logic_level, r.review_status,
        r.evidence_quote,
    )


def collect_candidates(
    records: list[Record], *, view: str | None = "onto2", known: set[str] | None = None,
) -> dict[str, list[Candidate]]:
    """每个概念（作为 subject）的候选关联概念。

    只算 subject 与 object 都是概念ID 的边；方向保持原样（A→B 不会生成 B→A）；
    自环（subject==object）丢弃；同一 (subject, object) 的多条断言合并为一个候选；
    给定 known（概念总表 ID 集）时，两端不在其中的边丢弃（悬空引用不能当候选）。
    """
    by_pair: dict[tuple[str, str], Candidate] = {}
    for r in records:
        if view and r.view and r.view != view:
            continue
        if not r.is_concept_edge or r.subject_id == r.object_id:
            continue
        if known is not None and (r.subject_id not in known or r.object_id not in known):
            continue
        cand = by_pair.setdefault((r.subject_id, r.object_id), Candidate(r.subject_id, r.object_id))
        cand.edges.append(to_edge(r))
    out: dict[str, list[Candidate]] = defaultdict(list)
    for (s, _o), cand in by_pair.items():
        out[s].append(cand)
    return dict(out)


def incoming_counts(candidates: dict[str, list[Candidate]]) -> Counter:
    c: Counter = Counter()
    for cands in candidates.values():
        for cand in cands:
            c[cand.object_id] += 1
    return c


# ---------------------------------------------------------------- 报告

@dataclass
class QualityReport:
    summary: dict = field(default_factory=dict)
    reference_issues: dict[str, int] = field(default_factory=dict)
    reference_samples: dict[str, list[dict]] = field(default_factory=dict)
    anchor_status: dict[str, dict[str, int]] = field(default_factory=dict)       # view -> 状态 -> 条数
    anchor_kinds: dict[str, dict[str, int]] = field(default_factory=dict)        # view -> 锚点类别 -> 条数
    definition_anchor: dict[str, dict] = field(default_factory=dict)             # view -> {total, title_row, references,...}
    definition_title_row: list[dict] = field(default_factory=list)
    definition_references: list[dict] = field(default_factory=list)
    quote_check: dict[str, dict[str, int]] = field(default_factory=dict)         # view -> 核对结果 -> 条数
    not_found_samples: dict[str, list[dict]] = field(default_factory=dict)       # view -> 前 20 条 not_found 样例
    cross_chapter_names: list[dict] = field(default_factory=list)
    zero_mention: dict = field(default_factory=dict)
    top3_coverage: dict = field(default_factory=dict)
    issue_types: dict[str, int] = field(default_factory=dict)
    student_set: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_markdown(self) -> str:
        return _render_markdown(self)


def validate_references(
    records: list[Record], concepts: ConceptTable, anchors: AnchorIndex | None,
    excluded_issue_types: tuple[str, ...] = ("D 非概念噪声", "删去"),
) -> tuple[Counter, dict[str, list[dict]]]:
    by_id = concepts.by_id
    issues: Counter = Counter()
    samples: dict[str, list[dict]] = defaultdict(list)

    def flag(kind: str, r: Record, detail: str) -> None:
        issues[kind] += 1
        if len(samples[kind]) < SAMPLE_LIMIT:
            samples[kind].append({"record_id": r.record_id, "view": r.view, "detail": detail})

    seen: set[tuple[str, str]] = set()
    for r in records:
        key = (r.view, r.record_id)
        if key in seen:
            flag(DUPLICATE_RECORD_ID, r, r.record_id)
        seen.add(key)

        for side, nid in (("subject", r.subject_id), ("object", r.object_id)):
            kind = node_kind(nid)
            if kind == "empty":
                flag(EMPTY_ENDPOINT, r, side)
            elif kind == "concept":
                c = by_id.get(nid)
                if c is None:
                    flag(UNKNOWN_CONCEPT, r, f"{side}={nid}")
                else:
                    if c.issue_type in excluded_issue_types:
                        flag(EXCLUDED_CONCEPT_REF, r, f"{side}={nid}《{c.name}》{c.issue_type}")
                    file_ch = r.chapter_id[2:4] if r.chapter_id.startswith("Ch") else ""
                    if file_ch.isdigit() and int(file_ch) != int(nid[1:3]):
                        # 跨章引用本身合法（概念可关联别章概念），仅在 subject 侧要求同章
                        if side == "subject":
                            flag(CHAPTER_MISMATCH, r, f"{nid} 出现在 {r.chapter_id}")
            elif kind == "other":
                flag(UNKNOWN_NODE, r, f"{side}={nid}")

        if anchors is not None:
            ak, n = classify_anchor(r.source_anchor)
            if ak == "paragraph" and n is not None and anchors.info(n).status == OUT_OF_RANGE:
                flag(ANCHOR_OUT_OF_RANGE, r, r.source_anchor)
    return issues, dict(samples)


def not_found_samples(records: list[Record], anchors: AnchorIndex, view: str, limit: int = 20) -> list[dict]:
    """引文核对为 not_found 的前 limit 条：区分「锚点偏移」（全书别处能找到）与「引文被改写」（全书都找不到）。"""
    out: list[dict] = []
    for r in records:
        if r.view != view or anchors.check_record(r) != NOT_FOUND:
            continue
        _k, n = classify_anchor(r.source_anchor)
        u = anchors.unit(n) if n else None
        hits = anchors.find_quote(r.evidence_quote)
        out.append({
            "record_id": r.record_id, "anchor": r.source_anchor,
            "quote_head": r.evidence_quote[:40], "anchor_text_head": (u.text[:40] if u else ""),
            "found_in_book": bool(hits), "found_at": [f"P{h:04d}" for h in hits[:5]],
            "offset": (hits[0] - n) if hits and n else None,
        })
        if len(out) >= limit:
            break
    return out


def _definition_audit(records: list[Record], anchors: AnchorIndex, concepts: dict[str, Concept]):
    total = 0
    status: Counter = Counter()
    title_rows: list[dict] = []
    refs: list[dict] = []
    for r in records:
        if r.predicate_id != DEFINITION_PREDICATE:
            continue
        total += 1
        info = anchors.anchor_info(r.source_anchor)
        if info is None:
            status["non_paragraph_anchor"] += 1
            continue
        status[info.status] += 1
        if info.status in (TITLE_ROW, REFERENCES):
            n = classify_anchor(r.source_anchor)[1]
            u = anchors.unit(n) if n else None
            row = {
                "record_id": r.record_id, "view": r.view, "subject_id": r.subject_id,
                "subject_label": r.subject_label, "anchor": r.source_anchor,
                "anchor_text": (u.text[:40] if u else ""),
                "concept_known": r.subject_id in concepts,
            }
            (title_rows if info.status == TITLE_ROW else refs).append(row)
    return {"total": total, **dict(status)}, title_rows, refs


def _cross_chapter_names(concepts: list[Concept]) -> list[dict]:
    groups: dict[str, list[Concept]] = defaultdict(list)
    for c in concepts:
        groups[norm(split_name(c.name)[0])].append(c)
    out = []
    for key, cs in groups.items():
        if len({c.chapter for c in cs}) > 1:
            out.append({
                "name": split_name(cs[0].name)[0],
                "chapters": sorted({c.chapter for c in cs}),
                "ids": [c.id for c in sorted(cs, key=lambda c: c.id)],
                "labels_differ": len({c.name for c in cs}) > 1,
            })
    out.sort(key=lambda d: (-len(d["ids"]), d["name"]))
    return out


def _mention_counts(book, aliases: AliasTable) -> dict[str, int]:
    text = "\n".join(u.text for u in book.units if u.zone in MENTION_ZONES and u.kind != "heading")
    return aliases.count_mentions(text)


def _zero_mention(concepts: list[Concept], mentions: dict[str, int], aliases: AliasTable,
                  excluded: tuple[str, ...]) -> dict:
    zero, no_alias = [], []
    for c in concepts:
        if c.issue_type in excluded:
            continue
        if not aliases.aliases_of(c.id):
            no_alias.append(c.id)       # 所有别名都被 blocked：无法判断提及
        elif mentions.get(c.id, 0) == 0:
            zero.append(c.id)
    by_chapter: Counter = Counter(int(i[1:3]) for i in zero)
    by_id = {c.id: c for c in concepts}
    return {
        "count": len(zero),
        "by_chapter": dict(sorted(by_chapter.items())),
        "sample": [{"id": i, "name": by_id[i].name, "issue_type": by_id[i].issue_type} for i in zero[:SAMPLE_LIMIT * 2]],
        "ids": zero,
        "no_matchable_alias": no_alias,
    }


def _top3_coverage(concepts: list[Concept], candidates: dict[str, list[Candidate]],
                   excluded: tuple[str, ...]) -> dict:
    inc = incoming_counts(candidates)
    live = [c for c in concepts if c.issue_type not in excluded]
    dist: Counter = Counter()
    zero_out_with_in: list[str] = []
    for c in live:
        n = len(candidates.get(c.id, []))
        dist[str(n) if n < 4 else "4+"] += 1
        if n == 0 and inc.get(c.id, 0) > 0:
            zero_out_with_in.append(c.id)
    by_ch: dict[int, Counter] = defaultdict(Counter)
    for c in live:
        n = len(candidates.get(c.id, []))
        by_ch[c.chapter]["has_3plus" if n >= 3 else "lt3"] += 1
    multi_edge = sum(1 for cs in candidates.values() for cand in cs if len(cand.edges) > 1)
    preds: Counter = Counter(e.predicate_id for cs in candidates.values() for cand in cs for e in cand.edges)
    unknown_subjects = sorted(set(candidates) - {c.id for c in concepts})
    return {
        "concepts_considered": len(live),
        "out_degree_distribution": {k: dist[k] for k in ("0", "1", "2", "3", "4+") if k in dist},
        "with_at_least_3": sum(v for k, v in dist.items() if k in ("3", "4+")),
        "fewer_than_3": sum(v for k, v in dist.items() if k in ("0", "1", "2")),
        "no_outgoing_but_has_incoming": len(zero_out_with_in),
        "candidate_pairs": sum(len(v) for v in candidates.values()),
        "pairs_with_multiple_edges": multi_edge,
        "reverse_pairs_present": _reverse_pairs(candidates),
        "predicate_counts": dict(preds.most_common()),
        "by_chapter": {ch: dict(c) for ch, c in sorted(by_ch.items())},
        "subjects_not_in_concept_table": unknown_subjects[:SAMPLE_LIMIT],
    }


def _reverse_pairs(candidates: dict[str, list[Candidate]]) -> dict:
    pairs = {(c.subject_id, c.object_id) for cs in candidates.values() for c in cs}
    both = sum(1 for (s, o) in pairs if (o, s) in pairs) // 2
    return {"directed_pairs": len(pairs), "bidirectional_pairs": both}


def _student_set_check(rows: list[dict[str, str]], records: list[Record]) -> dict:
    ids = {r.record_id for r in records}
    matched = machine = expert = other = 0
    unmatched: list[str] = []
    for row in rows:
        ref = row.get("来源编号", "")
        if not ref:
            other += 1
        elif ref in ids:
            matched += 1
        elif "_R" in ref and ref.startswith("Ch"):
            machine += 1
            if len(unmatched) < SAMPLE_LIMIT:
                unmatched.append(ref)
        else:
            expert += 1
    return {
        "rows": len(rows), "matched_record_id": matched, "machine_id_not_found": machine,
        "expert_style_id": expert, "no_ref": other, "unmatched_sample": unmatched,
    }


def build_report(
    concepts: ConceptTable,
    records: list[Record],
    aliases: AliasTable,
    book=None,
    student_rows: list[dict[str, str]] | None = None,
    excluded_issue_types: tuple[str, ...] = ("D 非概念噪声", "删去"),
) -> QualityReport:
    rep = QualityReport()
    anchors = AnchorIndex(book) if book is not None else None
    cs = concepts.concepts
    by_id = concepts.by_id
    views = sorted({r.view for r in records})

    rep.summary = {
        "concepts": len(cs),
        "concept_rows_skipped": len(concepts.skipped),
        "records": {v: sum(1 for r in records if r.view == v) for v in views},
        "textbook_paragraphs": anchors.n_paragraphs if anchors else None,
    }
    rep.issue_types = dict(Counter(c.issue_type or "(空)" for c in cs).most_common())

    issues, samples = validate_references(records, concepts, anchors, excluded_issue_types)
    rep.reference_issues, rep.reference_samples = dict(issues), samples

    for v in views:
        vr = [r for r in records if r.view == v]
        rep.anchor_kinds[v] = dict(Counter(classify_anchor(r.source_anchor)[0] for r in vr))
        if anchors is None:
            continue
        rep.anchor_status[v] = dict(Counter(
            (anchors.anchor_info(r.source_anchor).status if anchors.anchor_info(r.source_anchor) else NOT_PARAGRAPH)
            for r in vr
        ))
        rep.definition_anchor[v], tr, rf = _definition_audit(vr, anchors, by_id)
        rep.definition_title_row += tr
        rep.definition_references += rf
        rep.quote_check[v] = dict(Counter(anchors.check_record(r) for r in vr))
        rep.not_found_samples[v] = not_found_samples(vr, anchors, v)

    rep.cross_chapter_names = _cross_chapter_names(cs)
    if book is not None:
        rep.zero_mention = _zero_mention(cs, _mention_counts(book, aliases), aliases, excluded_issue_types)
    cands = collect_candidates(records, view="onto2", known=set(by_id))
    rep.top3_coverage = _top3_coverage(cs, cands, excluded_issue_types)

    rep.aliases = {
        "surfaces": len(aliases.by_key),
        "matchable": len(aliases.active()),
        "blocked": len(aliases.blocked()),
        "blocked_by_reason": dict(Counter(a.blocked for a in aliases.blocked())),
        "ambiguous": sum(1 for a in aliases.active() if a.ambiguous),
        "blocked_sample": [a.surface for a in aliases.blocked()][:SAMPLE_LIMIT * 3],
    }
    if student_rows is not None:
        rep.student_set = _student_set_check(student_rows, records)
    return rep


# ---------------------------------------------------------------- Markdown

def _kv(d: dict) -> str:
    return "、".join(f"{k}={v}" for k, v in d.items()) or "（无）"


def _render_markdown(r: QualityReport) -> str:
    L: list[str] = ["# 《数据要素》本体数据质量报告（M2）", ""]
    s = r.summary
    L += ["## 概览", f"- 概念总表有效行：{s.get('concepts')}（跳过非概念行 {s.get('concept_rows_skipped')}）",
          f"- 本体记录：{_kv(s.get('records', {}))}", f"- 教材段落数：{s.get('textbook_paragraphs')}",
          f"- 问题类型分布：{_kv(r.issue_types)}", ""]

    L += ["## ID 引用校验"]
    L += [f"- {k}：{v}" for k, v in r.reference_issues.items()] or ["- 无问题"]
    for k, ss in r.reference_samples.items():
        L += [f"  - {k} 示例：" + "；".join(f"{x['record_id']}（{x['detail']}）" for x in ss[:3])]
    L.append("")

    L += ["## 证据锚点"]
    for v, st in r.anchor_status.items():
        L.append(f"- {v} 锚点状态：{_kv(st)}")
    for v, st in r.anchor_kinds.items():
        L.append(f"- {v} 锚点类别：{_kv(st)}")
    for v, st in r.quote_check.items():
        L.append(f"- {v} 引文核对：{_kv(st)}")
    for v, ss in r.not_found_samples.items():
        if not ss:
            continue
        moved = sum(1 for x in ss if x["found_in_book"])
        L.append(f"- {v} 引文 not_found 前 {len(ss)} 条：全书能找到 {moved}（锚点偏移）、找不到 {len(ss) - moved}（疑似改写）")
        for x in ss:
            where = f"→{','.join(x['found_at'])}（偏移{x['offset']:+d}）" if x["found_in_book"] else "全书未找到"
            L.append(f"  - {x['record_id']} {x['anchor']} 引文《{x['quote_head']}》 锚点段《{x['anchor_text_head']}》 {where}")
    L.append("")

    L += ["## hasDefinition 证据落点"]
    for v, d in r.definition_anchor.items():
        L.append(f"- {v}：{_kv(d)}")
    L.append(f"- 锚在标题行：{len(r.definition_title_row)} 条；锚在参考文献：{len(r.definition_references)} 条")
    for x in r.definition_title_row[:5]:
        L.append(f"  - 标题行 {x['record_id']} {x['subject_label']} → {x['anchor']}《{x['anchor_text']}》")
    for x in r.definition_references:
        L.append(f"  - 参考文献 {x['record_id']} {x['subject_label']} → {x['anchor']}《{x['anchor_text']}》")
    L.append("")

    L += ["## 跨章同名概念", f"- 共 {len(r.cross_chapter_names)} 组"]
    for x in r.cross_chapter_names[:10]:
        L.append(f"  - {x['name']}：{'、'.join(x['ids'])}")
    L.append("")

    z = r.zero_mention
    if z:
        L += ["## 零提及概念（正文+小结中无可匹配别名出现）", f"- {z['count']} 个；按章：{_kv(z['by_chapter'])}",
              f"- 别名全被 blocked、无法判断：{len(z['no_matchable_alias'])} 个", ""]

    t = r.top3_coverage
    L += ["## Top3 候选覆盖（onto2，概念→概念，有向）",
          f"- 参与统计的概念：{t['concepts_considered']}；候选对：{t['candidate_pairs']}"
          f"（含多条断言的 {t['pairs_with_multiple_edges']}）",
          f"- 出度分布（候选数）：{_kv(t['out_degree_distribution'])}",
          f"- ≥3 个候选：{t['with_at_least_3']}；不足 3 个：{t['fewer_than_3']}；仅有入边无出边：{t['no_outgoing_but_has_incoming']}",
          f"- 双向成对：{t['reverse_pairs_present']}", f"- 谓词分布：{_kv(t['predicate_counts'])}", ""]

    a = r.aliases
    L += ["## 别名表", f"- 表面形式 {a['surfaces']}：可匹配 {a['matchable']}（同形多概念 {a['ambiguous']}），"
          f"blocked {a['blocked']}（{_kv(a['blocked_by_reason'])}）", ""]
    if r.student_set:
        L += ["## 学生知识集合核对", f"- {_kv(r.student_set)}", ""]
    return "\n".join(L)
