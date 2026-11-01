"""Step 1 的本地词表工具：search_vocab / get_concept / get_neighbors（另含 Step 4 用的 relations_among）。

词表 = 概念总表去掉「D 非概念噪声」「删去」条目，再去掉别名全部被 blocked 的泛词概念（数据、技术等）。
不用 Embedding：检索 = 别名字面匹配 + BM25（jieba 或字符 bigram）。
关系保持有向：A→B 不会生成 B→A。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .aliases import AliasTable, norm, split_name
from .bm25 import BM25, tokenize
from .ontology import ConceptTable, Record

EXCLUDED_ISSUES = ("D 非概念噪声", "删去")
SNIPPET_CHARS = 60
VIEWS = ("onto1", "onto2", "both")

SCORE_ALIAS_EXACT = 1000.0
SCORE_ALIAS_IN_QUERY = 500.0
SCORE_QUERY_IN_ALIAS = 300.0

REJECT_UNKNOWN = "unknown_concept_id"
REJECT_EXCLUDED = "excluded_issue_type"
REJECT_BLOCKED = "blocked_generic"


@dataclass(frozen=True)
class VocabHit:
    concept_id: str
    name: str
    chapter: int
    ctype: str
    snippet: str
    score: float
    via: str          # alias_exact | alias_in_query | query_in_alias | bm25

    def to_dict(self) -> dict:
        return {"concept_id": self.concept_id, "name": self.name, "chapter": self.chapter,
                "type": self.ctype, "snippet": self.snippet, "via": self.via, "score": round(self.score, 3)}


@dataclass(frozen=True)
class ConceptCard:
    concept_id: str
    name: str
    chapter: int
    topic: str
    ctype: str
    context: str
    aliases: list[str]
    same_name_ids: list[str]
    selectable: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Neighbor:
    concept_id: str
    name: str
    chapter: int
    predicate_id: str
    direction: str    # out：当前概念 → 邻居；in：邻居 → 当前概念
    view: str

    def to_dict(self) -> dict:
        return asdict(self)


class VocabIndex:
    def __init__(self, concepts: ConceptTable, aliases: AliasTable, records: list[Record], *,
                 excluded: tuple[str, ...] = EXCLUDED_ISSUES, use_jieba: bool | None = None):
        self.concepts = concepts.by_id
        self.aliases = aliases
        self.excluded = tuple(excluded)
        self._use_jieba = use_jieba
        self.selectable: set[str] = {
            c.id for c in concepts.concepts if c.issue_type not in self.excluded and aliases.aliases_of(c.id)
        }
        self._doc_ids = [c.id for c in concepts.concepts if c.id in self.selectable]
        self._bm25 = BM25([tokenize(self._doc_text(i), use_jieba) for i in self._doc_ids])
        self._name_groups: dict[str, list[str]] = defaultdict(list)
        for c in concepts.concepts:
            if c.issue_type not in self.excluded:
                self._name_groups[norm(split_name(c.name)[0])].append(c.id)
        self._neighbors: dict[str, list[Neighbor]] = defaultdict(list)
        self._edge_map: dict[tuple[str, str], dict[tuple[str, str], str]] = defaultdict(dict)
        self._build_edges(records)

    # ---- 构建 ----
    def _doc_text(self, cid: str) -> str:
        c = self.concepts[cid]
        surfaces = [a.surface for a in self.aliases.aliases_of(cid)]
        return " ".join([c.name] * 3 + surfaces * 2 + [(c.context or c.excerpt)[:300]])

    def _build_edges(self, records: list[Record]) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        for r in records:
            if not r.is_concept_edge or r.subject_id == r.object_id:
                continue
            s, o = r.subject_id, r.object_id
            if s not in self.selectable or o not in self.selectable:
                continue
            self._edge_map[(s, o)].setdefault((r.view, r.predicate_id), r.predicate_label)
            key = (s, o, r.predicate_id, r.view)
            if key in seen:
                continue
            seen.add(key)
            cs, co = self.concepts[s], self.concepts[o]
            self._neighbors[s].append(Neighbor(o, co.name, co.chapter, r.predicate_id, "out", r.view))
            self._neighbors[o].append(Neighbor(s, cs.name, cs.chapter, r.predicate_id, "in", r.view))
        for lst in self._neighbors.values():
            lst.sort(key=lambda n: (n.view, n.direction, n.predicate_id, n.concept_id))

    # ---- 校验 ----
    def reject_reason(self, concept_id: str) -> str | None:
        """None 表示可选；否则返回拒绝原因。"""
        c = self.concepts.get(concept_id)
        if c is None:
            return REJECT_UNKNOWN
        if c.issue_type in self.excluded:
            return REJECT_EXCLUDED
        if concept_id not in self.selectable:
            return REJECT_BLOCKED
        return None

    # ---- 工具 ----
    def search_vocab(self, query: str, chapter: int | None = None, limit: int = 30) -> list[VocabHit]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 30
        try:
            chapter = int(chapter) if chapter not in (None, "") else None
        except (TypeError, ValueError):
            chapter = None

        best: dict[str, tuple[float, str]] = {}

        def bump(cid: str, score: float, via: str) -> None:
            if cid in self.selectable and (cid not in best or score > best[cid][0]):
                best[cid] = (score, via)

        qk = norm(q)
        for key, a in self.aliases.by_key.items():
            if a.blocked:
                continue
            if key == qk:
                score, via = SCORE_ALIAS_EXACT, "alias_exact"
            elif key in qk:
                score, via = SCORE_ALIAS_IN_QUERY + len(key), "alias_in_query"
            elif len(qk) >= 2 and qk in key:
                score, via = SCORE_QUERY_IN_ALIAS + len(qk), "query_in_alias"
            else:
                continue
            for cid in a.concept_ids:
                bump(cid, score, via)
        for i, sc in self._bm25.scores(tokenize(q, self._use_jieba)).items():
            bump(self._doc_ids[i], sc, "bm25")

        hits = []
        for cid, (score, via) in best.items():
            c = self.concepts[cid]
            if chapter is not None and c.chapter != chapter:
                continue
            hits.append(VocabHit(cid, c.name, c.chapter, c.ctype, (c.context or c.excerpt)[:SNIPPET_CHARS], score, via))
        hits.sort(key=lambda h: (h.via == "bm25", -h.score, h.concept_id))   # 名称/别名命中一律排在 BM25 命中之前
        return hits[:limit]

    def get_concept(self, concept_id: str) -> ConceptCard | None:
        c = self.concepts.get(concept_id)
        if c is None:
            return None
        same = [i for i in self._name_groups.get(norm(split_name(c.name)[0]), []) if i != concept_id]
        return ConceptCard(
            c.id, c.name, c.chapter, c.topic, c.ctype, c.context or c.excerpt,
            [a.surface for a in self.aliases.aliases_of(c.id)], same, concept_id in self.selectable,
        )

    def get_neighbors(self, concept_id: str, view: str = "both") -> list[Neighbor]:
        if view not in VIEWS:
            raise ValueError(f"view 必须是 {VIEWS} 之一")
        return [n for n in self._neighbors.get(concept_id, []) if view == "both" or n.view == view]

    def relations_among(self, concept_ids) -> list[dict]:
        """给定概念集合内部的有向关系（Step 4 的本体结构信息），按 (主语, 宾语, 视角, 谓词) 去重排序。"""
        ids = [i for i in dict.fromkeys(concept_ids) if i in self.concepts]
        out = []
        for s in ids:
            for o in ids:
                for (view, pred), label in sorted(self._edge_map.get((s, o), {}).items()):
                    out.append({
                        "subject_id": s, "subject": self.concepts[s].name, "predicate_id": pred,
                        "predicate_label": label, "object_id": o, "object": self.concepts[o].name, "view": view,
                    })
        return out
