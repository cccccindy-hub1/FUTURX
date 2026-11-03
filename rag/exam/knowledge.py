"""把每道题挂到学生知识集合的知识点上——试卷库的颗粒度与学生知识集合对齐。

学生知识集合（三层集合之一，17346 条）的最小单位是一条知识三元组：
  章节 · 主体 —关系→ 客体（如 Ch01 · 数据要素 —hasDefinition→ 「根据特定生产需求……」）。
试卷库的检索单位是一道完整的题（不拆题），但每道题带上它考查的知识点：
- concepts：题目考查的概念（学生知识集合的「主体」名），一题可多个（简答 / 论述常考两三个概念的关系）；
- kp_ids：这些概念在学生知识集合里的知识三元组 ID（Student_xxxxxx），只取内容性关系
  （定义、特征、构成、类型、方法、流程、风险、区别……），不取 instanceOf / hasSourceSpan 等结构性关系；
  两个概念之间若有直接三元组（如 A —distinguishesFrom→ B），优先挂这条。
于是「按知识点找题」和「按题找知识点」都在学生知识集合的同一粒度上进行。

匹配顺序（同章优先；题库第 11 章「数据工程师」教材没有对应章，只能跨章匹配）：
1. 引号概念与主体名完全相同；
2. 复合概念按「、/与/和/及」拆开后逐个完全匹配（「数据权属、数据权利与数据权益」→ 三个概念）；
3. 包含匹配：主体名（≥ EXAM_MENTION_MIN_CHARS 字）出现在概念里，或概念出现在主体名里；
4. 题干字面提及：没有引号概念的题（多为判断题）用题干里出现的主体名（≥ EXAM_MENTION_MIN_CHARS 字）。
匹配方式写入 link，供审阅；挂不上的题 concepts 为空，检索时只走 BM25。
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..course.ontology import load_concepts, load_student_set
from . import config
from .bank import Question, norm

# 结构性 / 教学元数据关系：不算知识点
_STRUCTURAL = frozenset({
    "instanceOf", "hasSourceSpan", "extractedFrom", "hasConcept", "hasSource", "hasSection", "hasSubsection",
    "hasKnowledgePoint", "hasCoreKnowledgePoint", "answersGuidingQuestion", "hasGuidingQuestion",
    "hasLearningTag", "hasLearningOutcome", "hasTeachingObjective", "supportsTeachingObjective",
    "recommendedFor", "hasTopic",
})
# 关系优先级：定义 > 特征 / 构成 / 类型 > 其余
_REL_RANK = {"hasDefinition": 0, "hasFeature": 1, "hasPart": 1, "hasType": 1, "distinguishesFrom": 1,
             "hasMethod": 2, "hasProcess": 2, "hasStep": 2, "hasStage": 2, "hasPrinciple": 2, "hasRisk": 2,
             "hasRight": 2}
_SPLIT_RE = re.compile(r"[、与和及＋+]")
_HEADING_RE = re.compile(r"^(?:第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)])")
_SUBJECT_MAX = 20   # 更长的「主体」多是判例名、节标题，不当概念用


@dataclass(frozen=True)
class KnowledgePoint:
    kp_id: str
    chapter: int
    subject: str
    relation: str
    obj: str
    axiom: str


@dataclass
class StudentIndex:
    points: dict[str, KnowledgePoint]
    by_subject: dict[str, list[str]]                         # norm(主体) -> kp_ids（内容性关系，按优先级排）
    chapters_of: dict[str, set[int]]                          # norm(主体) -> 出现的章
    display: dict[str, str]                                   # norm(主体) -> 原名
    names_by_chapter: dict[int, list[str]] = field(default_factory=dict)   # 章 -> norm 主体名（长的在前）

    @classmethod
    def load(cls, path: str | Path | None = None) -> "StudentIndex":
        return cls.from_rows(load_student_set(path or config.EXAM_STUDENT_SET))

    @classmethod
    def from_rows(cls, rows: list[dict[str, str]]) -> "StudentIndex":
        points: dict[str, KnowledgePoint] = {}
        by_subject: dict[str, list[str]] = defaultdict(list)
        chapters_of: dict[str, set[int]] = defaultdict(set)
        display: dict[str, str] = {}
        for r in rows:
            rel = (r.get("关系") or "").strip()
            subj = (r.get("主体") or "").strip()
            m = re.search(r"\d+", r.get("章节") or "")
            if not subj or not m or rel in _STRUCTURAL:
                continue
            ch = int(m.group())
            kp = KnowledgePoint(kp_id=r.get("集合ID", ""), chapter=ch, subject=subj, relation=rel,
                                obj=(r.get("客体") or "").strip(), axiom=(r.get("公理") or "").strip())
            points[kp.kp_id] = kp
            key = norm(subj)
            by_subject[key].append(kp.kp_id)
            chapters_of[key].add(ch)
            display.setdefault(key, subj)
        for key, ids in by_subject.items():
            ids.sort(key=lambda i: (_REL_RANK.get(points[i].relation, 3), i))
        names_by_chapter: dict[int, list[str]] = defaultdict(list)
        for key, chs in chapters_of.items():
            if len(key) > _SUBJECT_MAX or _HEADING_RE.match(display[key]):
                continue
            for ch in chs:
                names_by_chapter[ch].append(key)
        for ch in names_by_chapter:
            names_by_chapter[ch].sort(key=len, reverse=True)
        return cls(points=points, by_subject=dict(by_subject), chapters_of=dict(chapters_of), display=display,
                   names_by_chapter=dict(names_by_chapter))

    # ---- 概念匹配 ----
    def _usable(self, key: str) -> bool:
        return key in self.by_subject and len(key) <= _SUBJECT_MAX and not _HEADING_RE.match(self.display[key])

    def _all_names(self) -> list[str]:
        if not hasattr(self, "_names_cache"):
            names = {k for k in self.by_subject if self._usable(k)}
            self._names_cache = sorted(names, key=len, reverse=True)
        return self._names_cache

    def _contained(self, text: str, chapter: int | None, min_chars: int) -> list[str]:
        """text 中出现的主体名（最长优先，已被更长匹配覆盖的位置不再匹配）。"""
        pool = self.names_by_chapter.get(chapter, []) if chapter else self._all_names()
        taken = [False] * len(text)
        out: list[str] = []
        for key in pool:
            if len(key) < min_chars:
                break
            start = text.find(key)
            while start >= 0:
                if not any(taken[start:start + len(key)]):
                    for i in range(start, start + len(key)):
                        taken[i] = True
                    if key not in out:
                        out.append(key)
                    break
                start = text.find(key, start + 1)
        return out

    def match_concept(self, concept: str, chapter: int | None) -> tuple[list[str], str]:
        """一个引号概念 → (主体 key 列表, 匹配方式)。同章优先，同章没有再跨章；chapter=None（教材无对应章）只跨章。"""
        key = norm(concept).strip("“”\"《》")
        min_chars = config.EXAM_MENTION_MIN_CHARS
        for scope, tag in _scopes(chapter):
            def ok(k: str) -> bool:
                return self._usable(k) and (scope is None or scope in self.chapters_of[k])
            if ok(key):
                return [key], f"exact:{tag}"
            parts = [p for p in _SPLIT_RE.split(key) if len(p) >= 2]
            if len(parts) > 1:
                hits = [p for p in parts if ok(p)]
                if hits:
                    return hits, f"split:{tag}"
            hits = [k for k in self._contained(key, scope, min_chars) if ok(k)]
            if hits:
                return hits, f"contains:{tag}"
            if len(key) >= min_chars:
                pool = self.names_by_chapter.get(scope, []) if scope else self._all_names()
                sup = sorted((k for k in pool if key in k and ok(k)), key=len)[:2]
                if sup:
                    return sup, f"within:{tag}"
        return [], "none"

    def knowledge_points(self, keys: list[str], chapter: int | None, per_concept: int) -> list[str]:
        """主体 key → kp_ids：两概念之间的直接三元组优先，其后每个概念按关系优先级取前 per_concept 条，同章优先。"""
        out: list[str] = []
        names = [self.display[k] for k in keys]
        if len(keys) > 1:
            for k in keys:
                for i in self.by_subject.get(k, []):
                    kp = self.points[i]
                    if any(n != kp.subject and n in kp.obj for n in names) and i not in out:
                        out.append(i)
        for k in keys:
            ids = self.by_subject.get(k, [])
            same = [i for i in ids if self.points[i].chapter == chapter]
            for i in (same or ids)[:per_concept]:
                if i not in out:
                    out.append(i)
        return out


@dataclass
class Link:
    concepts: list[str]           # 学生知识集合中的主体原名
    kp_ids: list[str]
    method: str                   # exact / split / contains / within / mention（:same 同章、:cross 跨章），多种用 + 连接


def _scopes(chapter: int | None) -> tuple:
    return ((chapter, "same"), (None, "cross")) if chapter else ((None, "cross"),)


def link_question(q: Question, idx: StudentIndex, book_chapter: int | None = -1) -> Link:
    """book_chapter：该题所在题库章对应的教材章（None = 教材无对应章，只跨章挂接；缺省 = 与题库章号相同）。"""
    ch = q.chapter if book_chapter == -1 else book_chapter
    keys: list[str] = []
    methods: list[str] = []
    for c in q.concepts:
        hits, how = idx.match_concept(c, ch)
        if hits:
            methods.append(how)
            keys.extend(k for k in hits if k not in keys)
    if not keys:   # 没有引号概念或都没匹配上：用题干字面提及
        text = norm(q.stem if q.qtype != "case" else q.stem.split("【分析问题】")[-1])
        for scope, tag in _scopes(ch):
            hits = idx._contained(text, scope, config.EXAM_MENTION_MIN_CHARS)[:4]
            if hits:
                keys, methods = hits, [f"mention:{tag}"]
                break
    kp_ids = idx.knowledge_points(keys, ch, config.EXAM_KP_PER_CONCEPT)
    return Link(concepts=[idx.display[k] for k in keys], kp_ids=kp_ids,
                method="+".join(dict.fromkeys(methods)) or "none")


def book_chapter_map(exam_titles: dict[int, str], concept_table: str | Path | None) -> dict[int, int | None]:
    """题库章号 → 教材章号：按章名与概念总表「章节主题」对应；总表不可用时按章号相同处理。"""
    path = Path(concept_table) if concept_table else None
    if path is None or not path.is_file():
        return {ch: ch for ch in exam_titles}
    topics: dict[str, int] = {}
    for c in load_concepts(path).concepts:
        topics.setdefault(norm(c.topic), c.chapter)
    return {ch: topics.get(norm(t)) for ch, t in exam_titles.items()}
