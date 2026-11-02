"""试卷库检索：结构化优先、BM25 兜底、低分不硬凑——不调用 LLM。

为什么不用自适应 RAG（CRAG 式「检索→LLM 判相关→不够再检索」）：
试卷库返回的是题目原文，不需要模型综合作答；用户要的通常是结构化条件（哪个知识点、第几章、
什么题型、几道、要不要答案），用元数据过滤比向量相似度 + LLM 打分更准、更快、零成本、结果可复现。
「自适应」体现在按问题选策略，而不是多轮 LLM 调用：
1. 题号直查：「第一章选择题第 3 题」→ 精确定位；
2. 知识点检索：问题里出现学生知识集合的概念 → 按概念找挂接了该知识点的题（与学生知识集合同粒度）；
3. BM25 兜底：概念没命中或题数不够 → 对题干 + 概念名 + 正确答案做 BM25；
4. 最高分低于 EXAM_BM25_MIN_SCORE → 返回 no_match，不拿不相关的题凑数。
课程问答后的「要不要来 1–3 道【基础】题」直接用 by_concepts(课程选中的概念, basic_only=True)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..course.bm25 import BM25, tokenize
from . import config
from .bank import BASIC_QTYPES, QTYPES, Question, _cn_int, compare_student, norm, parse_bank
from .knowledge import StudentIndex, book_chapter_map, link_question

_QTYPE_WORDS = (("名词解释", "term"), ("案例", "case"), ("论述", "essay"), ("简答", "short"),
                ("判断", "judge"), ("选择", "choice"), ("单选", "choice"))
_CHAPTER_RE = re.compile(r"第\s*([一二三四五六七八九十]+|\d{1,2})\s*章")
_NUMBER_RE = re.compile(r"第\s*(\d{1,3})\s*[题道]")
_COUNT_RE = re.compile(r"([1-9]|10|[一二两三四五六七八九十])\s*[道个]")
_STOP = frozenset("什么 是 的 有 哪些 吗 呢 了 给 我 来 出 几 道 题 题目 练习 练习题 习题 试题 考题 一下 关于 相关 请 "
                  "和 与 及 做 刷 测验 答案 参考 基础 章 第".split())
_CN_COUNT = {"两": 2}


@dataclass
class ExamItem:
    q: Question
    concepts: list[str]           # 学生知识集合主体名（挂接结果）
    kp_ids: list[str]
    link_method: str
    search_text: str = ""

    def view(self, *, include_answer: bool = False, kp_lookup=None) -> dict:
        q = self.q
        d = {"question_id": q.question_id, "chapter": q.chapter, "chapter_title": q.chapter_title,
             "qtype": q.qtype, "qtype_cn": q.qtype_cn, "number": q.number, "basic": q.basic, "stem": q.stem,
             "options": q.options, "sub_questions": q.sub_questions, "concepts": self.concepts,
             "kp_ids": self.kp_ids}
        if kp_lookup is not None:
            d["knowledge_points"] = [kp_lookup(i) for i in self.kp_ids]
        if include_answer:
            d.update(answer=q.answer, analysis=q.analysis, scoring=q.scoring)
        return d


@dataclass
class QueryIntent:
    chapter: int | None = None
    qtypes: tuple[str, ...] = ()
    basic_only: bool = False
    count: int | None = None
    number: int | None = None
    want_answer: bool = False
    concepts: list[str] = field(default_factory=list)


def _search_text(q: Question, concepts: list[str]) -> str:
    stem = q.stem
    if q.qtype == "case":   # 案例材料很长且多为情节，检索只看标题与分析问题
        title = re.search(r"【案例标题】(.*?)【", stem, re.S)
        stem = (title.group(1) if title else "") + " " + " ".join(q.sub_questions)
    right = q.options.get(q.answer, "") if q.qtype == "choice" else ""
    return " ".join([q.chapter_title, stem, right, " ".join(concepts), " ".join(q.concepts)])


class ExamBank:
    def __init__(self, items: list[ExamItem], idx: StudentIndex, stats: dict | None = None):
        self.items = items
        self.idx = idx
        self.stats = stats or {}
        self.by_id = {it.q.question_id: it for it in items}
        self.by_key = {(it.q.chapter, it.q.qtype, it.q.number): it for it in items}
        self.by_concept: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            for c in it.concepts:
                self.by_concept.setdefault(norm(c), []).append(i)
        self._bm25 = BM25([self._tokens(it.search_text) for it in items])

    # ---- 构建 ----
    @classmethod
    def build(cls, teacher_doc: str | Path | None = None, student_set: str | Path | None = None,
              student_doc: str | Path | None = None, concept_table: str | Path | None = None) -> "ExamBank":
        questions, stats = parse_bank(teacher_doc or config.EXAM_TEACHER_DOC)
        idx = StudentIndex.load(student_set)
        chapters = book_chapter_map({q.chapter: q.chapter_title for q in questions},
                                    concept_table if concept_table is not None else config.EXAM_CONCEPT_TABLE)
        stats["book_chapter_map"] = {str(k): v for k, v in sorted(chapters.items())}
        items = []
        for q in questions:
            link = link_question(q, idx, chapters.get(q.chapter))
            items.append(ExamItem(q, link.concepts, link.kp_ids, link.method, _search_text(q, link.concepts)))
        stats["linked"] = sum(1 for it in items if it.kp_ids)
        if student_doc is not None and Path(student_doc).exists():
            stats["student_check"] = compare_student(questions, student_doc)
        return cls(items, idx, stats)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in _STOP]

    def kp_view(self, kp_id: str) -> dict:
        kp = self.idx.points.get(kp_id)
        if kp is None:
            return {"kp_id": kp_id}
        return {"kp_id": kp_id, "chapter": kp.chapter, "subject": kp.subject, "relation": kp.relation,
                "object": kp.obj[:200]}

    # ---- 意图解析 ----
    def parse_intent(self, query: str) -> QueryIntent:
        it = QueryIntent()
        m = _CHAPTER_RE.search(query)
        if m:
            s = m.group(1)
            it.chapter = int(s) if s.isdigit() else _cn_int(s)
        it.qtypes = tuple(dict.fromkeys(code for word, code in _QTYPE_WORDS if word in query))
        it.basic_only = "基础" in query and not it.qtypes
        m = _NUMBER_RE.search(query)
        if m:
            it.number = int(m.group(1))
        m = _COUNT_RE.search(query)
        if m:
            s = m.group(1)
            it.count = int(s) if s.isdigit() else _CN_COUNT.get(s) or _cn_int(s)
        it.want_answer = any(w in query for w in ("答案", "解析", "评分", "怎么答", "参考答案"))
        text = norm(_CHAPTER_RE.sub("", query))
        it.concepts = [self.idx.display[k] for k in self.idx._contained(text, None, 3)
                       if norm(k) in self.by_concept]
        return it

    # ---- 检索 ----
    def _filter(self, it: ExamItem, chapter, qtypes, basic_only, exclude) -> bool:
        q = it.q
        return ((chapter is None or q.chapter == chapter) and (not qtypes or q.qtype in qtypes)
                and (not basic_only or q.qtype in BASIC_QTYPES) and q.question_id not in exclude)

    def by_concepts(self, concepts: list[str], k: int = 3, *, chapter: int | None = None,
                    qtypes: tuple[str, ...] = (), basic_only: bool = False, exclude_ids=()) -> list[ExamItem]:
        """挂接了这些知识点的题。排序：覆盖的请求概念多者先；题目考查的概念少（更聚焦）者先；
        未指定题型时按题型轮转取，避免全是选择题。"""
        wanted = [norm(c) for c in concepts]
        exclude = set(exclude_ids)
        scored: dict[int, tuple] = {}
        for rank, key in enumerate(wanted):
            for i in self.by_concept.get(key, []):
                it = self.items[i]
                if not self._filter(it, chapter, qtypes, basic_only, exclude):
                    continue
                own = [norm(c) for c in it.concepts]
                hit = sum(1 for w in wanted if w in own)
                primary = 1 if own and own[0] in wanted else 0
                scored[i] = (-hit, -primary, len(own), rank, i)
        ordered = [self.items[i] for i in sorted(scored, key=lambda i: scored[i])]
        if qtypes or len(ordered) <= k:
            return ordered[:k]
        # 题型轮转
        buckets: dict[str, list[ExamItem]] = {}
        for it in ordered:
            buckets.setdefault(it.q.qtype, []).append(it)
        order = [t for t in QTYPES.values() if t in buckets]
        out: list[ExamItem] = []
        while len(out) < k and any(buckets[t] for t in order):
            for t in order:
                if buckets[t] and len(out) < k:
                    out.append(buckets[t].pop(0))
        return out

    def bm25(self, query: str, k: int, *, chapter=None, qtypes=(), basic_only=False, exclude_ids=()):
        exclude = set(exclude_ids)
        ranked = sorted(self._bm25.scores(self._tokens(query)).items(), key=lambda x: -x[1])
        out = []
        for i, s in ranked:
            it = self.items[i]
            if self._filter(it, chapter, qtypes, basic_only, exclude):
                out.append((it, s))
                if len(out) >= k:
                    break
        return out

    def search(self, query: str, k: int | None = None, *, chapter: int | None = None,
               qtypes: tuple[str, ...] = (), basic_only: bool | None = None) -> dict:
        """自然语言找题。返回 {"status", "strategy", "intent", "items"}；参数显式给出时覆盖从问题里解析的条件。"""
        intent = self.parse_intent(query)
        chapter = chapter if chapter is not None else intent.chapter
        qtypes = tuple(qtypes) or intent.qtypes
        basic_only = intent.basic_only if basic_only is None else basic_only
        k = max(1, min(10, k or intent.count or config.EXAM_SEARCH_TOPK))
        base = {"intent": {"chapter": chapter, "qtypes": list(qtypes), "basic_only": basic_only, "count": k,
                           "number": intent.number, "want_answer": intent.want_answer,
                           "concepts": intent.concepts}}

        if intent.number is not None and chapter is not None and len(qtypes) == 1:
            hit = self.by_key.get((chapter, qtypes[0], intent.number))
            if hit:
                return {**base, "status": "ok", "strategy": "number", "items": [hit]}

        items: list[ExamItem] = []
        strategy = []
        if intent.concepts:
            items = self.by_concepts(intent.concepts, k, chapter=chapter, qtypes=qtypes, basic_only=basic_only)
            if items:
                strategy.append("concept")
        if len(items) < k:
            got = {it.q.question_id for it in items}
            hits = self.bm25(query, k - len(items), chapter=chapter, qtypes=qtypes, basic_only=basic_only,
                             exclude_ids=got)
            top = hits[0][1] if hits else 0.0
            # 绝对门槛挡住完全无关的问题；相对门槛（不足最高分的 EXAM_BM25_REL）挡住凑数的弱相关题
            floor = max(config.EXAM_BM25_MIN_SCORE, top * config.EXAM_BM25_REL)
            extra = [it for it, s in hits if s >= floor]
            if extra:
                items += extra
                strategy.append("bm25")
            base["bm25_top"] = round(top, 3)
        if not items and (chapter is not None or qtypes) and not intent.concepts:
            # 只给了章 / 题型、没有主题词（如「来几道第三章判断题」）：按条件顺序取
            items = [it for it in self.items if self._filter(it, chapter, qtypes, basic_only, set())][:k]
            if items:
                strategy.append("filter")
        status = "ok" if items else "no_match"
        return {**base, "status": status, "strategy": "+".join(strategy) or "none", "items": items[:k]}


def format_items(items: list[ExamItem], *, include_answer: bool = False) -> str:
    """给 /auto/ask 的 answer 字段：题目按题库原样排版；默认不附答案（先练后看）。"""
    lines = []
    for n, it in enumerate(items, 1):
        q = it.q
        head = f"{n}.【{q.qtype_cn}·第{q.chapter}章 {q.chapter_title}】{q.stem}"
        lines.append(head)
        for letter, text in q.options.items():
            lines.append(f"  {letter}. {text}")
        if include_answer:
            lines.append(f"  【参考答案】{q.answer}")
            if q.analysis:
                lines.append(f"  【解析】{q.analysis}")
        lines.append("")
    return "\n".join(lines).strip()
