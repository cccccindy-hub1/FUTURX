"""Step 1（single 模式，默认）：本地确定性生成候选概念 + 一次 LLM 调用选概念。

候选生成不调用 LLM：别名字面匹配 -> 概念名/定义 BM25 -> 对字面命中的概念沿本体扩一跳；去重排序后取前 N 个，
每个候选附概念 ID、名称、所在章节、一句话定义（截断到约 60 字）。
LLM 只能从候选里选：question_concepts 1–3 个，related_concepts 0–5 个；越界 ID 丢弃并记进 trace。
解析失败或一个都没选：不重试，直接用候选前 3 个作 question_concept 继续（trace 里标注 fallback）。
旧的多轮 agent 选词保留在 agent_select.py，COURSE_SELECT_MODE=agent 切换。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, prompts
from .agent_select import SelectResult, Selection, _log_round
from .llm import LLMClient, parse_json_object
from .vocab_tools import VocabIndex

MAX_QUESTION = 3
MAX_RELATED = 5
FALLBACK_QUESTION = 3
DEFINITION_CHARS = 60


@dataclass(frozen=True)
class Candidate:
    """一条候选 = 一个概念名。同名概念（分属不同章）合并为一条：concept_id 取代表（排序最靠前的一个），
    members 是全部同名概念 {concept_id: chapter}（含代表自身），模型选中后展开为各章的概念 ID。"""
    concept_id: str
    name: str
    chapter: int
    definition: str
    via: str                 # alias_exact | alias_in_query | query_in_alias | bm25 | neighbor
    score: float
    members: tuple = ()      # ((concept_id, chapter), ...) 按章号升序

    @property
    def chapters(self) -> list[int]:
        return sorted({ch for _, ch in self.members}) or [self.chapter]

    def line(self) -> str:
        if len(self.members) > 1:
            chs = "/".join(str(c) for c in self.chapters)
            return f"{self.concept_id}｜{self.name} — 第{chs}章｜{self.definition}"
        return f"{self.concept_id}｜{self.name}｜第{self.chapter}章｜{self.definition}"

    def to_dict(self) -> dict:
        d = {"concept_id": self.concept_id, "name": self.name, "chapter": self.chapter, "via": self.via,
             "score": round(self.score, 3)}
        if len(self.members) > 1:
            d["members"] = [{"concept_id": c, "chapter": ch} for c, ch in self.members]
        return d

    def expand(self) -> list[str]:
        """选中该候选后对应的全部概念 ID（同名各章），代表概念排第一。"""
        ids = [c for c, _ in self.members] or [self.concept_id]
        return [self.concept_id] + [c for c in ids if c != self.concept_id]


_VIA_RANK = {"alias_exact": 0, "alias_in_query": 1, "query_in_alias": 2, "neighbor": 3, "bm25": 4}


def build_candidates(question: str, index: VocabIndex, limit: int | None = None) -> list[Candidate]:
    """确定性：同一输入、同一词表，结果与顺序完全一致。别名命中 > 一跳邻居 > BM25；同档按得分降序、ID 升序。"""
    limit = config.COURSE_SELECT_CANDIDATES if limit is None else limit
    hits = index.search_vocab(question, None, 100)             # 别名字面匹配 + 概念名/定义 BM25（jieba）
    best: dict[str, tuple[str, float]] = {c.concept_id: (c.via, c.score) for c in hits}
    literal = [h for h in hits if h.via != "bm25"]
    for h in literal:                                           # 对字面命中的概念沿本体扩一跳
        for n in index.get_neighbors(h.concept_id, "both"):
            if n.concept_id in index.selectable and n.concept_id not in best:
                best[n.concept_id] = ("neighbor", h.score * 0.001)
    ranked = sorted(best.items(), key=lambda kv: (_VIA_RANK[kv[1][0]], -kv[1][1], kv[0]))
    out: list[Candidate] = []
    absorbed: set[str] = set()                                 # 已被某条候选收为成员的概念，不再单独成候选
    for cid, (via, score) in ranked:
        if cid in absorbed:                                     # 同名概念合并到已有候选，不占新名额
            continue
        if len(out) >= limit:
            break
        c = index.concepts[cid]
        members = tuple(sorted(((m, index.concepts[m].chapter) for m in index.same_name_selectable(cid)),
                               key=lambda x: (x[1], x[0])))
        absorbed.update(m for m, _ in members)
        absorbed.add(cid)
        defn = " ".join((c.context or c.excerpt or "").split())[:DEFINITION_CHARS]
        out.append(Candidate(cid, c.name, c.chapter, defn, via, score, members))
    return out


def _ids(raw, allowed: set[str], cap: int, kind: str, dropped: list[dict]) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        dropped.append({"field": kind, "reason": "not_a_list"})
        return []
    out: list[str] = []
    for x in raw:
        cid = str(x).strip()
        if cid not in allowed:
            dropped.append({"field": kind, "concept_id": cid, "reason": "not_in_candidates"})
        elif cid in out:
            dropped.append({"field": kind, "concept_id": cid, "reason": "duplicate"})
        elif len(out) >= cap:
            dropped.append({"field": kind, "concept_id": cid, "reason": "over_limit"})
        else:
            out.append(cid)
    return out


def select_single(question: str, index: VocabIndex, llm: LLMClient, *, model: str | None = None,
                  limit: int | None = None) -> SelectResult:
    model = model or config.COURSE_SELECT_MODEL
    cands = build_candidates(question, index, limit)
    result = SelectResult()
    result.candidates = [c.to_dict() for c in cands]
    if not cands:
        result.end_reason, result.fallback = "no_candidates", "no_candidates"
        return result
    by_rep = {c.concept_id: c for c in cands}
    allowed = set(by_rep) | {m for c in cands for m, _ in c.members}      # 模型也可能直接写某一章的同名 ID，同样视为选中该候选
    member_to_rep = {m: c.concept_id for c in cands for m, _ in c.members}
    messages = [{"role": "system", "content": prompts.SINGLE_SELECT_SYSTEM},
                {"role": "user", "content": prompts.SINGLE_SELECT_USER.format(
                    question=question, candidates="\n".join(c.line() for c in cands))}]
    res = llm.chat("select", messages, model=model, json_mode=True)       # 一次调用，不重试
    _log_round(result, res, llm)
    try:
        obj = parse_json_object(res.content)
    except ValueError:
        result.parse_failures += 1
        obj = None
    q_ids = r_ids = []
    if obj is not None:
        q_ids = _ids(obj.get("question_concepts"), allowed, MAX_QUESTION, "question_concepts", result.dropped)
        r_ids = [c for c in _ids(obj.get("related_concepts"), allowed, MAX_RELATED, "related_concepts", result.dropped)
                 if c not in q_ids]
        result.reason = str(obj.get("reason", ""))[:200]
    if q_ids:
        result.finished, result.end_reason = True, "single"
    else:                                                                 # 解析失败或没选 question_concept：用候选前 3 个
        result.fallback = "parse_failed" if obj is None else "empty_selection"
        result.final_text = (res.content or "")[:300] if obj is None else ""
        result.end_reason = "single_fallback"
        q_ids, r_ids = [c.concept_id for c in cands[:FALLBACK_QUESTION]], []
    result.selected = _expand(q_ids, r_ids, by_rep, member_to_rep, result)
    return result


def _expand(q_ids, r_ids, by_rep, member_to_rep, result: SelectResult) -> list[Selection]:
    """把选中的候选展开为同名各章的概念 ID；同一概念只出现一次，question 优先于 related。"""
    out: list[Selection] = []
    seen: set[str] = set()
    for role, ids in (("question_concept", q_ids), ("related_concept", r_ids)):
        for cid in ids:
            cand = by_rep[member_to_rep.get(cid, cid)]
            for mid in cand.expand():
                if mid in seen:
                    continue
                seen.add(mid)
                out.append(Selection(mid, "单轮选概念" + ("" if mid == cand.concept_id else "（同名各章展开）"), role))
            if len(cand.members) > 1:
                result.expanded.append({"candidate": cand.concept_id, "name": cand.name, "to": cand.expand()})
    return out
