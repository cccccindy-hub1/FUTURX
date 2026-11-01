"""Top3 排序方案草案（M2 仅供确认，未启用）。

「关联概念相关性 Top3」= 该概念最相关的 3 个关联概念。数据里没有排名字段，
expert_relevance_weight（1/0.8/0.6）是来源层级而非排名，所以排名需要计算。这里给出三种方案，
均为纯函数、确定性（同分按 object_id 排序）、方向敏感（只看 subject→object 的边，不对称化）。

方案：
- A 专家权重优先：(最高专家权重, 最高置信度, 断言条数)，字典序。简单透明，但会被「全是 1.0」的专家条目压平。
- B 证据可核实优先：先看是否有至少一条边的引文能在教材锚点处核实，再按 A。避免把教材里找不到依据的关联排前面。
- C 加权综合：score = Σ 因子加权，因子 = 专家权重、置信度、证据核实度、关系类型先验、多证据加成。
  关系类型先验（predicate_prior）数据里没有，默认全部 0.5（中性），需要你按谓词给出偏好后才有意义。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .evidence import (
    NOT_FOUND, NOT_PARAGRAPH, NO_QUOTE, VERIFIED, VERIFIED_COMPOSITE, VERIFIED_NEARBY, AnchorIndex,
)
from .quality import Candidate

DEFAULT_PRIOR = 0.5
VERIFY_SCORE = {VERIFIED: 1.0, VERIFIED_NEARBY: 0.7, VERIFIED_COMPOSITE: 0.7, NOT_FOUND: 0.0, NO_QUOTE: 0.3,
                NOT_PARAGRAPH: 0.3}


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    score: tuple | float
    detail: dict = field(default_factory=dict)


def _best(vals: list[float | None]) -> float:
    xs = [v for v in vals if v is not None]
    return max(xs) if xs else 0.0


def evidence_level(cand: Candidate, anchors: AnchorIndex | None) -> float:
    """候选下所有边中最好的证据核实得分；无教材索引时为 0。"""
    if anchors is None:
        return 0.0
    return max(VERIFY_SCORE.get(anchors.check_quote(e.source_anchor, e.evidence_quote), 0.0) for e in cand.edges)


def _basic(cand: Candidate) -> tuple[float, float, int]:
    return (
        _best([e.expert_weight for e in cand.edges]),
        _best([e.confidence for e in cand.edges]),
        len(cand.edges),
    )


def scheme_a(cand: Candidate, anchors: AnchorIndex | None = None) -> Scored:
    w, c, n = _basic(cand)
    return Scored(cand, (w, c, n), {"expert_weight": w, "confidence": c, "edges": n})


def scheme_b(cand: Candidate, anchors: AnchorIndex | None = None) -> Scored:
    w, c, n = _basic(cand)
    ev = evidence_level(cand, anchors)
    return Scored(cand, (ev, w, c, n), {"evidence": ev, "expert_weight": w, "confidence": c, "edges": n})


def make_scheme_c(
    weights: dict[str, float] | None = None,
    predicate_prior: dict[str, float] | None = None,
) -> Callable[[Candidate, AnchorIndex | None], Scored]:
    wt = {"expert": 0.35, "confidence": 0.2, "evidence": 0.25, "predicate": 0.15, "multi": 0.05, **(weights or {})}
    prior = predicate_prior or {}

    def scheme_c(cand: Candidate, anchors: AnchorIndex | None = None) -> Scored:
        w, c, n = _basic(cand)
        ev = evidence_level(cand, anchors)
        p = max(prior.get(e.predicate_id, DEFAULT_PRIOR) for e in cand.edges)
        multi = min(1.0, (n - 1) / 3)
        score = (wt["expert"] * w + wt["confidence"] * c + wt["evidence"] * ev
                 + wt["predicate"] * p + wt["multi"] * multi)
        return Scored(cand, round(score, 6), {
            "expert_weight": w, "confidence": c, "evidence": ev, "predicate_prior": p, "multi": multi,
        })

    return scheme_c


SCHEMES: dict[str, Callable[[Candidate, AnchorIndex | None], Scored]] = {
    "A": scheme_a,
    "B": scheme_b,
    "C": make_scheme_c(),
}


def rank(cands: list[Candidate], scheme: Callable, anchors: AnchorIndex | None = None, k: int = 3) -> list[Scored]:
    scored = [scheme(c, anchors) for c in cands]
    scored.sort(key=lambda s: (_neg(s.score), s.candidate.object_id))
    return scored[:k]


def _neg(score):
    return tuple(-x for x in score) if isinstance(score, tuple) else -score


def tie_at_cutoff(cands: list[Candidate], scheme: Callable, anchors: AnchorIndex | None = None, k: int = 3) -> bool:
    """第 k 名与第 k+1 名得分相同：说明该方案在此概念上无法区分，只能靠 object_id 兜底。"""
    scored = [scheme(c, anchors) for c in cands]
    scored.sort(key=lambda s: (_neg(s.score), s.candidate.object_id))
    return len(scored) > k and scored[k - 1].score == scored[k].score


def compare_schemes(
    candidates: dict[str, list[Candidate]], anchors: AnchorIndex | None = None, k: int = 3,
    schemes: dict[str, Callable] | None = None,
) -> dict:
    """各方案的 Top-k 重合度与并列截断统计，用于帮助选择。"""
    schemes = schemes or SCHEMES
    tops = {name: {cid: [s.candidate.object_id for s in rank(cs, fn, anchors, k)] for cid, cs in candidates.items()}
            for name, fn in schemes.items()}
    ties = {name: sum(1 for cs in candidates.values() if tie_at_cutoff(cs, fn, anchors, k)) for name, fn in schemes.items()}
    names = list(schemes)
    overlap: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = [cid for cid in candidates if len(candidates[cid]) > k]
            if not ids:
                continue
            same = sum(len(set(tops[a][cid]) & set(tops[b][cid])) for cid in ids)
            overlap[f"{a}-{b}"] = round(same / (k * len(ids)), 3)
    return {"concepts_with_more_than_k": sum(1 for cs in candidates.values() if len(cs) > k),
            "ties_at_cutoff": ties, "topk_overlap": overlap, "tops": tops}
