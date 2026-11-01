"""Step 3：LLM 给召回子块逐块定级（direct / background / irrelevant），按批调用后合并。

- 子块在 prompt 中用短编号 S1、S2…（全局递增，跨批不重复），LLM 看不到 chunk_id；本模块负责映射回 chunk_id；
- 只接受输入批次中出现过的编号，编造的编号丢弃并记录；
- 批内某个子块没有被判断，或等级不合法：不静默当作无用，记为 unjudged（默认不进入作答并记录）；
- 作答材料由 pick_for_answer 选出：先 direct 后 background，各自保持召回顺序，最多 max_chunks 个；
- 原生路径：response_format=json_object；回退路径：prompt 要求 JSON + 本地解析，失败带错误重试 1 次。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient, parse_json_value
from .recall import Recalled

_REF_RE = re.compile(r"^\[?\s*[Ss](\d+)\s*\]?$")

DIRECT, BACKGROUND, IRRELEVANT = "direct", "background", "irrelevant"
GRADES = (DIRECT, BACKGROUND, IRRELEVANT)


@dataclass(frozen=True)
class Verdict:
    chunk_id: str
    useful: bool
    reason: str
    ref: str = ""
    grade: str = ""

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "ref": self.ref, "grade": self.grade, "useful": self.useful,
                "reason": self.reason}


def _grade_of(it: dict) -> str | None:
    """读取等级；兼容旧格式 useful=true/false（true 视为 direct）。不合法返回 None。"""
    g = str(it.get("grade", "")).strip().lower()
    if g in GRADES:
        return g
    if "grade" not in it and isinstance(it.get("useful"), bool):
        return DIRECT if it["useful"] else IRRELEVANT
    return None


@dataclass
class FilterResult:
    verdicts: list[Verdict] = field(default_factory=list)
    dropped_unknown_ids: list[str] = field(default_factory=list)   # LLM 给出的、输入里不存在的编号
    unjudged: list[str] = field(default_factory=list)              # 输入了但 LLM 没给判断的 chunk_id
    duplicates: list[str] = field(default_factory=list)
    batches: list[dict] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)             # S# -> chunk_id（全部输入）

    @property
    def useful_ids(self) -> list[str]:
        return [v.chunk_id for v in self.verdicts if v.useful]

    def grade_of(self, chunk_id: str) -> str | None:
        return next((v.grade for v in self.verdicts if v.chunk_id == chunk_id), None)

    def to_dict(self) -> dict:
        return {"verdicts": [v.to_dict() for v in self.verdicts], "dropped_unknown_ids": self.dropped_unknown_ids,
                "unjudged": self.unjudged, "duplicates": self.duplicates, "batches": self.batches,
                "refs": self.refs}


def assign_refs(items: list[Recalled], start: int = 1) -> dict[str, str]:
    """按顺序给子块分配 S# 编号，返回 {S#: chunk_id}。"""
    return {f"S{start + i}": it.chunk_id for i, it in enumerate(items)}


def _norm_ref(raw) -> str:
    m = _REF_RE.match(str(raw or "").strip())
    return f"S{int(m.group(1))}" if m else str(raw or "").strip()


def _render_items(batch: list[Recalled], ref_of: dict[str, str]) -> str:
    return "\n\n".join(
        prompts.FILTER_ITEM.format(ref=ref_of[it.chunk_id], path=" › ".join(it.path), text=it.text) for it in batch)


def _parse_verdicts(content: str) -> list[dict]:
    val = parse_json_value(content)
    items = val.get("verdicts") if isinstance(val, dict) else val
    if not isinstance(items, list):
        raise ValueError("缺少 verdicts 数组")
    return items


def _judge_batch(question: str, batch: list[Recalled], ref_of: dict[str, str], llm: LLMClient, model: str,
                 native: bool, meta: dict) -> list[dict]:
    system = prompts.FILTER_SYSTEM + ("" if native else prompts.FILTER_JSON_FALLBACK)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompts.FILTER_USER.format(
                    question=question, items=_render_items(batch, ref_of))}]
    for attempt in (0, 1):
        res = llm.chat("filter", messages, model=model, json_mode=native)
        try:
            return _parse_verdicts(res.content)
        except ValueError as e:
            meta["parse_failures"] += 1
            if attempt == 1:
                return []
            messages.append({"role": "assistant", "content": res.content})
            messages.append({"role": "user", "content": prompts.RETRY_JSON.format(error=str(e)[:80])})
    return []


def filter_passages(question: str, batches: list[list[Recalled]], llm: LLMClient, *,
                    model: str | None = None, native_tools: bool | None = None) -> FilterResult:
    model = model or config.COURSE_FILTER_MODEL
    native = config.COURSE_LLM_NATIVE_TOOLS if native_tools is None else native_tools
    out = FilterResult()
    flat = [it for b in batches for it in b]
    out.refs = assign_refs(flat)
    ref_of = {cid: ref for ref, cid in out.refs.items()}
    judged: set[str] = set()
    for bi, batch in enumerate(batches):
        meta = {"index": bi, "size": len(batch), "chars": sum(len(i.text) for i in batch),
                "tier_min": min(i.tier for i in batch), "tier_max": max(i.tier for i in batch),
                "parse_failures": 0}
        allowed = {ref_of[i.chunk_id]: i.chunk_id for i in batch}
        for it in _judge_batch(question, batch, ref_of, llm, model, native, meta):
            if not isinstance(it, dict):
                continue
            ref = _norm_ref(it.get("id", it.get("ref", it.get("chunk_id"))))
            cid = allowed.get(ref)
            grade = _grade_of(it)
            if cid is None:
                out.dropped_unknown_ids.append(ref)
            elif cid in judged:
                out.duplicates.append(ref)
            elif grade is None:
                meta["invalid_grade"] = meta.get("invalid_grade", 0) + 1
            else:
                judged.add(cid)
                out.verdicts.append(Verdict(cid, grade != IRRELEVANT, str(it.get("reason", "")).strip(), ref, grade))
        miss = [i.chunk_id for i in batch if i.chunk_id not in judged]
        out.unjudged += miss
        meta["unjudged"] = len(miss)
        out.batches.append(meta)
    return out


def pick_for_answer(items: list[Recalled], fr: FilterResult, max_chunks: int | None = None,
                    ) -> tuple[list[Recalled], list[str]]:
    """选作答材料：先 direct 后 background，各自保持召回顺序，最多 max_chunks 个。返回 (材料, 因上限被截掉的 chunk_id)。"""
    max_chunks = config.COURSE_ANSWER_MAX_CHUNKS if max_chunks is None else max_chunks
    grade = {v.chunk_id: v.grade for v in fr.verdicts}
    ranked = [i for g in (DIRECT, BACKGROUND) for i in items if grade.get(i.chunk_id) == g]
    return ranked[:max_chunks], [i.chunk_id for i in ranked[max_chunks:]]


def cap_for_filter(items: list[Recalled], max_chunks: int | None = None, question: str = "",
                   concept_names: list[str] | None = None) -> tuple[list[Recalled], dict]:
    """筛选一次调用的输入上限；超出时按「与问题 / 已选概念的匹配程度」排序取前 max_chunks 块，不再按召回顺序截断。

    排序键（从高到低）：
      1. 命中档：有本体边证据（tier 0–2）> 仅 BM25/mention；这一档只是第一排序键，不再保证全部保留；
      2. 相关分 = 块文本与问题的字符 bigram 重合率 + 与已选概念名的重合率（本体块内部也据此区分）+ 该块的 BM25 分（若有）归一化；
      3. 并列时保持原召回顺序（稳定）。
    kept 返回时保持原召回顺序。丢弃块数、来源、被丢块中最高相关分写入统计，便于排查误丢。"""
    max_chunks = config.COURSE_FILTER_MAX_CHUNKS if max_chunks is None else max_chunks
    stats = {"limit": max_chunks, "total": len(items), "kept": len(items), "dropped": 0, "dropped_by_source": {},
             "ranking": "relevance"}
    if len(items) <= max_chunks:
        return items, stats
    order = {i.chunk_id: n for n, i in enumerate(items)}
    q_grams = _grams(question)
    name_grams = [_grams(n) for n in (concept_names or []) if n]
    bm = {i.chunk_id: max((r.score or 0.0 for r in i.reasons if r.via == "bm25"), default=0.0) for i in items}
    bm_max = max(bm.values()) or 1.0

    def score(i: Recalled) -> float:
        g = _grams(" ".join(i.path[-2:]) + i.text)
        overlap_q = len(g & q_grams) / (len(q_grams) or 1)
        overlap_n = max((len(g & ng) / (len(ng) or 1) for ng in name_grams), default=0.0)
        return overlap_q + 0.5 * overlap_n + 0.5 * bm[i.chunk_id] / bm_max

    scored = {i.chunk_id: score(i) for i in items}
    ranked = sorted(items, key=lambda i: (i.tier > 2, -scored[i.chunk_id], order[i.chunk_id]))
    keep_ids = {i.chunk_id for i in ranked[:max_chunks]}
    kept = [i for i in items if i.chunk_id in keep_ids]
    drop = [i for i in items if i.chunk_id not in keep_ids]
    for i in drop:
        stats["dropped_by_source"][i.source] = stats["dropped_by_source"].get(i.source, 0) + 1
    stats.update(kept=len(kept), dropped=len(drop), dropped_ids=[i.chunk_id for i in drop],
                 dropped_max_score=round(max((scored[i.chunk_id] for i in drop), default=0.0), 3),
                 kept_min_score=round(min((scored[i.chunk_id] for i in kept), default=0.0), 3))
    return kept, stats


def _grams(text: str) -> set[str]:
    """字符 bigram 集合（去掉标点空白）；中文问句与教材块的粗粒度重合度，不依赖分词器。"""
    t = "".join(ch for ch in (text or "") if ch.isalnum())
    return {t[k:k + 2] for k in range(len(t) - 1)} if len(t) > 1 else set(t)
