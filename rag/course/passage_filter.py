"""Step 3：LLM 逐块判断召回子块是否有助于回答，按批调用后合并。

- 子块在 prompt 中用短编号 S1、S2…（全局递增，跨批不重复），LLM 看不到 chunk_id；本模块负责映射回 chunk_id；
- 只接受输入批次中出现过的编号，编造的编号丢弃并记录；
- 批内某个子块没有被判断：不静默当作无用，记为 unjudged（默认视为 useful=False 并记录）；
- 原生路径：response_format=json_object；回退路径：prompt 要求 JSON + 本地解析，失败带错误重试 1 次。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient, parse_json_value
from .recall import Recalled

_REF_RE = re.compile(r"^\[?\s*[Ss](\d+)\s*\]?$")


@dataclass(frozen=True)
class Verdict:
    chunk_id: str
    useful: bool
    reason: str
    ref: str = ""

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "ref": self.ref, "useful": self.useful, "reason": self.reason}


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
            if cid is None:
                out.dropped_unknown_ids.append(ref)
            elif cid in judged:
                out.duplicates.append(ref)
            else:
                judged.add(cid)
                out.verdicts.append(Verdict(cid, it.get("useful") is True, str(it.get("reason", "")).strip(), ref))
        miss = [i.chunk_id for i in batch if i.chunk_id not in judged]
        out.unjudged += miss
        meta["unjudged"] = len(miss)
        out.batches.append(meta)
    return out
