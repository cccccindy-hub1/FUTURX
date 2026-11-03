"""名言检索与作答：按整次讲话检索，模型只挑段落编号，程序原文照录。

检索：以名言库为目录的 RAGService（一次讲话一块）做混合 + 重排，取前 count 次讲话；沿用 AdaptiveRAG 的置信度判断，
最高分低于阈值时改写查询重查一次，仍然不够就返回 not_found（「合集中没有找到相关表述」），不调用挑段落的模型。

作答（领导人讲话不能被模型改写）：
- 模型只做一件事：在命中的讲话里挑出最相关的段落编号，输出 JSON {"picks":[{"id":"Q1","paragraphs":[2,3]}]}；
- 程序原样输出这些段落，加出处：「……原文……」——2018 年，全国网络安全和信息化工作会议上的讲话；
- 不加模型的解读；不调用模型生成任何一个字。越界的讲话编号 / 段落编号丢弃；模型挑不出来或 JSON 非法时，
  退回每次讲话的第一段（程序决定，不重试）。
每题调用次数：路由 1 + 挑段落 1，共 2 次（改写查询时再加 1 次）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..course.llm import LLMClient, parse_json_object
from . import config

NOT_FOUND_TEXT = "合集中没有找到相关表述。"

PICK_SYSTEM = """你是引文挑选器，不写任何文字，只从给定的讲话里挑出与问题最相关的段落。
下面每次讲话以 [Q1]、[Q2]… 编号，其下每个段落以 (1)、(2)… 编号。
为每次讲话挑出最能回答问题的 1–3 个段落（只挑编号，不要改写、不要概括）；某次讲话没有相关段落就不要列出它。
只输出一个 JSON 对象：{"picks":[{"id":"Q1","paragraphs":[2,3]}]}"""


@dataclass
class SpeechHit:
    speech_id: str
    year: int
    title: str
    kind: str
    paragraphs: list[str]
    score: float = 0.0


@dataclass
class QuotePick:
    n: int                                   # Q 编号（1 起）
    speech: SpeechHit
    indices: list[int]                       # 被选中的段落序号（1 起）

    def texts(self) -> list[str]:
        return [self.speech.paragraphs[i - 1] for i in self.indices]


def source_line(s: SpeechHit) -> str:
    return f"{s.year}年，{s.title}"


class QuoteRetriever:
    def __init__(self, svc, *, min_rerank_score: float | None = None, max_distance: float | None = None):
        from ..adaptive import AdaptiveRAG
        self.svc = svc
        self._ad = AdaptiveRAG(svc, min_rerank_score=min_rerank_score, max_distance=max_distance)

    def _search(self, query: str, count: int, tm: dict):
        t0 = time.perf_counter()
        res = self.svc.retrieve_result(query, max(count, config.QUOTE_FETCH))
        rerank = float(res.debug.get("rerank_s", 0.0))
        tm["rerank_s"] += rerank
        tm["retrieve_s"] += max(time.perf_counter() - t0 - rerank, 0.0)
        ok, why = self._ad._confidence(res)
        scores = res.debug.get("rerank_scores") or {}
        hits: list[SpeechHit] = []
        seen: set[str] = set()
        for i, (cid, doc, meta) in enumerate(zip(res.ids, res.documents, res.metadatas)):
            sid = meta.get("speech_id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            s = scores.get(cid)
            if s is None:
                d = res.distances[i] if i < len(res.distances) else 1.0
                s = (1.0 - float(d)) if d else 1.0 / (i + 1)
            hits.append(SpeechHit(sid, int(meta.get("year", 0)), meta.get("title", ""), meta.get("kind", ""),
                                  _paragraphs(meta, doc), float(s)))
        return hits, ok, why

    def retrieve(self, question: str, count: int = 1, *, rewrite=None) -> dict:
        """返回 {"mode": search|not_found, "hits": [SpeechHit], "trace": {...}}。"""
        count = max(1, min(5, int(count or 1)))
        tm = {"retrieve_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0, "aux_llm_s": 0.0}
        t0 = time.perf_counter()
        trace: dict = {"retrievals": [], "count": count}
        hits, ok, why = self._search(question, count, tm)
        trace["retrievals"].append({"query": question, "reliable": ok, "speeches": len(hits), **why})
        if ok is False and rewrite is not None:                                   # 纠错：改写重查一次
            t1 = time.perf_counter()
            new_q = rewrite(question)
            tm["aux_llm_s"] += time.perf_counter() - t1
            if new_q and new_q != question:
                trace["rewritten"] = new_q
                hits, ok, why = self._search(new_q, count, tm)
                trace["retrievals"].append({"query": new_q, "reliable": ok, "speeches": len(hits), **why})
        found = bool(hits) and ok is not False
        total = round(time.perf_counter() - t0, 4)
        trace["timings"] = {**{k: round(v, 4) for k, v in tm.items()}, "total_s": total}
        return {"mode": "search" if found else "not_found", "hits": hits[:count] if found else [], "trace": trace}


def _paragraphs(meta: dict, doc: str) -> list[str]:
    """段落列表优先取元数据里的 JSON（建库时存入），否则从 [N] 标注的文本里切回来。"""
    import json
    import re
    raw = meta.get("paragraphs_json")
    if raw:
        try:
            v = json.loads(raw)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v
        except ValueError:
            pass
    parts = re.split(r"(?m)^\[(\d+)\]\s*", doc or "")
    return [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


# ---------------------------------------------------------------- 挑段落（1 次 LLM 调用）

def render_for_pick(hits: list[SpeechHit], first_n: int = 1) -> str:
    out = []
    for k, h in enumerate(hits, first_n):
        paras = "\n".join(f"({i}) {p}" for i, p in enumerate(h.paragraphs, 1))
        out.append(f"[Q{k}] {source_line(h)}（{h.kind}）\n{paras}")
    return "\n\n".join(out)


def pick_paragraphs(question: str, hits: list[SpeechHit], llm: LLMClient, *, model: str, first_n: int = 1) -> tuple[list[QuotePick], dict]:
    """一次调用，不重试。返回 (挑中的段落, 调试信息)。模型输出越界的编号一律丢弃；挑不出来就每次讲话取第一段。"""
    info: dict = {"dropped": [], "fallback": ""}
    n_of = {f"Q{k}": (k, h) for k, h in enumerate(hits, first_n)}
    msgs = [{"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": f"问题：{question}\n\n{render_for_pick(hits, first_n)}"}]
    picks: list[QuotePick] = []
    try:
        res = llm.chat("quote_pick", msgs, model=model, json_mode=True)
        obj = parse_json_object(res.content)
        raw = obj.get("picks")
        if not isinstance(raw, list):
            raise ValueError("no picks")
        seen: set[str] = set()
        for it in raw:
            if not isinstance(it, dict):
                continue
            key = str(it.get("id", "")).strip().upper().lstrip("[").rstrip("]")
            if key not in n_of or key in seen:
                info["dropped"].append({"id": key, "reason": "unknown_or_duplicate_speech"})
                continue
            seen.add(key)
            k, h = n_of[key]
            idx: list[int] = []
            for x in it.get("paragraphs") or []:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 1 <= i <= len(h.paragraphs) and i not in idx:
                    idx.append(i)
                else:
                    info["dropped"].append({"id": key, "paragraph": x, "reason": "out_of_range_or_duplicate"})
            if idx:
                picks.append(QuotePick(k, h, sorted(idx)[:3]))
    except Exception as e:  # noqa: BLE001 - 只记类型名
        info["fallback"] = type(e).__name__
        picks = []
    if not picks:                                                   # 挑不出来：程序退回每次讲话第一段（不重试、不让模型生成）
        info["fallback"] = info["fallback"] or "empty_picks"
        picks = [QuotePick(k, h, [1]) for k, h in n_of.values() if h.paragraphs]
    picks.sort(key=lambda p: p.n)
    return picks, info


# ---------------------------------------------------------------- 程序照录

def render_quote_answer(picks: list[QuotePick], numbered: bool = True) -> str:
    """原文照录：「……」——年份，讲话标题。多条用空行隔开；numbered 时在引文后附序号 [n]（来源列表里对应）。"""
    blocks = []
    for k, p in enumerate(picks, 1):
        text = "\n".join(p.texts())
        mark = f"[{k}]" if numbered else ""
        blocks.append(f"「{text}」{mark}\n——{source_line(p.speech)}")
    return "\n\n".join(blocks)
