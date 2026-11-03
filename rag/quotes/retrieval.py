"""名言检索（agentic 自适应 RAG）：不用向量库、不用 bge-m3，启动时直接读原始 docx（见 QuoteIndex.load）。

1. 本地初筛（程序）：BM25 检索全部讲话段落，取最相关的 QUOTE_CANDIDATES（12）段作候选；
2. 模型挑段（1 次调用，Flash、关闭思考）：只输出段落编号，最多 count 段；都不相关时输出一个改写检索词，
   回到第 1 步重查，**只重查一次**；仍没有 → not_found（「合集中没有找到相关表述」），不再调用任何模型；
3. 原文照录（程序）：被选中段落的文字由程序原样复制，加出处。模型不生成任何一个字。
每题调用次数：路由 1 + 挑段 1 = 2 次（改写重查时 3 次）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..course.llm import parse_json_object
from . import config

NOT_FOUND_TEXT = "合集中没有找到相关表述。"

PICK_SYSTEM = """你是引文挑选器，不写任何解释，只从候选段落里挑出能回答学生问题的段落。
每个候选以 [编号] 开头，附讲话出处。请挑出最相关的段落，最多 {count} 个，按相关程度从高到低；只能用候选里出现过的编号，不要改写、不要概括。
如果候选里没有一个直接相关，不要硬选：改为给出一个改写后的检索词 rewrite（换成更贴近讲话用语的关键词），picks 留空。
只输出一个 JSON 对象：{{"picks": [编号, ...], "rewrite": ""}}。picks 非空时 rewrite 填空串。"""

PICK_PARA_CHARS = 420


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


class QuoteIndex:
    """名言库内存索引：讲话 + 全部段落的 BM25（一个段落一个检索单位）。"""

    def __init__(self, speeches: list[dict]):
        from ..course.bm25 import BM25, tokenize
        self._tok = tokenize
        self.speeches = {sp["speech_id"]: sp for sp in speeches}
        self.units: list[tuple[str, int]] = []          # (speech_id, 段落序号 1 起)
        docs = []
        for sp in speeches:
            for i, para in enumerate(sp["paragraphs"], 1):
                self.units.append((sp["speech_id"], i))
                docs.append(tokenize(f"{sp['year']} {sp['title']} {para}"))
        self._bm25 = BM25(docs) if docs else None
        self.info: dict = {}

    @classmethod
    def load(cls, doc=None, *, use_cache: bool = True) -> "QuoteIndex | None":
        from ..docscache import load_or_build
        got = load_or_build(doc if doc is not None else config.QUOTE_DOC, "quotes", _speech_rows, version=PARSE_VERSION, use_cache=use_cache)
        if got is None:
            return None
        idx = cls(got[0])
        idx.info = got[1]
        return idx

    def speech_hit(self, speech_id: str) -> SpeechHit:
        sp = self.speeches[speech_id]
        return SpeechHit(speech_id, int(sp["year"]), sp["title"], sp["kind"], list(sp["paragraphs"]))

    def candidates(self, query: str, limit: int | None = None) -> list[tuple[str, int, float]]:
        if self._bm25 is None:
            return []
        limit = limit or config.QUOTE_CANDIDATES
        return [(self.units[i][0], self.units[i][1], sc) for i, sc in self._bm25.top(self._tok(query), limit)]


PARSE_VERSION = 1


def _speech_rows(doc) -> list[dict]:
    from .chunker import chunk_quotes
    from .parser import parse_quotes
    out = []
    for c in chunk_quotes(parse_quotes(doc)):
        out.append({"speech_id": c.speech_id, "year": c.year, "n": c.n, "title": c.title, "kind": c.kind, "paragraphs": c.paragraphs})
    return out


def pick_paragraphs(question: str, index: QuoteIndex, cands: list[tuple[str, int, float]], count: int, llm, *, model: str):
    """一次调用。返回 (选中的 [(speech_id, 段落序号)], 改写词, 调试信息)。越界 / 重复编号丢弃；输出不可用 → 一个都没选。"""
    info: dict = {"dropped": [], "error": ""}
    lines, key = [], {}
    for k, (sid, pi, _) in enumerate(cands, 1):
        sp = index.speeches[sid]
        lines.append(f"[{k}] {sp['year']}年，{sp['title']}\n{sp['paragraphs'][pi - 1][:PICK_PARA_CHARS]}")
        key[str(k)] = (sid, pi)
    msgs = [{"role": "system", "content": PICK_SYSTEM.format(count=count)},
            {"role": "user", "content": f"学生问题：{question}\n\n候选段落：\n" + "\n\n".join(lines)}]
    try:
        obj = parse_json_object(llm.chat("quote_pick", msgs, model=model, json_mode=True).content)
    except Exception as e:  # noqa: BLE001 - 只记类型名
        info["error"] = type(e).__name__
        return [], "", info
    picked: list[tuple[str, int]] = []
    raw = obj.get("picks")
    for x in raw if isinstance(raw, list) else []:
        v = key.get(str(x).strip().lstrip("[").rstrip("]"))
        if v is None or v in picked:
            info["dropped"].append({"pick": str(x)[:20], "reason": "unknown_or_duplicate"})
        elif len(picked) < count:
            picked.append(v)
    rewrite = str(obj.get("rewrite") or "").strip()[:60] if not picked else ""
    return picked, rewrite, info


def group_picks(index: QuoteIndex, picked: list[tuple[str, int]], first_n: int = 1) -> list[QuotePick]:
    """按讲话归并（保持首次出现的顺序），同一讲话的段落按原文顺序排。Q 编号从 first_n 起。"""
    order: list[str] = []
    by: dict[str, list[int]] = {}
    for sid, pi in picked:
        if sid not in by:
            order.append(sid)
        by.setdefault(sid, []).append(pi)
    return [QuotePick(first_n + k, index.speech_hit(sid), sorted(set(by[sid]))) for k, sid in enumerate(order)]


class QuoteRetriever:
    """index：QuoteIndex；llm：胜算云 OpenAICompatClient；model：挑段模型（Flash、关闭思考）。"""

    def __init__(self, index: QuoteIndex, llm, *, model: str):
        self.index, self.llm, self.model = index, llm, model

    def retrieve(self, question: str, count: int = 1, *, first_n: int = 1) -> dict:
        """返回 {"mode": search|not_found, "picks": [QuotePick], "trace": {...}}。"""
        count = max(1, min(5, int(count or 1)))
        t0 = time.perf_counter()
        trace: dict = {"count": count, "rounds": [], "llm_calls": 0}
        query = question
        for round_no in (1, 2):
            cands = self.index.candidates(query)
            rd: dict = {"query": query, "candidates": len(cands)}
            trace["rounds"].append(rd)
            if not cands:
                break
            t1 = time.perf_counter()
            picked, new_q, info = pick_paragraphs(question, self.index, cands, count, self.llm, model=self.model)
            trace["llm_calls"] += 1
            rd.update(pick_s=round(time.perf_counter() - t1, 4), picked=[f"{a}#{b}" for a, b in picked], rewrite=new_q, **info)
            if picked:
                trace["total_s"] = round(time.perf_counter() - t0, 4)
                return {"mode": "search", "picks": group_picks(self.index, picked, first_n), "trace": trace}
            if round_no == 1 and new_q and new_q != query:
                trace["rewritten"] = new_q
                query = new_q
                continue
            break
        trace["total_s"] = round(time.perf_counter() - t0, 4)
        return {"mode": "not_found", "picks": [], "trace": trace}


# ---------------------------------------------------------------- 程序照录

def render_quote_answer(picks: list[QuotePick], numbered: bool = True) -> str:
    """原文照录：「……」——年份，讲话标题。多条用空行隔开；numbered 时在引文后附序号 [n]（来源列表里对应）。"""
    blocks = []
    for k, p in enumerate(picks, 1):
        text = "\n".join(p.texts())
        mark = f"[{k}]" if numbered else ""
        blocks.append(f"「{text}」{mark}\n——{source_line(p.speech)}")
    return "\n\n".join(blocks)
