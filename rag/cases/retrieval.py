"""判例检索（自适应 RAG）。检索服务 svc 是一个以判例库为目录的 RAGService（见 rag.query），这里只做判例特有的逻辑：

1. 点名直达：问题里出现某案的当事人简称 / 案号 / 指导性案例编号 / 别名，直接取该案，不做向量检索；
2. 章节标签优先：路由给了 chapter 时先在该章判例里检索；该章可靠结果不够 count 个，再从其他章补足（判例常常跨章适用）；
3. 自适应检索（混合 + 重排，沿用 AdaptiveRAG 的置信度判断）→ 按判例聚合（取该案块的最高分）→ 取前 count 个判例；
4. 纠错：最高分低于阈值时改写查询重查一次；仍不够就返回 not_found，不调用作答模型；
5. 列表模式：「第六章有哪些判例」直接列出该章判例名，不检索、不调用作答模型。

向量库里的 chunk 元数据需含 case_id / seq / chapter / chapter_no / title / kind(card|body)；建库命令见 rag.cases.build。
每案材料 = 卡片 + 最相关的正文块（每案最多 CASE_MAX_BODY_CHUNKS 块）。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import config
from .parser import chapter_number

NOT_FOUND_TEXT = "判例库里没有找到与这个问题直接相关的判例。"


@dataclass
class CaseRef:
    """判例目录里的一项（来自 cases.jsonl，不含向量）。"""
    case_id: str
    seq: int
    chapter: str
    chapter_no: int
    title: str
    meta: dict = field(default_factory=dict)
    aliases: tuple = ()


@dataclass
class CaseHit:
    ref: CaseRef
    score: float                                   # 该案块的最高分（重排分或 1 - 距离）
    via: str = "search"                            # named（点名直达）/ search / search_other_chapter（跨章补足）
    card: str = ""                                 # 卡片全文（带前缀）
    bodies: list[str] = field(default_factory=list)      # 最相关的正文块
    body_headings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 判例目录与别名

_SUFFIX_RE = re.compile(r"(?:不正当竞争|侵权责任|合同|网络服务合同|著作权|侵害技术秘密|个人信息保护|隐私权|人格权|政府信息公开|执行实施|网络侵权责任)?纠纷(?:上诉)?案|案$|一审民事判决书|二审民事判决书|二审刑事判决书")
_CASE_NO_RE = re.compile(r"(?:指导性?案例)\s*(\d+)\s*号")
_DOC_NO_RE = re.compile(r"[（(]\s*\d{4}\s*[）)]\s*[^\s，。,；;]{1,10}?\d+\s*号")
_PAREN_RE = re.compile(r"[（(]([^（）()]{2,16})[）)]")
_STOP_ALIAS = {"案", "上诉案", "纠纷案", "一审", "二审", "民事判决书", "刑事判决书", "有限公司", "股份有限公司", "科技有限公司", "官方来源未披露"}
_BAD_TAIL = ("纠纷上", "纠纷", "上诉", "判决", "裁定", "侵害技术秘密", "不正当竞争", "侵权责任", "罪案", "罪")       # 以这些结尾的是被截断的案由残片，不是称呼
_PARTY_KEY = "party"                                                                                   # 当事人简称类别名的来源标记


def build_aliases(title: str, meta: dict) -> tuple:
    """从判例名和元数据提取点名用的别名：括号里的称呼（小旺神数据产品、大数据搬家软件案、DAKS数据库…）、
    指导性案例编号、案号、「X诉Y」两侧的公司 / 个人简称。过短 / 泛词 / 案由残片不收，避免误触发。
    当事人简称（最后一类）可能被多个案共用（同一家公司打了好几场官司），是否可用由 load_case_refs 统一裁定。"""
    out: list[str] = []

    def add(x: str):
        x = x.strip(" ，。、；：")
        if len(x) >= 3 and x not in _STOP_ALIAS and not x.endswith(_BAD_TAIL) and x not in out:
            out.append(x)

    for m in _PAREN_RE.finditer(title):
        inner = m.group(1)
        mm = _CASE_NO_RE.search(inner)
        if mm:
            add(f"指导性案例{mm.group(1)}号")
            add(f"指导案例{mm.group(1)}号")
        elif not inner.startswith("以下简称"):
            add(inner)
            if inner.endswith("案"):
                add(inner[:-1])
    for m in _CASE_NO_RE.finditer(title):
        add(f"指导性案例{m.group(1)}号")
        add(f"指导案例{m.group(1)}号")
    for key in ("case_number", "ref_case_no"):
        for no in re.split(r"[；;]", meta.get(key, "") or ""):
            no = re.sub(r"\s+", "", no)
            if no:
                add(no)
                m = _DOC_NO_RE.search(no)
                if m:
                    add(m.group(0))
    body = re.sub(r"[（(][^（）()]*[）)]", "", title)
    body = body.split("——")[0]
    for part in re.split(r"诉|与|、|和|等", body):
        part = _SUFFIX_RE.sub("", part).strip()
        part = re.sub(r"(?:有限责任公司|股份有限公司|有限公司|公司|科技|网络|信息|技术|软件|咨询|商务)+$", "", part).strip()
        if 3 <= len(part) <= 14 and not re.fullmatch(r"[某甲乙丙丁]+", part) and "某" not in part:
            add(part)
    return tuple(out)


def load_case_refs(rows: list[dict]) -> list[CaseRef]:
    """rows = cases.jsonl 的每一行。别名被多个案共用的（如同一家公司的多个案子）一律丢弃：
    点名要能唯一指向一个案；问「微博诉云智联」靠「云智联」这类唯一的简称，而不是「北京微梦创科」。"""
    out = []
    for r in rows:
        out.append(CaseRef(case_id=r["case_id"], seq=r["seq"], chapter=r["chapter"], chapter_no=r["chapter_no"],
                           title=r["title"], meta=r.get("meta") or {}, aliases=build_aliases(r["title"], r.get("meta") or {})))
    owners: dict[str, set[str]] = {}
    for ref in out:
        for a in ref.aliases:
            owners.setdefault(re.sub(r"\s+", "", a), set()).add(ref.case_id)
    for ref in out:
        ref.aliases = tuple(a for a in ref.aliases if len(owners[re.sub(r"\s+", "", a)]) == 1)
    return out


def find_named(question: str, refs: list[CaseRef]) -> list[CaseRef]:
    """点名直达：问题里出现某案的别名 / 案号 / 指导性案例编号。长别名优先；同一案只算一次；按问题里出现的先后排序。"""
    q = re.sub(r"\s+", "", question)
    found: dict[str, tuple[int, int, CaseRef]] = {}
    for ref in refs:
        for a in ref.aliases:
            pos = q.find(re.sub(r"\s+", "", a))
            if pos >= 0:
                prev = found.get(ref.case_id)
                if prev is None or len(a) > prev[1]:
                    found[ref.case_id] = (pos, len(a), ref)
    # 编号类别名（指导性案例 262 号）命中多个案时，按「更长的别名」自然区分；仍并列则都保留
    return [v[2] for v in sorted(found.values(), key=lambda v: (v[0], -v[1]))]


# ---------------------------------------------------------------- 列表模式

def list_chapter(refs: list[CaseRef], chapter_no: int | None) -> list[CaseRef]:
    return [r for r in refs if chapter_no is None or r.chapter_no == chapter_no]


def render_case_list(refs: list[CaseRef], chapter_no: int | None) -> str:
    """列表模式的回答正文（由程序生成，不调用模型）。"""
    items = list_chapter(refs, chapter_no)
    if not items:
        return NOT_FOUND_TEXT
    head = f"{items[0].chapter}共有 {len(items)} 个判例：" if chapter_no else f"判例库共有 {len(items)} 个判例："
    lines = [f"{k}. {r.title}" + (f"（{r.meta['case_number']}）" if r.meta.get("case_number") and len(r.meta["case_number"]) < 40 else "")
             for k, r in enumerate(items, 1)]
    return head + "\n" + "\n".join(lines)


# ---------------------------------------------------------------- 检索

class CaseRetriever:
    """svc：以判例库为目录的 RAGService（需有 retrieve_result / mode / collection）；refs：判例目录。"""

    def __init__(self, svc, refs: list[CaseRef], *, min_rerank_score: float | None = None, max_distance: float | None = None):
        from ..adaptive import AdaptiveRAG
        self.svc, self.refs = svc, refs
        self.by_id = {r.case_id: r for r in refs}
        self._ad = AdaptiveRAG(svc, min_rerank_score=min_rerank_score, max_distance=max_distance)

    # ---- 向量库里取某案全部块（点名直达 / 补卡片用）；用元数据过滤，不做向量相似度 ----
    def _fetch_case_chunks(self, case_id: str) -> list[tuple[str, dict]]:
        col = self.svc.collection
        got = col.get(where={"case_id": case_id}, include=["documents", "metadatas"])
        return list(zip(got["documents"], got["metadatas"]))

    def _hit_from_chunks(self, ref: CaseRef, chunks: list[tuple[str, dict]], score: float, via: str, ranked_bodies=None) -> CaseHit:
        card = next((d for d, m in chunks if m.get("kind") == "card"), "")
        bodies = ranked_bodies if ranked_bodies is not None else [(d, m) for d, m in chunks if m.get("kind") == "body"]
        bodies = bodies[: config.CASE_MAX_BODY_CHUNKS]
        return CaseHit(ref=ref, score=score, via=via, card=card, bodies=[d for d, _ in bodies],
                       body_headings=[m.get("heading", "") for _, m in bodies])

    # ---- 一次向量检索 + 按案聚合 ----
    def _search(self, query: str, where: dict | None, tm: dict) -> tuple[dict[str, dict], bool | None, dict]:
        """返回 ({case_id: {"score", "cards", "bodies"}}, 是否可靠, 置信度依据)。"""
        fetch = config.CASE_FETCH_CHUNKS
        saved = self.svc.where
        self.svc.where = where
        try:
            res = self._ad._retrieve(query, fetch, tm)
        finally:
            self.svc.where = saved
        ok, why = self._ad._confidence(res)
        scores = res.debug.get("rerank_scores") or {}
        agg: dict[str, dict] = {}
        for i, (cid, doc, meta) in enumerate(zip(res.ids, res.documents, res.metadatas)):
            case_id = meta.get("case_id")
            if not case_id:
                continue
            s = scores.get(cid)
            if s is None:                                   # 无重排分：用 1 - 距离（dense）或排名倒数
                d = res.distances[i] if i < len(res.distances) else 1.0
                s = (1.0 - float(d)) if d else 1.0 / (i + 1)
            a = agg.setdefault(case_id, {"score": s, "cards": [], "bodies": []})
            a["score"] = max(a["score"], s)                 # 取该案块的最高分
            (a["cards"] if meta.get("kind") == "card" else a["bodies"]).append((doc, meta, s))
        return agg, ok, why

    def retrieve(self, question: str, count: int = 1, chapter: int | None = None, *, rewrite=None) -> dict:
        """返回 {"mode": named|search|not_found, "hits": [CaseHit], "trace": {...}}。
        rewrite(query) -> str：纠错时改写查询（None = 不改写，直接判定不够）。"""
        count = max(1, min(5, int(count or 1)))
        tm = {"retrieve_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0, "aux_llm_s": 0.0}
        t0 = time.perf_counter()
        trace: dict = {"retrievals": [], "count": count, "chapter": chapter}
        # 1. 点名直达
        named = find_named(question, self.refs)
        if named:
            hits = []
            for ref in named[:count]:
                hits.append(self._hit_from_chunks(ref, self._fetch_case_chunks(ref.case_id), 1.0, "named"))
            trace.update(mode="named", named=[r.case_id for r in named], timings=_round(tm, t0))
            return {"mode": "named", "hits": hits, "trace": trace}
        # 2. 章节优先
        picked: dict[str, CaseHit] = {}
        order: list[str] = []
        query = question

        def collect(where, via, q):
            agg, ok, why = self._search(q, where, tm)
            trace["retrievals"].append({"query": q, "where": where, "reliable": ok, "cases": len(agg), **why})
            reliable = ok is not False
            for case_id, a in sorted(agg.items(), key=lambda kv: -kv[1]["score"]):
                if case_id in picked or case_id not in self.by_id or not reliable:
                    continue
                bodies = [(d, m) for d, m, _ in sorted(a["bodies"], key=lambda x: -x[2])]
                cards = [d for d, _, _ in a["cards"]]
                chunks = None
                card = cards[0] if cards else ""
                if not card:
                    chunks = self._fetch_case_chunks(case_id)           # 命中的只有正文块：补取卡片
                    card = next((d for d, m in chunks if m.get("kind") == "card"), "")
                h = CaseHit(ref=self.by_id[case_id], score=a["score"], via=via, card=card,
                            bodies=[d for d, _ in bodies[: config.CASE_MAX_BODY_CHUNKS]],
                            body_headings=[m.get("heading", "") for _, m in bodies[: config.CASE_MAX_BODY_CHUNKS]])
                picked[case_id] = h
                order.append(case_id)
            return reliable

        if chapter:
            collect({"chapter_no": int(chapter)}, "search", query)
        if len(picked) < count:                                           # 本章不够，跨章补足
            collect(None, "search_other_chapter" if chapter else "search", query)
        # 3. 纠错：仍不够 -> 改写查询重查一次
        if len(picked) < count and rewrite is not None:
            new_q = _timed(lambda: rewrite(query), tm)
            if new_q and new_q != query:
                trace["rewritten"] = new_q
                if chapter:
                    collect({"chapter_no": int(chapter)}, "search", new_q)
                if len(picked) < count:
                    collect(None, "search_other_chapter" if chapter else "search", new_q)
        hits = sorted(picked.values(), key=lambda h: (h.via != "search", -h.score))[:count]
        if chapter:                                                       # 本章结果排在前面，跨章补足的在后
            hits = sorted(picked.values(), key=lambda h: (h.via == "search_other_chapter", -h.score))[:count]
        trace.update(mode="search" if hits else "not_found", timings=_round(tm, t0))
        return {"mode": "search" if hits else "not_found", "hits": hits, "trace": trace}


def _timed(fn, tm):
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        tm["aux_llm_s"] += time.perf_counter() - t0


def _round(tm: dict, t0: float) -> dict:
    return {**{k: round(v, 4) for k, v in tm.items()}, "total_s": round(time.perf_counter() - t0, 4)}
