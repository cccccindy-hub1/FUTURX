"""判例检索（agentic 自适应 RAG）：不用向量库、不用 bge-m3，启动时直接读原始 docx（见 CaseIndex.load）。

1. 点名直达（程序）：问题里出现案名 / 案号 / 指导性案例编号 / 别名，直接取该案，跳到第 4 步；
2. 本地初筛（程序）：BM25 检索卡片和正文块，按案聚合，最多取 CASE_CANDIDATES（12）个候选案子；路由给了 chapter 时先查该章，不够再跨章补足；
3. 模型选案（1 次调用，Flash、关闭思考）：把候选案子的卡片交给模型，模型输出相关的案号，最多 count 个；
   都不相关时模型输出一个改写后的检索词，回到第 2 步重查，**只重查一次**；重查后仍没有 → not_found（不调用作答）；
4. 取材（程序）：对选中的案子，用 BM25 取最相关的几个正文块（每案最多 CASE_MAX_BODY_CHUNKS 个），再交给作答（1 次调用，Pro、思考 low）。
5. 列表模式：「第六章有哪些判例」直接列出该章判例名，不检索、不调用任何模型。
每案材料 = 卡片 + 最相关的正文块。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import config
from .parser import chapter_number

NOT_FOUND_TEXT = "判例库里没有找到相关判例。"


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


# ---------------------------------------------------------------- 索引：启动时读 docx（带 JSON 缓存），不建向量库

PARSE_VERSION = 1


def _chunk_rows(doc) -> dict:
    """docx -> 可 JSON 化的数据：判例目录 + 全部块。解析与切块沿用 rag.cases.parser / chunker。"""
    from .chunker import chunk_cases
    from .parser import parse_cases
    cases = parse_cases(doc)
    chunks = chunk_cases(cases)
    refs = [{"case_id": f"case:{c.seq:02d}", "seq": c.seq, "chapter": c.chapter, "chapter_no": c.chapter_no, "title": c.title,
             "meta": c.meta} for c in cases]
    return {"refs": refs, "chunks": [k.to_dict() for k in chunks]}


class CaseIndex:
    """判例库的内存索引：判例目录 + 卡片 / 正文块的 BM25。"""

    def __init__(self, refs: list[CaseRef], chunks: list[dict]):
        from ..course.bm25 import BM25, tokenize
        self.refs = refs
        self.by_id = {r.case_id: r for r in refs}
        self.chunks = chunks
        self._tok = tokenize
        self.cards = {c["case_id"]: c for c in chunks if c["kind"] == "card"}
        self.bodies: dict[str, list[dict]] = {}
        for c in chunks:
            if c["kind"] == "body":
                self.bodies.setdefault(c["case_id"], []).append(c)
        self._bm25 = BM25([tokenize(c["text"]) for c in chunks]) if chunks else None
        self.info: dict = {}

    @classmethod
    def load(cls, doc=None, *, use_cache: bool = True) -> "CaseIndex | None":
        """读 CASE_DOC（缺省取配置）；文件不存在返回 None（调用方跳过这个库，服务照常启动）。"""
        from ..docscache import load_or_build
        got = load_or_build(doc if doc is not None else config.CASE_DOC, "cases", _chunk_rows, version=PARSE_VERSION, use_cache=use_cache)
        if got is None:
            return None
        data, info = got
        idx = cls(load_case_refs(data["refs"]), data["chunks"])
        idx.info = info
        return idx

    # ---- BM25 ----
    def search_chunks(self, query: str, *, chapter_no: int | None = None, only_cases: set[str] | None = None, k: int = 60):
        if self._bm25 is None:
            return []
        qt = self._tok(query)
        out = []
        for i, s in self._bm25.top(qt, k * 4):
            c = self.chunks[i]
            if chapter_no is not None and c["chapter_no"] != chapter_no:
                continue
            if only_cases is not None and c["case_id"] not in only_cases:
                continue
            out.append((c, s))
            if len(out) >= k:
                break
        return out

    def candidates(self, query: str, *, chapter_no: int | None = None, limit: int | None = None) -> list[tuple[str, float]]:
        """按案聚合（取该案块的最高 BM25 分），返回 [(case_id, 分)]，最多 limit 个。"""
        limit = limit or config.CASE_CANDIDATES
        best: dict[str, float] = {}
        for c, s in self.search_chunks(query, chapter_no=chapter_no, k=120):
            best[c["case_id"]] = max(best.get(c["case_id"], 0.0), s)
        return sorted(best.items(), key=lambda kv: -kv[1])[:limit]

    def best_bodies(self, case_id: str, query: str, n: int | None = None) -> list[dict]:
        n = n or config.CASE_MAX_BODY_CHUNKS
        bodies = self.bodies.get(case_id, [])
        if len(bodies) <= n:
            return list(bodies)
        scores = {c["chunk_id"]: s for c, s in self.search_chunks(query, only_cases={case_id}, k=len(bodies))}
        return sorted(bodies, key=lambda c: (-scores.get(c["chunk_id"], 0.0), c["part"]))[:n]


# ---------------------------------------------------------------- 模型选案（1 次调用）

CASE_PICK_SYSTEM = """你是判例筛选器，只做选择，不写任何解释。下面是若干候选判例，每个以 [编号] 开头，附案名和卡片摘要。
请选出与学生问题直接相关的判例，最多 {count} 个，按相关程度从高到低；只能选候选里出现过的编号。
如果候选里没有一个直接相关，不要硬选：改为给出一个改写后的检索词 rewrite（换成更贴近判例用语的关键词，如「不正当竞争 数据抓取 平台」），选择留空。
只输出一个 JSON 对象：{{"picks": ["编号", ...], "rewrite": ""}}。picks 非空时 rewrite 填空串。"""

PICK_CARD_CHARS = 420


def render_candidates(index: "CaseIndex", ids: list[str]) -> tuple[str, dict[str, str]]:
    """候选案子 -> 给模型看的文本，同时返回 {显示编号: case_id}。显示编号用卡片序号（如 19），与案号区分开。"""
    lines, key = [], {}
    for cid in ids:
        ref = index.by_id[cid]
        card = (index.cards.get(cid) or {}).get("text", "")
        body = card.split("\n", 2)[-1] if card else ""
        lines.append(f"[{ref.seq}] {ref.chapter}｜{ref.title}\n{body[:PICK_CARD_CHARS]}")
        key[str(ref.seq)] = cid
    return "\n\n".join(lines), key


def pick_cases(question: str, index: "CaseIndex", ids: list[str], count: int, llm, *, model: str) -> tuple[list[str], str, dict]:
    """一次调用。返回 (选中的 case_id 列表, 改写词, 调试信息)。越界编号丢弃；输出不可用 -> 视为一个都没选、没有改写词（不重试）。"""
    from ..course.llm import parse_json_object
    info: dict = {"dropped": [], "error": ""}
    text, key = render_candidates(index, ids)
    msgs = [{"role": "system", "content": CASE_PICK_SYSTEM.format(count=count)},
            {"role": "user", "content": f"学生问题：{question}\n\n候选判例：\n{text}"}]
    try:
        res = llm.chat("case_pick", msgs, model=model, json_mode=True)
        obj = parse_json_object(res.content)
    except Exception as e:  # noqa: BLE001 - 只记类型名
        info["error"] = type(e).__name__
        return [], "", info
    picked: list[str] = []
    raw = obj.get("picks")
    for x in raw if isinstance(raw, list) else []:
        k = str(x).strip().lstrip("[").rstrip("]").lstrip("0") or "0"
        cid = key.get(k)
        if cid is None or cid in picked:
            info["dropped"].append({"pick": str(x)[:20], "reason": "unknown_or_duplicate"})
        elif len(picked) < count:
            picked.append(cid)
    rewrite = str(obj.get("rewrite") or "").strip()[:60] if not picked else ""
    return picked, rewrite, info


# ---------------------------------------------------------------- 检索入口

class CaseRetriever:
    """index：CaseIndex；llm：胜算云的 OpenAICompatClient（选案用，Flash、关闭思考）；model：选案模型。"""

    def __init__(self, index: CaseIndex, llm, *, model: str):
        self.index, self.llm, self.model = index, llm, model
        self.refs, self.by_id = index.refs, index.by_id

    def _hit(self, ref: CaseRef, query: str, via: str, score: float) -> CaseHit:
        bodies = self.index.best_bodies(ref.case_id, query)
        return CaseHit(ref=ref, score=score, via=via, card=(self.index.cards.get(ref.case_id) or {}).get("text", ""),
                       bodies=[b["text"] for b in bodies], body_headings=[b.get("heading", "") for b in bodies])

    def _candidates(self, query: str, chapter: int | None) -> tuple[list[str], dict[str, str]]:
        """章节优先：先查该章，不够 CASE_CANDIDATES 个再跨章补足。返回 ([case_id], {case_id: via})。"""
        ids: list[str] = []
        via: dict[str, str] = {}
        if chapter:
            for cid, _ in self.index.candidates(query, chapter_no=int(chapter)):
                ids.append(cid)
                via[cid] = "search"
        if len(ids) < config.CASE_CANDIDATES:
            for cid, _ in self.index.candidates(query, limit=config.CASE_CANDIDATES * 2):
                if cid not in via and len(ids) < config.CASE_CANDIDATES:
                    ids.append(cid)
                    via[cid] = "search_other_chapter" if chapter else "search"
        return ids, via

    def retrieve(self, question: str, count: int = 1, chapter: int | None = None, *, rewrite=None) -> dict:
        """返回 {"mode": named|search|not_found, "hits": [CaseHit], "trace": {...}}。
        rewrite 参数仅为兼容旧调用点而保留，不再使用：改写由选案模型在同一次调用里给出。"""
        count = max(1, min(5, int(count or 1)))
        t0 = time.perf_counter()
        trace: dict = {"count": count, "chapter": chapter, "rounds": [], "llm_calls": 0}
        named = find_named(question, self.refs)
        if named:                                                       # 1. 点名直达：不检索、不调用选案模型
            hits = [self._hit(r, question, "named", 1.0) for r in named[:count]]
            trace.update(mode="named", named=[r.case_id for r in named], total_s=round(time.perf_counter() - t0, 4))
            return {"mode": "named", "hits": hits, "trace": trace}
        query = question
        for round_no in (1, 2):                                         # 2.–3. 初筛 -> 模型选案；最多重查一次
            ids, via = self._candidates(query, chapter)
            rd: dict = {"query": query, "candidates": len(ids)}
            trace["rounds"].append(rd)
            if not ids:
                break
            t1 = time.perf_counter()
            picked, new_q, info = pick_cases(question, self.index, ids, count, self.llm, model=self.model)
            trace["llm_calls"] += 1
            rd.update(pick_s=round(time.perf_counter() - t1, 4), picked=picked, rewrite=new_q, **info)
            if picked:
                hits = [self._hit(self.by_id[cid], query, via.get(cid, "search"), 1.0 - 0.01 * n) for n, cid in enumerate(picked)]
                if chapter:                                             # 本章的排在前面，跨章补足的在后
                    hits.sort(key=lambda h: h.via == "search_other_chapter")
                trace.update(mode="search", total_s=round(time.perf_counter() - t0, 4))
                return {"mode": "search", "hits": hits, "trace": trace}
            if round_no == 1 and new_q and new_q != query:
                trace["rewritten"] = new_q
                query = new_q
                continue
            break
        trace.update(mode="not_found", total_s=round(time.perf_counter() - t0, 4))
        return {"mode": "not_found", "hits": [], "trace": trace}
