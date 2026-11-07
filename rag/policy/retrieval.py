"""政策库检索与作答（agentic 自适应 RAG）：不用向量库、不用 bge-m3，启动时直接读原始 docx（见 PolicyIndex.load）。

1. 点名直达（程序）：问题里出现文件名 / 简称 / 别名（含「数据二十条」）→ 锁定该文件；同时问了「第X条」→ 直接取该条，跳过选块；
2. 本地初筛（程序）：BM25 取最多 POLICY_CANDIDATES（15）个块；问题提到北京 / 上海 / 广东 / 广州 / 深圳 / 贵州 / 贵阳时先限定地区，不够再放宽；锁定了文件则只在该文件内查；
3. 模型选块（1 次调用，Flash、关闭思考）：最多 POLICY_PICK_MAX（6）个块编号；都不相关 → 给出改写检索词，回到第 2 步只重查一次；仍没有 → not_found（不调用作答）；
4. 作答（1 次调用，Pro、思考 low）：总—分—总（结尾问句已移除，改由 followups 管道统一生成）；每个要点用 [n] 引用，来源列表写「篇 › 文件名 › 第X条」；
   模型写在引号里的法条文字必须逐字出自入选块，否则删除引号内容、保留 [n]；征求意见稿由程序标注「非正式生效文本」；末尾由程序附「以汇编收录版本为准……」。
调用次数（含路由 1 次）：普通 3 次，改写重查 4 次，点名到具体条 2 次。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..course.llm import parse_json_object
from ..prompt_rules import ANSWER_STRUCTURE, COURSE_SAMPLES, with_history
from . import config
from .aliases import ALIASES
from .corpus import build_corpus, cn_to_int

NOT_FOUND_TEXT = "政策汇编中没有找到相关规定。"
CLOSING_NOTE = "以汇编收录版本为准，现行有效性请以官方发布为准。"
DRAFT_LABEL = "征求意见稿，非正式生效文本"
PARSE_VERSION = 1
PICK_CHUNK_CHARS = 360

REGION_WORDS = (("北京", "北京"), ("上海", "上海"), ("广东", "广东"), ("广州", "广东"), ("深圳", "深圳"), ("贵州", "贵州"), ("贵阳", "贵州"))
_ART_ASK_RE = re.compile(r"第\s*([零〇一二三四五六七八九十百\d]+)\s*条")

PICK_SYSTEM = """你是政策条文筛选器，只做选择，不写任何解释。下面是若干候选条文块，每个以 [编号] 开头，附「篇｜文件名｜条号」。
请选出与学生问题直接相关的块，最多 {count} 个，按相关程度从高到低；只能选候选里出现过的编号。
如果候选里没有一个直接相关，不要硬选：改为给出一个改写后的检索词 rewrite（换成更贴近政策用语的关键词），picks 留空。
只输出一个 JSON 对象：{{"picks": [编号, ...], "rewrite": ""}}。picks 非空时 rewrite 填空串。"""

ANSWER_SYSTEM = """你是 AI 李老师，《数据要素》课程的助教，现在回答数据政策法规方面的问题。只依据下面给出的政策条文材料回答，不使用材料之外的知识，直接作答。
- 材料用 [P1]、[P2]… 编号，每个编号是一个条文块，开头写明了「篇｜文件名｜条号」。
""" + ANSWER_STRUCTURE + """
- 分点时每个要点先点明依据的文件名和条号（如“《数据安全法》第二十一条规定……”），再用自己的话说明内容；
- 每个论断后用材料里实际出现的编号标注依据，如 [P2]；不要引用材料中没有的编号；
- 需要引用条文原话时，只能逐字摘录材料里的文字并放进引号，不得改写后放进引号；
- 材料标明「征求意见稿」的文件，要写明它是征求意见稿、尚未正式生效，不要当成现行规定；
- 不要写没有具体原因的套话式免责，结尾不要附加说明引用方式的话（系统会另行附注）；材料里没有的内容不要编造。

下面两个样例示范期望的结构与详略（只学格式，不要照抄内容）：

""" + COURSE_SAMPLES

ANSWER_USER = "问题：{question}\n\n政策条文材料：\n{materials}"


@dataclass
class PolicyHit:
    n: int                                       # P 编号（1 起）
    chunk: dict

    @property
    def draft(self) -> bool:
        return bool(self.chunk["draft"])


def source_label(c: dict) -> str:
    """来源条目：篇 › 文件名 › 第X条（没有条号的块只写到文件名；章节标题不进来源）。"""
    parts = [c["pian"], c["policy"]]
    if c.get("article_label"):
        parts.append(c["article_label"])
    return " › ".join(parts)


def render_materials(hits: list[PolicyHit], limit_chars: int = 2400) -> str:
    out = []
    for h in hits:
        c = h.chunk
        tag = f"（{DRAFT_LABEL}）" if c["draft"] else ""
        out.append(f"[P{h.n}] {source_label(c)}{tag}\n{c['body'][:limit_chars]}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- 索引

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _auto_aliases(f: dict) -> set[str]:
    out: set[str] = set()
    for base in (f["policy"], f["short"]):
        b = _norm(base)
        out.add(b)
        for strip in ("中华人民共和国", "中共中央国务院", "（试行）", "（征求意见稿）", "（暂行）"):
            out.add(b.replace(strip, ""))
    out = {a for a in out if len(a) >= 4}
    return out


class PolicyIndex:
    def __init__(self, corpus: dict):
        from ..course.bm25 import BM25, tokenize
        self._tok = tokenize
        self.files = {f["doc_no"]: f for f in corpus["files"]}
        self.chunks = corpus["chunks"]
        self.by_file: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_file.setdefault(c["doc_no"], []).append(i)
        self._bm25 = BM25([tokenize(c["text"]) for c in self.chunks]) if self.chunks else None
        owners: dict[str, set[str]] = {}
        for no, f in self.files.items():
            for a in _auto_aliases(f):
                owners.setdefault(a, set()).add(no)
        self.aliases: dict[str, str] = {a: next(iter(v)) for a, v in owners.items() if len(v) == 1}      # 共用的别名丢弃：点名必须唯一
        for a, no in ALIASES.items():
            if no in self.files:
                self.aliases[_norm(a)] = no
        self.info: dict = {}

    @classmethod
    def load(cls, doc=None, *, use_cache: bool = True) -> "PolicyIndex | None":
        from ..docscache import load_or_build
        got = load_or_build(doc if doc is not None else config.POLICY_DOC, "policy", build_corpus, version=PARSE_VERSION, use_cache=use_cache)
        if got is None:
            return None
        idx = cls(got[0])
        idx.info = got[1]
        return idx

    # ---- 点名 ----
    def find_files(self, question: str) -> list[str]:
        """问题里出现的文件名 / 简称 / 别名 → doc_no 列表（按出现先后）。长别名优先，已被长别名占用的位置不再匹配短别名。"""
        q = _norm(question)
        taken: list[tuple[int, int]] = []
        found: dict[str, int] = {}
        for a in sorted(self.aliases, key=len, reverse=True):
            start = 0
            while True:
                pos = q.find(a, start)
                if pos < 0:
                    break
                span = (pos, pos + len(a))
                if not any(s < span[1] and span[0] < e for s, e in taken):
                    taken.append(span)
                    no = self.aliases[a]
                    found[no] = min(found.get(no, pos), pos)
                start = pos + 1
        return sorted(found, key=found.get)

    def article_chunks(self, doc_no: str, number: int) -> list[dict]:
        return [self.chunks[i] for i in self.by_file.get(doc_no, []) if self.chunks[i]["article"] == number and self.chunks[i]["kind"] == "article"]

    # ---- BM25 ----
    def candidates(self, query: str, *, files: list[str] | None = None, regions: set[str] | None = None, limit: int | None = None) -> list[int]:
        if self._bm25 is None:
            return []
        limit = limit or config.POLICY_CANDIDATES
        allowed = None
        if files:
            allowed = {i for f in files for i in self.by_file.get(f, [])}
        out = []
        for i, _ in self._bm25.top(self._tok(query), max(limit * 12, 200)):
            if allowed is not None and i not in allowed:
                continue
            if regions and self.chunks[i]["region"] not in regions:
                continue
            out.append(i)
            if len(out) >= limit:
                break
        return out


def question_regions(question: str) -> set[str]:
    return {r for w, r in REGION_WORDS if w in question}


# ---------------------------------------------------------------- 模型选块

def pick_chunks(question: str, index: PolicyIndex, ids: list[int], llm, *, model: str, count: int | None = None):
    count = count or config.POLICY_PICK_MAX
    info: dict = {"dropped": [], "error": ""}
    lines, key = [], {}
    for k, i in enumerate(ids, 1):
        c = index.chunks[i]
        lines.append(f"[{k}] {source_label(c)}\n{c['body'][:PICK_CHUNK_CHARS]}")
        key[str(k)] = i
    msgs = [{"role": "system", "content": PICK_SYSTEM.format(count=count)},
            {"role": "user", "content": f"学生问题：{question}\n\n候选条文块：\n" + "\n\n".join(lines)}]
    try:
        obj = parse_json_object(llm.chat("policy_pick", msgs, model=model, json_mode=True).content)
    except Exception as e:  # noqa: BLE001 - 只记类型名
        info["error"] = type(e).__name__
        return [], "", info
    picked: list[int] = []
    raw = obj.get("picks")
    for x in raw if isinstance(raw, list) else []:
        i = key.get(str(x).strip().lstrip("[").rstrip("]"))
        if i is None or i in picked:
            info["dropped"].append({"pick": str(x)[:20], "reason": "unknown_or_duplicate"})
        elif len(picked) < count:
            picked.append(i)
    rewrite = str(obj.get("rewrite") or "").strip()[:60] if not picked else ""
    return picked, rewrite, info


class PolicyRetriever:
    def __init__(self, index: PolicyIndex, llm, *, model: str):
        self.index, self.llm, self.model = index, llm, model

    def _hits(self, idxs: list[int], first_n: int = 1) -> list[PolicyHit]:
        return [PolicyHit(first_n + k, self.index.chunks[i]) for k, i in enumerate(idxs)]

    def retrieve(self, question: str, *, first_n: int = 1) -> dict:
        """返回 {"mode": named_article|search|not_found, "hits": [PolicyHit], "trace": {...}}。"""
        t0 = time.perf_counter()
        ix = self.index
        trace: dict = {"rounds": [], "llm_calls": 0}
        files = ix.find_files(question)
        trace["files"] = files
        if files:
            m = _ART_ASK_RE.search(question)
            num = cn_to_int(m.group(1)) if m else None
            if num is not None:
                got = ix.article_chunks(files[0], num)
                if got:                                                         # 点名到具体条：直接取，跳过选块
                    trace.update(mode="named_article", article=num, total_s=round(time.perf_counter() - t0, 4))
                    return {"mode": "named_article", "hits": [PolicyHit(first_n + k, c) for k, c in enumerate(got)], "trace": trace}
                trace["article_missing"] = num
        regions = question_regions(question)
        query = question
        for round_no in (1, 2):
            ids = ix.candidates(query, files=files or None, regions=regions or None)
            if regions and not files and len(ids) < 5:                          # 地区限定后太少：放宽，补足
                ids += [i for i in ix.candidates(query) if i not in ids][: config.POLICY_CANDIDATES - len(ids)]
            rd: dict = {"query": query, "candidates": len(ids), "regions": sorted(regions), "files": files}
            trace["rounds"].append(rd)
            if not ids:
                break
            t1 = time.perf_counter()
            picked, new_q, info = pick_chunks(question, ix, ids, self.llm, model=self.model)
            trace["llm_calls"] += 1
            rd.update(pick_s=round(time.perf_counter() - t1, 4), picked=[ix.chunks[i]["chunk_id"] for i in picked], rewrite=new_q, **info)
            if picked:
                trace.update(mode="search", total_s=round(time.perf_counter() - t0, 4))
                return {"mode": "search", "hits": self._hits(picked, first_n), "trace": trace}
            if round_no == 1 and new_q and new_q != query:
                trace["rewritten"] = new_q
                query = new_q
                continue
            break
        trace.update(mode="not_found", total_s=round(time.perf_counter() - t0, 4))
        return {"mode": "not_found", "hits": [], "trace": trace}


# ---------------------------------------------------------------- 作答

@dataclass
class PolicyAnswer:
    text: str = ""
    body: str = ""
    sources: list[dict] = field(default_factory=list)
    removed_quotes: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)


def normalize_markers(text: str, n_hits: int) -> str:
    """[P#] → [S#]（交给统一的 CitationRewriter）；越界编号映射成不存在的 S0，由改写器当无效引用去掉。"""
    text = text.translate(str.maketrans({"［": "[", "］": "]", "【": "[", "】": "]", "〔": "[", "〕": "]"}))
    return re.sub(r"\[\s*[Pp](\d+)\s*\]", lambda m: f"[S{int(m.group(1))}]" if 1 <= int(m.group(1)) <= n_hits else "[S0]", text)


def source_entry(h: PolicyHit, n: int) -> dict:
    c = h.chunk
    path = source_label(c) + (f"（{DRAFT_LABEL}）" if c["draft"] else "")
    return {"n": n, "kind": "policy", "chunk_id": c["chunk_id"], "path": path, "title": c["policy"], "article": c["article_label"],
            "draft": c["draft"], "region": c["region"]}


def finish_answer(body: str, hits: list[PolicyHit], cited: dict[str, int]) -> tuple[str, list[dict], list[str]]:
    """校验引文、附注。返回 (正文 + 附注, 来源列表, 被删除的引号文字)。"""
    from ..combined import scrub_fabricated_quotes, tidy_after_removal
    originals = [h.chunk["body"] for h in hits]
    body, removed = scrub_fabricated_quotes(body, originals)
    if removed:
        body = tidy_after_removal(body)
    by_key = {h.chunk["chunk_id"]: h for h in hits}
    sources = sorted((source_entry(by_key[cid], n) for cid, n in cited.items() if cid in by_key), key=lambda s: s["n"])
    notes = []
    drafts = list(dict.fromkeys(s["title"] for s in sources if s["draft"]))
    if drafts:
        notes.append("、".join(f"《{t}》" for t in drafts) + f"为{DRAFT_LABEL}。")
    notes.append(CLOSING_NOTE)
    return body.rstrip() + "\n\n" + "".join(notes), sources, removed


def answer_policy(question: str, hits: list[PolicyHit], llm, *, model: str, on_delta=None, on_source=None,
                  history: list[dict] | None = None) -> PolicyAnswer:
    """一次作答调用。on_delta：流式时回调学生可见文本（[n] 已实时编号）；引号校验与附注只作用于最终 answer。"""
    from ..course.answer import CitationRewriter
    refs = {f"S{h.n}": h.chunk["chunk_id"] for h in hits}
    by_id = {h.chunk["chunk_id"]: h for h in hits}
    msgs = with_history([{"role": "system", "content": ANSWER_SYSTEM},
                         {"role": "user",
                          "content": ANSWER_USER.format(question=question, materials=render_materials(hits))}],
                        history)
    rw = CitationRewriter(refs, on_assign=(lambda n, cid: on_source(source_entry(by_id[cid], n))) if on_source else None)
    n_hits = max((h.n for h in hits), default=0)

    def push(raw: str) -> None:
        shown = rw.feed(normalize_markers(raw, n_hits))
        if shown and on_delta:
            on_delta(shown)

    stream = getattr(llm, "chat_stream", None) if on_delta else None
    if stream:
        buf = {"t": ""}

        def feed(raw: str) -> None:
            buf["t"] += raw
            cut = buf["t"].rfind("[")
            if cut >= 0 and "]" not in buf["t"][cut:] and len(buf["t"]) - cut <= 12:
                safe, buf["t"] = buf["t"][:cut], buf["t"][cut:]
            else:
                safe, buf["t"] = buf["t"], ""
            if safe:
                push(safe)
        stream("policy_answer", msgs, model=model, on_delta=feed)
        if buf["t"]:
            push(buf["t"])
    else:
        push(llm.chat("policy_answer", msgs, model=model).content or "")
    tail = rw.finish()
    if tail and on_delta:
        on_delta(tail)
    text, sources, removed = finish_answer(rw.rendered_all.strip(), hits, dict(rw.numbers))
    return PolicyAnswer(text=text, body=rw.rendered_all.strip(), sources=sources, removed_quotes=removed, invalid=list(rw.invalid))
