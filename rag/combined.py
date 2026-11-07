"""跨库组合问答：一次路由出多个库（最多 2 个，仅 course / case / quote / policy 之间），各库并行检索、各自筛选，**只调一次作答**。

材料分区交给作答模型：
    【教材材料】[S1] …      课程库筛选后的教材块
    【判例材料】[C1] …      判例卡片 + 最相关正文块（头部标签由程序填写，模型不写）
    【讲话原文】[Q1] …      命中的讲话及已挑出的段落——只供引用，不得改写
    【政策条文】[P1] …      政策库选出的条文块（头部「篇｜文件名｜条号」由程序给出）
作答要求：先讲知识点，再用判例或讲话印证、说明二者的联系；不分成两个独立回答，不是每个库各出一个答案再拼接。

引用与来源：正文统一编号 [1] [2]（按首次出现；[S#]/[C#]/[Q#] 经同一个 CitationRewriter）；来源列表按库分区（教材 / 判例 / 讲话）。
讲话原文不能被改写：模型只能用 [Q#] 标记引用位置，不得自己写出讲话内容。程序在回答末尾按「讲话原文」分区原样贴出所引段落；
模型输出里若出现引号内的讲话文字，逐字核对挑出的原文，不一致就整段删除，只保留 [Q#] 标记。
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .prompt_rules import ANSWER_STRUCTURE, COURSE_SAMPLES, with_history

COMBINED_SYSTEM = """你是 AI 李老师，《数据要素》课程的助教。下面的材料分几个区（教材、判例、讲话原文、政策条文，有哪个区就有哪个），只依据这些材料回答，不使用材料之外的知识，直接作答，只写一个回答。
- 【教材材料】用 [S#] 编号，是知识点的依据；【判例材料】用 [C#] 编号，每个编号是一个判例；【讲话原文】用 [Q#] 编号，是习近平总书记的讲话原文；【政策条文】用 [P#] 编号，是政策法规的条文块，写到政策时要点明文件名和条号。
""" + ANSWER_STRUCTURE + """
- 分点时先把知识点讲清楚，再用判例或讲话来印证，并说明二者与知识点的联系；
  不要分成「教材部分」「判例部分」两个独立回答，也不要把同样的内容重复一遍；
- 每个论断后用对应编号标注依据，如 [S2]、[C1]、[Q1]、[P1]；不要引用材料中没有的编号。
- 讲话原文不能被改写或概括：引用讲话时只写 [Q#] 标记来指明位置，不要自己写出讲话里的话，也不要把讲话文字放进引号；原文由系统另行附在回答末尾。
- 判例的案号、法院、日期、金额等细节材料里没有就不要写；不要编造。
- 不要写没有具体原因的套话式免责，结尾不要附加说明引用方式的话。

下面两个样例示范期望的结构与详略（只学格式，不要照抄内容）：

""" + COURSE_SAMPLES

COMBINED_USER = "问题：{question}\n\n{materials}"

SECTION_TITLES = {"course": "【教材材料】", "case": "【判例材料】", "quote": "【讲话原文】", "policy": "【政策条文】"}
KIND_ORDER = ("course", "case", "quote", "policy")           # 来源列表的分区顺序
KIND_LABELS = {"course": "教材", "case": "判例", "quote": "讲话", "policy": "政策"}


# ---------------------------------------------------------------- 讲话引文校验：引号里的讲话文字必须逐字出自入选原文

_QUOTE_PAIRS = (("「", "」"), ("“", "”"), ("\"", "\""), ("『", "』"))
_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’「」『』（）()《》\[\]【】—-]+")
MIN_QUOTED_CHARS = 8                              # 引号里少于这么多字的（如「数据」二字的强调、书名号式引用）不当作讲话引文校验


def _squash(s: str) -> str:
    return _PUNCT.sub("", s or "")


def scrub_fabricated_quotes(text: str, originals: list[str]) -> tuple[str, list[str]]:
    """删除模型写在引号里、却不是入选讲话原文逐字内容的文字。
    判定：引号内文字（去掉标点空白后）不是任何入选段落原文的子串 → 视为改写 / 编造，整段连引号一起删除；
    子串关系成立的（逐字摘录）保留。过短的引号内容（< MIN_QUOTED_CHARS）不校验，避免误删普通强调词。
    返回 (清理后文本, 被删除的引号内文字列表)。"""
    pool = [_squash(p) for p in originals]
    removed: list[str] = []

    def keep(inner: str) -> bool:
        q = _squash(inner)
        return len(q) < MIN_QUOTED_CHARS or any(q in p for p in pool)

    out = text
    for left, right in _QUOTE_PAIRS:
        if left == right:
            pat = re.compile(re.escape(left) + r"([^" + re.escape(left) + r"\n]{1,400})" + re.escape(right))
        else:
            pat = re.compile(re.escape(left) + r"([^" + re.escape(left + right) + r"\n]{1,400})" + re.escape(right))

        def repl(m, _removed=removed):
            if keep(m.group(1)):
                return m.group(0)
            _removed.append(m.group(1))
            return ""
        out = pat.sub(repl, out)
    return out, removed


# ---------------------------------------------------------------- 材料

@dataclass
class Materials:
    course_text: str = ""                              # 【教材材料】区正文（[S#] …）
    course_refs: dict[str, str] = field(default_factory=dict)       # S# -> chunk_id
    course_kept: list = field(default_factory=list)                 # Recalled 列表（来源条目用）
    relations: list = field(default_factory=list)
    case_hits: list = field(default_factory=list)                   # CaseHit
    quote_picks: list = field(default_factory=list)                 # QuotePick（Q 编号从 1 起）
    policy_hits: list = field(default_factory=list)                 # PolicyHit（P 编号从 1 起）
    traces: dict = field(default_factory=dict)                      # 每个库的检索 / 筛选 trace
    empty: list = field(default_factory=list)                       # 没有找到材料的库

    @property
    def sections(self) -> list[str]:
        return [k for k in KIND_ORDER if (k == "course" and self.course_kept) or (k == "case" and self.case_hits)
                or (k == "quote" and self.quote_picks) or (k == "policy" and self.policy_hits)]


def render_materials(m: Materials) -> str:
    """分区材料文本，顺序固定：教材 → 判例 → 讲话。讲话区只放挑出的段落（不是整次讲话），并标明「只供引用，不得改写」。"""
    from .cases.answer import render_materials as render_cases
    from .quotes.retrieval import source_line
    parts = []
    if m.course_kept:
        parts.append(SECTION_TITLES["course"] + "\n" + m.course_text)
    if m.case_hits:
        parts.append(SECTION_TITLES["case"] + "\n" + render_cases(m.case_hits))
    if m.quote_picks:
        body = "\n\n".join(f"[Q{p.n}] {source_line(p.speech)}（只供引用，不得改写）\n" + "\n".join(p.texts()) for p in m.quote_picks)
        parts.append(SECTION_TITLES["quote"] + "\n" + body)
    if m.policy_hits:
        from .policy.retrieval import render_materials as render_policy
        parts.append(SECTION_TITLES["policy"] + "\n" + render_policy(m.policy_hits))
    return "\n\n".join(parts)


def tidy_after_removal(text: str) -> str:
    """删掉引号文字后，清理残句：「总书记说：。」→「总书记说。」，连续标点收拢，空括号去掉。"""
    t = re.sub(r"([：:，,])\s*([。；！？.;!?])", r"\2", text)
    t = re.sub(r"([。；！？])\1+", r"\1", t)
    t = re.sub(r"[（(]\s*[）)]", "", t)
    return re.sub(r"[ \t]{2,}", " ", t)


# ---------------------------------------------------------------- 一次作答

@dataclass
class CombinedAnswer:
    text: str = ""                                          # 学生可见正文（[1] [2]）+ 末尾「讲话原文」分区（若引用了讲话）
    body: str = ""                                          # 仅模型写的正文（不含末尾讲话原文分区）
    sources: list[dict] = field(default_factory=list)       # [{n, kind: course|case|quote, chunk_id, path, ...}]，按序号排列
    removed_quotes: list[str] = field(default_factory=list)  # 被删除的、与讲话原文不一致的引号文字
    invalid: list[str] = field(default_factory=list)
    dropped_markers: list[str] = field(default_factory=list)


def build_refs(m: Materials) -> tuple[dict[str, str], dict[str, tuple[str, object]]]:
    """统一的引用表：marker(S1/C1/Q1) -> 全局唯一 id；以及 id -> (kind, 对象)。CitationRewriter 只认 S# 形式，
    所以对外统一映射成 S{序号}（教材 S 保持原号，判例接在后面，讲话再接在后面），解析完再按 kind 归位。"""
    refs: dict[str, str] = {}
    info: dict[str, tuple[str, object]] = {}
    k = 0
    for ref, cid in m.course_refs.items():
        k += 1
        key = f"course:{cid}"
        refs[f"S{k}"] = key
        info[key] = ("course", next((c for c in m.course_kept if c.chunk_id == cid), None))
    for i, h in enumerate(m.case_hits, 1):
        k += 1
        key = f"case:{h.ref.case_id}"
        refs[f"S{k}"] = key
        info[key] = ("case", h)
    for p in m.quote_picks:
        k += 1
        key = f"quote:{p.speech.speech_id}"
        refs[f"S{k}"] = key
        info[key] = ("quote", p)
    for h in m.policy_hits:
        k += 1
        key = f"policy:{h.chunk['chunk_id']}"
        refs[f"S{k}"] = key
        info[key] = ("policy", h)
    return refs, info


def _normalize_markers(text: str, m: Materials) -> str:
    """把模型写的 [S#]/[C#]/[Q#] 都映射成统一的 [S#]（编号按 build_refs 的顺序）。越界编号映射成一个不存在的 S 号，由改写器当无效引用去掉。"""
    n_course, n_case = len(m.course_refs), len(m.case_hits)
    q_numbers = {p.n: idx for idx, p in enumerate(m.quote_picks, 1)}
    n_quote = len(m.quote_picks)
    p_numbers = {h.n: idx for idx, h in enumerate(m.policy_hits, 1)}
    text = text.translate(str.maketrans({"［": "[", "］": "]", "【": "[", "】": "]", "〔": "[", "〕": "]"}))

    def repl(mo):
        letter, num = mo.group(1).upper(), int(mo.group(2))
        if letter == "S":
            return f"[S{num}]" if 1 <= num <= n_course else "[S0]"
        if letter == "C":
            return f"[S{n_course + num}]" if 1 <= num <= n_case else "[S0]"
        if letter == "Q":
            return f"[S{n_course + n_case + q_numbers[num]}]" if num in q_numbers else "[S0]"
        if letter == "P":
            return f"[S{n_course + n_case + n_quote + p_numbers[num]}]" if num in p_numbers else "[S0]"
        return mo.group(0)
    return re.sub(r"\[\s*([SCQPscqp])\s*(\d+)\s*\]", repl, text)


def source_for(kind: str, obj, n: int, titles=None) -> dict:
    if kind == "course":
        from .course.answer import source_entry
        e = source_entry(n, obj, titles)
        e["kind"] = "course"
        return e
    if kind == "case":
        from .cases.answer import source_path
        meta = obj.ref.meta
        return {"n": n, "kind": "case", "chunk_id": obj.ref.case_id, "path": source_path(obj), "title": obj.ref.title,
                "case_number": meta.get("case_number", ""), "source_url": meta.get("source_url", "")}
    if kind == "policy":
        from .policy.retrieval import source_entry
        return source_entry(obj, n)
    from .quotes.retrieval import source_line
    return {"n": n, "kind": "quote", "chunk_id": obj.speech.speech_id, "path": source_line(obj.speech),
            "title": obj.speech.title, "year": obj.speech.year}


def quote_appendix(picks_cited: list) -> str:
    """回答末尾的「讲话原文」分区：程序原样贴出所引段落及出处，不含任何模型文字。"""
    from .quotes.retrieval import source_line
    if not picks_cited:
        return ""
    blocks = [f"「{chr(10).join(p.texts())}」\n——{source_line(p.speech)}" for p in picks_cited]
    return "【讲话原文】\n" + "\n\n".join(blocks)


def answer_combined(question: str, m: Materials, llm, *, model: str, on_delta=None, on_source=None, titles=None,
                     history: list[dict] | None = None) -> CombinedAnswer:
    """**一次**作答调用（不是每个库各一次）。on_delta(text)：流式时回调学生可见文本（[n] 已实时编号）；
    注意讲话引号校验需要完整正文，所以流式时引号里的文字也会先流出；校验只作用于 done 里的最终 answer（见 stream 说明）。"""
    from .course.answer import CitationRewriter
    refs, info = build_refs(m)
    messages = with_history([{"role": "system", "content": COMBINED_SYSTEM},
                             {"role": "user",
                              "content": COMBINED_USER.format(question=question, materials=render_materials(m))}],
                            history)
    rw = CitationRewriter(refs, on_assign=(lambda n, key: on_source(source_for(*info[key], n, titles))) if on_source else None)

    def push(raw: str) -> None:
        shown = rw.feed(_normalize_markers(raw, m))
        if shown and on_delta:
            on_delta(shown)

    stream = getattr(llm, "chat_stream", None) if on_delta else None
    if stream:
        # 流式：分片边界可能切开 [C1]；先在这里缓冲到标记闭合再做 C/Q -> S 的映射
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
        res = stream("answer", messages, model=model, on_delta=feed)
        if buf["t"]:
            push(buf["t"])
    else:
        res = llm.chat("answer", messages, model=model)
        push(res.content or "")
    tail = rw.finish()
    if tail and on_delta:
        on_delta(tail)
    body = rw.rendered_all.strip()
    originals = [para for p in m.quote_picks for para in p.texts()] + [h.chunk["body"] for h in m.policy_hits]
    body, removed = scrub_fabricated_quotes(body, originals)
    if removed:
        body = tidy_after_removal(body)
    cited = [(k, info[k]) for k in rw.citations]
    sources = []
    for key, (kind, obj) in cited:
        sources.append(source_for(kind, obj, rw.numbers[key], titles))
    sources.sort(key=lambda s: s["n"])
    cited_picks = [obj for _, (kind, obj) in cited if kind == "quote"]
    appendix = quote_appendix(cited_picks)
    text = body + ("\n\n" + appendix if appendix else "")
    if m.policy_hits and any(kind == "policy" for _, (kind, _o) in cited):            # 引用了政策：程序附注（征求意见稿标注 + 以汇编为准）
        from .policy.retrieval import CLOSING_NOTE, DRAFT_LABEL
        drafts = list(dict.fromkeys(o.chunk["policy"] for _, (kind, o) in cited if kind == "policy" and o.draft))
        text += "\n\n" + ("、".join(f"《{t}》" for t in drafts) + f"为{DRAFT_LABEL}。" if drafts else "") + CLOSING_NOTE
    return CombinedAnswer(text=text, body=body, sources=sources, removed_quotes=removed, invalid=list(rw.invalid))


# ---------------------------------------------------------------- 各库并行取材

def _course_material(svc, question: str, on_event=None) -> tuple[dict, dict]:
    """课程库：选概念 → 召回 → 筛选（answer=False，不作答）。返回 (course 结果, trace)。"""
    from .course.pipeline import ask, result_to_json
    res = ask(question, ctx=svc.course_ctx, llm=svc.course_llm, answer=False, on_event=on_event)
    trace = res.get("trace")
    return res, trace.to_dict() if hasattr(trace, "to_dict") else trace


def _case_material(svc, question: str, count: int, chapter: int | None) -> dict:
    t0 = time.perf_counter()
    out = svc.case_retriever.retrieve(question, count=count, chapter=chapter)
    out["trace"]["retrieve_s"] = round(time.perf_counter() - t0, 4)
    return out


def _quote_material(svc, question: str, count: int, first_n: int = 1) -> dict:
    t0 = time.perf_counter()
    out = svc.quote_retriever.retrieve(question, count=count, first_n=first_n)
    out["trace"]["retrieve_s"] = round(time.perf_counter() - t0, 4)
    return out


def _policy_material(svc, question: str, first_n: int = 1) -> dict:
    t0 = time.perf_counter()
    out = svc.policy_retriever.retrieve(question, first_n=first_n)
    out["trace"]["retrieve_s"] = round(time.perf_counter() - t0, 4)
    return out


def gather_materials(svc, question: str, decision, *, on_event=None) -> Materials:
    """routes 里的每个库在各自线程里并行检索 / 筛选。每个库的耗时写进 traces。库内出错整体抛出（由上层转成 error 事件，只带类型名）。"""
    routes = list(decision.routes)
    m = Materials()
    jobs = {}
    with ThreadPoolExecutor(max_workers=max(len(routes), 1), thread_name_prefix="combined") as pool:
        t0 = time.perf_counter()
        if "course" in routes:
            jobs["course"] = pool.submit(_course_material, svc, question, on_event)
        if "case" in routes:
            jobs["case"] = pool.submit(_case_material, svc, question, decision.count, decision.chapter)
        if "quote" in routes:
            # 讲话的 Q 编号从 1 起，与教材 / 判例的 S#、C# 互不冲突
            jobs["quote"] = pool.submit(_quote_material, svc, question, decision.count, 1)
        if "policy" in routes:
            jobs["policy"] = pool.submit(_policy_material, svc, question, 1)
        results = {k: f.result() for k, f in jobs.items()}
        wall = round(time.perf_counter() - t0, 4)
    if "course" in results:
        res, tr = results["course"]
        kept = res.get("kept") or []
        m.course_kept, m.relations = kept, res.get("relations") or []
        m.course_refs = {f"S{i}": k.chunk_id for i, k in enumerate(kept, 1)}
        from .course.prompts import ANSWER_PASSAGE
        m.course_text = "\n\n".join(ANSWER_PASSAGE.format(ref=ref, path=" › ".join(k.path), text=k.text) for (ref, _), k in zip(m.course_refs.items(), kept))
        m.traces["course"] = {"fallback": res.get("fallback"), "trace": tr, "selected_concepts": res.get("selected_concepts"),
                              "recalled": len(res.get("recalled") or []), "kept": len(kept)}
        if not kept:
            m.empty.append("course")
    if "case" in results:
        out = results["case"]
        m.case_hits = out["hits"]
        m.traces["case"] = {"mode": out["mode"], **out["trace"]}
        if not out["hits"]:
            m.empty.append("case")
    if "quote" in results:
        out = results["quote"]
        m.quote_picks = out["picks"]
        m.traces["quote"] = {"mode": out["mode"], **out["trace"]}
        if not out["picks"]:
            m.empty.append("quote")
    if "policy" in results:
        out = results["policy"]
        m.policy_hits = out["hits"]
        m.traces["policy"] = {"mode": out["mode"], **out["trace"]}
        if not out["hits"]:
            m.empty.append("policy")
    m.traces["wall_s"] = wall
    return m
