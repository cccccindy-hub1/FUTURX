"""Step 4：只依据筛选后的子块作答；引用只来自这些子块；区分教材原文与本体结构信息。

LLM 看到并使用的是短编号 [S1][S2]…。对学生可见的正文里，[S#] 改成按首次出现顺序编号的 [1][2]…
（CitationRewriter：同一块用同一序号，相邻的重复引用合并；非教材块的标记如 [本体]、[onto1] 一律去掉）；
chunk_id 只保留在 sources / citations / trace 里，正文不出现。编号不在材料内的引用从正文移除并记录。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient
from ..prompt_rules import with_history
from .recall import Recalled

# 全角 / 其他括号 -> 半角方括号：模型偶尔写出 ［S3］ 或 【S3】，不归一化就匹配不上、引用留在正文里
_BRACKET_MAP = str.maketrans({"［": "[", "］": "]", "【": "[", "】": "]", "〔": "[", "〕": "]"})
_CITE_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")
_REF_RE = re.compile(r"^[Ss](\d+)$")
ONTOLOGY_TAG = "本体"
# 防御性写作的免责短语：命中只记进 trace 的 warnings，不自动删改正文
DISCLAIMER_PHRASES = ("这部分不是教材原文", "不是教材原文", "非教材原文", "只是本体中记录", "本体中仅记录", "不能作为教材",
                      "不能作为教材原文证据", "不能作为……依据", "不能作为依据", "本体关系材料中未提供", "材料未提供",
                      "材料中未提供", "材料不足以", "教材并没有列出", "不属于教材原文", "属于本体推导关系")
INSUFFICIENT_PHRASE = "教材材料不足以回答"


@dataclass
class AnswerResult:
    text: str = ""                                            # 学生可见正文：[S#] 已改成按首次出现编号的 [n]，不含 chunk_id
    citations: list[str] = field(default_factory=list)        # chunk_id，按首次出现顺序（= 序号 1..n），仅限 kept 子块
    invalid_citations: list[str] = field(default_factory=list)  # 答案里引用了材料外的编号（已从文本中移除）
    uses_ontology: bool = False
    insufficient: bool = False
    refs: dict[str, str] = field(default_factory=dict)        # S# -> chunk_id（本次提供给 LLM 的全部材料）
    warnings: list[str] = field(default_factory=list)         # 命中的免责短语（不改正文）
    sources: list[dict] = field(default_factory=list)         # [{n, chunk_id, path}]，按序号排列，只含正文实际引用过的块

    def to_dict(self) -> dict:
        return {"text": self.text, "citations": self.citations, "invalid_citations": self.invalid_citations,
                "uses_ontology": self.uses_ontology, "insufficient": self.insufficient, "refs": self.refs,
                "warnings": self.warnings, "sources": self.sources}


def disclaimer_warnings(text: str) -> list[str]:
    """回答里出现的免责短语（去重、按出现顺序）。仅用于告警，不修改正文。"""
    return [p for p in dict.fromkeys(DISCLAIMER_PHRASES) if p in (text or "")]


def render_relations(relations: list[dict]) -> str:
    return "\n".join(prompts.ANSWER_RELATION.format(
        subject=r["subject"], predicate=r.get("predicate_label") or r["predicate_id"], object=r["object"])
        for r in relations)


class CitationRewriter:
    """把模型输出里的 [S#] 改成学生可见的序号 [n]；流式与非流式共用，保证两边正文逐字一致。

    - 序号按「首次出现」分配，同一块始终用同一个序号；
    - 相邻的重复引用合并：连续引用同一块时只保留一个（中间只隔空白、标点或其它被去掉的标记也算相邻）；
    - 编号不在 refs 的 [S#]、占位符 [S#]、直接写出的 chunk_id（de:…）、[本体] / [onto1] / [onto2] 这类非教材块标记，
      一律从正文去掉（前两类、chunk_id 记入 invalid_citations；本体标记只记 uses_ontology）；
    - 章节路径等普通方括号内容原样保留；
    - 流式：feed() 逐片喂入，返回可以立即输出的文本；一个 [S12] 被拆在两个分片时，未闭合的 "[…" 缓冲到下一片，
      最长缓冲 MAX_PENDING 个字符，超过就当作普通文本放出（避免正文里一个孤立的 "[" 把后面整段扣住）。
    - on_assign(n, chunk_id)：第一次给某块分配序号时回调（流式用它发 materials 事件）。
    """

    MAX_PENDING = 130

    def __init__(self, refs: dict[str, str], on_assign=None):
        self.refs, self.on_assign = refs, on_assign
        self.numbers: dict[str, int] = {}            # chunk_id -> 序号（按首次出现）
        self.invalid: list[str] = []
        self.uses_ontology = False
        self._buf = ""                               # 尚未能确定的尾部（以 "[" 开头、还没闭合）
        self._last_cited: str | None = None          # 紧邻的上一处输出是否为引用（用于相邻合并）
        self._out_tail = ""                          # 已输出文本的尾部，用于判断「相邻」
        self._rendered: list[str] = []               # 已输出的全部学生可见文本
        self._held = ""                              # 暂扣的尾部空白：等看到后面是不是会被整体去掉的标记再决定输不输出

    # ---- 对外 ----
    def feed(self, chunk: str) -> str:
        self._buf += (chunk or "").translate(_BRACKET_MAP)
        return self._emit(self._drain(final=False))

    def finish(self) -> str:
        out = self._drain(final=True)
        self._held = ""                                # 文末的空白不输出
        return self._emit(out)

    def _emit(self, text: str) -> str:
        self._rendered.append(text)
        return text

    @property
    def rendered_all(self) -> str:
        return "".join(self._rendered)

    @property
    def citations(self) -> list[str]:
        return sorted(self.numbers, key=self.numbers.get)

    # ---- 内部 ----
    def _drain(self, final: bool) -> str:
        out = []
        buf = self._buf
        pos = 0
        while pos < len(buf):
            lb = buf.find("[", pos)
            if lb < 0:
                out.append(self._plain(buf[pos:]))
                pos = len(buf)
                break
            if lb > pos:
                out.append(self._plain(buf[pos:lb]))
                pos = lb
            rb = buf.find("]", lb + 1)

            if rb < 0:                                # 还没闭合
                if final or len(buf) - lb > self.MAX_PENDING:
                    out.append(self._plain(buf[lb:lb + 1]))      # 当作普通 "["，继续往后扫
                    pos = lb + 1
                    continue
                break
            inner = buf[lb + 1:rb]
            if "\n" in inner or "[" in inner or len(inner) > 120:     # 不是合法的 [...] 引用样式：当普通文本
                out.append(self._plain(buf[lb:lb + 1]))
                pos = lb + 1
                continue
            piece = self._tag(inner.strip(), buf[lb:rb + 1])
            if piece:
                out.append(self._held + piece)         # 有输出：暂扣的空白在它前面补回去
            self._held = ""                            # 无输出（被整体去掉）：暂扣的空白一并丢弃
            pos = rb + 1
        self._buf = buf[pos:]
        return "".join(out)

    def _plain(self, text: str) -> str:
        if text.strip(" \t，。、；：,.;:"):               # 有实质内容的普通文本打断「相邻」
            self._last_cited = None
        text = self._held + text
        self._held = ""
        body = text.rstrip(" \t")
        if len(body) != len(text):                     # 尾部空白扣住：流式下后面可能紧跟 [本体] 这类会被整体去掉的标记
            self._held = text[len(body):]
        self._out_tail = (self._out_tail + body)[-8:]
        return body

    def _tag(self, tag: str, raw: str) -> str:
        if tag == ONTOLOGY_TAG or re.fullmatch(r"onto[12]?", tag, flags=re.I):
            self.uses_ontology = True
            return ""
        if tag in ("S#", "S＃"):                       # 模型照抄提示词里的占位符：直接去掉，不计入无效引用
            return ""
        sm = _REF_RE.match(tag)
        if sm:
            ref = f"S{int(sm.group(1))}"
            cid = self.refs.get(ref)
            if cid is None:
                if ref not in self.invalid:
                    self.invalid.append(ref)
                return ""
            new = cid not in self.numbers
            if new:
                self.numbers[cid] = len(self.numbers) + 1
                if self.on_assign:
                    self.on_assign(self.numbers[cid], cid)
            if self._last_cited == cid:                # 相邻重复引用同一块：合并
                return ""
            self._last_cited = cid
            return f"[{self.numbers[cid]}]"
        if tag.startswith("de:"):                      # 绕过编号直接写 chunk_id：一律不信任
            if tag not in self.invalid:
                self.invalid.append(tag)
            return ""
        self._last_cited = None
        return raw


def sanitize_citations(text: str, refs: dict[str, str]) -> tuple[str, list[str], list[str], bool]:
    """一次性版本：返回 (学生可见正文, 有效引用 chunk_id 按首次出现顺序, 无效引用, 是否出现过本体标记)。"""
    rw = CitationRewriter(refs)
    rw.feed(text or "")
    rw.finish()
    out = rw.rendered_all
    return out, rw.citations, rw.invalid, rw.uses_ontology


_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百]+章")
_CHUNK_CHAPTER_RE = re.compile(r":ch(\d+)(?:[.:]|$)")


def chapter_titles(chunks) -> dict[int, str]:
    """章号 -> 章标题（如 6 -> "第六章 数据市场"），取自路径首级就是章标题的子块。"""
    out: dict[int, str] = {}
    for c in chunks:
        if c.chapter is not None and c.chapter not in out and c.path and _CHAPTER_RE.match(c.path[0]):
            out[c.chapter] = c.path[0]
    return out


def source_entry(n: int, chunk: Recalled, titles: dict[int, str] | None = None) -> dict:
    """学生可见的来源条目：序号 n、chunk_id（给评测/调试）、章名与小节路径。
    教材里「第一节」或「一、」用了一级标题样式的章节，标题栈会把章标题挤掉，路径首级不是章名；这里按 chunk_id 的章号补上。"""
    path = list(chunk.path)
    m = _CHUNK_CHAPTER_RE.search(chunk.chunk_id)
    if titles and m and not (path and _CHAPTER_RE.match(path[0])):
        title = titles.get(int(m.group(1)))
        if title:
            path.insert(0, title)
    return {"n": n, "chunk_id": chunk.chunk_id, "path": " › ".join(path)}


def answer_question(question: str, kept: list[Recalled], relations: list[dict], llm: LLMClient, *,
                    model: str | None = None, on_delta=None, on_source=None,
                    titles: dict[int, str] | None = None, solve: bool | str = False,
                    history: list[dict] | None = None, profile: str = "") -> AnswerResult:
    """solve=True：学生带着具体题目来问，系统提示词追加“判断 / 选择题先给结论”一条；solve="explain"：题库命中后讲解，
    追加“与参考答案一致”一条；默认 False，提示词与原来逐字相同。
    on_delta(text)：流式时逐段回调「学生可见」文本（[S#] 已实时改成 [n]，一个被拆在两片里的 [S12] 会先缓冲到闭合）；
    on_source(entry)：某块第一次被引用、分配序号时回调（entry 同 source_entry）。返回值与非流式相同。
    profile：这个学生的画像（rag.profile.summary），非空时追加到 system 末尾；为空时提示词逐字不变。"""
    model = model or config.COURSE_ANSWER_MODEL
    refs = {f"S{i + 1}": k.chunk_id for i, k in enumerate(kept)}
    by_id = {k.chunk_id: k for k in kept}
    passages = "\n\n".join(
        prompts.ANSWER_PASSAGE.format(ref=ref, path=" › ".join(k.path), text=k.text)
        for (ref, _cid), k in zip(refs.items(), kept)) or "（无）"
    hint = prompts.ANSWER_EXPLAIN_HINT if solve == "explain" else (prompts.ANSWER_SOLVE_HINT if solve else "")
    messages = with_history([
        {"role": "system", "content": prompts.ANSWER_SYSTEM + hint},
        {"role": "user", "content": prompts.ANSWER_USER.format(question=question, passages=passages)
         + (prompts.ANSWER_ONTOLOGY.format(ontology=render_relations(relations)) if relations else "")},
    ], history, profile=profile)
    rw = CitationRewriter(refs, on_assign=(lambda n, cid: on_source(source_entry(n, by_id[cid], titles))) if on_source else None)

    def push(raw: str) -> None:
        shown = rw.feed(raw)
        if shown and on_delta:
            on_delta(shown)

    stream = getattr(llm, "chat_stream", None) if on_delta else None
    if stream:
        res = stream("answer", messages, model=model, on_delta=push)
        tail = rw.finish()
        if tail and on_delta:
            on_delta(tail)
        text = rw.rendered_all
    else:
        res = llm.chat("answer", messages, model=model)
        push(res.content or "")
        tail = rw.finish()
        if tail and on_delta:
            on_delta(tail)
        text = rw.rendered_all
    sources = [source_entry(rw.numbers[c], by_id[c], titles) for c in rw.citations]
    return AnswerResult(text=text.strip(), citations=rw.citations, invalid_citations=rw.invalid,
                        uses_ontology=rw.uses_ontology, insufficient=INSUFFICIENT_PHRASE in text, refs=refs,
                        warnings=disclaimer_warnings(text), sources=sources)
