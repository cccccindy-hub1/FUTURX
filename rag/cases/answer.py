"""判例单库作答。

每个判例输出（设计文档第二节）：
    【第六章 数据市场】北京微梦创科…不正当竞争纠纷案
    案号：（2021）……　审理法院：……　裁判日期：……
    基本案情：两三句。
    裁判要点：两三句。
    与你的问题的关系：一两句。
- 前两行（章节标签、判例名、案号、法院、日期）由程序从元数据填写，绝不让模型生成，避免编错案号；
- 后三项由模型根据检索到的判例原文写，精炼；引用用 [C#]（材料编号），清理后改成统一的 [1] [2]（按首次出现编号，
  与课程库同一套 CitationRewriter）；来源列表带章节标签、案名、案号和来源链接；
- 一次调用同时写完所有判例（count 个），不是每个判例调用一次。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..course.llm import LLMClient
from ..persona import PERSONA_BLOCK
from .retrieval import CaseHit

CASE_ANSWER_SYSTEM = PERSONA_BLOCK + """
现在是判例讲解：只依据下面给出的判例材料写，不使用材料之外的知识，不要编造案号、法院、日期、金额。
材料按 [C1]、[C2]… 编号，每个编号是一个判例。整个回答采用“总—分—总”结构：
- lead：开头一句总起，直接回答学生问题，可用一句话概括这些判例的共同裁判立场；不超过 60 字；
- 每个判例写下面三项，每项一两句话到两三句话，精炼、不复述原文：
  基本案情：……
  裁判要点：……
  与你的问题的关系：……
- close：结尾一句收束，点出这些判例对回答该问题的意义；不超过 60 字，不得重复 lead 或任何判例原句。
判例的标题、案号、法院、日期由系统另行填写，你不要写。每项里的论断后面用该判例的编号标注依据，如 [C1]。
不要写没有具体原因的套话式免责。
输出严格 JSON：{"lead":"…","cases":[{"id":"C1","facts":"…","holding":"…","relation":"…"}],"close":"…"}，
每个判例一项，id 必须是材料里出现过的编号。"""

CASE_ANSWER_USER = "学生问题：{question}\n\n判例材料：\n{materials}"


@dataclass
class CaseAnswer:
    text: str = ""
    sources: list[dict] = field(default_factory=list)       # [{n, kind:"case", chunk_id, path, title, ...}]
    parsed: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)


def header_lines(h: CaseHit) -> list[str]:
    """程序填写的两行：【章节标签】判例名 / 案号、法院、日期。字段缺失就不写那一项，不让模型补。"""
    m = h.ref.meta
    parts = []
    for label, key in (("案号", "case_number"), ("审理法院", "court"), ("裁判日期", "judgment_date")):
        if m.get(key):
            parts.append(f"{label}：{m[key]}")
    lines = [f"【{h.ref.chapter}】{h.ref.title}"]
    if parts:
        lines.append("　".join(parts))
    return lines


def source_path(h: CaseHit) -> str:
    """来源条目的显示文字：章节标签 + 案名 + 案号。"""
    m = h.ref.meta
    return f"{h.ref.chapter} › {h.ref.title}" + (f"（{m['case_number']}）" if m.get("case_number") else "")


def render_materials(hits: list[CaseHit], first_n: int = 1, limit_chars: int = 3200) -> str:
    out = []
    for k, h in enumerate(hits, first_n):
        parts = [h.card] + h.bodies
        text = "\n".join(p for p in parts if p)
        out.append(f"[C{k}] {text[:limit_chars]}")
    return "\n\n".join(out)


def build_messages(question: str, hits: list[CaseHit], first_n: int = 1) -> list[dict]:
    return [{"role": "system", "content": CASE_ANSWER_SYSTEM},
            {"role": "user", "content": CASE_ANSWER_USER.format(question=question, materials=render_materials(hits, first_n))}]


def answer_cases(question: str, hits: list[CaseHit], llm: LLMClient, *, model: str,
                 history: list[dict] | None = None, profile: str = "") -> CaseAnswer:
    """一次作答调用。模型只填 facts / holding / relation；头两行由程序拼。解析失败时只返回程序填写的头部和原文要点，不重试。
    profile：这个学生的画像，非空时追加到 system 末尾；为空时提示词逐字不变。"""
    from ..course.answer import CitationRewriter
    from ..course.llm import parse_json_object
    msgs = build_messages(question, hits)
    if history or profile:
        from ..prompt_rules import with_history

        msgs = with_history(msgs, history, profile=profile)
    res = llm.chat("case_answer", msgs, model=model, json_mode=True)
    refs = {f"C{k}": h.ref.case_id for k, h in enumerate(hits, 1)}
    rw = CitationRewriter(refs)
    by_ref = {f"C{k}": (k, h) for k, h in enumerate(hits, 1)}
    out = CaseAnswer()
    items: dict[str, dict] = {}
    lead, close = "", ""
    try:
        obj = parse_json_object(res.content)
        if isinstance(obj, dict):
            lead = str(obj.get("lead") or "").strip()
            close = str(obj.get("close") or "").strip()
        for it in obj.get("cases") or []:
            key = str(it.get("id", "")).strip().upper().lstrip("[").rstrip("]") if isinstance(it, dict) else ""
            if key in by_ref and key not in items:
                items[key] = it
            else:
                out.dropped.append({"id": key, "reason": "unknown_or_duplicate"})
    except Exception as e:  # noqa: BLE001 - 只记类型名
        out.dropped.append({"reason": type(e).__name__})
    if not items and not out.dropped:
        out.dropped.append({"reason": "no_usable_cases"})
    # 只有模型写的三项交给 CitationRewriter（[C#] -> [n]，且按全文首次出现顺序统一编号）；
    # 程序填写的头部（【第六章 数据市场】…、案号…）原样拼回，绝不经过改写器——它会把全角【】归一化成半角 []。
    rw.refs = {k.replace("C", "S", 1): v for k, v in refs.items()}
    blocks = []
    for key, (k, h) in by_ref.items():
        it = items.get(key)
        lines = header_lines(h)
        if it:
            for label, f in (("基本案情", "facts"), ("裁判要点", "holding"), ("与你的问题的关系", "relation")):
                body = str(it.get(f, "")).strip()
                if body:
                    lines.append(f"{label}：" + _rewrite_c_markers(body, rw))
        blocks.append("\n".join(lines))
    # lead → 三项式 → close；lead / close 是模型写的整句，也要过一遍 [C#]→[n] 改写。
    # 字段缺失（旧缓存 / 模型未给）时优雅降级：只省略该段，不影响正文。
    body = "\n\n".join(blocks)
    parts = []
    for piece in (lead, body, close):
        if piece:
            parts.append(_rewrite_c_markers(piece, rw) if piece is not body else piece)
    out.text = "\n\n".join(parts)
    out.parsed = [{"id": k, "ok": k in items} for k in by_ref]
    out.sources = [{"n": rw.numbers[cid], "kind": "case", "chunk_id": cid, "path": source_path(h),
                    "title": h.ref.title, "case_number": h.ref.meta.get("case_number", ""),
                    "source_url": h.ref.meta.get("source_url", "")}
                   for cid, h in ((h.ref.case_id, h) for _, h in by_ref.values()) if cid in rw.numbers]
    out.sources.sort(key=lambda s: s["n"])
    return out


_BRACKETS = str.maketrans({"［": "[", "］": "]", "【": "[", "】": "]", "〔": "[", "〕": "]"})


def _rewrite_c_markers(text: str, rw) -> str:
    """CitationRewriter 只认 [S#]；判例材料用 [C#]。把 [C#] 映射成 [S#] 再交给同一个改写器（调用方已设好 rw.refs），
    保证编号规则（首次出现、同块同号、相邻合并、去掉非教材标记）与课程库完全一致。每一项独立 flush，
    不会把半截标记带到下一项。"""
    text = text.translate(_BRACKETS)
    mapped = re.sub(r"\[\s*[Cc](\d+)\s*\]", lambda m: f"[S{int(m.group(1))}]", text)
    shown = rw.feed(mapped) + rw.finish()
    rw._buf = ""
    rw._last_cited = None            # 每一项（基本案情 / 裁判要点 / 与问题的关系）之间隔着程序写的标题行，不算「相邻」，不能合并
    return shown.strip()
