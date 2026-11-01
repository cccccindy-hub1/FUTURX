"""Step 4：只依据筛选后的子块作答；引用只来自这些子块；区分教材原文与本体结构信息。

LLM 看到并使用的是短编号 [S1][S2]…；本模块把有效的 [S#] 换回 chunk_id 后再输出，
编号不在材料内的引用从正文移除并记录。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient
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
    text: str = ""                                            # 已把 [S#] 换回 [chunk_id]
    citations: list[str] = field(default_factory=list)        # chunk_id，按首次出现顺序，仅限 kept 子块
    invalid_citations: list[str] = field(default_factory=list)  # 答案里引用了材料外的编号（已从文本中移除）
    uses_ontology: bool = False
    insufficient: bool = False
    refs: dict[str, str] = field(default_factory=dict)        # S# -> chunk_id（本次提供给 LLM 的全部材料）
    warnings: list[str] = field(default_factory=list)         # 命中的免责短语（不改正文）

    def to_dict(self) -> dict:
        return {"text": self.text, "citations": self.citations, "invalid_citations": self.invalid_citations,
                "uses_ontology": self.uses_ontology, "insufficient": self.insufficient, "refs": self.refs,
                "warnings": self.warnings}


def disclaimer_warnings(text: str) -> list[str]:
    """回答里出现的免责短语（去重、按出现顺序）。仅用于告警，不修改正文。"""
    return [p for p in dict.fromkeys(DISCLAIMER_PHRASES) if p in (text or "")]


def render_relations(relations: list[dict]) -> str:
    return "\n".join(prompts.ANSWER_RELATION.format(
        subject=r["subject"], subject_id=r["subject_id"], predicate=r.get("predicate_label") or r["predicate_id"],
        object=r["object"], object_id=r["object_id"], view=r["view"]) for r in relations)


def sanitize_citations(text: str, refs: dict[str, str]) -> tuple[str, list[str], list[str], bool]:
    """把 [S#] 换回 [chunk_id]；编号不在 refs 的 [S#] 以及直接写出的 chunk_id（de:…）移除并记录。

    返回 (清理后文本, 有效引用 chunk_id 列表, 无效引用列表, 是否使用本体标记)。
    章节路径等普通方括号内容（不是 S# 也不是 de:…）原样保留。
    """
    valid: list[str] = []
    invalid: list[str] = []
    uses_ontology = False
    text = (text or "").translate(_BRACKET_MAP)

    def repl(m: re.Match) -> str:
        nonlocal uses_ontology
        tag = m.group(1).strip()
        if tag == ONTOLOGY_TAG:
            uses_ontology = True
            return m.group(0)
        sm = _REF_RE.match(tag)
        if sm:
            ref = f"S{int(sm.group(1))}"
            cid = refs.get(ref)
            if cid is None:
                if ref not in invalid:
                    invalid.append(ref)
                return ""
            if cid not in valid:
                valid.append(cid)
            return f"[{cid}]"
        if tag in ("S#", "S＃"):           # 模型照抄提示词里的占位符：直接去掉，不计入无效引用
            return ""
        if tag.startswith("de:"):          # 绕过编号直接写 chunk_id：一律不信任
            if tag not in invalid:
                invalid.append(tag)
            return ""
        return m.group(0)

    return _CITE_RE.sub(repl, text), valid, invalid, uses_ontology


def answer_question(question: str, kept: list[Recalled], relations: list[dict], llm: LLMClient, *,
                    model: str | None = None) -> AnswerResult:
    model = model or config.COURSE_ANSWER_MODEL
    refs = {f"S{i + 1}": k.chunk_id for i, k in enumerate(kept)}
    passages = "\n\n".join(
        prompts.ANSWER_PASSAGE.format(ref=ref, path=" › ".join(k.path), text=k.text)
        for (ref, _cid), k in zip(refs.items(), kept)) or "（无）"
    messages = [{"role": "system", "content": prompts.ANSWER_SYSTEM},
                {"role": "user", "content": prompts.ANSWER_USER.format(question=question, passages=passages)
                 + (prompts.ANSWER_ONTOLOGY.format(ontology=render_relations(relations)) if relations else "")}]
    res = llm.chat("answer", messages, model=model)
    text, valid, invalid, onto = sanitize_citations(res.content, refs)
    return AnswerResult(text=text.strip(), citations=valid, invalid_citations=invalid, uses_ontology=onto,
                        insufficient=INSUFFICIENT_PHRASE in text, refs=refs, warnings=disclaimer_warnings(text))
