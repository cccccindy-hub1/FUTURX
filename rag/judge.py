"""答案打分：词法基线（字符 F1）/ ROUGE-L 与 LLM-as-judge。

- 词法指标用于离线、零成本的快速对比；对中文按字符计算，无需分词依赖。
- LLM-as-judge 用「1~5 分 + 是否答对」的严格 JSON 输出；解析失败时降级为
  词法分数，保证评测不会因单条模型的格式抖动而崩掉。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "你是严谨的学术答案评审。给「模型答案」与「标准答案」的一致性打分。"
    "只输出一个 JSON 对象，不要任何其他文字、不要代码块围栏。格式："
    '{"score": <1-5 的整数>, "correct": <true|false>, "reason": "<20字以内的中文理由>"}。'
    "5=完全覆盖要点；4=基本覆盖；3=部分相关但有遗漏；2=仅沾边；1=错误或无关。"
)


@dataclass
class JudgeResult:
    score: float          # 5 分制
    correct: bool
    reason: str = ""
    method: str = "llm"   # llm | lexical


# ---------------------------------------------------------------- 词法指标

def _char_ngrams(text: str, n: int = 2) -> list[str]:
    t = re.sub(r"\s+", "", text)
    if len(t) < n:
        return [t] if t else []
    return [t[i : i + n] for i in range(len(t) - n + 1)]


def char_f1(pred: str, gold: str) -> float:
    """字符 bigram 的 F1；对中文顺序不敏感近似的召回/精确平衡。"""
    p, g = _char_ngrams(pred), _char_ngrams(gold)
    if not p or not g:
        return 0.0
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    """字符级 LCS 的 ROUGE-L F1（保留顺序）。"""
    a = re.sub(r"\s+", "", pred)
    b = re.sub(r"\s+", "", gold)
    if not a or not b:
        return 0.0
    # 滚动数组求 LCS 长度
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[len(b)]
    precision = lcs / len(a)
    recall = lcs / len(b)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def lexical_score(pred: str, gold: str) -> JudgeResult:
    """无需 LLM 的基线打分：F1 映射到 5 分制，阈值判断是否答对。"""
    f1 = char_f1(pred, gold)
    return JudgeResult(
        score=round(1 + 4 * f1, 3),
        correct=f1 >= 0.5,
        reason=f"char_f1={f1:.2f}",
        method="lexical",
    )


# ---------------------------------------------------------------- LLM 评审

def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):  # 容忍模型加代码块围栏
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_answer(question: str, pred: str, gold: str, llm, model: str) -> JudgeResult:
    """用 LLM 打分；llm 为 None 或解析失败时降级为词法分数。"""
    if llm is None or not pred.strip():
        return lexical_score(pred, gold)
    user = f"问题：{question}\n\n标准答案：{gold}\n\n模型答案：{pred}"
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        data = _parse_json(resp.choices[0].message.content or "")
        if not data or "score" not in data:
            return lexical_score(pred, gold)
        return JudgeResult(
            score=float(data["score"]),
            correct=bool(data.get("correct", float(data["score"]) >= 3)),
            reason=str(data.get("reason", ""))[:40],
            method="llm",
        )
    except Exception as e:
        logger.warning("LLM 评审失败，降级为词法打分：%s", e)
        return lexical_score(pred, gold)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0
