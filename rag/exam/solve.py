"""“解题”与“找题”的区分，以及解题前的题库预匹配（纯程序，不调用任何模型）。

- 找题：“出三道判断题”“第四章有哪些名词解释”——没有具体题干，走原来的试卷库检索；
- 解题：问题里带着一道具体的题（“判断这道题是否正确：……”“名词解释：专有数据、共享数据与开放数据”）。
  先把问题去掉外壳（解题指令、题号、题型前缀），剩下的题干至少 SOLVE_MIN_CHARS（12）个字时，和题库的题干模糊匹配：
  命中 → 直接输出“题目 + 答案 + 解析”，不路由、不调用任何模型（调用 0 次）；没命中 → 走原来的路由，
  路由把它判为解题时送课程库，课程作答提示词追加一条“判断题 / 选择题第一句先给结论”。
匹配：双方归一化（NFKC、去标点空格、题号、题型前缀）；用字符 4-gram 倒排索引取候选（不逐题比较，耗时 ms 级），
对前 N 个候选算：完全包含（较短者不少于较长者的一半），或相似度 ≥ 0.9。
"""
from __future__ import annotations

import re
import time
import unicodedata
from difflib import SequenceMatcher

SOLVE_MIN_CHARS = 12
SIMILARITY = 0.9
NGRAM = 4
MAX_CANDIDATES = 8
CONTAIN_RATIO = 0.5

# 解题指令外壳：出现任一个，就说明学生是在让我们解答一道具体的题
SOLVE_SHELLS = (
    r"判断(?:一下)?(?:这道题|这个题|这题|下面(?:这)?(?:句话|道题|题)?|这句话)?(?:是否|对不对|是不是)?(?:正确|对|成立)?",
    r"这句话(?:是否|对不对|是不是)?(?:正确|对|成立)?(?:吗|呢)?", r"(?:是否|对不对|是不是)正确", r"对不对", r"正确吗", r"对吗",
    r"名词解释", r"(?:请|帮我|帮忙|麻烦)?(?:做|解|答|解答|回答|分析)(?:一下)?(?:这道题|这个题|这题|下面(?:这)?道题)",
    r"这道题(?:怎么(?:做|答|选)|选什么|的答案|答案)?", r"(?:答案|解析)(?:是什么|是啥|是多少)?", r"参考答案", r"选哪个", r"选什么",
)
_SHELL_RE = re.compile("(?:" + "|".join(SOLVE_SHELLS) + ")")
# 讲解类外壳：只用于剥离题干（预匹配），不参与 is_solve_question 的判定（否则普通课程问题也会被当成解题、改动课程提示词）
_EXPLAIN_RE = re.compile(r"(?:从课程(?:的)?角度[，,]?\s*)?(?:请|帮我|帮忙|麻烦)?(?:讲解|讲讲|讲一下|解释|分析|理解)(?:一下)?(?:这道题|这个题|这题)?"
                         r"|这道题(?:怎么|如何)理解|(?:怎么|如何)理解(?:这道题|这个题|这题)?")
_SHELL_WORDS = ("判断", "这句话", "正确吗", "对吗", "对不对", "是否正确", "名词解释", "这道题", "这个题", "这题", "答案", "解析", "怎么做", "选什么", "选哪个", "帮我做", "解答")
_NUM_PREFIX = re.compile(r"^(?:第\s*\d{1,3}\s*[题道]|[（(]?\d{1,3}[）).、．.])")
_TYPE_PREFIX = re.compile(r"^(?:【[^】]{1,12}】|(?:单选题|多选题|选择题|判断题|名词解释|简答题|论述题|案例题)[：:])")
_PUNCT = re.compile(r"[\W_]+", re.UNICODE)
_FIND_WORDS = ("出几道", "出一道", "出两道", "出三道", "出四道", "出五道", "来几道", "来一道", "来几题", "有哪些", "练习题", "出题", "出道")


def normalize(text: str) -> str:
    """去标点、空格；题号、题型前缀已在 strip_shell 里处理，这里只做字符层面的归一。"""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text or "")).casefold()


def strip_shell(question: str) -> tuple[str, bool]:
    """去掉解题外壳，返回 (剩余题干, 是否出现过外壳)。"""
    q = unicodedata.normalize("NFKC", question or "").strip()
    had = bool(_SHELL_RE.search(q)) and any(w in q for w in _SHELL_WORDS)
    rest = q
    for _ in range(3):
        prev = rest
        rest = _TYPE_PREFIX.sub("", _NUM_PREFIX.sub("", rest.strip())).strip()
        if rest == prev:
            break
    rest = _SHELL_RE.sub(" ", _EXPLAIN_RE.sub(" ", rest))
    rest = re.sub(r"^\s*(?:请|帮我|帮忙|麻烦|下面|以下|这道题|这个题)\s*", "", rest)
    return rest.strip(), had


def stem_text(question: str) -> str:
    return normalize(strip_shell(question)[0])


def is_solve_question(question: str) -> bool:
    """带具体题干的解题问题：出现解题外壳，且剩余题干 ≥12 个字，且不是“出题 / 找题”的说法。"""
    rest, had = strip_shell(question)
    return had and len(normalize(rest)) >= SOLVE_MIN_CHARS and not any(w in question for w in _FIND_WORDS)


# 要讲解而不只是要答案：命中题库后先给参考答案，再走课程流程讲解
EXPLAIN_WORDS = ("怎么理解", "如何理解", "讲解", "讲讲", "讲一下", "解释", "为什么", "为啥", "分析一下", "分析这", "从课程角度", "理解一下")


def wants_explanation(question: str) -> bool:
    return any(w in (question or "") for w in EXPLAIN_WORDS)


# 指代上一轮列出的某道题：“讲解一下第四道题”“第2题选什么”“这道题怎么理解”（上一轮只有一道题时）
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ORDINAL_RE = re.compile(r"第\s*([0-9]{1,2}|十?[一二两三四五六七八九]|十)\s*(?:道|题|个|小题)")
_THIS_ITEM_RE = re.compile(r"(?:这|那|该|此|上面那?|刚才那?|前面那?)\s*(?:一)?\s*[道个]?\s*题")
_LIST_ITEM_RE = re.compile(r"^\s*(\d{1,2})\.\s*【[^】]{0,40}】\s*(.+?)\s*$", re.M)     # format_items 的题目行：1.【判断题·第4章 …】题干
_ANSWER_ONLY = ("答案", "选什么", "选哪个", "选哪项", "对不对", "正确吗", "对吗", "是否正确")


def _ordinal(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if text.startswith("十"):
        return 10 + _CN_NUM.get(text[1:], 0)
    return _CN_NUM.get(text)


def referenced_stem(question: str, history) -> str | None:
    """问题指代上一轮回答里列出的某道题时，返回那道题的题干（纯程序，不调用模型）；取不到返回 None。
    history：[{"role", "content"}]，取最近一条带题目列表的助手回答。"""
    m = _ORDINAL_RE.search(question or "")
    n = _ordinal(m.group(1)) if m else None
    if n is None and not _THIS_ITEM_RE.search(question or ""):
        return None
    last = next((h.get("content") or "" for h in reversed(history or [])
                 if isinstance(h, dict) and h.get("role") == "assistant" and _LIST_ITEM_RE.search(h.get("content") or "")), "")
    items = {int(k): v for k, v in _LIST_ITEM_RE.findall(last)}
    if not items:
        return None
    if n is None:                                      # “这道题”：上一轮恰好只有一道题才能确定
        if len(items) != 1:
            return None
        n = next(iter(items))
    return items.get(n)


def wants_answer_only(question: str) -> bool:
    return any(w in (question or "") for w in _ANSWER_ONLY) and not wants_explanation(question)


def is_prematch_candidate(question: str) -> bool:
    """值得去题库预匹配的问题：解题问题，或要讲解且带着一段题干（≥12 字）；出题 / 找题的说法除外。"""
    rest, had = strip_shell(question)
    return (had or wants_explanation(question)) and len(normalize(rest)) >= SOLVE_MIN_CHARS \
        and not any(w in question for w in _FIND_WORDS)


def _ngrams(s: str) -> set[str]:
    return {s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1)}


class StemMatcher:
    """题库题干的 4-gram 倒排索引。items 为任意带 .q.stem 的对象列表（ExamItem）。"""

    def __init__(self, items):
        self.norms: list[str] = []
        self.exact: dict[str, int] = {}                  # 归一化题干 -> 下标（完全相同的题干直接命中，不依赖投票）
        self.index: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            s = normalize(_TYPE_PREFIX.sub("", _NUM_PREFIX.sub("", it.q.stem or "")))
            s = re.sub(r"(?:下列)?(?:哪一?个?选项|选项).*$", "", s) if False else s
            self.norms.append(s)
            if len(s) >= SOLVE_MIN_CHARS:
                self.exact.setdefault(s, i)
                for g in _ngrams(s):
                    self.index.setdefault(g, []).append(i)

    def match(self, question: str) -> dict:
        """返回 {"hit": 题目下标或 None, "similarity", "how": exact|contain|similar|"", "stem_chars", "ms", "candidates"}。"""
        t0 = time.perf_counter()
        rest, _ = strip_shell(question)
        nq = normalize(rest)
        out = {"hit": None, "similarity": 0.0, "how": "", "stem_chars": len(nq), "candidates": 0}
        if len(nq) >= SOLVE_MIN_CHARS:
            votes: dict[int, int] = {}
            for g in _ngrams(nq):
                for i in self.index.get(g, ()):
                    votes[i] = votes.get(i, 0) + 1
            # 票数相同时偏向长度更接近的题干（术语也出现在大量长选择题题干里，不能被它们挤出候选）
            cands = sorted(votes, key=lambda i: (-votes[i], abs(len(self.norms[i]) - len(nq))))[:MAX_CANDIDATES]
            if nq in self.exact and self.exact[nq] not in cands:
                cands.append(self.exact[nq])
            out["candidates"] = len(cands)
            best = (0.0, None, "")
            for i in cands:
                ns = self.norms[i]
                short, long_ = (nq, ns) if len(nq) <= len(ns) else (ns, nq)
                if nq == ns:
                    score, how = 1.0, "exact"                            # 题干与题库完全一致：优先于“被更长题干包含”
                elif short in long_ and len(short) >= CONTAIN_RATIO * len(long_):
                    score, how = 0.99, "contain"
                else:
                    sm = SequenceMatcher(None, nq, ns, autojunk=False)
                    if sm.real_quick_ratio() < SIMILARITY or sm.quick_ratio() < SIMILARITY:
                        continue
                    score, how = sm.ratio(), "similar"
                    if score < SIMILARITY:
                        continue
                if score > best[0]:
                    best = (score, i, how)
            if best[1] is not None:
                out.update(hit=best[1], similarity=round(best[0], 4), how=best[2])
        out["ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out
