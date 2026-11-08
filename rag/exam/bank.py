"""题库 docx → 题目列表：一题一条记录（题干 + 选项 + 参考答案 + 评分要点），不跨题、不拆题。

题库结构（教师版，2015 题）有两种写法，同一份文件里可以混用，解析器两种都认：
- 旧格式（10.31 修订版全书；11.7 题干修订版的第十一章仍是它）：
  「第X章 章名（共 N 题）」→「一、选择题（共 N 题）」…；选项可在一个段落里「A. …B. …C. …D. …」；
  答案段「【参考答案】…【评分要点与解析】…」（可在同一段或相邻两段）；案例分析题答案按【问题（n）参考答案】分小问，可跨多段。
- 新格式（11.7 题干修订版前十章）：
  「第X章 章名」+「本章共 N 题」→「一 选择题  共 N 题」…；每个选项一段；
  答案「参考答案  X」+「解析  …」（选择 / 判断）或「评分要点：（1）…（50%）；…」（名词解释 / 简答 / 论述 / 案例），
  再加「书稿依据：…」（出处，记入 source，不当作答案）；案例题以【教学情境】开头，【分析问题】单独成段。
- 全书最后的「书稿待核说明」及其后各段不是题目，读到即停。

学生版题干与教师版相同、没有答案段，只用于核对（compare_student）。
只读 word/document.xml（书稿里个别图片条目 CRC 损坏，不能用 python-docx 整包打开），开头的统计表跳过。
question_id = ex:{版本}:ch{章:02d}:{题型代码}:{题号:03d}:{题干 sha1 前 8 位}。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from ..course.textbook import _W, _collect_text
from . import config

_CN = "一二三四五六七八九十"
CHAPTER_RE = re.compile(rf"^第([{_CN}]+)章\s*(.+?)\s*[（(]共\s*\d+\s*题[）)]\s*$")
# 新格式章标题：「第一章 数据要素」（章名前必须有空白，且不含句读，避开「第十一章数据工程师保留原有209题…」这类说明段）
CHAPTER_NEW_RE = re.compile(rf"^第([{_CN}]+)章\s+([^\s（(，。；：、]{{2,20}})$")
CHAPTER_COUNT_RE = re.compile(r"^本章共\s*(\d+)\s*题")
SECTION_RE = re.compile(rf"^[{_CN}]+\s*[、]?\s*(选择题|判断题|名词解释|简答题|论述题|案例分析题)\s*[（(]?\s*共\s*(\d+)\s*题[）)]?\s*$")
END_MARK = "书稿待核说明"
NEW_ANSWER_RE = re.compile(r"^参考答案\s{2,}(.*)$", re.S)
NEW_ANALYSIS_RE = re.compile(r"^解析\s{2,}(.*)$", re.S)
NEW_SCORING_RE = re.compile(r"^评分要点[：:]\s*(.*)$", re.S)
SOURCE_RE = re.compile(r"^书稿依据[：:]\s*(.*)$", re.S)
OLD_ANALYSIS_RE = re.compile(r"^【评分要点与解析】(.*)$", re.S)
NEW_WEIGHT_RE = re.compile(r"（(\d{1,2})）\s*(.*?)\s*[（(]\s*(\d{1,3})\s*%\s*[）)]", re.S)
ITEM_RE = re.compile(r"^(\d{1,3})[.．]\s*(.+)$", re.S)
ANSWER_RE = re.compile(r"^【参考答案】(.*?)(?:【评分要点与解析】(.*))?$", re.S)
OPTION_START_RE = re.compile(r"^A[.．]\s*")
WEIGHT_RE = re.compile(r"[（(]\s*占\s*(\d{1,3})\s*%\s*[）)]")
QUOTED_RE = re.compile(r"[“\"]([^”\"]{2,40})[”\"]")
CASE_PART_RE = re.compile(r"【(案例标题|案例材料|分析问题)】")
SUBQ_RE = re.compile(r"[（(](\d)[）)]")
SUBANS_RE = re.compile(r"【问题[（(](\d)[）)]参考答案】")

QTYPES = {"选择题": "choice", "判断题": "judge", "名词解释": "term", "简答题": "short", "论述题": "essay",
          "案例分析题": "case"}
# 老师说的【基础】题：客观题与名词解释；简答 / 论述 / 案例为提高
BASIC_QTYPES = ("choice", "judge", "term")

_CN_NUM = {c: i + 1 for i, c in enumerate(_CN)}


def _cn_int(s: str) -> int:
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十"):
        return 10 + _CN_NUM.get(s[1:], 0)
    return _CN_NUM[s[0]] * 10 + _CN_NUM.get(s[-1], 0) if len(s) > 1 else 0


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


@dataclass
class Question:
    question_id: str
    chapter: int
    chapter_title: str
    qtype: str                  # choice / judge / term / short / essay / case
    qtype_cn: str
    number: int                 # 章内、题型内的题号（与题库一致）
    stem: str
    options: dict[str, str] = field(default_factory=dict)       # 仅选择题
    answer: str = ""            # 选择题为字母，判断题为「正确 / 错误」，其余为参考答案全文
    analysis: str = ""          # 【评分要点与解析】全文
    scoring: list[dict] = field(default_factory=list)           # [{"point": 文本, "weight": 百分比}]
    sub_questions: list[str] = field(default_factory=list)      # 仅案例题：分析问题（1）（2）…
    concepts: list[str] = field(default_factory=list)           # 题干中引号括出的概念 / 名词解释的词条
    flags: list[str] = field(default_factory=list)
    source: str = ""            # 新格式的「书稿依据」出处（只供审阅，不当作答案）

    @property
    def basic(self) -> bool:
        return self.qtype in BASIC_QTYPES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["basic"] = self.basic
        return d


def read_paragraphs(path: str | Path) -> list[str]:
    """body 下的直接段落正文（跳过表格），去掉空段。"""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(_W + "body")
    out: list[str] = []
    for el in body if body is not None else []:
        if el.tag != _W + "p":
            continue
        parts: list[str] = []
        _collect_text(el, parts, [])
        text = "".join(parts).strip()
        if text:
            out.append(text)
    return out


def split_options(text: str) -> dict[str, str]:
    """「A. …B. …C. …D. …」→ {"A": …}。按字母顺序依次找下一个「X.」，前一个字符不能是英文字母或数字。"""
    letters = "ABCDEFGH"
    pos = [0]
    for letter in letters[1:]:
        m = re.compile(rf"(?<![A-Za-z0-9]){letter}[.．]\s*").search(text, pos[-1] + 2)
        if not m:
            break
        pos.append(m.start())
    pos.append(len(text))
    out: dict[str, str] = {}
    for i in range(len(pos) - 1):
        seg = text[pos[i]:pos[i + 1]]
        out[letters[i]] = re.sub(rf"^{letters[i]}[.．]\s*", "", seg).strip()
    return out


def _parse_scoring_new(analysis: str) -> list[dict]:
    """新格式「（1）要点（50%）；（2）要点（50%）。允许同义表述……」→ [{"point", "weight"}]。"""
    return [{"point": m.group(2).strip("；; "), "weight": int(m.group(3))} for m in NEW_WEIGHT_RE.finditer(analysis)]


def parse_scoring(analysis: str) -> list[dict]:
    """「（1）（占40%）……；（2）（占35%）……」或新格式「（1）……（50%）；……」→ [{"point", "weight"}]；没有占比标注的返回空。"""
    marks = list(WEIGHT_RE.finditer(analysis))
    if not marks:
        return _parse_scoring_new(analysis)
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(analysis)
        seg = analysis[m.end():end]
        seg = re.sub(r"[；;]?\s*[（(]\d{1,2}[）)]\s*$", "", seg.strip())  # 去掉下一条的序号
        out.append({"point": seg.strip("；; "), "weight": int(m.group(1))})
    return out


def _concepts(qtype: str, stem: str) -> list[str]:
    if qtype == "term":
        return [stem.strip()]
    body = stem
    if qtype == "case":   # 案例题只看分析问题里点名的理论，案例材料里的引号多是引语
        m = re.search(r"【分析问题】(.*)$", stem, re.S)
        body = m.group(1) if m else ""
    seen: list[str] = []
    for c in QUOTED_RE.findall(body):
        c = c.strip()
        if c and c not in seen:
            seen.append(c)
    return seen


def _sub_questions(stem: str) -> list[str]:
    m = re.search(r"【分析问题】(.*)$", stem, re.S)
    if not m:
        return []
    parts = SUBQ_RE.split(m.group(1))
    # split 结果：[前缀, "1", 文本, "2", 文本, ...]
    return [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2) if parts[i + 1].strip()]


def _qid(chapter: int, qtype: str, number: int, stem: str) -> str:
    h = hashlib.sha1(norm(stem).encode("utf-8")).hexdigest()[:8]
    return f"ex:{config.EXAM_BANK_VERSION}:ch{chapter:02d}:{qtype}:{number:03d}:{h}"


def _header(text: str):
    """章 / 题型标题的识别（新旧两种写法）。返回 ("chapter", 章号, 章名) / ("section", 题型中文, 声明题数) / None。"""
    m = CHAPTER_RE.match(text) or CHAPTER_NEW_RE.match(text)
    if m:
        return "chapter", _cn_int(m.group(1)), m.group(2).strip()
    m = SECTION_RE.match(text)
    if m:
        return "section", m.group(1), int(m.group(2))
    return None


def _is_next_item(text: str, cur: "Question | None") -> re.Match | None:
    """「N. ……」是否开始新的一题：当前没有题、序号正好是下一题、或当前题还没有答案（沿用旧行为）时才算；
    答案跨段时，答案里偶然出现的「1. ……」不会被当成新题。"""
    m = ITEM_RE.match(text)
    if not m:
        return None
    if cur is None or int(m.group(1)) == cur.number + 1 or not (cur.answer or cur.analysis):
        return m
    return None


def parse_bank(path: str | Path) -> tuple[list[Question], dict]:
    """解析教师版题库（新旧格式均可）。返回 (题目列表, 统计)，统计含各章各题型的声明数与实际数，便于核对。"""
    paras = read_paragraphs(path)
    questions: list[Question] = []
    declared: dict[str, int] = {}
    chapter, chapter_title, qtype_cn = 0, "", ""
    cur: Question | None = None
    unparsed: list[str] = []
    part = ""          # 当前题正在接收哪一段：stem / options / answer / analysis

    def close():
        nonlocal cur, part
        if cur is not None:
            if not cur.answer and not cur.analysis:
                cur.flags.append("no_answer")
            if cur.qtype == "choice" and len(cur.options) < 2:
                cur.flags.append("options_missing")
            if cur.qtype == "choice" and cur.answer and cur.answer not in cur.options:
                cur.flags.append("answer_not_in_options")
            questions.append(cur)
        cur, part = None, ""

    def set_answer(q: Question, ans: str) -> None:
        ans = ans.strip()
        if q.qtype == "choice":
            lm = re.match(r"^\s*([A-H])", ans)
            q.answer = lm.group(1) if lm else ans
        elif q.qtype == "judge":
            q.answer = "正确" if ans.startswith(("正确", "对", "√")) else ("错误" if ans else "")
        else:
            q.answer = ans

    def set_analysis(q: Question, text: str) -> None:
        q.analysis = text.strip()
        q.scoring = parse_scoring(q.analysis)

    for i, text in enumerate(paras):
        if text.strip() == END_MARK:
            break                                                   # 书稿待核说明：不是题目
        h = _header(text)
        if h and h[0] == "chapter":
            close()
            chapter, chapter_title, qtype_cn = h[1], h[2], ""
            continue
        if h and h[0] == "section" and chapter:
            close()
            qtype_cn = h[1]
            declared[f"ch{chapter:02d}:{QTYPES[qtype_cn]}"] = h[2]
            continue
        if not chapter or not qtype_cn:
            continue   # 封面、说明、统计表前后的文字，以及「本章共 N 题」
        qtype = QTYPES[qtype_cn]
        # ---- 答案（旧：【参考答案】…【评分要点与解析】…；新：参考答案  X / 解析  … / 评分要点：…）----
        m = ANSWER_RE.match(text)
        if m and cur is not None:
            set_answer(cur, m.group(1))
            set_analysis(cur, m.group(2) or "")
            part = "answer" if not m.group(2) else "analysis"
            continue
        m = NEW_ANSWER_RE.match(text)
        if m and cur is not None:
            set_answer(cur, m.group(1))
            part = "answer"
            continue
        m = (NEW_ANALYSIS_RE.match(text) or NEW_SCORING_RE.match(text) or OLD_ANALYSIS_RE.match(text)) if cur is not None else None
        if m:
            set_analysis(cur, m.group(1))
            part = "analysis"
            continue
        m = SOURCE_RE.match(text) if cur is not None else None
        if m:
            cur.source = m.group(1).strip()
            part = "source"
            continue
        # ---- 选项：每段一个（新），或整段「A. …B. …」（旧）----
        if cur is not None and qtype == "choice" and part in ("", "stem", "options") and re.match(r"^[A-H][.．]\s*", text):
            letter = text[0]
            nxt = paras[i + 1] if i + 1 < len(paras) else ""
            if letter == "A" and not cur.options and not re.match(r"^B[.．]\s*", nxt):
                cur.options = split_options(text)                   # 旧格式：A–D 在同一段
            else:
                cur.options[letter] = re.sub(r"^[A-H][.．]\s*", "", text).strip()
            part = "options"
            continue
        m = _is_next_item(text, cur)
        if m:
            close()
            number, stem = int(m.group(1)), m.group(2).strip()
            cur = Question(
                question_id=_qid(chapter, qtype, number, stem), chapter=chapter, chapter_title=chapter_title,
                qtype=qtype, qtype_cn=qtype_cn, number=number, stem=stem,
                sub_questions=_sub_questions(stem) if qtype == "case" else [],
                concepts=_concepts(qtype, stem),
            )
            part = "stem"
            continue
        if cur is not None:
            if part == "answer":                                    # 答案跨段（案例题的分小问答案）
                cur.answer = f"{cur.answer}\n{text}".strip()
                continue
            if part == "analysis":                                  # 评分要点与解析跨段
                set_analysis(cur, f"{cur.analysis}\n{text}")
                continue
            if part == "source":
                cur.source = f"{cur.source}\n{text}".strip()
                continue
            # 题干跨段（案例材料、【分析问题】及其小问分段）：接回题干；question_id 在题干定稿后重算
            if not cur.answer and not cur.options:
                cur.stem = f"{cur.stem}\n{text}"
                cur.concepts = _concepts(cur.qtype, cur.stem)
                if cur.qtype == "case":
                    cur.sub_questions = _sub_questions(cur.stem)
                continue
        unparsed.append(text[:80])
    close()
    for q in questions:                                             # 题干跨段接回之后再算 id，id 只取决于最终题干
        q.question_id = _qid(q.chapter, q.qtype, q.number, q.stem)

    actual: dict[str, int] = {}
    for q in questions:
        key = f"ch{q.chapter:02d}:{q.qtype}"
        actual[key] = actual.get(key, 0) + 1
    ids = [q.question_id for q in questions]
    stats = {
        "questions": len(questions),
        "declared_total": sum(declared.values()),
        "count_mismatch": {k: {"declared": declared.get(k), "actual": actual.get(k)}
                           for k in sorted(set(declared) | set(actual)) if declared.get(k) != actual.get(k)},
        "duplicate_ids": len(ids) - len(set(ids)),
        "flags": _count_flags(questions),
        "unparsed_paragraphs": unparsed[:20],
        "unparsed_count": len(unparsed),
    }
    return questions, stats


def _count_flags(questions: list[Question]) -> dict[str, int]:
    out: dict[str, int] = {}
    for q in questions:
        for f in q.flags:
            out[f] = out.get(f, 0) + 1
    return out


def compare_student(questions: list[Question], student_path: str | Path) -> dict:
    """学生版与教师版逐题核对：按（章, 题型, 题号）对齐，比较题干。学生版不应出现答案段。"""
    paras = read_paragraphs(student_path)
    stems: dict[tuple[int, str, int], str] = {}
    chapter, qtype_cn, answer_paras = 0, "", 0
    last_no = 0
    for text in paras:
        if text.strip() == END_MARK:
            break
        h = _header(text)
        if h and h[0] == "chapter":
            chapter, qtype_cn, last_no = h[1], "", 0
            continue
        if h and h[0] == "section" and chapter:
            qtype_cn, last_no = h[1], 0
            continue
        if not chapter or not qtype_cn:
            continue
        if text.startswith(("【参考答案】", "参考答案")):
            answer_paras += 1
            continue
        m = ITEM_RE.match(text)
        if m and (int(m.group(1)) == last_no + 1 or last_no == 0):
            last_no = int(m.group(1))
            stems[(chapter, QTYPES[qtype_cn], last_no)] = m.group(2).strip()
    mismatched = []
    for q in questions:
        s = stems.get((q.chapter, q.qtype, q.number))
        # 学生版题干可能跨段，只比较开头
        if s is None or not norm(q.stem).startswith(norm(s)[:60]):
            mismatched.append(q.question_id)
    teacher_keys = {(q.chapter, q.qtype, q.number) for q in questions}
    return {
        "student_questions": len(stems),
        "teacher_questions": len(questions),
        "stem_mismatch": len(mismatched),
        "stem_mismatch_ids": mismatched[:20],
        "only_in_student": len(set(stems) - teacher_keys),
        "student_answer_paragraphs": answer_paras,
    }
