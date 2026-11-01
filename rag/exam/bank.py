"""题库 docx → 题目列表：一题一条记录（题干 + 选项 + 参考答案 + 评分要点），不跨题、不拆题。

题库结构（教师版，2015 题，10.31 修订版）：
- 「第X章 章名（共 N 题）」→「一、选择题（共 N 题）」…「六、案例分析题」；
- 每题 = 题干段「N. ……」+（选择题）选项段「A. …B. …C. …D. …」+ 答案段「【参考答案】…【评分要点与解析】…」；
- 案例分析题题干内含【案例标题】【案例材料】【分析问题】，答案按【问题（n）参考答案】分小问。

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
SECTION_RE = re.compile(rf"^[{_CN}]+、\s*(选择题|判断题|名词解释|简答题|论述题|案例分析题)\s*[（(]共\s*(\d+)\s*题[）)]\s*$")
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


def parse_scoring(analysis: str) -> list[dict]:
    """「（1）（占40%）……；（2）（占35%）……」→ [{"point", "weight"}]；没有占比标注的返回空。"""
    marks = list(WEIGHT_RE.finditer(analysis))
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


def parse_bank(path: str | Path) -> tuple[list[Question], dict]:
    """解析教师版题库。返回 (题目列表, 统计)，统计含各章各题型的声明数与实际数，便于核对。"""
    paras = read_paragraphs(path)
    questions: list[Question] = []
    declared: dict[str, int] = {}
    chapter, chapter_title, qtype_cn = 0, "", ""
    cur: Question | None = None
    unparsed: list[str] = []

    def close():
        nonlocal cur
        if cur is not None:
            if not cur.answer and not cur.analysis:
                cur.flags.append("no_answer")
            if cur.qtype == "choice" and len(cur.options) < 2:
                cur.flags.append("options_missing")
            if cur.qtype == "choice" and cur.answer and cur.answer not in cur.options:
                cur.flags.append("answer_not_in_options")
            questions.append(cur)
        cur = None

    for text in paras:
        m = CHAPTER_RE.match(text)
        if m:
            close()
            chapter, chapter_title, qtype_cn = _cn_int(m.group(1)), m.group(2).strip(), ""
            continue
        m = SECTION_RE.match(text)
        if m and chapter:
            close()
            qtype_cn = m.group(1)
            declared[f"ch{chapter:02d}:{QTYPES[qtype_cn]}"] = int(m.group(2))
            continue
        if not chapter or not qtype_cn:
            continue   # 封面、说明、统计表前后的文字
        qtype = QTYPES[qtype_cn]
        m = ANSWER_RE.match(text)
        if m and cur is not None:
            ans, analysis = m.group(1).strip(), (m.group(2) or "").strip()
            if qtype == "choice":
                lm = re.match(r"^\s*([A-H])", ans)
                cur.answer = lm.group(1) if lm else ans
            elif qtype == "judge":
                cur.answer = "正确" if ans.startswith(("正确", "对", "√")) else ("错误" if ans else "")
            else:
                cur.answer = ans
            cur.analysis = analysis
            cur.scoring = parse_scoring(analysis)
            continue
        if cur is not None and qtype == "choice" and not cur.options and OPTION_START_RE.match(text):
            cur.options = split_options(text)
            continue
        m = ITEM_RE.match(text)
        if m:
            close()
            number, stem = int(m.group(1)), m.group(2).strip()
            cur = Question(
                question_id=_qid(chapter, qtype, number, stem), chapter=chapter, chapter_title=chapter_title,
                qtype=qtype, qtype_cn=qtype_cn, number=number, stem=stem,
                sub_questions=_sub_questions(stem) if qtype == "case" else [],
                concepts=_concepts(qtype, stem),
            )
            continue
        if cur is not None:
            # 题干跨段（个别案例材料分段）：接回题干
            if not cur.answer and not cur.options:
                cur.stem = f"{cur.stem}\n{text}"
                cur.concepts = _concepts(cur.qtype, cur.stem)
                if cur.qtype == "case":
                    cur.sub_questions = _sub_questions(cur.stem)
                continue
        unparsed.append(text[:80])
    close()

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
    for text in paras:
        m = CHAPTER_RE.match(text)
        if m:
            chapter, qtype_cn = _cn_int(m.group(1)), ""
            continue
        m = SECTION_RE.match(text)
        if m and chapter:
            qtype_cn = m.group(1)
            continue
        if not chapter or not qtype_cn:
            continue
        if text.startswith("【参考答案】"):
            answer_paras += 1
            continue
        m = ITEM_RE.match(text)
        if m:
            stems[(chapter, QTYPES[qtype_cn], int(m.group(1)))] = m.group(2).strip()
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
