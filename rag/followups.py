"""追问（followups）管道：在回答末尾附 2-4 条延展性追问。

两种来源，互斥：
- make_followups(question, answer, route, llm)：一次 json_mode 小调用，模型用 filter 同档（便宜、快）；
- exam_followups(item)：纯程序模板，基于命中题的知识点轮换三类角度，零 LLM 调用——
  exam 预匹配的核心卖点是 0 调用、毫秒级返回，不能被追问毁掉。

约束（规格）：2–4 条、每条 ≤25 字、必须是疑问句、不与已答内容重复。
off_topic / quote 路径不生成（由 unified 侧跳过）。
"""
from __future__ import annotations

import re

from . import prompt_rules as rules

FOLLOWUP_SYSTEM = """你是《数据要素》课程的助教，现在要为学生推荐下一步可以追问的问题。

要求：
- 给出 2 到 4 条追问，每条不超过 25 个字；
- 每条都必须是疑问句，以问号结尾；
- 角度要多样：相邻概念、实际应用、对比辨析、考试怎么考，不要四条都问同一个方向；
- 不要重复回答里已经讲过的内容，也不要问回答里已给出答案的问题；
- 只输出 JSON 对象，不要解释、不要 Markdown 代码围栏。

示例：
学生问题：什么是数据确权？
回答：数据确权是明确数据归谁持有、谁能使用、如何流转的制度安排……（略）
输出：{"followups": ["数据确权与数据所有权有什么区别？", "确权登记具体怎么办？", "企业怎么证明自己有权使用数据？"]}"""

# 论文 / 写作类问题的追问：角度换成学术写作，不要出现「考试怎么考」——线上实测在「好选题的标准」
# 之后追问「考试会怎么考选题设计」，与学生的实际问题完全不在一个语境。
FOLLOWUP_SYSTEM_THESIS = """你是学位论文写作的助教，现在要为学生推荐下一步可以追问的问题。

要求：
- 给出 2 到 4 条追问，每条不超过 25 个字；
- 每条都必须是疑问句，以问号结尾；
- 角度要多样：具体怎么操作、有什么判断标准、常见坑、和相邻环节的关系；不要问“考试怎么考”；
- 不要重复回答里已经讲过的内容，也不要问回答里已给出答案的问题；
- 只输出 JSON 对象，不要解释、不要 Markdown 代码围栏。

示例：
学生问题：什么样的选题算是好的选题？
回答：好选题要同时满足问题真实、边界清晰、你能做……（略）
输出：{"followups": ["怎么判断选题是不是太大？", "选题依据部分该写多少字？", "换方向还来得及吗？"]}"""

# 按路由选择追问提示词：论文单独一套，其余仍用课程那一套
# （off_topic / quote 在 unified 侧就被跳过了，不会走到这里）
THESIS_ROUTES = ("thesis",)


def system_for(route: str) -> str:
    return FOLLOWUP_SYSTEM_THESIS if route in THESIS_ROUTES else FOLLOWUP_SYSTEM

FOLLOWUP_USER = """学生问题：{question}

已有回答：
{answer}

请按上述要求给出追问的 JSON。"""

# 角度轮换模板（exam 路径，零调用）
_EXAM_ANGLES = (
    ("知识点延伸", "这道题还涉及哪些知识点？"),
    ("相邻考点对比", "这个考点容易和什么混淆？"),
    ("实务应用", "这个知识点在实际中怎么用？"),
)


def _clean(items, question: str, answer: str) -> list[str]:
    """去重、截断、去掉与问题重复或非疑问句的条目，最终夹在 2–4 条。"""
    q = (question or "").strip()
    a = (answer or "").strip()
    out: list[str] = []
    for raw in items or []:
        s = str(raw or "").strip().strip('"').strip("“”")
        if not s:
            continue
        if len(s) > rules.FOLLOWUP_MAX_CHARS:
            s = s[: rules.FOLLOWUP_MAX_CHARS - 1].rstrip("，、；： ") + "？"
        if not s.endswith(("？", "?")):
            s = s.rstrip("。！；，、") + "？"
        if s == q or s in a or s in out:
            continue
        out.append(s)
        if len(out) >= rules.FOLLOWUP_MAX:
            break
    return out


def make_followups(question: str, answer: str, route: str, llm) -> list[str]:
    """一次小调用生成追问。失败/解析不出来时返回空列表（调用方静默省略，不影响主答案）。"""
    from .course.config import LIB_PICK_MODEL
    from .course.llm import parse_json_object

    if llm is None or not (answer or "").strip():
        return []
    msgs = [{"role": "system", "content": system_for(route)},
            {"role": "user", "content": FOLLOWUP_USER.format(question=question, answer=(answer or "")[:1200])}]
    try:
        res = llm.chat("followup", msgs, model=LIB_PICK_MODEL, json_mode=True)
        obj = parse_json_object(res.content)
    except Exception:  # noqa: BLE001 - 追问失败不能影响主答案
        return []
    items = obj.get("followups") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    got = _clean(items, question, answer)
    return got if len(got) >= rules.FOLLOWUP_MIN else []


def exam_followups(item) -> list[str]:
    """题库命中题的追问：用题干挂接的知识点拼三类角度，零 LLM 调用。

    item 可以是 exam.ExamItem，也可以是它 view(include_answer=True) 得到的 dict
    （unified 的 exam 事件里就是 dict），两种形态都能取到 concepts / qtype_cn。
    """
    if item is None:
        return []
    if isinstance(item, dict):
        concepts = [c for c in (item.get("concepts") or []) if c]
        qtype_cn = item.get("qtype_cn") or "这道题"
    else:
        q = getattr(item, "q", None)
        concepts = [c for c in (getattr(item, "concepts", None) or []) if c]
        qtype_cn = getattr(q, "qtype_cn", "") or "这道题"
    name = concepts[0] if concepts else ""
    out: list[str] = []
    if name:
        out.append(f"{name}还涉及哪些考点？")
        out.append(f"{name}和企业实际业务怎么结合？")
    out.append(f"这类{qtype_cn}考试时怎么答才满分？")
    if len(out) < rules.FOLLOWUP_MIN:
        out.append("这个知识点容易和什么混淆？")
    seen: list[str] = []
    for s in out:
        s = s if len(s) <= rules.FOLLOWUP_MAX_CHARS else s[: rules.FOLLOWUP_MAX_CHARS - 1] + "？"
        if s not in seen:
            seen.append(s)
    return seen[: rules.FOLLOWUP_MAX]


__all__ = ["make_followups", "exam_followups", "system_for", "FOLLOWUP_SYSTEM", "FOLLOWUP_SYSTEM_THESIS",
           "FOLLOWUP_USER"]
