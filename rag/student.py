"""学生自述的身份：从一句话里认出「本科生 / 编辑出版专业 / 新闻传播 / 选题阶段」。

为什么单独一个模块：这段话不是问题，是一个前提。线上实测「我是本科生 编辑出版专业（新闻传播大类）」
被路由成 off_topic，模型把它当成「介绍一下这个专业」去答，讲了一段考研方向；紧接着问「什么样的选题
算好选题」，回答里全是公共管理的例子——因为它不知道说话的人是谁，只能拿库里的 MPA 资料硬套。

所以这里只做两件事：
1. 认出来（parse）：只认明确的说法（「我是…本科生」「…专业」「论文在选题阶段」），不猜；
2. 判断这句话是不是「只在自我介绍」（only_intro）：是的话回一句确认 + 问需要什么，不调模型。

认不出就返回空，调用方什么都不做——宁可不记，也不要给学生安一个他没说过的专业。
"""
from __future__ import annotations

import re

# 学历：长的写在前面，避免「硕士研究生」被「硕士」抢先匹配后留下残字
_LEVELS = (("博士研究生", "博士"), ("博士", "博士"), ("硕士研究生", "硕士"), ("硕士", "硕士"),
           ("研究生", "硕士"), ("本科生", "本科"), ("本科", "本科"),
           ("大四", "本科"), ("大三", "本科"), ("大二", "本科"), ("大一", "本科"),
           ("专科", "专科"), ("高职", "专科"))
_SELF = ("我是", "本人是", "我读", "我在读", "我目前", "我现在", "我这边", "学生是", "我们学校")
_STAGES = ("选题", "开题", "写作", "撰写", "初稿", "修改", "中期", "定稿", "答辩")
# 常见学科门类：只用来兜底（「我是新闻传播的」这种没写「专业」二字的说法）
_FIELDS = ("新闻传播", "管理学", "法学", "经济学", "文学", "历史学", "哲学", "教育学", "社会学",
           "政治学", "公共管理", "图书情报", "计算机", "统计学", "艺术学", "医学", "理学", "工学", "农学")
# 「专业」前面这些词不是专业名
_BAD_MAJOR = ("这个", "那个", "什么", "哪些", "哪个", "我们", "你们", "他们", "本", "该", "跨", "转")
_MAJOR_RE = re.compile(r"([\u4e00-\u9fa5A-Za-z]{2,12}?)(?:专业|系)")
_PAREN_RE = re.compile(r"[（(]([^（）()]{2,16}?)[)）]")
_STAGE_ALT = "|".join(_STAGES)
# 论文阶段：必须带「论文 / 开题 / 初稿」这类上下文，或明确的「在…阶段」，避免「选题」一词被当成阶段。
# 三支分别捕获：论文在X / 正在X / 卡在X（「卡在文献综述」里的「综述」不是阶段，所以不列）。
_STAGE_RE = re.compile(
    rf"(?:论文|毕业设计|大作业|课题|初稿)[^，。；]{{0,8}}?({_STAGE_ALT})"
    rf"|(?:正在|还在|刚刚|卡在|处于|到了)({_STAGE_ALT})"
    rf"|({_STAGE_ALT})(?:阶段|环节)")
_QUESTION_MARK = re.compile(r"[？?]")


def parse(text: str) -> dict:
    """从一句话里认出身份字段；认不出就是空字符串。返回 {"level","major","field","stage"}。"""
    s = " ".join((text or "").split())
    out = {"level": "", "major": "", "field": "", "stage": ""}
    if not s:
        return out
    for word, mapped in _LEVELS:
        if word in s:
            out["level"] = mapped
            break
    m = _MAJOR_RE.search(s)
    if m:
        name = m.group(1)
        # 「我是编辑出版专业」里的「我是」不是专业名的一部分
        for p in _SELF:
            if name.startswith(p):
                name = name[len(p):]
        if name and name not in _BAD_MAJOR:
            out["major"] = name.strip("的 ，。")
    p = _PAREN_RE.search(s)
    if p:
        inner = p.group(1).strip("的大类门类学科 ")
        out["field"] = inner[:12]
    if not out["field"]:
        for f in _FIELDS:
            if f in s and f not in out["major"]:
                out["field"] = f
                break
    if out["field"] and out["field"] in out["major"]:
        out["field"] = ""
    st = _STAGE_RE.search(s)
    if st:
        out["stage"] = st.group(1) or st.group(2) or st.group(3)
    else:
        # 「论文在选题阶段」被 _STAGE_RE 的第一支匹配到时 group(1) 为空，逐组取第一个非空
        out["stage"] = next((g for g in (st.groups() if st else ()) if g), "")
    return out


def only_intro(text: str) -> bool:
    """这句话是不是「只在自我介绍」：没有追问、长度短、且认出了学历 / 专业 / 阶段。

    「我是本科生，编辑出版专业，想问选题怎么定」不算——后半句有真问题，按真问题走。
    """
    s = (text or "").strip()
    if not s or len(s) > 60 or _QUESTION_MARK.search(s):
        return False
    if any(k in s for k in ("吗", "呢", "怎么", "如何", "什么", "哪些", "为什么", "能不能", "可不可以", "想问", "请问")):
        return False
    got = parse(s)
    return bool(got["level"] or got["major"] or got["stage"])


def confirm_text(ident: dict) -> str:
    """自述之后的短回复：确认记住了什么，再把话头递回去。不调模型。"""
    parts = []
    level, major, field, stage = (ident.get(k, "") for k in ("level", "major", "field", "stage"))
    if level:
        parts.append(f"{level}生" if level in ("本科", "硕士", "博士") else level)
    if major:
        parts.append(f"{major}专业" + (f"（{field}）" if field and field not in major else ""))
    if stage:
        parts.append(f"论文在{stage}阶段")
    head = "记住了：" + "，".join(parts) + "。" if parts else "记住了。"
    return head + "接下来想先解决哪一件？比如选题怎么定、开题报告怎么写、文献综述怎么梳理。"


__all__ = ["parse", "only_intro", "confirm_text"]
