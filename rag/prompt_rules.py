"""回答格式规范与 few-shot 样例（集中管理点）。

所有生成路径（course / policy / combined / thesis）的 ANSWER_SYSTEM 拼接这里的常量，
格式条款只写一份，避免各库措辞漂移（此前 course 400 字、policy 500 字、combined 500 字各自为政）。

样例只借鉴句式骨架，不含教材或题库原文；题库预匹配（exam）返回预写答案，不走本规则。
"""
from __future__ import annotations

# ---- 结构规则：总—分—总 + 400 字上限（列举类有界例外 600 字）----
ANSWER_STRUCTURE = """- 回答一律用“总—分—总”结构，要精炼，全文不超过 400 字：
  · 总：开头用一两句话直接回答问题，不写引言；
  · 分：分点 2–4 条；若内容要点超过 4 条，必须按维度归并为最多 4 条，条内可用顿号并列列举，
    不得一项一条展开，每个要点不超过三句，说明其含义或依据，不复述材料原文；
  · 总：用一句话收束要点，不重复前文原句；
- 问列举、构成、环节、类型、方法、关系、区别、原因、作用时，“分”的部分必须分点展开，
  要列全要点，不能只给结论或只罗列名称；
- 例外：简单定义、是非判断、两个概念的简要对比这类问题，可以用压缩格式——总起直接作答，
  接一两句展开，再一句收束，不必强行分点；
- 字数放宽：仅当问题明确要求完整列举较多条目（如多种方法、类型、步骤）时，可在分点内用顿号
  压缩列举，全文至多 600 字；其余情况仍不超过 400 字；
- 样例仅示范结构与口吻，须针对当前问题重新组织语言，不得复用样例原句。"""

# ---- course / policy / combined 共用样例 ----
COURSE_SAMPLES = """【样例一·概念解释型】
问题：什么是数据确权？
回答：
数据确权是指明确数据归谁持有、谁能使用、如何流转的一整套制度安排，它是数据进入市场流通的前提。
- **确什么权**：不是笼统的“数据归谁”，而是拆成持有、加工使用、产品经营等具体权利分别界定；
- **怎么确**：按“数据二十条”的思路，淡化所有权、强化使用权，通过分级分类和登记制度落地；
- **为什么难**：数据可复制、可共享，同一份数据上往往叠着多方投入，权属边界天然模糊。
可以说，确权解决的是“数据能不能动、谁说了算”的问题——它定了，交易和入表才有依据。

【样例二·流程列举型】
问题：企业把数据变成资产，大致要经过哪几步？
回答：
从原始数据到报表上的数据资产，一般要走“资源化—产品化—资产化”三步。
- **资源化**：汇聚、清洗、治理内外部数据，让杂乱数据成为可用的数据资源；
- **产品化**：围绕具体场景加工开发，形成可复用、可交付的数据产品或服务；
- **资产化**：满足确认条件后计入无形资产或存货，并通过评估、交易实现价值变现。
三步层层递进：前一步的治理与加工越扎实，后一步的确认与估值就越有依据。"""

# ---- thesis 共用样例 ----
THESIS_SAMPLES = """【样例三·方法论建议型】
问题：开题时导师说我选题太大，怎么收窄？
回答：
收窄选题最有效的办法是给研究问题加边界，从对象、场景、问题类型三个方向收紧。
- **限定对象**：把“基层治理数字化”收成“某市街道政务服务中心的工作人员”；
- **限定场景**：只研究一个政策周期或一类具体业务，不做全景式扫描；
- **换问题类型**：把“现状如何”的描述题改成“X 如何影响 Y”的机制题，范围自然可控。
收窄不是牺牲意义，而是让问题在论文体量内能被真正回答——小问题做深，比大问题做浅更站得住。

【样例四·压缩格式示范】
问题：文献综述和文献述评是一回事吗？
回答：
不是一回事：综述是“梳理性”的，按主题或时间把已有研究归类呈现；述评是“批判性”的，要在梳理之上指出共识、分歧和研究缺口。开题报告里导师要的通常是后者——只有指出缺口，你的研究问题才有立足点。"""

# ---- 追问条数/长度约束（followups 管道与检查脚本共用） ----
FOLLOWUP_MIN = 2
FOLLOWUP_MAX = 4
FOLLOWUP_MAX_CHARS = 25

# ---- 多轮对话：把历史插进已有消息列表 ----
def with_history(messages: list[dict], history: list[dict] | None, profile: str = "") -> list[dict]:
    """在最后一条 user 消息之前插入历史轮次（system 保持在最前）。

    各作答路径（课程 / 政策 / 判例 / 组合）都按 [system, user(资料+问题)] 组装，历史统一插在
    这两者之间，模型看到的就是真实的 user / assistant 交替，而不是被压成一段文字。
    history 元素为 {"role": "user"|"assistant", "content": str}；空内容与非法角色直接跳过。

    profile：这个学生的画像（rag.profile.summary 的结果），非空时追加到 system 提示末尾。
    为空时 messages 原样返回（逐字不变），没有画像的学生不受影响。
    """
    if profile:
        messages = _with_profile(messages, profile)
    if not history:
        return messages
    pairs = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if role in ("user", "assistant") and text:
            pairs.append({"role": role, "content": text})
    if not pairs:
        return messages
    return messages[:-1] + pairs + messages[-1:]


def _with_profile(messages: list[dict], profile: str) -> list[dict]:
    """把画像追加到第一条 system 之后；没有 system 时不加（避免把画像塞进 user 里当用户说的话）。"""
    if not messages or messages[0].get("role") != "system":
        return messages
    first = dict(messages[0])
    first["content"] = str(first.get("content") or "") + "\n\n" + profile.strip()
    return [first] + list(messages[1:])


__all__ = ["ANSWER_STRUCTURE", "COURSE_SAMPLES", "THESIS_SAMPLES",
           "FOLLOWUP_MIN", "FOLLOWUP_MAX", "FOLLOWUP_MAX_CHARS", "with_history"]
