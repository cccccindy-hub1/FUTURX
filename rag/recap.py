"""长会话的前情摘要：把送出模型窗口的旧轮次压成两三句，跟画像一起注入。

为什么需要：会话历史只送最近 CHAT_HISTORY_MAX_TURNS 轮（默认 6）。一个 19 轮的会话，
前 13 轮模型完全看不见——学生问「前面那个抽样方案」，它只能答「你说的是哪个」。
画像补不了这个缺口：画像只记「问过哪些概念」，不记「聊到哪儿了、定了什么」。

做法（增量，不重复摘要）：
- 每次问答结束后检查：会话轮数 - 窗口轮数 = 应该被摘要的旧轮次；
- 已经摘到第 recap_upto 轮，就只把「新的那几轮」补进去，旧摘要作为上下文一起给模型，
  让它合并成一段新摘要——不是每次重摘全量，轮数越多越省；
- 摘要存 conversations.recap，跟画像一起拼进 system 末尾。

只保留能承接对话的东西（模块里的 SUMMARY_SYSTEM 写死了）：在做什么题、卡在哪一步、
已经定下的决定与约束、明确的偏好。不保留过程、不摘原话、不做水平推断。

失败静默：模型未配置、超时、写库失败都只记 warning，本轮回答照常返回，摘要留到下次再补。
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# 摘要只记「承接对话需要的」事实，不记过程也不评价学生——与画像同一条原则：
# 没有依据的判断写进提示词，模型会顺着编。
SUMMARY_SYSTEM = """你在整理一段师生对话的「前情提要」，供后面的回答接上下文用。

只保留这几类信息，其余一律不写：
1. 学生正在做什么（学位层次、专业、论文题目或研究主题）；
2. 已经聊到哪一步、卡在哪里；
3. 已经定下来的决定与结论（选了什么方法、定了什么口径、排除了什么方案）；
4. 学生明确说过的限制与偏好（时间、数据条件、导师要求、写作偏好）。

不要写：客套、追问过程、你的评价（「基础较弱」这类）、原话复述、资料引用编号。
用 2–3 句中文，第三人称陈述（「学生…」），不超过 120 字。只输出摘要本身，
不要前缀、不要标题、不要换行分段。"""

SUMMARY_USER = """已有前情提要（可能为空）：
{recap}

需要合并进来的新对话：
{transcript}

输出合并后的前情提要约（2–3 句，≤120 字）。"""

SUMMARY_MAX_CHARS = 400        # 存库上限（比 120 字宽，留出多轮叠加的余地）
SUMMARY_TIMEOUT = 20.0         # 秒；超时就跳过，下一轮再补
SUMMARY_MAX_TURNS = 12         # 一次最多合并多少轮新对话（防止一次补太多拖慢）
SUMMARY_MAX_CHARS_IN = 6000    # 一次最多读多少字旧对话


def _transcript(pairs: list[tuple[str, str]]) -> str:
    """把旧轮次拼成「学生：… / 老师：…」的文本；每条截断，避免长回答撑爆摘要输入。"""
    lines = []
    for role, text in pairs:
        who = "学生" if role == "user" else "老师"
        body = " ".join((text or "").split())[:400]
        if body:
            lines.append(f"{who}：{body}")
    return "\n".join(lines)


def maybe_update(conversation_id: str | None, *, keep_turns: int, llm=None, model: str = "",
                 timeout: float = SUMMARY_TIMEOUT) -> str:
    """会话超出窗口时更新前情摘要，返回最新的摘要（不需要更新时返回已有摘要）。

    keep_turns：模型能看到的最新轮数（与 CHAT_HISTORY_MAX_TURNS 一致）。
    llm：能 chat(step, messages, model=...) 返回 .content 的客户端；为 None 时只读不写。
    """
    if not conversation_id:
        return ""
    old = ""
    try:
        from . import conversation as cstore

        conv = cstore.get(conversation_id)
        if conv is None:
            return ""
        old = conv.get("recap") or ""                            # 失败时回退到它，别把已有摘要丢掉
        total = int(conv.get("turn_count") or 0)
        if total <= keep_turns:
            return old                                           # 还没超出窗口，什么都不做
        already = cstore.recap_upto(conversation_id)
        # 本次应当覆盖到「总轮数 - 窗口轮数」；已经够新就不用再摘
        target = total - keep_turns
        if already >= target:
            return old
        older = cstore.old_turns(conversation_id, keep_turns=keep_turns, after=already,
                                 max_turns=SUMMARY_MAX_TURNS, max_chars=SUMMARY_MAX_CHARS_IN)
        if not older:
            return old
        if llm is None or not model:
            return old                                           # 没配模型：保留旧摘要，不阻塞回答
        transcript = _transcript(older)
        if not transcript:
            return old
        msgs = [{"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_USER.format(
                    recap=(old or "（无）"), transcript=transcript)}]
        try:
            res = llm.chat("recap", msgs, model=model)
        except TypeError:                                       # 兼容不接受 step 的客户端
            res = llm.chat(msgs, model=model) if hasattr(llm, "chat") else None
        recap = " ".join(str(getattr(res, "content", "") or "").split())[:SUMMARY_MAX_CHARS]
        if not recap:
            return old
        # 推进量按**本次实际覆盖的轮次**算（预算截断时少于 total-keep），否则下一轮会重复摘同一段
        covered = already + cstore.count_turns(older)
        cstore.update_recap(conversation_id, recap, covered)
        return recap
    except Exception as e:  # noqa: BLE001 - 摘要只是加分项
        log.warning("前情摘要更新失败（%s），本轮跳过", type(e).__name__)
        return old


def injection(recap: str) -> str:
    """拼进 system 的那一段；空摘要返回空串（提示词逐字不变）。"""
    text = " ".join((recap or "").split()).rstrip("。；;.")
    if not text:
        return ""
    return (f"【前情】这段对话之前聊过（早于下面保留的最近几轮）：{text}"
            "。回答时可以接着这些内容说，但不要复述、不要当成学生这一轮刚说的话。")


__all__ = ["maybe_update", "injection", "SUMMARY_SYSTEM", "SUMMARY_MAX_CHARS", "SUMMARY_TIMEOUT"]
