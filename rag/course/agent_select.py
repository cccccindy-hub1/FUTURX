"""Step 1：LLM 工具调用循环，从本体词表中选概念。

两条路径（COURSE_LLM_NATIVE_TOOLS 切换）：
- 原生：tools=TOOL_SPECS，读 tool_calls；
- 回退：prompt 约定每轮回复一个 {"tool","args"} JSON，本地解析；解析失败带错误重试 1 次，仍失败则以当前选择结束。
max_steps = 选词 LLM 调用次数上限（一次调用 = 一轮，其中可并行多个工具调用），含最后强制 finish 的那一次：
前 max_steps-1 次可以探索，用完后追加提示并只提供 finish 工具（truncated=True）；这一次仍不 finish 才以当前选择结束
（end_reason=truncated）。max_steps<=1 时第一次调用就是只能 finish 的强制提交。
较早轮次的工具返回会被压缩成只剩概念 ID 与名称，避免每轮输入 token 持续增长。
校验：finish 里的 concept_id 必须存在、未被排除、未被 blocked，否则丢弃并记录到 dropped。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient, LLMError, parse_json_object, reasoning_param
from .aliases import norm, split_name
from .vocab_tools import VocabIndex

ROLES = ("question_concept", "related_concept")
TOOL_NAMES = ("search_vocab", "get_concept", "get_neighbors", "finish")
MAX_NEIGHBORS_SHOWN = 60
FINISH_ONLY = [t for t in prompts.TOOL_SPECS if t["function"]["name"] == "finish"]


@dataclass(frozen=True)
class Selection:
    concept_id: str
    reason: str
    role: str

    def to_dict(self) -> dict:
        return {"concept_id": self.concept_id, "reason": self.reason, "role": self.role}


@dataclass
class SelectResult:
    selected: list[Selection] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    steps: int = 0                 # 已执行的工具调用轮数（不含 finish；一轮可含多个并行工具调用）
    truncated: bool = False        # 达到 max_steps，被要求立即 finish
    finished: bool = False         # LLM 主动调用了 finish
    parse_failures: int = 0
    tool_log: list[dict] = field(default_factory=list)
    # 结束原因：finish / truncated / no_tool_call（LLM 只回了文字）/ parse_failed（JSON 路径重试后仍无法解析）
    end_reason: str = ""
    fast_path: str = ""            # 非空表示走了快速路径（值为命中的别名），此时 0 次 LLM 调用
    rounds: list[dict] = field(default_factory=list)   # 每轮 LLM 调用的耗时与 token
    final_text: str = ""           # end_reason 为 no_tool_call / parse_failed 时 LLM 回复的前 300 字，便于排查

    def to_dict(self) -> dict:
        return {"selected": [s.to_dict() for s in self.selected], "dropped": self.dropped, "steps": self.steps,
                "truncated": self.truncated, "finished": self.finished, "parse_failures": self.parse_failures,
                "tool_log": self.tool_log, "fast_path": self.fast_path, "rounds": self.rounds, "end_reason": self.end_reason, "final_text": self.final_text}


def run_tool(index: VocabIndex, name: str, args: dict) -> dict:
    """执行一个本地词表工具，返回可 JSON 序列化的结果；参数错误返回 {"error": ...}（不抛异常，让 LLM 能自我纠正）。"""
    try:
        if name == "search_vocab":
            hits = index.search_vocab(str(args.get("query", "")), args.get("chapter"), args.get("limit", 30))
            return {"results": [h.to_dict() for h in hits]}
        if name == "get_concept":
            card = index.get_concept(str(args.get("concept_id", "")))
            return card.to_dict() if card else {"error": "concept_id 不存在"}
        if name == "get_neighbors":
            cid = str(args.get("concept_id", ""))
            if cid not in index.concepts:
                return {"error": "concept_id 不存在"}
            ns = index.get_neighbors(cid, args.get("view") or "both")
            return {"total": len(ns), "neighbors": [n.to_dict() for n in ns[:MAX_NEIGHBORS_SHOWN]],
                    "truncated": len(ns) > MAX_NEIGHBORS_SHOWN}
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    return {"error": f"未知工具 {name}"}


def _validate(raw, index: VocabIndex, result: SelectResult) -> None:
    if not isinstance(raw, list):
        result.dropped.append({"item": repr(raw)[:80], "reason": "selected_not_a_list"})
        return
    seen = {s.concept_id for s in result.selected}
    for it in raw:
        if not isinstance(it, dict):
            result.dropped.append({"item": repr(it)[:80], "reason": "not_an_object"})
            continue
        cid = str(it.get("concept_id", "")).strip()
        role = it.get("role")
        why = index.reject_reason(cid)
        if why:
            result.dropped.append({"concept_id": cid, "reason": why})
        elif cid in seen:
            result.dropped.append({"concept_id": cid, "reason": "duplicate"})
        elif role not in ROLES:
            result.dropped.append({"concept_id": cid, "reason": "invalid_role", "role": role})
        else:
            seen.add(cid)
            result.selected.append(Selection(cid, str(it.get("reason", "")).strip(), role))


def _finish(args: dict, index: VocabIndex, result: SelectResult) -> None:
    if "selected" not in args:
        result.dropped.append({"reason": "finish_missing_selected", "keys": sorted(args)})
    _validate(args.get("selected", []), index, result)
    result.finished, result.end_reason = True, "finish"


def compress_tool_output(name: str, out: dict) -> dict:
    """较早轮次的工具返回压缩：只保留候选概念的 ID 与名称（错误信息原样保留）。"""
    if "error" in out:
        return out
    if name == "search_vocab":
        return {"results": [{"concept_id": h["concept_id"], "name": h["name"]} for h in out.get("results", [])]}
    if name == "get_concept":
        return {"concept_id": out.get("concept_id"), "name": out.get("name")}
    if name == "get_neighbors":
        seen: dict[str, str] = {}
        for n in out.get("neighbors", []):
            seen.setdefault(n["concept_id"], n["name"])
        return {"total": out.get("total"), "truncated": out.get("truncated"),
                "neighbors": [{"concept_id": c, "name": nm} for c, nm in seen.items()]}
    return out


def _compress_pending(messages: list[dict], pending: list[tuple[int, str, dict]]) -> None:
    for idx, name, out in pending:
        m = messages[idx]
        packed = json.dumps(compress_tool_output(name, out), ensure_ascii=False)
        m["content"] = packed if m["role"] == "tool" else "工具结果（已压缩，仅 ID 与名称）：" + packed
    pending.clear()


def _related_one_hop(cid: str, index: VocabIndex, limit: int) -> list[Selection]:
    """cid 的一跳邻居（本体 onto1/onto2 任一视角、任一方向），按连边数多、同章优先、ID 排序取前 limit 个。"""
    if limit <= 0:
        return []
    base_ch = index.concepts[cid].chapter
    edges: dict[str, int] = {}
    for n in index.get_neighbors(cid, "both"):
        if n.concept_id != cid and index.reject_reason(n.concept_id) is None:
            edges[n.concept_id] = edges.get(n.concept_id, 0) + 1
    ranked = sorted(edges, key=lambda c: (-edges[c], index.concepts[c].chapter != base_ch, c))[:limit]
    name = index.concepts[cid].name
    return [Selection(c, f"与「{name}」在本体中一跳相连（快速路径补入）", "related_concept") for c in ranked]


def _log_round(result: SelectResult, res, llm=None) -> None:
    """记录一次选词调用：耗时、token、思考情况。reasoning_sent 是实际发给服务端的 reasoning 参数
    （None = 没有发送该参数，由服务端按模型默认处理）；reasoning_tokens / reasoning_chars 来自响应。"""
    fn = getattr(llm, "reasoning_for", None)
    mode = fn("select") if fn else None
    try:
        sent = reasoning_param(mode) if mode is not None else None
    except ValueError:
        sent = None
    result.rounds.append({"round": len(result.rounds) + 1, "latency_s": round(res.latency_s, 3),
                          "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                          "reasoning_tokens": res.reasoning_tokens, "reasoning_chars": res.reasoning_chars,
                          "reasoning_mode": mode, "reasoning_sent": sent,
                          "tools": [c.name for c in res.tool_calls]})


_MULTI_CUES = ("和", "与", "及", "以及", "或", "、", "区别", "不同", "异同", "比较", "对比", "关系", "联系", "之间",
               "分别", "各自", "哪些", "为什么", "如何", "怎么", "怎样")
_UNMATCHED_TERM_RE = re.compile(r"数据[一-鿿]{2,}")
FAST_MAX_QUESTION_CHARS = 30


def fast_select(question: str, index: VocabIndex) -> SelectResult | None:
    """保守快速路径：问题里只命中唯一、无歧义的概念别名时直接选中，不调用 LLM。

    必须同时满足（任一不满足返回 None，照常走多轮）：
    1. 去掉被更长命中别名包含的别名后，恰好剩 1 个别名；
    2. 该别名只对应 1 个概念（无同形多概念）；
    3. 该概念可选，且没有同名的跨章概念（_name_groups 中仅它一个）。
    命中后选 1 个 question_concept，再补入至多 COURSE_FAST_RELATED_MAX 个一跳关联概念（related_concept），
    以缓解只选单概念导致的覆盖面下降；关联概念只按本体连边排序，没有 LLM 判断相关性，可能夹带弱相关概念。
    """
    q = question.strip()
    if len(q) > FAST_MAX_QUESTION_CHARS or any(c in q for c in _MULTI_CUES):
        return None                    # 对比 / 并列 / 开放式问题可能涉及多个概念，交给多轮
    hits = index.aliases.find_aliases(question)
    keys = [a.key for a in hits]
    top = [a for a in hits if not any(a.key != k and a.key in k for k in keys)]
    if len(top) != 1 or top[0].ambiguous:
        return None
    cid = top[0].concept_ids[0]
    if index.reject_reason(cid) is not None:
        return None
    rest = norm(q).replace(top[0].key, "")
    if _UNMATCHED_TERM_RE.search(rest):    # 剩余文本里还有没匹配上的「数据××」术语：可能漏掉第二个概念
        return None
    c = index.concepts[cid]
    if len(index._name_groups.get(norm(split_name(c.name)[0]), [])) > 1:
        return None
    r = SelectResult(finished=True, end_reason="fast_path", fast_path=top[0].surface)
    r.selected.append(Selection(cid, f"问题直接命中唯一概念别名「{top[0].surface}」", "question_concept"))
    r.selected.extend(_related_one_hop(cid, index, config.COURSE_FAST_RELATED_MAX))
    return r


def select_concepts(question: str, active_concepts, index: VocabIndex, llm: LLMClient, *,
                    max_steps: int | None = None, model: str | None = None,
                    native_tools: bool | None = None, fast_path: bool | None = None) -> SelectResult:
    max_steps = config.COURSE_AGENT_MAX_STEPS if max_steps is None else max_steps
    model = model or config.COURSE_SELECT_MODEL
    native = config.COURSE_LLM_NATIVE_TOOLS if native_tools is None else native_tools
    if config.COURSE_SELECT_FAST_PATH if fast_path is None else fast_path:
        fast = fast_select(question, index)
        if fast is not None:
            return fast
    active = list(active_concepts or [])
    active_txt = "、".join(
        f"{c}《{index.concepts[c].name}》" if c in index.concepts else c for c in active) or "（无）"
    system = prompts.SELECT_SYSTEM.format(max_steps=max_steps, active_concepts=active_txt)
    if not native:
        system += prompts.SELECT_JSON_PROTOCOL.format()   # 模板里的 {{ }} 需经 format 还原为单花括号
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": prompts.SELECT_USER.format(question=question)}]
    result = SelectResult()
    if max_steps <= 1:                   # 只有一次调用机会：直接要求 finish
        result.truncated = True
        messages[-1]["content"] += "\n" + prompts.SELECT_FORCE_FINISH.format(max_steps=max_steps)
    loop = _loop_native if native else _loop_json
    loop(messages, index, llm, model, max_steps, result)
    return result


def _loop_native(messages, index, llm, model, max_steps, result: SelectResult) -> None:
    pending: list[tuple[int, str, dict]] = []     # 上一轮工具结果在 messages 里的位置，下一轮开始前压缩
    while True:
        res = llm.chat("select", messages, model=model, tools=FINISH_ONLY if result.truncated else prompts.TOOL_SPECS)
        _log_round(result, res, llm)
        calls = res.tool_calls
        if not calls:                    # LLM 没调工具也没 finish：以当前选择结束，记录回复便于排查
            result.end_reason, result.final_text = "no_tool_call", (res.content or "")[:300]
            return
        messages.append({
            "role": "assistant", "content": res.content or None,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                           for c in calls],
        })
        fin = next((c for c in calls if c.name == "finish"), None)
        if fin is not None:
            _finish(fin.args, index, result)
            return
        if result.truncated:             # 已要求立即 finish 仍未提交
            result.end_reason = "truncated"
            return
        result.steps += 1
        _compress_pending(messages, pending)
        for c in calls:
            out = run_tool(index, c.name, c.args)
            result.tool_log.append({"tool": c.name, "args": c.args, "ok": "error" not in out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out, ensure_ascii=False)})
            pending.append((len(messages) - 1, c.name, out))
        if result.steps >= max_steps - 1:      # 探索次数用完，下一次只能 finish
            result.truncated = True
            messages.append({"role": "user", "content": prompts.SELECT_FORCE_FINISH.format(max_steps=max_steps)})


def _loop_json(messages, index, llm, model, max_steps, result: SelectResult) -> None:
    retried = False
    pending: list[tuple[int, str, dict]] = []
    while True:
        res = llm.chat("select", messages, model=model)
        _log_round(result, res, llm)
        try:
            obj = parse_json_object(res.content)
            name, args = obj.get("tool"), obj.get("args") or {}
            if name not in TOOL_NAMES or not isinstance(args, dict):
                raise ValueError(f"tool 必须是 {list(TOOL_NAMES)} 之一，args 必须是对象")
        except ValueError as e:
            result.parse_failures += 1
            if retried:                  # 重试后仍失败：以当前选择结束
                result.end_reason, result.final_text = "parse_failed", (res.content or "")[:300]
                return
            retried = True
            messages.append({"role": "assistant", "content": res.content})
            messages.append({"role": "user", "content": prompts.RETRY_JSON.format(error=str(e)[:80])})
            continue
        retried = False
        messages.append({"role": "assistant", "content": res.content})
        if name == "finish":
            _finish(args, index, result)
            return
        if result.truncated:             # 已要求立即 finish 仍未提交
            result.end_reason = "truncated"
            return
        result.steps += 1
        _compress_pending(messages, pending)
        out = run_tool(index, name, args)
        result.tool_log.append({"tool": name, "args": args, "ok": "error" not in out})
        tail = ""
        if result.steps >= max_steps - 1:      # 探索次数用完，下一次只能 finish
            result.truncated = True
            tail = "\n" + prompts.SELECT_FORCE_FINISH.format(max_steps=max_steps)
        messages.append({"role": "user", "content": "工具结果：" + json.dumps(out, ensure_ascii=False) + tail})
        pending.append((len(messages) - 1, name, out))
