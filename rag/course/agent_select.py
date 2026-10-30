"""Step 1：LLM 工具调用循环，从本体词表中选概念。

两条路径（COURSE_LLM_NATIVE_TOOLS 切换）：
- 原生：tools=TOOL_SPECS，读 tool_calls；
- 回退：prompt 约定每轮回复一个 {"tool","args"} JSON，本地解析；解析失败带错误重试 1 次，仍失败则以当前选择结束。
最多 max_steps 轮工具调用（一轮 = 一次 LLM 回复，其中可并行多个工具调用）；达到上限后追加一条提示，
再给一次只能 finish 的机会（truncated=True）；这一轮仍不 finish 才以当前选择结束（end_reason=truncated）。
校验：finish 里的 concept_id 必须存在、未被排除、未被 blocked，否则丢弃并记录到 dropped。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LLMClient, LLMError, parse_json_object
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
    final_text: str = ""           # end_reason 为 no_tool_call / parse_failed 时 LLM 回复的前 300 字，便于排查

    def to_dict(self) -> dict:
        return {"selected": [s.to_dict() for s in self.selected], "dropped": self.dropped, "steps": self.steps,
                "truncated": self.truncated, "finished": self.finished, "parse_failures": self.parse_failures,
                "tool_log": self.tool_log, "end_reason": self.end_reason, "final_text": self.final_text}


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


def select_concepts(question: str, active_concepts, index: VocabIndex, llm: LLMClient, *,
                    max_steps: int | None = None, model: str | None = None,
                    native_tools: bool | None = None) -> SelectResult:
    max_steps = config.COURSE_AGENT_MAX_STEPS if max_steps is None else max_steps
    model = model or config.COURSE_SELECT_MODEL
    native = config.COURSE_LLM_NATIVE_TOOLS if native_tools is None else native_tools
    active = list(active_concepts or [])
    active_txt = "、".join(
        f"{c}《{index.concepts[c].name}》" if c in index.concepts else c for c in active) or "（无）"
    system = prompts.SELECT_SYSTEM.format(max_steps=max_steps, active_concepts=active_txt)
    if not native:
        system += prompts.SELECT_JSON_PROTOCOL.format()   # 模板里的 {{ }} 需经 format 还原为单花括号
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": prompts.SELECT_USER.format(question=question)}]
    result = SelectResult()
    loop = _loop_native if native else _loop_json
    loop(messages, index, llm, model, max_steps, result)
    return result


def _loop_native(messages, index, llm, model, max_steps, result: SelectResult) -> None:
    while True:
        res = llm.chat("select", messages, model=model, tools=FINISH_ONLY if result.truncated else prompts.TOOL_SPECS)
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
        for c in calls:
            out = run_tool(index, c.name, c.args)
            result.tool_log.append({"tool": c.name, "args": c.args, "ok": "error" not in out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out, ensure_ascii=False)})
        if result.steps >= max_steps:
            result.truncated = True
            messages.append({"role": "user", "content": prompts.SELECT_FORCE_FINISH.format(max_steps=max_steps)})


def _loop_json(messages, index, llm, model, max_steps, result: SelectResult) -> None:
    retried = False
    while True:
        res = llm.chat("select", messages, model=model)
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
        out = run_tool(index, name, args)
        result.tool_log.append({"tool": name, "args": args, "ok": "error" not in out})
        tail = ""
        if result.steps >= max_steps:
            result.truncated = True
            tail = "\n" + prompts.SELECT_FORCE_FINISH.format(max_steps=max_steps)
        messages.append({"role": "user", "content": "工具结果：" + json.dumps(out, ensure_ascii=False) + tail})
