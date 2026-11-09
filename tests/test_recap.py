"""前情摘要：会话超出送模型窗口后，把更旧的轮次压成两三句，跟画像一起注入。

为什么值得单独测：这一层是「长记忆」和「短记忆」之间唯一的东西。不测就很容易出现
重复摘要（每轮多一次调用）、摘要覆盖不全（旧内容丢掉）、或摘要串到别人会话上。
"""
from __future__ import annotations

import json

import pytest

from rag import conversation as cstore
from rag import recap
from rag.course.llm import LLMResult


class _RecapLLM:
    """记录每次摘要调用的输入，并按顺序返回预设摘要。"""

    def __init__(self, replies=None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])
        self.i = 0

    def chat(self, step, messages, *, model):
        self.calls.append({"step": step, "model": model, "system": messages[0]["content"],
                           "user": messages[1]["content"]})
        if self.replies:
            text = self.replies[self.i % len(self.replies)]
            self.i += 1
        else:
            text = "学生在做论文，卡在方法。"
        return LLMResult(content=text)


@pytest.fixture()
def conv(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_DB", str(tmp_path / "conv.sqlite3"))
    cstore.reset_connections()
    cid = cstore.create(key_id="k1", user_id="u1")["id"]
    return cid


def _fill(cid: str, n: int) -> None:
    for i in range(1, n + 1):
        cstore.append_turn(conversation_id=cid, question=f"第{i}轮问题", answer=f"第{i}轮回答",
                           key_id="k1", user_id="u1")


# ---------------------------------------------------------------- 边界

def test_short_conversation_gets_no_recap(conv):
    """没超出窗口就不摘要：不给短会话平白多一次模型调用。"""
    _fill(conv, 4)
    llm = _RecapLLM()
    assert recap.maybe_update(conv, keep_turns=6, llm=llm, model="m") == ""
    assert llm.calls == []
    assert cstore.recap_of(conv) == ""


def test_exactly_at_the_window_is_still_short(conv):
    _fill(conv, 6)
    llm = _RecapLLM()
    assert recap.maybe_update(conv, keep_turns=6, llm=llm, model="m") == ""
    assert llm.calls == []


def test_no_conversation_is_a_noop():
    llm = _RecapLLM()
    assert recap.maybe_update(None, keep_turns=6, llm=llm, model="m") == ""
    assert recap.maybe_update("conv_does_not_exist", keep_turns=6, llm=llm, model="m") == ""
    assert llm.calls == []


# ---------------------------------------------------------------- 切分

def test_only_the_turns_outside_the_window_are_summarised(conv):
    """第 7 轮以前才进摘要；最近 6 轮原样留给模型，不重复灌进摘要。"""
    _fill(conv, 10)
    older = cstore.old_turns(conv, keep_turns=6)
    texts = [t for _, t in older]
    assert len(older) == 8                                  # 4 轮 × 2 条消息
    assert "第1轮问题" in texts[0] and "第4轮回答" in texts[-1]
    assert not any("第5轮" in t or "第10轮" in t for t in texts)   # 窗口内的一律不进


def test_no_old_turns_when_within_window(conv):
    _fill(conv, 5)
    assert cstore.old_turns(conv, keep_turns=6) == []


def test_unanswered_question_still_counts_as_a_turn(conv):
    """上一轮失败没回答时，落单的 user 消息也算一轮，不能被吞掉。"""
    _fill(conv, 3)
    cstore.append_turn(conversation_id=conv, question="落单的问题", answer="", key_id="k1", user_id="u1")
    _fill_more = ["补充"]
    for i in range(4):
        cstore.append_turn(conversation_id=conv, question=f"补充{i}", answer=f"回答{i}",
                           key_id="k1", user_id="u1")
    older = cstore.old_turns(conv, keep_turns=2)
    assert any("落单的问题" in t for _, t in older) or any("第3轮" in t for _, t in older)


# ---------------------------------------------------------------- 增量

def test_recap_is_built_once_and_reused(conv):
    """同一批旧轮次不会重复摘要——每轮重摘全量会白白多花一次调用。"""
    _fill(conv, 10)
    llm = _RecapLLM(["第一版摘要。"])
    first = recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    assert first == "第一版摘要。"
    assert len(llm.calls) == 1
    assert cstore.recap_upto(conv) == 4                     # 10 轮 - 6 轮窗口

    again = recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    assert again == "第一版摘要。"
    assert len(llm.calls) == 1                              # 没有新轮次，不再调用


def test_new_turns_extend_the_recap_with_the_old_one_as_context(conv):
    """新增轮次时，旧摘要作为上下文一起给模型合并，而不是从头再摘一遍。"""
    _fill(conv, 10)
    llm = _RecapLLM(["第一版摘要。", "第二版摘要。"])
    recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    for i in range(11, 13):                                 # 再追加两轮 → 共 12 轮
        cstore.append_turn(conversation_id=conv, question=f"第{i}轮问题", answer=f"第{i}轮回答",
                           key_id="k1", user_id="u1")
    got = recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    assert got == "第二版摘要。"
    assert len(llm.calls) == 2
    assert "第一版摘要。" in llm.calls[1]["user"]           # 旧摘要进了输入
    assert cstore.recap_upto(conv) == 6                     # 12 - 6


def test_recap_system_prompt_forbids_judgements_and_verbatim_quotes(conv):
    """摘要只记能承接对话的事实，不能带评价——与画像同一条原则，否则模型会顺着编。"""
    _fill(conv, 10)
    llm = _RecapLLM()
    recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    system = llm.calls[0]["system"]
    assert "不要写" in system and "基础较弱" in system       # 明确禁止评价学生
    assert "已经定下来的决定" in system and "卡在哪里" in system
    assert llm.calls[0]["step"] == "recap"


# ---------------------------------------------------------------- 失败

def test_missing_llm_keeps_the_old_recap_and_does_not_fail(conv):
    """模型未配置时保留旧摘要，不抛异常、不阻塞回答。"""
    _fill(conv, 10)
    cstore.update_recap(conv, "旧摘要", 2)
    assert recap.maybe_update(conv, keep_turns=6, llm=None, model="m") == "旧摘要"
    assert cstore.recap_of(conv) == "旧摘要"


def test_llm_failure_keeps_the_old_recap(conv):
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("upstream down")

    _fill(conv, 10)
    cstore.update_recap(conv, "旧摘要", 2)
    assert recap.maybe_update(conv, keep_turns=6, llm=Boom(), model="m") == "旧摘要"


def test_empty_model_reply_does_not_overwrite_a_good_recap(conv):
    _fill(conv, 10)
    cstore.update_recap(conv, "旧摘要", 2)
    llm = _RecapLLM([""])
    assert recap.maybe_update(conv, keep_turns=6, llm=llm, model="m") == "旧摘要"
    assert cstore.recap_of(conv) == "旧摘要"


# ---------------------------------------------------------------- 注入

def test_injection_is_empty_for_empty_recap():
    """没有摘要时返回空串：短会话的提示词必须逐字不变。"""
    assert recap.injection("") == ""
    assert recap.injection("   ") == ""


def test_injection_has_one_sentence_end():
    text = recap.injection("学生在做论文，卡在抽样。")
    assert "抽样。。" not in text
    assert text.startswith("【前情】") and "不要当成学生这一轮刚说的话" in text


def test_recap_and_profile_are_both_in_the_system_prompt(conv, tmp_path, monkeypatch):
    """前情与画像一起拼进 system 末尾；两者都在时用空行分开，不混成一句话。"""
    from rag import profile as learner
    from rag import prompt_rules
    from rag.unified import _recap_text, _history_messages

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    _fill(conv, 10)
    learner.notice_identity("u1", level="硕士", major="公共管理")

    recap_text = _recap_text(conv, _RecapLLM(["学生在做满意度研究，已定问卷法。"]), "m")
    assert recap_text.startswith("【前情】")
    prof = learner.summary("u1")
    merged = f"{prof}\n\n{recap_text}".strip()
    msgs = prompt_rules.with_history(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
        _history_messages(conv), profile=merged)
    system = msgs[0]["content"]
    assert "【这个学生的情况】" in system and "【前情】" in system
    assert system.index("【这个学生的情况】") < system.index("【前情】")
    assert len([m for m in msgs if m["role"] == "user"]) == 7      # 6 轮历史 + 本轮


def test_recap_is_not_injected_into_short_conversations(conv):
    from rag.unified import _recap_text
    _fill(conv, 3)
    llm = _RecapLLM()
    assert _recap_text(conv, llm, "m") == ""
    assert llm.calls == []


def test_recap_never_leaks_across_conversations(conv):
    """摘要按会话存，别人的会话读不到。"""
    _fill(conv, 10)
    recap.maybe_update(conv, keep_turns=6, llm=_RecapLLM(["A 的摘要。"]), model="m")
    other = cstore.create(key_id="k1", user_id="u2")["id"]
    assert cstore.recap_of(other) == ""
    assert cstore.recap_of(conv) == "A 的摘要。"


def test_recap_upto_advances_by_the_turns_actually_covered(conv):
    """推进量要按本次实际覆盖的轮次算，不能用「总数 - 窗口」推算。

    线上实测：8 轮会话第一次摘要时，预算截断只覆盖了 1 轮，但按推算写了 2，
    下一轮就会把第 2 轮重复摘一遍。
    """
    _fill(conv, 8)
    llm = _RecapLLM(["第一版。"])
    recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    first = cstore.recap_upto(conv)
    covered = cstore.count_turns(cstore.old_turns(conv, keep_turns=6))
    assert first == covered == 2

    cstore.append_turn(conversation_id=conv, question="第9轮问题", answer="第9轮回答",
                       key_id="k1", user_id="u1")
    recap.maybe_update(conv, keep_turns=6, llm=llm, model="m")
    assert cstore.recap_upto(conv) == 3                     # 只补第 3 轮，不是重算
    assert len(llm.calls) == 2
    assert "第1轮问题" not in llm.calls[1]["user"]          # 已摘过的第 1 轮不再进输入


def test_already_summarised_turns_are_skipped(conv):
    _fill(conv, 10)
    cstore.update_recap(conv, "旧摘要", 3)
    older = cstore.old_turns(conv, keep_turns=6, after=3)
    texts = [t for _, t in older]
    assert not any("第1轮问题" in t or "第3轮问题" in t for t in texts)
    assert any("第4轮问题" in t for t in texts)
