"""论文库：文献综述类问题追加回答逻辑（其他问题提示词逐字不变）；资料与引用编号统一为 [n]。全部离线。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag import query as q


def _svc():
    return q.RAGService.__new__(q.RAGService)          # 不加载模型：prepare_from 只做拼装


def _result(n=2):
    return SimpleNamespace(documents=[f"片段{i}" for i in range(n)],
                           metadatas=[{"source": f"论文{i}.pdf", "section": f"第{i}节"} for i in range(n)],
                           distances=[0.1] * n)


@pytest.mark.parametrize("question", ["文献综述怎么写？", "第三次文献综述是什么", "怎么做文献回顾", "How to write a Literature Review?"])
def test_lit_review_questions_get_the_logic_appended(question):
    _, _, messages = _svc().prepare_from(question, _result())
    system = messages[0]["content"]
    assert system.startswith(q._SYSTEM_PROMPT) and system.endswith(q.LIT_REVIEW_LOGIC)
    assert "房子" in system and "第三次" in system and "建立研究空间" in system and "证实或证伪" in system


@pytest.mark.parametrize("question", ["开题时选题太大，怎么收窄？", "扎根理论的三级编码是什么"])
def test_other_thesis_questions_keep_the_prompt_unchanged(question):
    _, _, messages = _svc().prepare_from(question, _result())
    assert messages[0]["content"] == q._SYSTEM_PROMPT
    assert q.system_prompt_for(question) == q._SYSTEM_PROMPT


def test_materials_and_citation_rule_use_plain_numbers():
    _, sources, messages = _svc().prepare_from("扎根理论的三级编码是什么", _result(3))
    user = messages[1]["content"]
    assert "[1] 论文0.pdf（第0节）" in user and "[3] 论文2.pdf（第2节）" in user and "来源1" not in user
    assert "[来源N]" not in q._SYSTEM_PROMPT and "[1][2]" in q._SYSTEM_PROMPT
    assert len(sources) == 3


def test_logic_is_not_a_source_and_never_asks_for_a_citation():
    assert "不标引用" in q.LIT_REVIEW_LOGIC and "不得编造文献" in q.LIT_REVIEW_LOGIC
    assert "600 字" in q.LIT_REVIEW_LOGIC
