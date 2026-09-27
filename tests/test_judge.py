"""答案打分测试：词法基线 + LLM 评审（含解析失败降级）。"""
from __future__ import annotations

from rag.judge import char_f1, judge_answer, lexical_score, mean, rouge_l
from tests.conftest import StubLLM


def test_char_f1_identical_is_one():
    assert char_f1("扎根理论三级编码", "扎根理论三级编码") == 1.0


def test_char_f1_disjoint_is_zero():
    assert char_f1("完全不同的内容", "另一段无关文字") == 0.0


def test_char_f1_partial_between_zero_and_one():
    s = char_f1("扎根理论强调三级编码", "扎根理论强调开放式编码")
    assert 0.0 < s < 1.0


def test_char_f1_empty_inputs():
    assert char_f1("", "abc") == 0.0
    assert char_f1("abc", "") == 0.0


def test_rouge_l_rewards_subsequence_order():
    # 顺序保留的答案应高于顺序打乱的答案
    ordered = rouge_l("开放式编码 主轴编码 选择性编码", "开放式编码 主轴编码 选择性编码")
    assert ordered == 1.0
    assert rouge_l("主轴编码 开放式编码", "开放式编码 主轴编码") < 1.0


def test_rouge_l_empty():
    assert rouge_l("", "abc") == 0.0


def test_lexical_score_maps_f1_to_five_scale():
    r = lexical_score("扎根理论三级编码", "扎根理论三级编码")
    assert r.method == "lexical"
    assert r.score == 5.0
    assert r.correct is True


def test_judge_without_llm_falls_back_to_lexical():
    r = judge_answer("q?", "扎根理论三级编码", "扎根理论三级编码", None, "m")
    assert r.method == "lexical"
    assert r.correct is True


def test_judge_parses_model_json():
    llm = StubLLM('{"score": 4, "correct": true, "reason": "基本覆盖"}')
    r = judge_answer("q?", "pred", "gold", llm, "m")
    assert r.method == "llm"
    assert r.score == 4.0
    assert r.correct is True


def test_judge_tolerates_code_fence():
    llm = StubLLM('```json\n{"score": 2, "correct": false, "reason": "沾边"}\n```')
    r = judge_answer("q?", "pred", "gold", llm, "m")
    assert r.method == "llm"
    assert r.score == 2.0


def test_judge_bad_json_degrades_to_lexical():
    llm = StubLLM("我觉得还不错")
    r = judge_answer("q?", "pred 内容", "gold 内容", llm, "m")
    assert r.method == "lexical"


def test_mean_ignores_empty():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
