"""判分闭环（P2）：学生作答 → 判分 → 回写画像 → 下一轮作答带上「还没答对」。

覆盖四件事：
1. 判分用的是题库参考答案，学生答案确实进了判分提示；
2. 判分失败（题不存在 / 空答案 / 没参考答案）给错误码，不伪装成 0 分；
3. 判分结果回写 profile，答错进「还没答对」，重做答对后消失；
4. HTTP 接口按错误码给状态码，且判分不改变原有找题接口的行为。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.course.llm import LLMResult
from rag.exam.store import ExamBank, ExamItem, Question, StudentIndex
from rag.exam.api import router as exam_router

JUDGE = "个人权益、公共数据与公平竞争是数据要素市场健康发展的三个基础。"
TERM = "专有数据、共享数据与开放数据"


def _q(qtype, cn, n, stem, answer, analysis, options=None, chapter=1, concepts=()):
    it = ExamItem(Question(f"ex:t:{qtype}:{n}", chapter, "数据要素", qtype, cn, n, stem, options or {}, answer, analysis),
                  list(concepts), [], "none", stem)
    return it


def make_bank() -> ExamBank:
    return ExamBank([_q("judge", "判断题", 1, JUDGE, "正确", "见教材第一章。", concepts=["数据要素市场"]),
                     _q("judge", "判断题", 2, "数据资源经确权入表后可成为数据资产。", "正确", "见教材。",
                        concepts=["数据资源", "数据资产"]),
                     _q("term", "名词解释", 1, TERM, "（1）专有数据……（2）共享数据……", "评分要点。",
                        concepts=["专有数据"])],
                    StudentIndex.from_rows([]), {})


class JudgeLLM:
    """判分模型替身：记录发给模型的 messages，返回固定 JSON。"""

    def __init__(self, payload='{"score": 2, "correct": false, "reason": "漏了要点"}'):
        self.payload = payload
        self.calls: list[dict] = []

    def chat(self, step, messages, *, model, tools=None, json_mode=False):
        self.calls.append({"step": step, "messages": messages})
        return LLMResult(content=self.payload)


def test_grade_uses_the_bank_reference_and_the_students_own_answer(tmp_path, monkeypatch):
    from rag import profile as learner
    from rag import grade as grader

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    bank, llm = make_bank(), JudgeLLM()
    got = grader.grade(bank, "ex:t:judge:1", "只说了个人权益，没说公共数据和公平竞争",
                       user_id="stu-g", llm=llm)
    assert got["ok"] and got["score"] == 2.0 and got["correct"] is False and got["method"] == "llm"
    assert got["reference"] == "正确"
    sent = "\n".join(m["content"] for m in llm.calls[0]["messages"])
    assert "只说了个人权益" in sent and "正确" in sent and JUDGE in sent
    assert llm.calls[0]["step"] == "grade"


def test_grade_failures_are_errors_not_zero_scores(tmp_path, monkeypatch):
    from rag import profile as learner
    from rag import grade as grader

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    bank, llm = make_bank(), JudgeLLM()
    assert grader.grade(bank, "nope", "答案", llm=llm)["error"] == "not_found"
    assert grader.grade(bank, "ex:t:judge:1", "   ", llm=llm)["error"] == "empty_answer"
    assert grader.grade(None, "ex:t:judge:1", "答案", llm=llm)["error"] == "no_bank"
    assert llm.calls == []                                    # 判不了就不该调模型
    assert learner.summary("stu-g") == ""                     # 也不该留下「答错了」的记录


def test_wrong_answer_enters_the_profile_and_a_redo_clears_it(tmp_path, monkeypatch):
    from rag import profile as learner
    from rag import grade as grader

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    bank = make_bank()
    grader.grade(bank, "ex:t:judge:2", "不太清楚", user_id="stu-g", llm=JudgeLLM())
    text = learner.summary("stu-g")
    assert "还没答对" in text and "数据资源" in text and "数据资产" in text
    assert "基础薄弱" not in text                             # 只陈述判分事实，不评价水平

    grader.grade(bank, "ex:t:judge:2", "数据资源经确权、计量后计入报表就成为数据资产",
                 user_id="stu-g", llm=JudgeLLM('{"score": 5, "correct": true, "reason": "对"}'))
    assert "还没答对" not in learner.summary("stu-g")          # 重做答对后不再进提示词


def test_profile_stays_silent_for_a_student_who_never_submitted_an_answer(tmp_path, monkeypatch):
    from rag import profile as learner

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    learner.notice_question("stu-n", concepts=["数据确权"])     # 只问过问题、没交过作业
    assert learner.summary("stu-n") == ""
    assert learner.weaknesses("stu-n") == []


def test_grade_without_a_model_falls_back_to_lexical_scoring(tmp_path, monkeypatch):
    from rag import profile as learner
    from rag import grade as grader

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    got = grader.grade(make_bank(), "ex:t:judge:1", "正确", user_id="stu-x", llm=None)
    assert got["ok"] and got["method"] == "lexical" and got["correct"] is True


# ---------------------------------------------------------------- HTTP

def _client(bank, llm=None):
    app = FastAPI()
    app.state.exam = bank
    app.state.lib_llm = llm
    app.include_router(exam_router)
    return TestClient(app)


def test_grade_endpoint_returns_the_score_and_error_codes(tmp_path, monkeypatch):
    from rag import profile as learner

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    c = _client(make_bank(), JudgeLLM())
    r = c.post("/exam/grade", json={"question_id": "ex:t:judge:1", "answer": "只说了个人权益"})
    assert r.status_code == 200 and r.json()["score"] == 2.0 and r.json()["verdict"] == "只沾边"
    assert c.post("/exam/grade", json={"question_id": "nope", "answer": "x"}).status_code == 404
    assert c.post("/exam/grade", json={"question_id": "ex:t:judge:1", "answer": "   "}).status_code == 422


def test_grade_endpoint_identifies_the_student_from_the_uuid_header(tmp_path, monkeypatch):
    from rag import profile as learner

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    c = _client(make_bank(), JudgeLLM())
    c.post("/exam/grade", headers={"X-User-Uuid": "stu-h"},
           json={"question_id": "ex:t:judge:2", "answer": "不知道"})
    assert "还没答对" in learner.summary("stu-h")
    assert learner.summary("stu-other") == ""                 # 判分记录按人隔离


def test_grading_does_not_touch_the_existing_question_endpoints(tmp_path, monkeypatch):
    """判分是新加的接口，找题 / 取题照旧不返回答案（先练后看）。"""
    from rag import profile as learner

    monkeypatch.setenv("PROFILE_DB", str(tmp_path / "profile.sqlite3"))
    learner.reset()
    c = _client(make_bank(), JudgeLLM())
    q = c.get("/exam/question/ex:t:judge:1").json()
    assert "answer" not in q and "analysis" not in q
