"""原论文指导 RAG 的自适应流程（不是整套 agentic）。

simple：一次增强检索（沿用 RAGService 的混合 / 重排配置）→ 生成。
纠错（CRAG 思路）：用检索置信度判断是否靠谱，不靠谱则改写查询重查一次；仍不靠谱回答「资料不足」，不调用生成、不硬编。
  置信度来源（_confidence）：
  - 开了重排：重排最高分 >= ADAPTIVE_MIN_RERANK_SCORE；
  - 没开重排且为 dense：最近邻余弦距离 <= ADAPTIVE_MAX_DISTANCE；
  - 没开重排且为 hybrid / bm25：没有可比较的分数，跳过纠错（视为可靠）。
complex：LLM 拆成最多 ADAPTIVE_MAX_SUBQUESTIONS 个子问题；第 1 轮对所有子问题各检索一次，
  第 2..ADAPTIVE_MAX_ROUNDS 轮只对仍不可靠的子问题改写重查；轮数有硬上限；合并去重后一次汇总作答。
  这里没有复用 rag/course/agent_select.py：它的工具循环绑定课程本体词表，这里只需要「拆分 + 有界补查」。
trace 记录各环节耗时：retrieve_s（不含重排）/ rerank_s / generate_s / 其他 LLM 辅助调用 aux_llm_s。
"""
from __future__ import annotations

import logging
import time

from . import config
from .course.llm import parse_json_object
from .enhance import rewrite_query
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "你是检索规划器。把用户的复合问题拆成最多 {n} 个相互独立、可单独检索的子问题（每个不超过40字，自成一句、不含代词）。"
    "只输出 JSON：{{\"sub_questions\": [\"...\", \"...\"]}}。如果问题其实不需要拆分，就只给 1 个。"
)
_SYNTH_NOTE = "\n\n（注意：以下子问题在资料中未找到依据，回答时请明确说明这部分资料不足，不要编造：{missing}）"


class AdaptiveRAG:
    def __init__(self, svc, *, min_rerank_score: float | None = None, max_distance: float | None = None,
                 max_subquestions: int | None = None, max_rounds: int | None = None):
        self.svc = svc
        self.min_rerank = config.ADAPTIVE_MIN_RERANK_SCORE if min_rerank_score is None else min_rerank_score
        self.max_dist = config.ADAPTIVE_MAX_DISTANCE if max_distance is None else max_distance
        self.max_subs = config.ADAPTIVE_MAX_SUBQUESTIONS if max_subquestions is None else max_subquestions
        self.max_rounds = config.ADAPTIVE_MAX_ROUNDS if max_rounds is None else max_rounds

    # ---------------------------------------------------------------- 基础环节

    def _confidence(self, result: RetrievalResult) -> tuple[bool | None, dict]:
        """返回 (是否可靠 / None=无法判断而跳过, 依据说明)。"""
        if not result.ids:
            return False, {"kind": "empty"}
        rs = result.debug.get("rerank_scores")
        if rs:
            top = max(rs.values())
            return top >= self.min_rerank, {"kind": "rerank", "top": round(top, 4), "threshold": self.min_rerank}
        if self.svc.mode == "dense":
            nearest = min(result.distances)
            return nearest <= self.max_dist, {"kind": "distance", "nearest": round(nearest, 4),
                                              "threshold": self.max_dist}
        return None, {"kind": "skipped", "why": f"{self.svc.mode} 且未开重排，无可比较分数"}

    def _retrieve(self, query: str, top_k: int, tm: dict) -> RetrievalResult:
        t0 = time.perf_counter()
        res = self.svc.retrieve_result(query, top_k)
        total = time.perf_counter() - t0
        rerank = float(res.debug.get("rerank_s", 0.0))
        tm["rerank_s"] += rerank
        tm["retrieve_s"] += max(total - rerank, 0.0)
        return res

    def _aux(self, fn, tm: dict):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            tm["aux_llm_s"] += time.perf_counter() - t0

    def _generate(self, question: str, result: RetrievalResult, tm: dict, extra: str = ""):
        docs, sources, messages = self.svc.prepare_from(question, result)
        if extra:
            messages[-1]["content"] += extra
        if not self.svc.llm:
            return None, sources, docs
        from .config import LLM_MODEL
        t0 = time.perf_counter()
        resp = self.svc.llm.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2)
        tm["generate_s"] += time.perf_counter() - t0
        return resp.choices[0].message.content, sources, docs

    # ---------------------------------------------------------------- 入口

    def run(self, question: str, top_k: int | None = None, complexity: str = "simple") -> dict:
        top_k = top_k or config.TOP_K
        tm = {"retrieve_s": 0.0, "rerank_s": 0.0, "generate_s": 0.0, "aux_llm_s": 0.0}
        trace: dict = {"path": "simple", "retrievals": []}
        t0 = time.perf_counter()
        if complexity == "complex" and self.svc.llm is not None:
            out = self._complex(question, top_k, tm, trace)
        else:
            out = self._simple(question, top_k, tm, trace)
        out["trace"] = trace
        trace["timings"] = {**{k: round(v, 4) for k, v in tm.items()}, "total_s": round(time.perf_counter() - t0, 4)}
        return out

    def _insufficient(self) -> dict:
        return {"status": "insufficient", "answer": config.ADAPTIVE_INSUFFICIENT_TEXT, "sources": [], "contexts": []}

    def _ok(self, ans, sources, docs) -> dict:
        return {"status": "ok" if ans is not None else "no_llm", "answer": ans, "sources": sources, "contexts": docs}

    def _corrective(self, query: str, top_k: int, tm: dict, trace: dict, label: str):
        """检索 + 纠错：不可靠则改写重查一次。返回 (result, reliable)，reliable=False 表示仍然不靠谱。"""
        res = self._retrieve(query, top_k, tm)
        ok, why = self._confidence(res)
        trace["retrievals"].append({"label": label, "query": query, "reliable": ok, **why})
        if ok is not False:
            return res, True
        new_q = self._aux(lambda: rewrite_query(query, self.svc.llm, config.REWRITE_MODEL), tm) \
            if self.svc.llm else query
        if new_q == query:          # 没有 LLM 或改写无变化：重查没有意义
            return res, False
        res2 = self._retrieve(new_q, top_k, tm)
        ok2, why2 = self._confidence(res2)
        trace["retrievals"].append({"label": label + ":retry", "query": new_q, "reliable": ok2, **why2})
        return res2, ok2 is not False

    # ---------------------------------------------------------------- simple

    def _simple(self, question: str, top_k: int, tm: dict, trace: dict) -> dict:
        res, ok = self._corrective(question, top_k, tm, trace, "q")
        if not ok:
            trace["path"] = "simple:insufficient"
            return self._insufficient()
        return self._ok(*self._generate(question, res, tm))

    # ---------------------------------------------------------------- complex

    def _decompose(self, question: str, tm: dict) -> list[str]:
        def call():
            resp = self.svc.llm.chat.completions.create(
                model=config.REWRITE_MODEL,
                messages=[{"role": "system", "content": _DECOMPOSE_SYSTEM.format(n=self.max_subs)},
                          {"role": "user", "content": question}], temperature=0.0)
            return parse_json_object(resp.choices[0].message.content or "")
        try:
            subs = self._aux(call, tm).get("sub_questions", [])
            subs = [str(s).strip() for s in subs if isinstance(s, str) and s.strip()]
        except Exception as e:  # noqa: BLE001 - 拆分失败不影响主流程，退化为单问题
            logger.warning("子问题拆分失败（%s），按 simple 处理", type(e).__name__)
            return []
        return subs[: self.max_subs]

    def _complex(self, question: str, top_k: int, tm: dict, trace: dict) -> dict:
        subs = self._decompose(question, tm)
        if len(subs) < 2:           # 拆不出多个子问题 -> 走 simple
            trace["path"] = "complex->simple"
            return self._simple(question, top_k, tm, trace)
        trace.update(path="complex", sub_questions=subs, max_rounds=self.max_rounds)
        results: dict[int, RetrievalResult] = {}
        pending = list(range(len(subs)))
        rounds = 0
        while pending and rounds < self.max_rounds:
            rounds += 1
            still = []
            for i in pending:
                q = subs[i]
                if rounds > 1:      # 补查轮：改写后重查；改写无变化则放弃该子问题
                    new_q = self._aux(lambda: rewrite_query(q, self.svc.llm, config.REWRITE_MODEL), tm)
                    if new_q == q:      # 无法再改写：保留为未解决，最终在汇总里如实说明
                        still.append(i)
                        continue
                    q = new_q
                res = self._retrieve(q, top_k, tm)
                ok, why = self._confidence(res)
                trace["retrievals"].append({"label": f"sub{i + 1}:r{rounds}", "query": q, "reliable": ok, **why})
                if ok is not False:
                    results[i] = res
                else:
                    still.append(i)
            pending = still
        trace["rounds"] = rounds
        missing = [subs[i] for i in pending]
        if not results:
            trace["path"] = "complex:insufficient"
            return self._insufficient()
        merged = self._merge([results[i] for i in sorted(results)], top_k * len(results))
        extra = _SYNTH_NOTE.format(missing="；".join(missing)) if missing else ""
        trace["unanswered_sub_questions"] = missing
        return self._ok(*self._generate(question, merged, tm, extra))

    @staticmethod
    def _merge(results: list[RetrievalResult], cap: int) -> RetrievalResult:
        out, seen = RetrievalResult(), set()
        for r in results:
            for k, id_ in enumerate(r.ids):
                if id_ in seen:
                    continue
                seen.add(id_)
                out.ids.append(id_)
                out.documents.append(r.documents[k])
                out.metadatas.append(r.metadatas[k])
                out.distances.append(r.distances[k])
                out.scores.append(r.scores[k])
        for name in ("ids", "documents", "metadatas", "distances", "scores"):
            setattr(out, name, getattr(out, name)[:cap])
        return out
