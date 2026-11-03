"""四步管线：选词（LLM）→ 召回（确定性）→ 筛选（LLM）→ 作答（LLM），返回结构化结果。

返回 {question, selected_concepts, recalled, filtered, answer, citations, fallback, trace}。
回退（只返回 insufficient_evidence，不编造；HybridRetriever 兜底留接口，尚未实现）：
- no_valid_concepts：Step 1 无有效概念；
- no_recall：选中概念召回不到任何子块；
- all_filtered_out：Step 3 全部判为无用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from . import config
from .agent_select import SelectResult, select_concepts
from .aliases import AliasTable, build_alias_table
from .answer import answer_question, chapter_titles
from .chunker import ChunkSet, chunk_textbook
from .evidence import AnchorIndex
from .fallback import ChunkBM25
from .llm import LLMClient, Trace, TracingLLM
from .ontology import ConceptTable, OntologyPaths, Record, load_concepts, load_records
from .passage_filter import cap_for_filter, filter_passages, pick_for_answer, skip_filter
from .recall import DEFAULT_STRATEGY, Recaller, batch_by_budget, dry_run_report
from .textbook import Textbook, parse_docx
from .vocab_tools import VocabIndex

FALLBACK_NEXT = "hybrid_retriever(stub)"


@dataclass
class CourseContext:
    concepts: ConceptTable
    records: list[Record]
    book: Textbook
    chunkset: ChunkSet
    aliases: AliasTable
    index: VocabIndex
    anchors: AnchorIndex
    recaller: Recaller
    bm25: ChunkBM25 | None = None     # 兜底召回索引（COURSE_BM25_TOPK=0 时仍构建，只是不使用）

    @cached_property
    def chapter_titles(self) -> dict[int, str]:
        """章号 -> 章标题，给来源路径补章名用。"""
        return chapter_titles(self.chunkset.chunks)

    @classmethod
    def build(cls, concepts: ConceptTable, records: list[Record], book: Textbook,
              chunkset: ChunkSet | None = None, *, use_jieba: bool | None = None) -> "CourseContext":
        chunkset = chunkset or chunk_textbook(book)
        aliases = build_alias_table(concepts.concepts)
        anchors = AnchorIndex(book)
        return cls(concepts, records, book, chunkset, aliases,
                   VocabIndex(concepts, aliases, records, use_jieba=use_jieba), anchors,
                   Recaller(records, chunkset, anchors, aliases, concepts), ChunkBM25(chunkset, use_jieba=use_jieba))

    @classmethod
    def load(cls, root: Path | None = None, textbook: Path | None = None) -> "CourseContext":
        paths = OntologyPaths(Path(root) if root else config.COURSE_ONTO_ROOT)
        concepts = load_concepts(paths.concept_xlsx)
        records = [r for v in ("onto1", "onto2") for r in load_records(paths.onto_dir(v), v)]
        tb = textbook or paths.find_textbook() or config.COURSE_TEXTBOOK
        if not Path(tb).is_file():
            raise FileNotFoundError(f"教材文件不存在：{tb}（用 --textbook 或 COURSE_TEXTBOOK 指定）")
        return cls.build(concepts, records, parse_docx(tb))


def _fallback(reason: str, detail: str) -> dict:
    return {"status": "insufficient_evidence", "reason": reason, "detail": detail, "next": FALLBACK_NEXT}


def _empty_result(question: str, trace: Trace) -> dict:
    return {"question": question, "selected_concepts": [], "recalled": [], "filtered": [], "answer": None,
            "citations": [], "fallback": None, "trace": trace}


def ask(question: str, *, ctx: CourseContext, llm: LLMClient | None = None, active_concepts=(),
        strategy: str = DEFAULT_STRATEGY, dry_run: bool = False, select: list[tuple[str, str]] | None = None,
        max_chars: int | None = None, on_event=None, answer: bool = True) -> dict:
    """dry_run=True：不调用任何 LLM（llm 可为 None）；选中概念由 select=[(id, role)] 手动指定。

    on_event(dict)：流式输出用的回调（rag.unified 的 /auto/ask/stream）。依次收到
    stage=select / filter 的进度、作答正文的 delta 片段（学生可见文本：[S#] 已实时改成按首次出现编号的 [n]），
    以及每块第一次被引用时的 materials（n、chunk_id、章节路径）；不给时行为与原来完全一样。

    answer=False：只做选概念 → 召回 → 筛选，不调用作答模型；筛选后的作答材料放在结果的 "kept"（Recalled 列表）、
    "relations"、"concept_names" 里，由跨库组合问答（rag.combined）拿去与判例 / 讲话材料一起交给**一次**作答。"""
    emit = on_event or (lambda _e: None)
    max_chars = max_chars or config.COURSE_RECALL_MAX_CHARS
    trace = Trace()
    res = _empty_result(question, trace)

    if dry_run:
        sel = [(c, r) for c, r in (select or []) if ctx.index.reject_reason(c) is None]
        res["selected_concepts"] = [{"concept_id": c, "role": r} for c, r in sel]
        res["dry_run"] = dry_run_report(ctx.recaller, sel, max_chars=max_chars,
                                        max_steps=config.COURSE_AGENT_MAX_STEPS)
        trace.stages["recall"] = {"dry_run": True}
        res["trace"] = trace
        return res

    if llm is None:
        raise ValueError("非 dry-run 需要提供 llm")
    traced = TracingLLM(llm, trace)

    # Step 1
    sr: SelectResult = select_concepts(question, active_concepts, ctx.index, traced)
    trace.stages["select"] = {k: v for k, v in sr.to_dict().items() if k != "selected"}
    res["selected_concepts"] = [s.to_dict() for s in sr.selected]
    emit({"type": "stage", "stage": "select",
          "concepts": [ctx.index.concepts[s.concept_id].name for s in sr.selected if s.concept_id in ctx.index.concepts]})
    use_bm25 = ctx.bm25 is not None and config.COURSE_BM25_TOPK > 0
    if not sr.selected and not use_bm25:
        res["fallback"] = _fallback("no_valid_concepts", "Step 1 没有选出任何有效概念")
        return res

    # Step 2
    rr = ctx.recaller.recall([(s.concept_id, s.role) for s in sr.selected], strategy)
    onto_ids = {i.chunk_id for i in rr.items}
    bm25_stats = None
    if use_bm25:
        bm25_stats = ctx.recaller.merge_extra(rr, ctx.bm25.search(question, config.COURSE_BM25_TOPK))
    batches = batch_by_budget(rr.items, max_chars)
    trace.stages["recall"] = {
        "strategy": strategy, "chunks": len(rr.items), "batches": [
            {"index": i, "size": len(b), "chars": sum(len(x.text) for x in b),
             "tier_min": b[0].tier, "tier_max": b[-1].tier} for i, b in enumerate(batches)],
        "excluded": rr.excluded, "sunk": rr.sunk, "zero_mention": rr.zero_mention,
        "ontology_only": rr.ontology_only, "bm25": bm25_stats,
        "sources": {"ontology": sum(i.source == "ontology" for i in rr.items), "bm25": sum(i.source == "bm25" for i in rr.items),
                    "both": sum(i.source == "both" for i in rr.items)}, "ontology_chunks": len(onto_ids)}
    res["recalled"] = [i.to_dict(with_text=False) for i in rr.items]
    if not rr.items:
        res["fallback"] = _fallback("no_valid_concepts" if not sr.selected else "no_recall",
                                    "Step 1 没有选出任何有效概念，BM25 兜底也没有命中" if not sr.selected
                                    else "选中概念没有召回到任何教材子块")
        return res

    # Step 3：所有召回块一次调用送筛选（single 模式）；超上限先截断，不再分批
    if config.COURSE_SELECT_MODE != "agent":
        to_filter, cap_stats = cap_for_filter(rr.items, question=question,
                                              concept_names=[ctx.index.concepts[x.concept_id].name for x in sr.selected])
        batches = [to_filter] if to_filter else []
    else:
        cap_stats = None
    skipped = (config.COURSE_SELECT_MODE != "agent" and 0 < len(rr.items) <= config.COURSE_FILTER_SKIP_MAX)
    fr = skip_filter(rr.items) if skipped else filter_passages(question, batches, traced)
    trace.stages["filter"] = {k: v for k, v in fr.to_dict().items() if k != "verdicts"}
    trace.stages["filter"]["cap"] = cap_stats
    trace.stages["filter"]["skipped"] = skipped
    res["filtered"] = [v.to_dict() for v in fr.verdicts]
    kept, capped = pick_for_answer(rr.items, fr, question=question,
                                   concept_names=[ctx.index.concepts[x.concept_id].name for x in sr.selected
                                                  if x.concept_id in ctx.index.concepts])
    emit({"type": "stage", "stage": "filter", "recalled": len(rr.items), "kept": len(kept), "skipped": skipped})
    if not kept:
        res["fallback"] = _fallback("all_filtered_out", "召回的子块被全部判为无助于回答")
        return res

    # Step 4
    relations = ctx.index.relations_among([s.concept_id for s in sr.selected])
    if not answer:                                          # 组合问答：材料交给上层，不在这里作答
        res["kept"], res["relations"], res["capped"] = kept, relations, capped
        res["concept_names"] = [ctx.index.concepts[x.concept_id].name for x in sr.selected if x.concept_id in ctx.index.concepts]
        return res
    ar = answer_question(question, kept, relations, traced, titles=ctx.chapter_titles,
                         on_delta=(lambda t: emit({"type": "delta", "text": t})) if on_event else None,
                         on_source=(lambda e: emit({"type": "materials", "n": e["n"], "chunk_id": e["chunk_id"],
                                                    "path": e["path"]})) if on_event else None)
    trace.stages["answer"] = {"kept": len(kept), "kept_grades": {g: sum(fr.grade_of(k.chunk_id) == g for k in kept)
                                                                 for g in ("direct", "background")},
                              "capped": capped, "relations": len(relations),
                              "invalid_citations": ar.invalid_citations, "warnings": ar.warnings}
    res["answer"] = ar.to_dict()
    # citations：与 answer.sources 同序（序号 1..n），保留 chunk_id 供评测脚本使用；path 仍是列表形式（向后兼容）
    res["citations"] = [{"n": src["n"], "chunk_id": src["chunk_id"], "path": next(k.path for k in kept if k.chunk_id == src["chunk_id"])}
                        for src in ar.sources]
    res["sources"] = ar.sources
    return res


def result_to_json(result: dict) -> dict:
    """可序列化版本（Trace 转 dict）。"""
    out = dict(result)
    t = out.get("trace")
    out["trace"] = t.to_dict() if isinstance(t, Trace) else t
    return out
