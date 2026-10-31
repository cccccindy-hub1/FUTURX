"""兜底召回：用问题对教材子块做 BM25（jieba），与本体召回结果合并去重后再进入筛选。

动机：本体召回依赖选词 LLM 选对概念，漏选一个概念，该概念所在章节的教材块就整块丢失；
BM25 不依赖选词，直接按问题字面找子块，作为独立的第二路来源。
合并（Recaller.merge_extra）后每个子块带来源标记 Recalled.source：ontology / bm25 / both。
BM25 命中的 tier 记为 3（与 mention 同档，排在有证据的本体召回之后送筛选），不会挤掉本体召回的块。
COURSE_BM25_TOPK=0 关闭。
"""
from __future__ import annotations

from .bm25 import BM25, tokenize
from .chunker import Chunk, ChunkSet

# 疑问词 / 虚词：对检索没有区分度，且会把无关块顶上来
_STOP = frozenset("什么 是 的 有 哪些 吗 呢 了 为什么 怎么 如何 怎样 在 哪 和 与 及 或 对 中 哪里 多少 请 讲讲 一下".split())


class ChunkBM25:
    def __init__(self, chunkset: ChunkSet, *, use_jieba: bool | None = None):
        self.chunks: list[Chunk] = list(chunkset.chunks)
        self._use_jieba = use_jieba
        self._bm25 = BM25([tokenize(c.embed_text, use_jieba) for c in self.chunks])

    def query_tokens(self, question: str) -> list[str]:
        return [t for t in tokenize(question, self._use_jieba) if t not in _STOP]

    def search(self, question: str, k: int) -> list[tuple[Chunk, float]]:
        if k <= 0:
            return []
        return [(self.chunks[i], s) for i, s in self._bm25.top(self.query_tokens(question), k)]
