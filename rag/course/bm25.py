"""轻量 BM25（Okapi），不依赖 rank_bm25。分词优先 jieba，未安装时退回字符 bigram。"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_RUN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")
_jieba = None
_jieba_checked = False


def _get_jieba():
    global _jieba, _jieba_checked
    if not _jieba_checked:
        _jieba_checked = True
        try:
            import jieba

            try:
                jieba.setLogLevel(60)
            except Exception:
                pass
            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def tokenize(text: str, use_jieba: bool | None = None) -> list[str]:
    """NFKC + 小写；英文数字按连续串；中文用 jieba（若可用）或字符 bigram（单字保留为 unigram）。

    use_jieba=None 自动检测，False 强制 bigram（测试用）。索引与查询必须用同一模式。
    """
    text = unicodedata.normalize("NFKC", text or "").casefold()
    jb = _get_jieba() if use_jieba is not False else None
    out: list[str] = []
    for run in _RUN_RE.findall(text):
        if run.isascii():
            out.append(run)
        elif jb is not None:
            out.extend(t for t in jb.lcut(run) if t.strip())
        elif len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for tok, f in Counter(d).items():
                self.postings.setdefault(tok, []).append((i, f))
        self.idf = {
            tok: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for tok, p in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        """返回有得分的文档下标 -> 得分（未命中的文档不出现）。"""
        out: dict[int, float] = {}
        if not self.n:
            return out
        for tok in set(query_tokens):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, f in self.postings[tok]:
                norm = 1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0)
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out

    def top(self, query_tokens: list[str], n: int = 10) -> list[tuple[int, float]]:
        sc = self.scores(query_tokens)
        return sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
