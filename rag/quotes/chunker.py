"""名言库切块：一次讲话一块（颗粒度最粗）= 年份 + 讲话标题 + 该次讲话全部段落（段落级去重）。

块里同时保存段落列表（paragraphs），供作答环节「挑段落编号、程序原文照录」使用；
text = 年份 + 标题 + 带编号的段落（入库、检索用）。讲话类型作为元数据。
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from .parser import Speech


@dataclass
class QuoteChunk:
    chunk_id: str
    speech_id: str
    year: int
    n: int
    title: str
    kind: str
    text: str                                   # 检索用文本：年份 + 标题 + 全部段落（段落前带 [编号]）
    paragraphs: list[str] = field(default_factory=list)       # 原文段落，序号从 1 起，与 text 里的 [编号] 对应
    chars: int = 0
    removed_duplicates: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def header(s: Speech) -> str:
    return f"{s.year}年　{s.title}（{s.kind}）"


def chunk_speech(s: Speech) -> QuoteChunk:
    body = "\n".join(f"[{i}] {p}" for i, p in enumerate(s.paragraphs, 1))
    text = f"{header(s)}\n{body}"
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return QuoteChunk(chunk_id=f"{s.speech_id}:{sha}", speech_id=s.speech_id, year=s.year, n=s.n, title=s.title, kind=s.kind,
                      text=text, paragraphs=list(s.paragraphs), chars=s.chars, removed_duplicates=len(s.removed_duplicates))


def chunk_quotes(speeches: list[Speech]) -> list[QuoteChunk]:
    return [chunk_speech(s) for s in speeches]
