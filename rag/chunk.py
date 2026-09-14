"""分块层：面向中文方法论/规范文档的语义分块。

策略：
1. 先按空行拆成段落块；
2. 识别标题行（数字/汉字序号、"第X章"、常见章节词），标题作为后续 chunk 的 section 元数据；
3. 按句子边界（。！？；）累积到 chunk_size 上限，相邻 chunk 带 overlap 重叠。
"""
from __future__ import annotations

import re

_HEADING_RE = [
    re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*[章节部分篇]"),
    re.compile(r"^[一二三四五六七八九十]+\s*[、.．]"),
    re.compile(r"^\d{1,2}([.．]\d{1,2})*\s*[、.．:：]?\s*\S"),
    re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]"),
]
_SECTION_KEYS = {
    "摘要", "关键词", "引言", "绪论", "前言", "文献综述", "研究方法", "研究设计",
    "结论", "结语", "参考文献", "致谢", "附录", "目录", "正文", "选题", "开题报告",
}
_SENT_SEP = "。！？；!?;"


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if s in _SECTION_KEYS:
        return True
    return any(r.match(s) for r in _HEADING_RE)


def _split_long(text: str, size: int) -> list[str]:
    """把单行文本按句子边界切成 <= size 的片段；无标点则硬切。"""
    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENT_SEP and len(buf) >= size // 2:
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf)
    final: list[str] = []
    for piece in out:
        while len(piece) > size:
            final.append(piece[:size])
            piece = piece[size:]
        if piece:
            final.append(piece)
    return final


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[tuple[str, str]]:
    """返回 [(chunk_text, section_title), ...]。"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[tuple[str, str]] = []
    section = ""
    buf = ""

    def flush() -> None:
        nonlocal buf
        buf = buf.strip()
        if buf:
            chunks.append((buf, section))
            buf = ""

    def add(line: str) -> None:
        nonlocal buf
        for piece in _split_long(line, chunk_size):
            while piece:
                room = chunk_size - len(buf)
                if len(piece) <= room:
                    buf += piece
                    piece = ""
                else:
                    buf += piece[:room]
                    flush()
                    # 重叠：保留上一块尾部 overlap 字符，避免语义被截断
                    buf = chunks[-1][0][-overlap:] if chunks else ""
                    piece = piece[room:]

    for block in blocks:
        for line in block.splitlines():
            s = line.strip()
            if not s:
                continue
            if is_heading(s):
                flush()
                section = s
                continue
            add(s)
    flush()
    return chunks
