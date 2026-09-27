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
    # 安全约束：overlap 必须严格小于 chunk_size，否则切分无法推进（会死循环）。
    chunk_size = max(int(chunk_size), 1)
    overlap = max(0, min(int(overlap), chunk_size - 1))

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
                if room <= 0:  # 缓冲区已满（overlap 残留导致），先冲刷再继续
                    flush()
                    buf = chunks[-1][0][-overlap:] if (chunks and overlap) else ""
                    continue
                if len(piece) <= room:
                    buf += piece
                    piece = ""
                else:
                    buf += piece[:room]
                    flush()
                    # 重叠：保留上一块尾部 overlap 字符，避免语义被截断
                    buf = chunks[-1][0][-overlap:] if (chunks and overlap) else ""
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


# ---------------------------------------------------------------- 父子分块

_HEADER_TMPL = "〔{parent_id}〕\n"


def _parent_key(uid: str, parent_index: int) -> str:
    """父块唯一键，同时作为子块 header 标记与 sidecar 映射键。"""
    return f"{uid}::p{parent_index}"


def _split_by_size(text: str, size: int) -> list[str]:
    """把文本按 size 字符切成若干块（尽量落在句子/换行边界）。"""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        out.extend(_split_long(para, size))
    return out


def _pack(pieces: list[str], size: int) -> list[str]:
    """把片段贪心打包成 <= size 的块（尽量保留段落边界）。"""
    packed: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > size:
            packed.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n{piece}" if buf else piece
    if buf:
        packed.append(buf)
    return packed


def chunk_parent_child(
    text: str,
    uid: str,
    child_size: int = 600,
    child_overlap: int = 100,
    parent_size: int = 1500,
) -> tuple[list[dict], list[dict]]:
    """父子分块：小 chunk 入向量库检索，父块保留更完整上下文。

    返回 (children, parents)：
    - children[i] = {"text", "section", "parent_id", "parent_offset", "parent_length"}
      text 为「header 标记 + 子块正文」，header 让子块自包含且父块可无歧义定位。
    - parents[i]  = {"parent_id", "text"}，写入 sidecar JSON，不入向量库。
    """
    parents: list[dict] = []
    children: list[dict] = []
    for parent_index, parent_text in enumerate(_pack(_split_by_size(text, parent_size), parent_size)):
        pid = _parent_key(uid, parent_index)
        parents.append({"parent_id": pid, "text": parent_text})
        header = _HEADER_TMPL.format(parent_id=pid)
        for body, section in chunk_text(parent_text, child_size, child_overlap):
            off, length = _locate(parent_text, body)
            children.append(
                {
                    "text": header + body,
                    "section": section,
                    "parent_id": pid,
                    "parent_offset": off,
                    "parent_length": length,
                }
            )
    return children, parents


def _locate(parent_text: str, body: str) -> tuple[int, int]:
    """在父块中定位子块正文，返回 (起始偏移, 长度)。

    先做直接子串匹配；匹配不到（父块按换行拼接、子块按行拼接导致换行差异）时，
    退化为「忽略换行」的逐字符对齐，并返回覆盖父块中对应区间的 (offset, length)。
    """
    offset = parent_text.find(body)
    if offset >= 0:
        return offset, len(body)

    compact = body.replace("\n", "")
    if not compact:
        return -1, 0

    # 跳过父块开头的空白，从首个非空白字符开始对齐
    start = 0
    while start < len(parent_text) and parent_text[start].isspace():
        start += 1
    k, i = 0, start
    first = last = -1
    while i < len(parent_text) and k < len(compact):
        if parent_text[i].isspace():  # 父块的换行不消耗 compact
            i += 1
            continue
        if parent_text[i] == compact[k]:
            if first < 0:
                first = i
            last = i
            k += 1
        else:  # 失配则从当前位置重新开始
            first = last = -1
            start = i + 1
            k = 0
        i += 1
    if first >= 0 and k == len(compact):
        return first, last - first + 1
    return -1, 0
