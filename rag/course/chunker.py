"""结构感知的父子分块（方案 E）。

- 父块 = 一个样式标题下的叶子小节（block_id 相同的连续正文单元），超过 PARENT_MAX
  时只在单元（段落/表格）边界切开，优先在编号子标题处切；
- 子块 = 以段落为单位打包：短段（< CHILD_MIN）与相邻段合并，打包上限 CHILD_MAX；
  只有超过 PARA_SPLIT 的单段才按句子切开——教材中的定义句总落在单个段落内，
  因此不会被切断；子块不跨编号子标题（section_id）边界，也不做字符级重叠；
- 表格单独成子块，过长时按行分组并重复表头；
- 目录（front）与引用/参考文献（references）不入库，小结标为 chunk_type=summary。

ID 设计（稳定、可追溯）：
- parent_id = de:{版本}:{block_id}:p{k}
- chunk_id  = de:{版本}:{section_id}:{seq:03d}:{正文 sha1 前 8 位}
  修改某段只会改变该小节内的 chunk_id，其他章节不受影响；换教材版本则整体换代。
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from .config import BOOK_VERSION, CHILD_MAX, CHILD_MIN, PARA_SPLIT, PARENT_MAX
from .textbook import Textbook, Unit

_CONTENT_KINDS = {"paragraph", "caption", "table", "textbox"}
_INDEXED_ZONES = {"body", "summary"}
_SENT_SEP = "。！？；!?;"
PATH_SEP = " › "


@dataclass
class Parent:
    parent_id: str
    block_id: str
    chapter: int | None
    path: list[str]
    text: str
    anchors: list[str]


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    section_id: str
    block_id: str
    chapter: int | None
    path: list[str]
    chunk_type: str             # body | summary | table
    text: str                   # 返回给 Agent 的正文（不含路径前缀）
    embed_text: str             # 向量化 / BM25 用：标题路径 + 正文
    anchors: list[str]          # 覆盖的段落/表格锚点，如 ["P1984", "P1985"]
    split_part: int | None = None  # 长段按句切开时的分片序号，未切为 None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkSet:
    book_version: str
    source_sha256: str
    parents: list[Parent] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


def _sha8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def split_sentences(text: str, size: int) -> list[str]:
    """按句末标点把长文本切成 <= size 的片段；单句超长时硬切。"""
    sents: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENT_SEP:
            sents.append(buf)
            buf = ""
    if buf.strip():
        sents.append(buf)
    out: list[str] = []
    cur = ""
    for s in sents:
        while len(s) > size:  # 单句超长：先冲刷再硬切
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:size])
            s = s[size:]
        if cur and len(cur) + len(s) > size:
            out.append(cur)
            cur = s
        else:
            cur += s
    if cur.strip():
        out.append(cur)
    return [p.strip() for p in out if p.strip()]


def _blocks(units: list[Unit]) -> list[list[Unit]]:
    """把可入库的正文单元按 block_id 分成连续的组（标题本身不入组）。"""
    groups: list[list[Unit]] = []
    cur_id = None
    for u in units:
        if u.zone not in _INDEXED_ZONES or u.kind not in _CONTENT_KINDS:
            # 标题/子标题只切换路径；front/references 中断当前组
            if u.zone not in _INDEXED_ZONES:
                cur_id = None
            continue
        if u.block_id != cur_id:
            groups.append([])
            cur_id = u.block_id
        groups[-1].append(u)
    return groups


def _split_parent(units: list[Unit], limit: int) -> list[list[Unit]]:
    """父块过长时在单元边界切开，优先选择编号子标题（section_id 变化）处。"""
    total = sum(len(u.text) for u in units)
    if total <= limit:
        return [units]
    parts: list[list[Unit]] = []
    cur: list[Unit] = []
    size = 0
    for i, u in enumerate(units):
        boundary = bool(cur) and u.section_id != cur[-1].section_id
        if cur and (size + len(u.text) > limit or (boundary and size >= limit // 2)):
            parts.append(cur)
            cur, size = [], 0
        cur.append(u)
        size += len(u.text)
    if cur:
        parts.append(cur)
    return parts


def _table_chunks(u: Unit, size: int) -> list[str]:
    lines = [" | ".join(r) for r in u.rows] or [u.text]
    head, rest = lines[0], lines[1:]
    if len(u.text) <= size or not rest:
        return [u.text]
    out: list[str] = []
    cur = head
    for line in rest:
        if len(cur) + len(line) + 1 > size and cur != head:
            out.append(cur)
            cur = head
        cur = f"{cur}\n{line}"
    out.append(cur)
    return out


def _pack_section(units: list[Unit]) -> list[tuple[str, list[str], str, int | None]]:
    """同一 section_id 内打包子块，返回 [(正文, 锚点, 类型, 分片序号)]。"""
    out: list[tuple[str, list[str], str, int | None]] = []
    buf: list[Unit] = []

    def flush() -> None:
        if buf:
            out.append(("\n".join(x.text for x in buf), [x.anchor for x in buf], "body", None))
            buf.clear()

    for u in units:
        if u.kind == "table":
            # 紧邻表格之前的表题随表格走（每个分片都带表题），避免表题孤立成块
            caption: list[Unit] = []
            while buf and buf[-1].kind == "caption":
                caption.insert(0, buf.pop())
            flush()
            prefix = "".join(x.text + "\n" for x in caption)
            anchors = [x.anchor for x in caption] + [u.anchor]
            pieces = _table_chunks(u, CHILD_MAX)
            for k, piece in enumerate(pieces):
                out.append((prefix + piece, anchors, "table", k if len(pieces) > 1 else None))
            continue
        if len(u.text) > PARA_SPLIT:
            flush()
            for k, piece in enumerate(split_sentences(u.text, CHILD_MAX)):
                out.append((piece, [u.anchor], "body", k))
            continue
        size = sum(len(x.text) for x in buf)
        if buf and size >= CHILD_MIN and size + len(u.text) > CHILD_MAX:
            flush()
        buf.append(u)
    flush()

    # 尾部过短的打包块并回前一个正文块（不并入表格或长段分片）
    if len(out) >= 2:
        text, anchors, kind, part = out[-1]
        ptext, panchors, pkind, ppart = out[-2]
        if (
            kind == "body" and part is None and pkind == "body" and ppart is None
            and len(text) < CHILD_MIN and len(ptext) + len(text) <= CHILD_MAX + CHILD_MIN
        ):
            out[-2:] = [(f"{ptext}\n{text}", panchors + anchors, "body", None)]
    return out


def chunk_textbook(book: Textbook, version: str = BOOK_VERSION) -> ChunkSet:
    result = ChunkSet(book_version=version, source_sha256=book.sha256)
    seq: dict[str, int] = {}
    for block in _blocks(book.units):
        for k, part in enumerate(_split_parent(block, PARENT_MAX)):
            first = part[0]
            pid = f"de:{version}:{first.block_id}:p{k}"
            # 父块路径取到样式标题一级（不含编号子标题）
            depth = len(first.path) - sum(1 for s in first.section_id.split(".") if s.startswith("i"))
            result.parents.append(
                Parent(
                    parent_id=pid,
                    block_id=first.block_id,
                    chapter=first.chapter,
                    path=list(first.path[:depth]),
                    text="\n".join(u.text for u in part),
                    anchors=[u.anchor for u in part],
                )
            )
            # 子块：按 section_id 连续分段后各自打包
            runs: list[list[Unit]] = []
            for u in part:
                if not runs or runs[-1][-1].section_id != u.section_id:
                    runs.append([])
                runs[-1].append(u)
            for run in runs:
                head = run[0]
                ctype_default = "summary" if head.zone == "summary" else "body"
                for text, anchors, kind, piece in _pack_section(run):
                    n = seq.get(head.section_id, 0)
                    seq[head.section_id] = n + 1
                    ctype = "table" if kind == "table" else ctype_default
                    result.chunks.append(
                        Chunk(
                            chunk_id=f"de:{version}:{head.section_id}:{n:03d}:{_sha8(text)}",
                            parent_id=pid,
                            section_id=head.section_id,
                            block_id=head.block_id,
                            chapter=head.chapter,
                            path=list(head.path),
                            chunk_type=ctype,
                            text=text,
                            embed_text=PATH_SEP.join(head.path) + "\n" + text,
                            anchors=anchors,
                            split_part=piece,
                        )
                    )
    return result
