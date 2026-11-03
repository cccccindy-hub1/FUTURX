"""数据政策汇编 docx → 政策库的块（chunk）与文件目录。可重复：同一份 docx 永远得到同样的块与编号。

切法（每块正文前都带「篇名｜文件名」前缀，单独拿出来也知道出处）：
- 条款式文件（≥5 个「第X条」）：一条一块，带所在的编 / 章 / 节标题和条号；条文之后的附件按事项累积；
  单条超过 POLICY_ARTICLE_CAP 才按段拆开，后续片仍带同一个条号；
- 非条款式文件（意见、规划、白皮书、通知）：按段落累积成 POLICY_CHUNK_MIN–POLICY_CHUNK_CAP 字的块；
- 表格按原文顺序并入所在文件的块里（行用 | 连接）。
元数据：篇、地区（国家 / 北京 / 上海 / 广东 / 深圳 / 贵州）、文件名、文件简称、条号、发布 / 通过日期（只取文件开头的括号注，没有就留空，绝不猜）、
是否征求意见稿（看标题）。解析与清洗沿用 rag.policy.chunker（只用标准库读 docx）。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..course.chunker import split_sentences
from . import config
from .chunker import ART_RE, Line, _blocks_article, _blocks_opinion, _render_table, clean_lines, parse_policy_docx

PIAN_REGION = {"1": "国家", "2": "国家", "3": "北京", "4": "上海", "5": "广东", "6": "深圳", "7": "贵州"}
_CN_DIGIT = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"〇": 0}
_OPEN_DATE_RE = re.compile(r"^[（(][^）)]*?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^）)]*[）)]$")
_SHELL_RE = re.compile(r"关于(?:印发|转发|公布|发布)[（(]?《(.+?)》[）)]?的?(?:通知|公告)?")


def cn_to_int(s: str) -> int | None:
    """第二十一条 / 21 / 一百零五 → 数字；识别不了返回 None。"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if not s or any(c not in "零〇一二三四五六七八九十百" for c in s):
        return None
    total, cur = 0, 0
    for c in s:
        if c == "百":
            total += (cur or 1) * 100
            cur = 0
        elif c == "十":
            total += (cur or 1) * 10
            cur = 0
        else:
            cur = _CN_DIGIT[c]
    return total + cur


def article_number(label: str) -> int | None:
    m = re.match(r"第\s*([零〇一二三四五六七八九十百\d]+)\s*条", label)
    return cn_to_int(m.group(1)) if m else None


def short_name(title: str) -> str:
    """文件简称：去掉汇编编号、外层书名号、「关于印发《X》的通知」外壳。"""
    name = re.sub(r"^\d+(?:\.\d+)?\s+", "", title).strip()
    m = _SHELL_RE.search(name)
    if m:
        return m.group(1).strip()
    inner = re.search(r"《([^《》]+)》", name)
    if inner and ("印发" in name or "发布" in name or name.startswith("《")):
        return inner.group(1).strip()
    return name.replace("《", "").replace("》", "").strip()


def opening_date(lines: list[Line]) -> str:
    """只认文件开头几行里整行是括号注的日期（通过 / 公布 / 施行日期）；没有就返回空串。"""
    for ln in lines[:6]:
        if ln.table is None:
            m = _OPEN_DATE_RE.match(ln.text.strip())
            if m:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""


def _line_text(ln: Line) -> str:
    return _render_table(ln.table) if ln.table is not None else ln.text


def _accumulate(lines: list[Line]) -> list[tuple[str, list[str]]]:
    """段落累积成 MIN–CAP 字的块。返回 [(正文, 锚点)]。"""
    lo, cap = config.POLICY_CHUNK_MIN, config.POLICY_CHUNK_CAP
    out: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    anchors: list[str] = []
    size = 0

    def flush():
        nonlocal buf, anchors, size
        if buf:
            out.append(("\n".join(buf), sorted(set(anchors))))
        buf, anchors, size = [], [], 0

    for ln in lines:
        t = _line_text(ln).strip()
        if not t:
            continue
        pieces = split_sentences(t, config.POLICY_PART_TARGET) if len(t) > cap else [t]
        for p in pieces:
            if buf and size + len(p) > cap:
                flush()
            buf.append(p)
            anchors.append(ln.anchor)
            size += len(p) + 1
            if size >= lo:
                flush()
    flush()
    if len(out) > 1 and len(out[-1][0]) < lo // 4 and len(out[-2][0]) + len(out[-1][0]) + 1 <= cap + lo // 4:
        a, b = out.pop(), out.pop()
        out.append((b[0] + "\n" + a[0], sorted(set(a[1] + b[1]))))
    return out


def _split_article(text: str) -> list[str]:
    cap = config.POLICY_ARTICLE_CAP
    if len(text) <= cap:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n"):
        for p in (split_sentences(para, config.POLICY_PART_TARGET) if len(para) > cap else [para]):
            if buf and len(buf) + len(p) + 1 > cap:
                parts.append(buf)
                buf = ""
            buf = buf + ("\n" if buf else "") + p
    if buf:
        parts.append(buf)
    return parts


def build_corpus(path: Path | str) -> dict:
    """返回 {"files": [...], "chunks": [...]}，两者都可直接 JSON 序列化。"""
    sha, docs = parse_policy_docx(Path(path))
    files, chunks = [], []
    for d in docs:
        dropped: Counter = Counter()
        lines = clean_lines(d.lines, dropped)
        n_art = sum(bool(ART_RE.match(ln.text)) for ln in lines if ln.table is None)
        short = short_name(d.title)
        draft = "征求意见稿" in d.title
        pian = d.level
        region = PIAN_REGION.get(re.sub(r"\D.*", "", pian.replace("第", "", 1)), "国家")
        date = opening_date(lines)
        head = f"{pian}｜{d.policy}"
        rows: list[tuple[str, str, int | None, list[str], list[str]]] = []      # (kind, 正文, 条号, 章节路径, 锚点)
        if n_art >= 5:
            blocks, rest = _blocks_article(lines)
            for b in blocks:
                body = "\n".join(_line_text(ln) for ln in b.lines).strip()
                if b.kind == "article":
                    for part in _split_article(body):
                        rows.append(("article", part, article_number(b.title), b.path, sorted({ln.anchor for ln in b.lines})))
                elif len(body) >= 100 or "。" in body:
                    rows.append(("preamble", body, None, b.path, sorted({ln.anchor for ln in b.lines})))
            for body, anc in _accumulate(rest):
                rows.append(("appendix", body, None, [], anc))
        else:
            for body, anc in _accumulate(lines):
                rows.append(("body", body, None, [], anc))
        f_chunks = []
        for seq, (kind, body, art, path_, anc) in enumerate(rows, 1):
            label = ART_RE.match(body).group(0) if kind == "article" and ART_RE.match(body) else ""
            if kind == "article" and not label:                                    # 拆开的后续片：沿用上一片的条号标签
                label = f_chunks[-1]["article_label"] if f_chunks else ""
            prefix = "｜".join([head, *path_, *([label] if label and not body.startswith(label) else [])])
            f_chunks.append({"chunk_id": f"pol:{d.doc_no}:{seq:03d}", "doc_no": d.doc_no, "pian": pian, "region": region,
                             "policy": d.policy, "short": short, "article": art, "article_label": label,
                             "path": path_, "kind": kind, "date": date, "draft": draft, "anchors": anc,
                             "body": body, "text": prefix + "\n" + body})
        chunks.extend(f_chunks)
        files.append({"doc_no": d.doc_no, "title": d.title, "policy": d.policy, "short": short, "pian": pian, "region": region,
                      "date": date, "draft": draft, "doc_type": "article" if n_art >= 5 else "other",
                      "n_chunks": len(f_chunks), "n_articles": sum(c["kind"] == "article" for c in f_chunks)})
    return {"source_sha256": sha, "files": files, "chunks": chunks}


def corpus_stats(corpus: dict) -> dict:
    ch, fl = corpus["chunks"], corpus["files"]
    lens = sorted(len(c["body"]) for c in ch)
    return {"files": len(fl), "chunks": len(ch), "article_chunks": sum(c["kind"] == "article" for c in ch),
            "chunks_by_region": dict(Counter(c["region"] for c in ch)), "files_by_region": dict(Counter(f["region"] for f in fl)),
            "chunks_by_kind": dict(Counter(c["kind"] for c in ch)), "draft_files": sum(f["draft"] for f in fl),
            "files_with_date": sum(bool(f["date"]) for f in fl),
            "chunk_len": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]} if lens else {},
            "duplicate_chunk_ids": len(ch) - len({c["chunk_id"] for c in ch}), "source_sha256": corpus["source_sha256"]}
