"""名言库 docx 解析：按年份分组，再以「N.」开头的段落为界，切成一次讲话一条。

结构（见设计文档）：
- 目录部分：toc 1 =「2018年」，toc 2 =「1.全国网络安全和信息化工作会议上的讲话」→ 每次讲话的标题，按（年份, 序号）索引；
- 正文部分（「三、检索结果」之后）：Heading 1 =「2018年」；以「1.」「2.」开头的段落开始一次新讲话，后面不带编号的段落属于同一次讲话；
- 同一次讲话内的段落去重（保留首次出现，记录去掉了几段）；
- 讲话类型（讲话 / 贺信 / 指示 / 文章）按标题关键词判断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

NUM_RE = re.compile(r"^(\d+)\.(?!\d)\s*")           # 「1.」，但不是「1.5倍」之类的小数
YEAR_RE = re.compile(r"^(\d{4})\s*年")
TOC_RE = re.compile(r"^(\d+)\.\s*(.+?)\s*(?:\t\s*\d+)?\s*$")
RESULT_HEADING = "三、检索结果"

# 类型判断：按优先级，先匹配到的为准。「致…的贺信」「向…致贺信」「…开幕视频致贺」都是贺信。
_TYPE_RULES = (
    ("贺信", ("贺信", "致贺", "贺词", "致信", "回信")),
    ("指示", ("重要指示", "作出重要指示", "指示", "批示")),
    ("文章", ("习近平.", "习近平．", "文章", "署名", "新闻联播")),
    ("讲话", ("讲话", "演讲", "致辞", "发言", "主旨", "座谈会", "学习时", "会议上", "会议")),
)


@dataclass
class Speech:
    year: int
    n: int                                   # 该年内的序号（与目录一致）
    title: str
    kind: str                                # 讲话 / 贺信 / 指示 / 文章
    paragraphs: list[str] = field(default_factory=list)       # 去重后的段落
    removed_duplicates: list[str] = field(default_factory=list)   # 被去掉的重复段（前 40 字，便于核对）
    raw_paragraph_count: int = 0

    @property
    def speech_id(self) -> str:
        return f"quote:{self.year}:{self.n:02d}"

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


def classify_kind(title: str) -> str:
    for kind, keys in _TYPE_RULES:
        if any(k in title for k in keys):
            return kind
    return "讲话"


def _norm(p: str) -> str:
    return re.sub(r"\s+", "", NUM_RE.sub("", p, count=1))


def parse_toc(paragraphs) -> dict[tuple[int, int], str]:
    """目录 -> {(年份, 序号): 标题}。"""
    out: dict[tuple[int, int], str] = {}
    year = None
    for p in paragraphs:
        name, text = p.style.name, p.text.strip()
        if name == "toc 1":
            m = YEAR_RE.match(text)
            year = int(m.group(1)) if m else None
        elif name == "toc 2" and year:
            m = TOC_RE.match(text)
            if m:
                out[(year, int(m.group(1)))] = m.group(2).strip()
    return out


def parse_quotes(path: str | Path) -> list[Speech]:
    import docx
    doc = docx.Document(str(path))
    paras = doc.paragraphs
    toc = parse_toc(paras)
    start = next((i for i, p in enumerate(paras) if p.style.name == "Heading 1" and p.text.strip() == RESULT_HEADING), None)
    if start is None:
        raise ValueError(f"找不到「{RESULT_HEADING}」标题，无法定位正文")
    speeches: list[Speech] = []
    year = None
    cur: Speech | None = None
    for p in paras[start + 1:]:
        text = p.text.strip()
        if p.style.name == "Heading 1":
            m = YEAR_RE.match(text)
            year = int(m.group(1)) if m else year
            cur = None
            continue
        if not text or year is None:
            continue
        m = NUM_RE.match(text)
        if m:                                             # 「N.」开头 = 新的一次讲话
            n = int(m.group(1))
            title = toc.get((year, n))
            if title is None:
                raise ValueError(f"{year} 年第 {n} 条在目录里找不到标题")
            cur = Speech(year=year, n=n, title=title, kind=classify_kind(title))
            speeches.append(cur)
        elif cur is None:
            raise ValueError(f"{year} 年出现了不属于任何讲话的段落：{text[:30]}")
        cur.raw_paragraph_count += 1
        body = NUM_RE.sub("", text, count=1).strip() if m else text
        seen = {_norm(x) for x in cur.paragraphs}
        if _norm(body) in seen:                            # 同一次讲话内的重复段落
            cur.removed_duplicates.append(body[:40])
        else:
            cur.paragraphs.append(body)
    return speeches
