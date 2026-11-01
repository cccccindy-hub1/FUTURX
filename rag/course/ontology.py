"""本体数据加载（只读）：概念总表（xlsx）、onto1/onto2 结构化导入 CSV、学生知识集合（核对用）。

约定（均来自 9.11 数据实况）：
- CSV 为 UTF-8 with BOM，每章一个文件，两个视角表头相同；confidence / expert_relevance_weight 是字符串；
- 概念总表第 2 行是「通过核验」占位行，只保留概念ID 匹配 C\\d\\d-\\d{4} 的行；
- 概念总表的「教材定位」指向旧版分章文件的段号，不是书稿 0906 的 P####，这里原样保存、不参与锚点核对；
- 不用 pandas（环境里 numpy 版本冲突），只用 csv + openpyxl。
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

CONCEPT_ID_RE = re.compile(r"^C\d\d-\d{4}$")
_PARA_ANCHOR_RE = re.compile(r"^P(\d{1,6})$")
_EXPERT_CELL_RE = re.compile(r"^onto\d+!", re.IGNORECASE)

CONCEPT_XLSX = "第1-10章_concept总表_问题标注版.xlsx"
CONCEPT_SHEET = "concept总表"
STUDENT_XLSX = Path("9.11update") / "三层集合" / "学生知识集合.xlsx"
VIEWS = ("onto1", "onto2")

# 概念总表表头 -> Concept 属性
_CONCEPT_COLS = {
    "概念ID": "id", "章号": "chapter", "章节主题": "topic", "概念名称": "name", "概念类型": "ctype",
    "上位或关联项": "parent_hint", "释义或知识语境": "context", "收录依据": "basis",
    "教材定位": "textbook_loc", "教材原文摘录": "excerpt", "专家原词": "expert_terms",
    "专家定位": "expert_loc", "归并及版本说明": "merge_note", "收录分组": "group",
    "问题类型": "issue_type", "问题说明": "issue_note",
}

_RECORD_STR_FIELDS = (
    "record_id", "chapter_id", "profile", "subject_id", "subject_label", "subject_class_id",
    "predicate_id", "predicate_label", "object_id", "object_label", "object_class_id",
    "assertion_type", "source_file", "source_anchor", "retrieval_anchor", "source_section",
    "evidence_quote", "evidence_requirement", "logic_level", "flow_chain", "driving_mechanism",
    "review_status", "normalization_note",
)
_RECORD_REQUIRED = _RECORD_STR_FIELDS + ("confidence", "expert_relevance_weight")


@dataclass(frozen=True)
class Concept:
    id: str
    chapter: int
    topic: str
    name: str
    ctype: str
    parent_hint: str
    context: str
    basis: str
    textbook_loc: str
    excerpt: str
    expert_terms: str      # 原始字符串（全角分号分隔），拆分见 aliases.split_expert_terms
    expert_loc: str
    merge_note: str
    group: str
    issue_type: str
    issue_note: str


@dataclass
class ConceptTable:
    concepts: list[Concept]
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (Excel 行号, 概念ID 单元格值)

    @property
    def by_id(self) -> dict[str, Concept]:
        return {c.id: c for c in self.concepts}


@dataclass(frozen=True)
class Record:
    record_id: str
    chapter_id: str
    profile: str
    subject_id: str
    subject_label: str
    subject_class_id: str
    predicate_id: str
    predicate_label: str
    object_id: str
    object_label: str
    object_class_id: str
    assertion_type: str
    source_file: str
    source_anchor: str
    retrieval_anchor: str
    source_section: str
    evidence_quote: str
    evidence_requirement: str
    logic_level: str
    flow_chain: str
    driving_mechanism: str
    review_status: str
    normalization_note: str
    confidence: float | None
    expert_relevance_weight: float | None
    view: str = ""         # onto1 / onto2，取自所在目录

    @property
    def is_concept_edge(self) -> bool:
        return is_concept_id(self.subject_id) and is_concept_id(self.object_id)


def is_concept_id(value: str) -> bool:
    return bool(CONCEPT_ID_RE.match(value or ""))


_NODE_PATTERNS = (
    ("KP", re.compile(r"^Ch\d\d_KP_")),
    ("SEC", re.compile(r"^Ch\d\d_SEC_")),
    ("DEF", re.compile(r"^Ch\d\d_DEF_")),
    ("PolicyEntry", re.compile(r"^PolicyLibrary_Entry_")),
    ("Case", re.compile(r"^CaseLibrary_Case_")),
    ("Class", re.compile(r"^Class::")),
    ("Node", re.compile(r"^Ch\d\d_[A-Za-z]{1,4}_\w+")),   # 通用非概念节点：Ch09_N_<hash> 等
    ("Chapter", re.compile(r"^Ch\d\d_Chapter$")),          # 章节点
    ("Course", re.compile(r"^Course_\w+")),                # 课程节点：Course_DataElementCirculationIntro
)


def node_kind(node_id: str) -> str:
    """节点 ID 类别：concept | KP | SEC | DEF | PolicyEntry | Case | Class | Node | Chapter | Course | empty | other。

    KP/SEC/DEF 等具名前缀优先；Ch??_<字母>_<hash>（如 Ch09_N_xxx）归为通用非概念节点 Node；
    Ch??_Chapter 是章节点、Course_* 是课程节点（均为教材结构节点）。
    """
    nid = (node_id or "").strip()
    if not nid:
        return "empty"
    if is_concept_id(nid):
        return "concept"
    for kind, pat in _NODE_PATTERNS:
        if pat.match(nid):
            return kind
    return "other"


def classify_anchor(anchor: str) -> tuple[str, int | None]:
    """返回 (类别, 段落序号)。类别：paragraph | expert_cell | policy | case | concept | empty | other。"""
    a = (anchor or "").strip()
    if not a:
        return "empty", None
    m = _PARA_ANCHOR_RE.match(a)
    if m:
        return "paragraph", int(m.group(1))
    if _EXPERT_CELL_RE.match(a):
        return "expert_cell", None
    if a.startswith("PolicyEntry") or a.startswith("PolicyLibrary"):
        return "policy", None
    if a.startswith("CaseLibrary"):
        return "case", None
    if is_concept_id(a):
        return "concept", None
    return "other", None


def _cell(v) -> str:
    return "" if v is None else str(v).strip()


def _chapter_of(value, concept_id: str) -> int:
    m = re.search(r"\d+", _cell(value))
    return int(m.group()) if m else int(concept_id[1:3])


def _open_xlsx(path: Path):
    from openpyxl import load_workbook

    return load_workbook(path, read_only=True, data_only=True)


def load_concepts(path: str | Path) -> ConceptTable:
    """读取概念总表，跳过非 C##-#### 行（如「通过核验」占位行）。"""
    path = Path(path)
    wb = _open_xlsx(path)
    try:
        if CONCEPT_SHEET not in wb.sheetnames:
            raise ValueError(f"{path.name} 缺少 sheet「{CONCEPT_SHEET}」，现有：{wb.sheetnames}")
        rows = wb[CONCEPT_SHEET].iter_rows(values_only=True)
        header = [_cell(h) for h in next(rows, ())]
        missing = [h for h in _CONCEPT_COLS if h not in header]
        if missing:
            raise ValueError(f"概念总表缺少列：{missing}")
        col = {_CONCEPT_COLS[h]: header.index(h) for h in _CONCEPT_COLS}
        out: list[Concept] = []
        skipped: list[tuple[int, str]] = []
        for i, row in enumerate(rows, start=2):
            get = lambda attr: row[col[attr]] if col[attr] < len(row) else None  # noqa: E731
            cid = _cell(get("id"))
            if not is_concept_id(cid):
                if any(v not in (None, "") for v in row):
                    skipped.append((i, cid))
                continue
            vals = {a: _cell(get(a)) for a in col if a != "chapter"}
            out.append(Concept(chapter=_chapter_of(get("chapter"), cid), **vals))
        return ConceptTable(out, skipped)
    finally:
        wb.close()


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_records(csv_dir: str | Path, view: str) -> list[Record]:
    """读取一个视角目录下所有 Ch*_{view}_9.11.csv，按文件名顺序拼接。"""
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob(f"Ch*_{view}_9.11.csv"))
    if not files:
        raise FileNotFoundError(f"{csv_dir} 下没有 Ch*_{view}_9.11.csv")
    out: list[Record] = []
    for f in files:
        with f.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in _RECORD_REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{f.name} 缺少列：{missing}")
            for row in reader:
                strs = {k: (row.get(k) or "").strip() for k in _RECORD_STR_FIELDS}
                out.append(
                    Record(
                        **strs,
                        confidence=_to_float(row.get("confidence", "")),
                        expert_relevance_weight=_to_float(row.get("expert_relevance_weight", "")),
                        view=view,
                    )
                )
    return out


def load_student_set(path: str | Path) -> list[dict[str, str]]:
    """学生知识集合：表头不在第 1 行（前有标题/说明），按「集合ID」所在行定位。"""
    wb = _open_xlsx(Path(path))
    try:
        ws = wb.worksheets[0]
        header: list[str] | None = None
        out: list[dict[str, str]] = []
        for row in ws.iter_rows(values_only=True):
            cells = [_cell(v) for v in row]
            if header is None:
                if "集合ID" in cells:
                    header = cells
                continue
            if any(cells):
                out.append({h: cells[i] for i, h in enumerate(header) if h and i < len(cells)})
        if header is None:
            raise ValueError("学生知识集合中找不到表头行（含「集合ID」）")
        return out
    finally:
        wb.close()


@dataclass(frozen=True)
class OntologyPaths:
    root: Path

    @property
    def concept_xlsx(self) -> Path:
        return self.root / CONCEPT_XLSX

    def onto_dir(self, view: str) -> Path:
        return self.root / "9.11update" / view

    @property
    def student_xlsx(self) -> Path:
        return self.root / STUDENT_XLSX

    def find_textbook(self) -> Path | None:
        hits = sorted((self.root / "教材").glob("书稿*0906*.docx"))
        return hits[0] if hits else None
