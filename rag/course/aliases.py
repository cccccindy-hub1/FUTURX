"""别名表：概念名称 + 专家原词 + 英文括注 → 概念ID，泛词标记为 blocked。

- 专家原词用全角分号「；」分隔，「XX的定义」是对定义的标注而不是别名，排除；
- 名称中的括注：含英文字母的作为英文别名（「数据要素（Data Element）」→ 中文名 + 英文名），
  纯中文括注（如「（简称…）」）不当别名，只在主名里去掉；
- 泛词（「数据」「技术」等）与单字、过短纯英文只登记、不参与匹配（status=blocked），供人工复核；
- 同一表面形式对应多个概念ID（跨章同名）时 ambiguous=True，匹配时返回全部候选，不擅自选一个。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .ontology import Concept

EXPERT_SEP_RE = re.compile(r"[；;]")
_DEFINITION_SUFFIX_RE = re.compile(r"的定义\s*$")
_PAREN_RE = re.compile(r"[（(]([^（）()]*)[）)]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_HAN_RE = re.compile(r"[一-鿿]")

# 默认泛词：单独出现几乎不指向某个具体概念。可按需追加，改动后重新建表即可。
GENERIC_TERMS: frozenset[str] = frozenset(
    {
        "数据", "技术", "信息", "系统", "管理", "方法", "模式", "机制", "平台", "服务", "应用", "业务",
        "企业", "政府", "市场", "价值", "安全", "风险", "资源", "产品", "资产", "要素", "权利", "规则",
        "制度", "标准", "国家", "社会", "经济", "个人", "组织", "主体", "活动", "过程", "能力", "主体",
        "对象", "内容", "类型", "形式", "功能", "目标", "原则", "环节", "行为", "责任", "政策", "法律",
    }
)
MIN_ASCII_LEN = 3  # 纯英文别名短于此（如 "AI"）视为泛词

# 别名来源
SRC_NAME = "name"
SRC_EXPERT = "expert"
SRC_ENGLISH = "english"

# 默认不为这些问题类型的概念建别名（非概念噪声、已决定删去）；传 exclude_issue_types=() 可关闭
DEFAULT_EXCLUDE_ISSUES = ("D 非概念噪声", "删去")


def norm(text: str) -> str:
    """别名/文本的匹配规范形：NFKC、小写、去掉全部空白。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).casefold()


def split_expert_terms(raw: str) -> list[str]:
    """拆分专家原词，去重保序，排除空项与「XX的定义」。"""
    out: list[str] = []
    for part in EXPERT_SEP_RE.split(raw or ""):
        t = part.strip()
        if not t or _DEFINITION_SUFFIX_RE.search(t):
            continue
        if t not in out:
            out.append(t)
    return out


def split_name(name: str) -> tuple[str, list[str]]:
    """「数据要素（Data Element）」→ ("数据要素", ["Data Element"])。"""
    english: list[str] = []

    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        if inner and _ASCII_ALPHA_RE.search(inner) and not _HAN_RE.search(inner):
            english.append(inner)
        return ""

    base = _PAREN_RE.sub(repl, name or "").strip()
    return (base or (name or "").strip()), english


def blocked_reason(surface: str, generic: frozenset[str] = GENERIC_TERMS) -> str | None:
    key = norm(surface)
    if not key:
        return "empty"
    if key in {norm(g) for g in generic}:
        return "generic"
    if len(key) == 1:
        return "single_char"
    if key.isascii() and len(key) < MIN_ASCII_LEN:
        return "short_ascii"
    return None


@dataclass
class Alias:
    surface: str                         # 代表性原写法（首次出现）
    key: str                             # norm(surface)
    concept_ids: list[str] = field(default_factory=list)
    sources: dict[str, set[str]] = field(default_factory=dict)  # concept_id -> {name, expert, english}
    blocked: str | None = None           # None 表示可匹配，否则为原因

    @property
    def ambiguous(self) -> bool:
        return len(self.concept_ids) > 1


class AliasTable:
    def __init__(self, generic: frozenset[str] = GENERIC_TERMS):
        self._generic = generic
        self.by_key: dict[str, Alias] = {}
        self.by_concept: dict[str, list[str]] = {}   # concept_id -> [key...]

    def add(self, surface: str, concept_id: str, source: str) -> None:
        surface = surface.strip()
        key = norm(surface)
        if not key:
            return
        a = self.by_key.get(key)
        if a is None:
            a = self.by_key[key] = Alias(surface, key, blocked=blocked_reason(surface, self._generic))
        if concept_id not in a.concept_ids:
            a.concept_ids.append(concept_id)
        a.sources.setdefault(concept_id, set()).add(source)
        keys = self.by_concept.setdefault(concept_id, [])
        if key not in keys:
            keys.append(key)

    # ---- 查询 ----
    def active(self) -> list[Alias]:
        return [a for a in self.by_key.values() if a.blocked is None]

    def blocked(self) -> list[Alias]:
        return [a for a in self.by_key.values() if a.blocked is not None]

    def lookup(self, surface: str, include_blocked: bool = False) -> list[str]:
        a = self.by_key.get(norm(surface))
        if a is None or (a.blocked and not include_blocked):
            return []
        return list(a.concept_ids)

    def aliases_of(self, concept_id: str, include_blocked: bool = False) -> list[Alias]:
        out = [self.by_key[k] for k in self.by_concept.get(concept_id, [])]
        return out if include_blocked else [a for a in out if a.blocked is None]

    # ---- 文本提及 ----
    def count_mentions(self, text: str) -> dict[str, int]:
        """在 text 中统计各概念的提及次数（可匹配别名的出现次数；同形多概念时每个候选各计一次）。"""
        hay = norm(text)
        counts: dict[str, int] = {}
        for a in self.active():
            n = hay.count(a.key)
            if n:
                for cid in a.concept_ids:
                    counts[cid] = counts.get(cid, 0) + n
        return counts

    def find_aliases(self, text: str) -> list[Alias]:
        """text 中出现的可匹配别名（不做最长匹配消歧，供上层决定）。"""
        hay = norm(text)
        return [a for a in self.active() if a.key in hay]


def build_alias_table(
    concepts: list[Concept],
    generic: frozenset[str] = GENERIC_TERMS,
    exclude_issue_types: tuple[str, ...] = DEFAULT_EXCLUDE_ISSUES,
) -> AliasTable:
    table = AliasTable(generic)
    for c in concepts:
        if c.issue_type in exclude_issue_types:
            continue
        base, english = split_name(c.name)
        table.add(base, c.id, SRC_NAME)
        if c.name.strip() != base:
            table.add(c.name, c.id, SRC_NAME)  # 保留带括注的原写法
        for e in english:
            table.add(e, c.id, SRC_ENGLISH)
        for term in split_expert_terms(c.expert_terms):
            base_t, english_t = split_name(term)
            table.add(base_t, c.id, SRC_EXPERT)
            for e in english_t:
                table.add(e, c.id, SRC_ENGLISH)
    return table
