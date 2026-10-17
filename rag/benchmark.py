"""data_benchmark 切分：把专家问答 CSV 清洗、按用户分组切分为 train/test。

规则来自 `docs/data_benchmark_数据评估报告.md`：
1. 清洗：去掉答复尾部 `...`、去 BOM、去重；
2. 过滤：提问 < 5 字、答复 < 20 字视为无效样本；
3. **按用户分组**切分——同一用户的全部问答不跨池，避免相似问题泄漏；
4. 分层：在「用户」粒度上按主题分层抽样，使 test 的主题分布接近整体；
5. 固定随机种子，保证可复现。

`data_benchmark/` 缺失时所有函数优雅降级（返回空/不报错），便于在无数据环境跑测试。
"""
from __future__ import annotations

import csv
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import (
    BENCHMARK_DIR,
    BENCHMARK_SEED,
    MAX_SOURCE_SHARE,
    MAX_TEST_RATIO,
    PROBE_SIZE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

QUESTION_COL = "用户提问"
ANSWER_COL = "智慧员工答复"
USER_COL = "来源姓名"
TOPIC_COL = "主题"
EMP_COL = "智慧员工"

MIN_QUESTION_CHARS = 5
MIN_ANSWER_CHARS = 20

# 答复首/尾的导出残留：`...` / `……` / `。。。`
_LEADING_DOTS = re.compile(r"^[.。·…]{2,}\s*")
_TRAILING_DOTS = re.compile(r"\s*[.。·…]{2,}$")


@dataclass
class QAItem:
    id: str
    question: str
    answer: str
    user: str = ""
    topic: str = ""
    employee: str = ""
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 读取与清洗

def list_csv_files(bench_dir: Path | None = None) -> list[Path]:
    d = Path(bench_dir or BENCHMARK_DIR)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.csv"))


def clean_answer(text: str) -> str:
    """去掉首尾导出残留的点号串与 BOM/空白（报告称 15% 的答复以 `...` 开头）。"""
    text = text.lstrip("\ufeff").strip()
    text = _LEADING_DOTS.sub("", text)
    text = _TRAILING_DOTS.sub("", text)
    return text.strip()


def load_raw(bench_dir: Path | None = None) -> list[dict]:
    """读取所有 CSV 为字典列表；文件缺失/无表头时返回 []。"""
    rows: list[dict] = []
    for path in list_csv_files(bench_dir):
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or QUESTION_COL not in reader.fieldnames:
                    logger.warning("跳过 %s：缺少字段 %s", path.name, QUESTION_COL)
                    continue
                for r in reader:
                    r["_source_file"] = path.name
                    rows.append(r)
        except Exception as e:
            logger.warning("读取 %s 失败：%s", path.name, e)
    return rows


def build_items(raw: list[dict]) -> list[QAItem]:
    """清洗 + 过滤 + 去重，返回 QAItem 列表。"""
    items: list[QAItem] = []
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(raw):
        q = (r.get(QUESTION_COL) or "").strip()
        a = clean_answer(r.get(ANSWER_COL) or "")
        if len(q) < MIN_QUESTION_CHARS or len(a) < MIN_ANSWER_CHARS:
            continue
        key = (q, a)
        if key in seen:  # 完全重复的问答对只保留一条
            continue
        seen.add(key)
        items.append(
            QAItem(
                id=f"bm{i:05d}",
                question=q,
                answer=a,
                user=(r.get(USER_COL) or "").strip(),
                topic=(r.get(TOPIC_COL) or "").strip(),
                employee=(r.get(EMP_COL) or "").strip(),
                meta={"source_file": r.get("_source_file", "")},
            )
        )
    return items


# ---------------------------------------------------------------- 切分

def _group_key(it: QAItem) -> str:
    """分组键：来源姓名（提问专家）；为空的行按自身 id 视为独立来源。"""
    return it.user or f"__anon__{it.id}"


def _topic_buckets(by_user: dict[str, list[QAItem]]) -> dict[str, list[str]]:
    """按来源的主主题（出现最多的主题）归桶。"""
    buckets: dict[str, list[str]] = {}
    for user, rows in by_user.items():
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.topic] = counts.get(r.topic, 0) + 1
        main_topic = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        buckets.setdefault(main_topic, []).append(user)
    return buckets


def _spread_topics(items: list[QAItem], k: int) -> set[str]:
    """种子评估集：按主题分桶轮流取样，使 k 条问题覆盖尽可能多的主题。

    主题在原始数据里高度分散（每个来源常对应一个独立研究主题），单纯按来源
    抽两户会让评估集只覆盖两个主题。这里在「问题」粒度上铺开，仅用于评估
    检索覆盖度，不参与切分。
    """
    by_topic: dict[str, list[QAItem]] = {}
    for it in items:
        by_topic.setdefault(it.topic, []).append(it)
    picked: set[str] = set()
    queues = [sorted(rows, key=lambda r: r.id) for _, rows in sorted(by_topic.items())]
    i = 0
    while len(picked) < k and any(i < len(q) for q in queues):
        for q in queues:
            if i < len(q) and len(picked) < k:
                picked.add(q[i].id)
        i += 1
    return picked


def split_by_user(
    items: list[QAItem],
    test_size: float = TEST_SIZE,
    seed: int = BENCHMARK_SEED,
    max_test_ratio: float = MAX_TEST_RATIO,
    max_source_share: float = MAX_SOURCE_SHARE,
) -> tuple[list[QAItem], list[QAItem]]:
    """按来源（CSV「来源姓名」，即提问专家）分组分层切分，返回 (train, test)。

    同一来源的问答不跨池，避免相似问题泄漏到测试集。由于来源是整体进出，
    单个来源的体量就是测试集容量的最小粒度：数据里最大的来源占 16.9%，
    远高于 8% 的目标，任何包含它的测试集都会超出 ratio 上限。因此：
    1. 体量 > max_source_share 的来源整户留在 train（不进测试集），
       这部分仅用于训练，不参与评测；
    2. 其余来源按主主题分层、在桶内按行数配额抽样；
    3. 抽样后按行数从大到小填充，直到进入 [test_size, max_test_ratio] 区间
       （区间内即可，不必精确等于 test_size）；
    4. 兜底：至少保留一个来源，测试集永不为空。
    """
    total = len(items)
    if not total:
        return [], []

    rng = random.Random(seed)
    by_user: dict[str, list[QAItem]] = {}
    for it in items:
        by_user.setdefault(_group_key(it), []).append(it)

    # 1. 体量超限的来源不进测试集
    eligible = {u: rows for u, rows in by_user.items() if len(rows) <= max_source_share * total}
    if not eligible:  # 极端情况：所有来源都超限，退回全体
        eligible = dict(by_user)

    lo = total * test_size
    hi = total * max_test_ratio

    # 2. 主题分层配额（行数）：桶内来源按行数降序，先放体量大的以尽快接近下限
    buckets = _topic_buckets(eligible)
    test_users: set[str] = set()
    for topic in sorted(buckets):
        users = sorted(buckets[topic], key=lambda u: (-len(eligible[u]), u))
        topic_rows = sum(len(eligible[u]) for u in users)
        quota = topic_rows / total * lo
        picked, acc = [], 0
        for u in users:
            if acc >= quota or acc + len(eligible[u]) > hi:
                continue
            picked.append(u)
            acc += len(eligible[u])
        if not picked and users and len(eligible[users[0]]) <= hi:
            picked = [users[0]]  # 该主题至少给一个（若放得下）
        test_users.update(picked)

    # 3. 未达下限则继续填充：先试大来源（尽快进入区间），再试小来源（补足下限）
    remaining = sorted(set(eligible) - test_users, key=lambda u: (-len(eligible[u]), u))
    chosen = sum(len(eligible[u]) for u in test_users)

    def _fill(cands: list[str], chosen: int) -> int:
        for u in cands:
            if chosen >= lo:
                break
            if u in test_users or chosen + len(eligible[u]) > hi:
                continue
            test_users.add(u)
            chosen += len(eligible[u])
        return chosen

    chosen = _fill(remaining, chosen)
    if chosen < lo:
        chosen = _fill(remaining[::-1], chosen)
    # 越界则回收（优先收回最小的，保留大来源以减少来源数）
    while chosen > hi and len(test_users) > 1:
        u = min(test_users, key=lambda x: (len(eligible[x]), x))
        test_users.discard(u)
        chosen -= len(eligible[u])

    # 4. 兜底：非空
    if not test_users:
        test_users.add(min(eligible, key=lambda u: (len(eligible[u]), u)))

    train = [it for it in items if _group_key(it) not in test_users]
    test = [it for it in items if _group_key(it) in test_users]
    return train, test


# ---------------------------------------------------------------- 持久化

def write_jsonl(items: list[QAItem], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[QAItem]:
    import json

    items: list[QAItem] = []
    if not Path(path).exists():
        return items
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(QAItem(**json.loads(line)))
    return items


def build_split(
    bench_dir: Path | None = None,
    out_dir: Path | None = None,
    probe_size: int = PROBE_SIZE,
) -> dict:
    """一站式：读取 -> 清洗 -> 切分 -> 落盘。返回统计摘要。

    额外产出 `probe.jsonl`：不参与切分、按主题铺开的 k 条问题，用于检查
    「检索语料」对 data_benchmark 主题的覆盖度（避免主测试集只覆盖少数主题）。
    """
    bench_dir = Path(bench_dir or BENCHMARK_DIR)
    out_dir = Path(out_dir or bench_dir / "split")
    raw = load_raw(bench_dir)
    if not raw:
        return {"ok": False, "reason": f"未找到 CSV（目录：{bench_dir}）", "n_raw": 0}

    items = build_items(raw)
    train, test = split_by_user(items)
    probe = _spread_topics(items, probe_size) if probe_size else set()
    probe_items = [it for it in items if it.id in probe]
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    write_jsonl(probe_items, out_dir / "probe.jsonl")

    total = len(items)
    source_rows: dict[str, int] = {}
    for it in items:
        k = _group_key(it)
        source_rows[k] = source_rows.get(k, 0) + 1
    excluded = sorted(
        (u for u, n in source_rows.items() if total and n > MAX_SOURCE_SHARE * total),
        key=lambda u: -source_rows[u],
    )

    summary = {
        "ok": True,
        "n_raw": len(raw),
        "n_clean": len(items),
        "n_train": len(train),
        "n_test": len(test),
        "test_ratio": round(len(test) / total, 4) if total else 0.0,
        "ratio_target": f"{TEST_SIZE:.2f}~{MAX_TEST_RATIO:.2f}",
        "sources_total": len(source_rows),
        "sources_test": len({_group_key(it) for it in test}),
        "excluded_sources": len(excluded),
        "excluded_detail": [f"{u}({source_rows[u]})" for u in excluded],
        "n_probe": len(probe_items),
        "probe_topics": len({it.topic for it in probe_items}),
        "out_dir": str(out_dir),
    }
    return summary
