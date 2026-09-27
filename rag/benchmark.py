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

from .config import BENCHMARK_DIR, BENCHMARK_SEED, TEST_SIZE

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

def split_by_user(
    items: list[QAItem],
    test_size: float = TEST_SIZE,
    seed: int = BENCHMARK_SEED,
) -> tuple[list[QAItem], list[QAItem]]:
    """按用户分组分层切分，返回 (train, test)。

    - 用户为空的行按「自身 id」视为独立用户，保证每条都能被切分。
    - 分层：按该用户的主主题（出现最多的主题）归桶，再在各桶内按用户抽样，
      使 test 的主题分布接近整体，且同一用户不跨池。
    """
    rng = random.Random(seed)
    by_user: dict[str, list[QAItem]] = {}
    for it in items:
        by_user.setdefault(it.user or f"__anon__{it.id}", []).append(it)

    # 每个用户的主主题
    buckets: dict[str, list[str]] = {}
    for user, rows in by_user.items():
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.topic] = counts.get(r.topic, 0) + 1
        main_topic = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        buckets.setdefault(main_topic, []).append(user)

    target = max(1, round(len(items) * test_size)) if items else 0
    total = len(items)

    test_users: set[str] = set()
    chosen = 0
    for topic, users in sorted(buckets.items()):
        # 该主题应抽中的“行数”配额，再换算成用户数（按该主题平均每用户行数）
        topic_rows = sum(len(by_user[u]) for u in users)
        row_quota = round(topic_rows / total * target) if total else 0
        per_user = topic_rows / len(users)
        user_quota = min(round(row_quota / per_user), len(users)) if per_user else 0
        user_quota = max(0, user_quota)
        picked = rng.sample(sorted(users), user_quota)
        test_users.update(picked)
        chosen += sum(len(by_user[u]) for u in picked)

    # 抽样不足则从剩余用户中随机补齐
    if chosen < target:
        remaining = sorted(u for u in by_user if u not in test_users)
        rng.shuffle(remaining)
        for u in remaining:
            if chosen >= target:
                break
            test_users.add(u)
            chosen += len(by_user[u])

    train = [it for it in items if (it.user or f"__anon__{it.id}") not in test_users]
    test = [it for it in items if (it.user or f"__anon__{it.id}") in test_users]
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


def build_split(bench_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """一站式：读取 -> 清洗 -> 切分 -> 落盘。返回统计摘要。"""
    bench_dir = Path(bench_dir or BENCHMARK_DIR)
    out_dir = Path(out_dir or bench_dir / "split")
    raw = load_raw(bench_dir)
    if not raw:
        return {"ok": False, "reason": f"未找到 CSV（目录：{bench_dir}）", "n_raw": 0}

    items = build_items(raw)
    train, test = split_by_user(items)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    summary = {
        "ok": True,
        "n_raw": len(raw),
        "n_clean": len(items),
        "n_train": len(train),
        "n_test": len(test),
        "test_ratio": round(len(test) / len(items), 4) if items else 0.0,
        "users_total": len({it.user or it.id for it in items}),
        "users_test": len({it.user or it.id for it in test}),
        "out_dir": str(out_dir),
    }
    return summary
