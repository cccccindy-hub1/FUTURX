"""data_benchmark 切分与清洗测试：用户不跨池、分层、可复现、缺数据降级。"""
from __future__ import annotations

import csv

import pytest

from rag.benchmark import (
    BENCHMARK_DIR,
    QAItem,
    build_items,
    build_split,
    clean_answer,
    load_raw,
    read_jsonl,
    split_by_user,
)

HEADER = ["统计周期", "智慧员工", "对话时间", "来源姓名", "来源部门", "省", "市", "主题", "用户提问", "智慧员工答复"]


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)


@pytest.fixture
def bench_dir(tmp_path):
    d = tmp_path / "data_benchmark"
    d.mkdir()
    rows = []
    # 20 个用户，每个 5 条；前 3 个用户是“大户”，主题各不相同
    for u in range(20):
        for i in range(5):
            rows.append([
                "2025-10", "AI李老师", "2025-10-01 10:00", f"user_{u:02d}", "", "江苏", "南京",
                f"主题{u % 3}", f"这是第{u}个用户提出的一个足够长的问题{i}？",
                f"这是专家给出的足够长的答复内容，用于测试清洗与切分逻辑，编号{u}-{i}。" * 2,
            ])
    # 无效样本：短问题、短答复、重复行、带 ... 前缀
    rows.append(["2025-10", "AI李老师", "", "user_00", "", "", "", "主题0", "谢谢", "足够长的答复内容" * 3])
    rows.append(["2025-10", "AI李老师", "", "user_00", "", "", "", "主题0", "这是一个足够长的问题？", "太短"])
    _write_csv(d / "a.csv", rows)
    return d


def test_clean_answer_strips_trailing_dots():
    assert clean_answer("...这是答复内容...") == "这是答复内容"
    assert clean_answer("答复内容……") == "答复内容"
    assert clean_answer("正常答复") == "正常答复"


def test_load_raw_and_build_items_filters_invalid(bench_dir, monkeypatch):
    monkeypatch.setattr("rag.benchmark.BENCHMARK_DIR", bench_dir)
    raw = load_raw(bench_dir)
    assert len(raw) == 102  # 100 + 2
    items = build_items(raw)
    # 短问题、短答复各剔除 1 条
    assert len(items) == 100
    for it in items:
        assert len(it.question) >= 5
        assert len(it.answer) >= 20
        assert not it.answer.endswith("...")
    assert len({it.id for it in items}) == len(items), "id 必须唯一"


def test_build_items_dedupes_identical_pairs():
    r = {"用户提问": "一个足够长的问题？", "智慧员工答复": "一条足够长的答复内容" * 2}
    items = build_items([dict(r), dict(r)])
    assert len(items) == 1


def test_split_keeps_users_on_one_side(bench_dir):
    items = build_items(load_raw(bench_dir))
    train, test = split_by_user(items, test_size=0.2, seed=42, max_source_share=1.0)
    assert train and test
    train_users = {it.user for it in train}
    test_users = {it.user for it in test}
    assert not (train_users & test_users), "同一用户不得跨 train/test"
    assert len(train) + len(test) == len(items)


def test_split_is_reproducible(bench_dir):
    items = build_items(load_raw(bench_dir))
    a = split_by_user(items, test_size=0.2, seed=7, max_source_share=1.0)
    b = split_by_user(items, test_size=0.2, seed=7, max_source_share=1.0)
    assert [i.id for i in a[1]] == [i.id for i in b[1]]


def test_split_test_ratio_near_target(bench_dir):
    items = build_items(load_raw(bench_dir))
    _, test = split_by_user(items, test_size=0.2, seed=42, max_source_share=1.0)
    ratio = len(test) / len(items)
    assert 0.1 <= ratio <= 0.35, f"测试集占比偏离目标过大：{ratio:.2f}"


def _items_with_big_source(n_big: int = 300, n_small_users: int = 200, per_small: int = 1, n_topics: int = 1):
    items = [
        QAItem(id=f"b{i}", question="q" * 10, answer="a" * 30, user="big", topic="t0")
        for i in range(n_big)
    ]
    for u in range(n_small_users):
        for i in range(per_small):
            items.append(
                QAItem(
                    id=f"u{u}_{i}",
                    question="q" * 10,
                    answer="a" * 30,
                    user=f"u{u}",
                    topic=f"t{i % n_topics}",
                )
            )
    return items


def test_split_excludes_oversized_source():
    """体量超过 max_source_share 的来源整户留在 train，测试集比例因此可控。"""
    items = _items_with_big_source()
    train, test = split_by_user(items, test_size=0.08, seed=42, max_source_share=0.06)
    assert "big" not in {it.user for it in test}
    assert "big" in {it.user for it in train}
    assert not ({it.user for it in train} & {it.user for it in test})
    assert 0.08 - 1e-9 <= len(test) / len(items) <= 0.10 + 1e-9


def test_split_never_returns_empty_test():
    """候选池里所有来源都超限时也不能把测试集清空。"""
    items = [
        QAItem(id=f"a{i}", question="q" * 10, answer="a" * 30, user="a", topic="t")
        for i in range(100)
    ]
    items += [
        QAItem(id=f"b{i}", question="q" * 10, answer="a" * 30, user="b", topic="t")
        for i in range(100)
    ]
    _, test = split_by_user(items, test_size=0.08, seed=42, max_source_share=0.06)
    assert test


def test_split_fills_ratio_from_small_sources():
    """只有小来源时，比例必须落进 [test_size, max_test_ratio]。"""
    items = _items_with_big_source(n_big=0, n_small_users=300, per_small=3)
    _, test = split_by_user(items, test_size=0.08, seed=42, max_source_share=0.06)
    assert 0.08 - 1e-9 <= len(test) / len(items) <= 0.10 + 1e-9


def test_probe_spreads_over_topics():
    from rag.benchmark import _spread_topics

    items = [
        QAItem(id=f"u{u}_{i}", question="q" * 10, answer="a" * 30, user=f"u{u}", topic=f"t{(u + i) % 10}")
        for u in range(60)
        for i in range(1)
    ]
    picked = _spread_topics(items, 30)
    topics = {it.topic for it in items if it.id in picked}
    assert len(picked) == 30
    assert len(topics) == 10, "种子评估集应尽量铺开到全部主题"


def test_build_split_writes_jsonl(bench_dir, tmp_path):
    summary = build_split(bench_dir=bench_dir, out_dir=tmp_path / "split", probe_size=10)
    assert summary["ok"]
    train = read_jsonl(tmp_path / "split" / "train.jsonl")
    test = read_jsonl(tmp_path / "split" / "test.jsonl")
    assert len(train) + len(test) == summary["n_clean"]
    assert {it.user for it in train} & {it.user for it in test} == set()
    probe = read_jsonl(tmp_path / "split" / "probe.jsonl")
    assert len(probe) == 10
    assert summary["n_probe"] == 10


def test_build_split_missing_dir_degrades(tmp_path):
    summary = build_split(bench_dir=tmp_path / "nope")
    assert summary["ok"] is False
    assert "reason" in summary


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_qa_item_roundtrip(tmp_path):
    from rag.benchmark import write_jsonl

    item = QAItem(id="x1", question="q?", answer="a" * 30, user="u", topic="t")
    path = tmp_path / "t.jsonl"
    write_jsonl([item], path)
    assert read_jsonl(path)[0].question == "q?"
