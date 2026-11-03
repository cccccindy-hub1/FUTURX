"""文档解析缓存：按源文件 mtime + 大小判断有效性；docx 缺失返回 None；缓存损坏 / 写不进去都不影响使用。"""
from __future__ import annotations

import json
import os

from rag.docscache import CACHE_VERSION, cache_path, fingerprint, load_or_build


def make_doc(tmp_path, content=b"hello"):
    p = tmp_path / "资料.docx"
    p.write_bytes(content)
    return p


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self, doc):
        self.n += 1
        return {"size": doc.stat().st_size, "items": ["甲", "乙"]}


def test_first_call_parses_and_writes_cache_second_call_reads_cache(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    data, info = load_or_build(doc, "cases", b)
    assert b.n == 1 and info["source"] == "parsed" and cache_path(doc, "cases").is_file() and data["items"] == ["甲", "乙"]
    data2, info2 = load_or_build(doc, "cases", b)
    assert b.n == 1 and info2["source"] == "cache" and data2 == data                       # 没有重新解析


def test_cache_invalidated_when_size_changes(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    load_or_build(doc, "cases", b)
    doc.write_bytes(b"hello world, now longer")
    data, info = load_or_build(doc, "cases", b)
    assert b.n == 2 and info["source"] == "parsed" and data["size"] == doc.stat().st_size


def test_cache_invalidated_when_only_mtime_changes_with_same_size(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    load_or_build(doc, "cases", b)
    st = doc.stat()
    os.utime(doc, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))                      # 大小不变，只改 mtime
    assert fingerprint(doc)["size"] == st.st_size
    _, info = load_or_build(doc, "cases", b)
    assert b.n == 2 and info["source"] == "parsed"


def test_cache_invalidated_when_parser_version_changes(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    load_or_build(doc, "cases", b, version=1)
    load_or_build(doc, "cases", b, version=2)
    assert b.n == 2


def test_missing_docx_returns_none_without_calling_builder(tmp_path):
    b = Counter()
    assert load_or_build(tmp_path / "nope.docx", "cases", b) is None and load_or_build(None, "cases", b) is None and b.n == 0


def test_corrupted_cache_falls_back_to_reparse_and_heals(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    load_or_build(doc, "quotes", b)
    cache_path(doc, "quotes").write_text("{坏掉的 json", encoding="utf-8")
    data, info = load_or_build(doc, "quotes", b)
    assert info["source"] == "parsed" and b.n == 2 and data["items"]
    assert load_or_build(doc, "quotes", b)[1]["source"] == "cache"                          # 下次又能用缓存了


def test_unwritable_cache_dir_does_not_break_loading(tmp_path, monkeypatch):
    doc, b = make_doc(tmp_path), Counter()
    import rag.docscache as dc
    monkeypatch.setattr(dc.os, "replace", lambda *a, **k: (_ for _ in ()).throw(PermissionError("ro")))
    data, info = load_or_build(doc, "policy", b)
    assert data["items"] and info["source"] == "parsed" and info["cache_write_error"] == "PermissionError"


def test_each_library_has_its_own_cache_file_and_cache_lives_beside_the_docx(tmp_path):
    doc = make_doc(tmp_path)
    for kind in ("cases", "quotes", "policy"):
        load_or_build(doc, kind, Counter())
    names = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".cache.json"))
    assert names == ["资料.cases.cache.json", "资料.policy.cache.json", "资料.quotes.cache.json"]
    assert json.loads((tmp_path / "资料.cases.cache.json").read_text(encoding="utf-8"))["cache_version"] == CACHE_VERSION


def test_use_cache_false_always_parses_and_writes_nothing(tmp_path):
    doc, b = make_doc(tmp_path), Counter()
    load_or_build(doc, "cases", b, use_cache=False)
    load_or_build(doc, "cases", b, use_cache=False)
    assert b.n == 2 and not cache_path(doc, "cases").exists()
