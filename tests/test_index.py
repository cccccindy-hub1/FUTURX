"""入库层 _parse 测试：解析 + 分块 + 主键/元数据（不加载模型）。"""
from __future__ import annotations

from rag import index


def test_parse_produces_unique_ids_and_metadata(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    assert files, "应收集到合成语料"

    rows, skipped, parents = index._parse(files)
    assert rows, "应解析出 chunk"
    assert skipped == []
    assert parents == {}, "默认（非父子分块）不应产生父块"

    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "chunk 主键必须唯一"

    rid, text, meta = rows[0]
    assert "::chunk" in rid
    assert text.strip()
    for key in ("source", "filename", "category", "section", "ext"):
        assert key in meta
    assert "parent_id" not in meta
    assert meta["category"] == "006/研究方法/扎根理论"


def test_parse_parent_child_mode(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    monkeypatch.setattr(index, "PARENT_CHILD_ENABLED", True)
    monkeypatch.setattr(index, "PARENT_SIZE", 120)

    rows, skipped, parents = index._parse(index._collect_files())
    assert rows and parents, "父子分块应同时产出子块与父块"
    assert skipped == []

    for rid, text, meta in rows:
        assert meta["parent_id"] in parents, "子块必须指向存在的父块"
        assert text.startswith(f"〔{meta['parent_id']}〕"), "子块应带父块 header 标记"
        parent = parents[meta["parent_id"]]
        off, length = meta["parent_offset"], meta["parent_length"]
        assert off >= 0
        # 父块保留段落换行、子块按行拼接，故忽略换行后应完全一致
        body = text.split("\n", 1)[1] if "\n" in text else text
        snippet = parent[off : off + length]
        assert snippet.replace("\n", "") == body.replace("\n", "")


def test_parse_reports_skipped_files(fake_corpus, monkeypatch, tmp_path):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    bad = fake_corpus / "006" / "研究方法" / "broken.doc"
    bad.write_bytes(b"\xd0\xcf\x11\xe0 not a real doc")
    monkeypatch.setattr("shutil.which", lambda _n: None)
    monkeypatch.setattr("rag.extract.find_soffice", lambda: None)

    files = index._collect_files()
    rows, skipped, _parents = index._parse(files)
    assert any(name == "broken.doc" for name, _ in skipped), "损坏的 .doc 应被记录为跳过"
    assert rows, "其余文件仍应正常解析"


def test_signature_changes_with_chunk_params(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    s1 = index._signature(files)
    monkeypatch.setattr(index, "CHUNK_SIZE", index.CHUNK_SIZE + 1)
    assert index._signature(files) != s1


def test_signature_changes_with_parent_child_flag(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    s1 = index._signature(files)
    monkeypatch.setattr(index, "PARENT_CHILD_ENABLED", not index.PARENT_CHILD_ENABLED)
    assert index._signature(files) != s1
