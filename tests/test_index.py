"""入库层 _parse 测试：解析 + 分块 + 主键/元数据（不加载模型）。"""
from __future__ import annotations

from rag import index


def test_parse_produces_unique_ids_and_metadata(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    assert files, "应收集到合成语料"

    rows, skipped = index._parse(files)
    assert rows, "应解析出 chunk"
    assert skipped == []

    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "chunk 主键必须唯一"

    rid, text, meta = rows[0]
    assert "::chunk" in rid
    assert text.strip()
    for key in ("source", "filename", "category", "section", "ext"):
        assert key in meta
    assert meta["category"] == "006/研究方法/扎根理论"


def test_parse_reports_skipped_files(fake_corpus, monkeypatch, tmp_path):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    bad = fake_corpus / "006" / "研究方法" / "broken.doc"
    bad.write_bytes(b"\xd0\xcf\x11\xe0 not a real doc")
    monkeypatch.setattr("shutil.which", lambda _n: None)
    monkeypatch.setattr("rag.extract.find_soffice", lambda: None)

    files = index._collect_files()
    rows, skipped = index._parse(files)
    assert any(name == "broken.doc" for name, _ in skipped), "损坏的 .doc 应被记录为跳过"
    assert rows, "其余文件仍应正常解析"


def test_signature_changes_with_chunk_params(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    files = index._collect_files()
    s1 = index._signature(files)
    monkeypatch.setattr(index, "CHUNK_SIZE", index.CHUNK_SIZE + 1)
    assert index._signature(files) != s1
