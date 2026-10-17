"""入库层 _parse 测试：解析 + 分块 + 主键/元数据（不加载模型）。"""
from __future__ import annotations

import pytest

from rag import index


@pytest.fixture(autouse=True)
def _no_external_doc_backend(monkeypatch):
    """测试不依赖本机 Word/LibreOffice：批量解析不走 COM/soffice，跳过纯文本文件。"""
    from rag import extract

    monkeypatch.setattr(index, "extract_files", lambda files: extract.extract_files(files, doc_backend="none"))
    monkeypatch.setattr(extract, "_has_pywin32", lambda: False)
    monkeypatch.setattr(extract, "find_soffice", lambda: None)


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

    files = index._collect_files()
    rows, skipped, _parents = index._parse(files)
    assert any(name == "broken.doc" for name, _ in skipped), "损坏的 .doc 应被记录为跳过"
    assert rows, "其余文件仍应正常解析"


def test_parse_reports_empty_text_files(fake_corpus, monkeypatch):
    """解析成功但无文本的文件（如扫描版 PDF）必须进跳过清单，不能静默丢失。"""
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    from rag import extract

    empty_pdf = fake_corpus / "006" / "研究方法" / "scan.pdf"
    empty_pdf.write_bytes(b"%PDF-1.4 fake")
    real_extract = extract.extract_file

    def fake_extract(path, doc_backend="auto"):
        if path.name == "scan.pdf":
            return {"path": str(path), "text": "  \n", "ext": ".pdf"}
        return real_extract(path, doc_backend=doc_backend)

    monkeypatch.setattr(extract, "extract_file", fake_extract)

    _rows, skipped, _parents = index._parse(index._collect_files())
    assert ("scan.pdf", "解析结果为空") in skipped


def test_check_broken_names_rejects_mojibake(fake_corpus, monkeypatch):
    """GBK 环境解压出的乱码文件名应在入库前报错，而不是把乱码写进 Chroma。"""
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    bad = fake_corpus / "006" / "研究方法" / "鍏稿瀷闂\ue1c0.doc"
    bad.write_text("正文", encoding="utf-8")

    files = index._collect_files()
    assert index._is_broken_name("鍏稿瀷闂\ue1c0.doc")
    with pytest.raises(SystemExit, match="文件名编码损坏"):
        index._check_broken_names(files)


def test_check_broken_names_accepts_normal_files(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    index._check_broken_names(index._collect_files())  # 不应抛错


def test_cache_roundtrip_keeps_skipped_list(fake_corpus, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", fake_corpus)
    sig = index._signature(index._collect_files())
    rows = [("id1::x::chunk0", "文本", {"source": "x"})]
    skipped = [("broken.doc", "解析失败")]
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(index, "CACHE_DIR", fake_corpus.parent / "cache_for_test")
    index._save_cache(sig, rows, np.zeros((1, 4), dtype="float32"), skipped)

    loaded_rows, _emb, loaded_skipped = index._load_cache(sig)
    assert loaded_rows == rows
    assert loaded_skipped == skipped


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


def _make_store(root, chunks):
    (root / "chroma_db").mkdir(parents=True)
    (root / "chroma_db" / "chroma.sqlite3").write_bytes(b"x" * chunks)


def test_snapshot_store_keeps_recent_n(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _make_store(store, 10)
    monkeypatch.setattr(index, "STORE_DIR", store / "chroma_db")
    monkeypatch.setattr(index, "SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(index, "SNAPSHOT_KEEP", 2)

    snaps = [index._snapshot_store(stamp=s) for s in ("20260101-000000", "20260102-000000", "20260103-000000")]
    assert snaps[0] is not None
    left = sorted(p.name for p in (tmp_path / "snaps").iterdir())
    assert left == ["20260102-000000", "20260103-000000"], "只保留最近 SNAPSHOT_KEEP 份"


def test_snapshot_store_skips_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "STORE_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(index, "SNAPSHOT_DIR", tmp_path / "snaps")
    assert index._snapshot_store() is None
    assert not (tmp_path / "snaps").exists()


def test_restore_store_reverts_half_written_db(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _make_store(store, 10)
    monkeypatch.setattr(index, "STORE_DIR", store / "chroma_db")
    monkeypatch.setattr(index, "SNAPSHOT_DIR", tmp_path / "snaps")
    snap = index._snapshot_store(stamp="good")

    (store / "chroma_db" / "chroma.sqlite3").write_bytes(b"")  # 模拟写坏
    index._restore_store(snap)
    assert (store / "chroma_db" / "chroma.sqlite3").read_bytes() == b"x" * 10


def test_build_rejects_missing_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", tmp_path / "nope")
    with pytest.raises(SystemExit, match="资料目录不存在"):
        index.build()


def test_build_rejects_empty_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit, match="没有可入库的文件"):
        index.build()
