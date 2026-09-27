"""解析层测试：Windows 可移植性 + 能力探测。"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.extract import extract_file, file_category, parser_status, find_soffice
from conftest import make_docx, make_pdf


def test_extract_pdf_roundtrip(tmp_path):
    p = make_pdf(tmp_path / "a.pdf", "digital governance research")
    rec = extract_file(p)
    assert rec["ext"] == ".pdf"
    assert "digital governance" in rec["text"]


def test_extract_docx_roundtrip(tmp_path):
    p = make_docx(tmp_path / "a.docx", ["第一段", "第二段：研究方法"])
    rec = extract_file(p)
    assert rec["ext"] == ".docx"
    assert "研究方法" in rec["text"]


def test_extract_file_unknown_extension(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_file(p)


def test_parser_status_reports_each_supported_ext():
    status = parser_status()
    for ext in (".pdf", ".docx", ".doc", ".pptx", ".ppt"):
        assert ext in status, f"{ext} 应出现在能力探测结果中"


def test_find_soffice_respects_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "soffice.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("SOFFICE_BIN", str(fake))
    assert find_soffice() == str(fake)


def test_find_soffice_absent(monkeypatch):
    monkeypatch.delenv("SOFFICE_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert find_soffice() is None


def test_doc_without_any_backend_raises_readable_error(monkeypatch, tmp_path):
    """无 LibreOffice / Word 时，.doc 应抛出可读异常而非静默失败。"""
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("rag.extract.find_soffice", lambda: None)
    p = tmp_path / "legacy.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0 garbage")
    with pytest.raises(RuntimeError, match="LibreOffice|Word|无法解析"):
        extract_file(p)


def test_file_category_derives_from_relative_path(tmp_path):
    root = tmp_path / "nlp_"
    f = root / "006" / "研究方法" / "扎根理论" / "x.pdf"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("", encoding="utf-8")
    assert file_category(f, root) == "006/研究方法/扎根理论"
