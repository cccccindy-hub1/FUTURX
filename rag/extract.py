"""解析层：把不同格式的文档统一抽成纯文本 + 元数据。

- .pdf  -> pymupdf（资料均为文本型，无需 OCR）
- .doc / .docx -> macOS 自带 textutil
- .ppt / .pptx -> textutil 不支持 PowerPoint，会报错并被上层跳过（仅 1 个文件）
"""
from __future__ import annotations

import subprocess
from pathlib import Path

try:  # 兼容 pymupdf 新旧两种导入名
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


def _extract_pdf(path: Path) -> str:
    doc = fitz.open(path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages)


def _extract_textutil(path: Path) -> str:
    proc = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore").strip())
    return proc.stdout.decode("utf-8", "ignore")


def extract_file(path: Path) -> dict:
    """返回 {"path": str, "text": str, "ext": str}。解析失败会抛出异常。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext in (".doc", ".docx"):
        text = _extract_textutil(path)
    elif ext in (".ppt", ".pptx"):
        text = _extract_textutil(path)  # 通常会失败，由调用方捕获
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    return {"path": str(path), "text": text, "ext": ext}


def file_category(path: Path, data_dir: Path) -> str:
    """从相对路径推导类别，如 nlp_/006/研究方法/扎根理论/x.pdf -> 006/研究方法/扎根理论。"""
    rel = path.relative_to(data_dir)
    parts = list(rel.parts)[:-1]  # 去掉文件名
    return "/".join(parts) if parts else ""
