"""解析层：把不同格式的文档统一抽成纯文本 + 元数据（跨平台）。

按扩展名 + 能力探测选择解析器：

- .pdf   -> pymupdf
- .docx  -> python-docx
- .pptx  -> python-pptx
- .doc   -> Windows 且装了 Word 时走 COM；否则 LibreOffice `soffice` 转换；都没有则报错跳过
- .ppt   -> LibreOffice `soffice` 转换；没有则报错跳过

设计原则：
- 可选依赖（python-docx / python-pptx / pywin32 / pymupdf）全部延迟导入并 try/except，
  缺依赖时标记为不可用，而不是在 import 阶段直接崩溃。
- 解析失败抛出带可读原因的异常，由调用方（index._parse）捕获并记录跳过。
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# pymupdf 兼容新旧两种导入名
try:
    import pymupdf as fitz

    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover
    try:
        import fitz  # type: ignore

        _HAS_PYMUPDF = True
    except ImportError:
        _HAS_PYMUPDF = False

_SOFFICE_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
)


class UnsupportedFormat(RuntimeError):
    """无法解析该格式（缺少后端）。"""


# ---------------------------------------------------------------- 后端探测

def find_soffice() -> str | None:
    """定位 LibreOffice 可执行文件；可用环境变量 SOFFICE_BIN 覆盖。"""
    env = os.environ.get("SOFFICE_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("soffice")
    if found:
        return found
    for cand in _SOFFICE_CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


def _has_pywin32() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import win32com.client  # noqa: F401

        return True
    except ImportError:
        return False


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def parser_status() -> dict[str, str]:
    """返回每个扩展名当前可用的解析后端，供 build 打日志与自检。"""
    soffice = find_soffice()
    status: dict[str, str] = {}
    status[".pdf"] = "available" if _HAS_PYMUPDF else (
        "needs_libreoffice" if soffice else "unsupported"
    )
    status[".docx"] = "available" if _has("docx") else "unsupported"
    status[".pptx"] = "available" if _has("pptx") else "unsupported"
    if _has_pywin32():
        status[".doc"] = "available"
    elif soffice:
        status[".doc"] = "needs_libreoffice"
    else:
        status[".doc"] = "needs_libreoffice_or_word"
    status[".ppt"] = "needs_libreoffice" if soffice else "unsupported"
    return status


# ---------------------------------------------------------------- 各格式解析

def _extract_pdf(path: Path) -> str:
    if _HAS_PYMUPDF:
        doc = fitz.open(path)
        try:
            pages = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        return "\n\n".join(pages)
    soffice = find_soffice()
    if soffice:
        return _soffice_convert_text(path, soffice)
    raise UnsupportedFormat("解析 PDF 需要 pymupdf，或安装 LibreOffice")


def _extract_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as e:
        raise UnsupportedFormat("解析 .docx 需要 python-docx（pip install python-docx）") from e

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as e:
        raise UnsupportedFormat("解析 .pptx 需要 python-pptx（pip install python-pptx）") from e

    prs = Presentation(str(path))
    parts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _soffice_run(args: list[str], soffice: str) -> subprocess.CompletedProcess:
    return subprocess.run([soffice, "--headless", *args], capture_output=True)


def _soffice_convert_text(path: Path, soffice: str) -> str:
    with tempfile.TemporaryDirectory() as out:
        proc = _soffice_run(
            ["--convert-to", "txt:Text (encoded):UTF8", "--outdir", out, str(path)], soffice
        )
        produced = Path(out) / (path.stem + ".txt")
        if proc.returncode != 0 or not produced.exists():
            raise UnsupportedFormat(f"LibreOffice 转换失败：{path.name}")
        return produced.read_text(encoding="utf-8", errors="ignore")


def _soffice_convert_to(path: Path, target: str, soffice: str) -> Path:
    """用 LibreOffice 把文件转成 target 格式（如 docx/pptx），返回产物路径。"""
    with tempfile.TemporaryDirectory() as out:
        proc = _soffice_run(
            ["--convert-to", target, "--outdir", out, str(path)], soffice
        )
        produced = Path(out) / f"{path.stem}.{target}"
        if proc.returncode != 0 or not produced.exists():
            raise UnsupportedFormat(f"LibreOffice 转换 {target} 失败：{path.name}")
        # 复制出临时目录，供后续解析
        keep = Path(tempfile.gettempdir()) / f"_rag_{os.getpid()}_{produced.name}"
        shutil.copyfile(produced, keep)
        return keep


def _extract_doc_via_soffice(path: Path, soffice: str) -> str:
    converted = _soffice_convert_to(path, "docx", soffice)
    try:
        return _extract_docx(converted)
    finally:
        converted.unlink(missing_ok=True)


def _extract_ppt_via_soffice(path: Path, soffice: str) -> str:
    converted = _soffice_convert_to(path, "pptx", soffice)
    try:
        return _extract_pptx(converted)
    finally:
        converted.unlink(missing_ok=True)


def _extract_doc_via_com(path: Path) -> str:
    """Windows + 已安装 Word 时用 COM 自动化读取 .doc。"""
    if not _has_pywin32():
        raise UnsupportedFormat("COM 解析需要 pywin32")
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore

    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(path.resolve()), ReadOnly=True)
        return doc.Content.Text
    except Exception as e:  # COM 失败交给上层回退
        raise UnsupportedFormat(f"Word COM 解析失败：{e}") from e
    finally:
        try:
            if doc is not None:
                doc.Close(False)
            if word is not None:
                word.Quit()
        finally:
            pythoncom.CoUninitialize()


# ---------------------------------------------------------------- 对外接口

def extract_file(path: Path) -> dict:
    """返回 {"path": str, "text": str, "ext": str}。解析失败会抛出异常。"""
    path = Path(path)
    ext = path.suffix.lower()
    soffice = find_soffice()

    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext == ".docx":
        text = _extract_docx(path)
    elif ext == ".pptx":
        text = _extract_pptx(path)
    elif ext == ".doc":
        if _has_pywin32():
            try:
                text = _extract_doc_via_com(path)
            except UnsupportedFormat:
                if not soffice:
                    _raise_doc(soffice)
                text = _extract_doc_via_soffice(path, soffice)
        elif soffice:
            text = _extract_doc_via_soffice(path, soffice)
        else:
            _raise_doc(soffice)
    elif ext == ".ppt":
        if not soffice:
            raise UnsupportedFormat(
                f"无法解析 {path.name}：.ppt 需要安装 LibreOffice（或将源文件另存为 .pptx）"
            )
        text = _extract_ppt_via_soffice(path, soffice)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    return {"path": str(path), "text": text, "ext": ext}


def _raise_doc(soffice: str | None) -> None:
    msg = "无法解析 .doc：请安装 LibreOffice（或 Windows 中安装 Word + pywin32）"
    if soffice:
        msg += "，或将源文件另存为 .docx"
    raise UnsupportedFormat(msg)


def file_category(path: Path, data_dir: Path) -> str:
    """从相对路径推导类别，如 nlp_/006/研究方法/扎根理论/x.pdf -> 006/研究方法/扎根理论。"""
    rel = path.relative_to(data_dir)
    parts = list(rel.parts)[:-1]  # 去掉文件名
    return "/".join(parts) if parts else ""
