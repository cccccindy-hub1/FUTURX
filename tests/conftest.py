"""测试夹具：全部离线，不加载真实模型、不联网。

设计要点：
- 用 `stub_embedder` 替换 SentenceTransformer，返回确定性的小向量，避免下载 2.3GB 的 BGE-m3。
- 用 `mock_llm` 替换 OpenAI 客户端，回答固定内容，避免真实 API 调用。
- 用 `fake_corpus` 造合成资料（PyMuPDF 可写 PDF、python-docx 可写 docx），
  使解析层能在没有 `nlp_/` 的情况下被真实验证。
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------- 路径隔离

@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """把所有落盘路径指向 tmp_path，避免污染真实 rag/ 目录。"""
    data_dir = tmp_path / "nlp_"
    store_dir = tmp_path / "chroma_db"
    cache_dir = tmp_path / "cache"
    for d in (data_dir, store_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("STORE_DIR", str(store_dir))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))

    # 让已导入的 config 模块重新读取环境变量；同时刷新各模块里 from-import 的名字
    import rag.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(cfg, "STORE_DIR", store_dir)
    monkeypatch.setattr(cfg, "CACHE_DIR", cache_dir)

    for mod_name in ("rag.index", "rag.query", "rag.retrieval", "rag.eval"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "DATA_DIR"):
            monkeypatch.setattr(mod, "DATA_DIR", data_dir)
        if hasattr(mod, "STORE_DIR"):
            monkeypatch.setattr(mod, "STORE_DIR", store_dir)
        if hasattr(mod, "CACHE_DIR"):
            monkeypatch.setattr(mod, "CACHE_DIR", cache_dir)
    return {"data": data_dir, "store": store_dir, "cache": cache_dir}


# ---------------------------------------------------------------- 假嵌入模型

class StubEmbedder:
    """确定性伪嵌入：按字符 hash 生成固定维向量，语义无关但可复现。"""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False, **_):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for r, text in enumerate(items):
            for ch in str(text):
                h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
                out[r, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out[0] if single else out


@pytest.fixture
def stub_embedder():
    return StubEmbedder()


# ---------------------------------------------------------------- 假 LLM

class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)


class _StubCompletion:
    def __init__(self, content):
        self.choices = [_StubChoice(content)]


class _StubCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **_kwargs):
        return _StubCompletion(self._content)


class _StubChat:
    def __init__(self, content):
        self.completions = _StubCompletions(content)


class StubLLM:
    """替身 OpenAI 客户端，记录调用次数，返回固定回答。"""

    def __init__(self, content: str = "这是测试回答 [来源1]。"):
        self.chat = _StubChat(content)
        self.calls = 0

    def _count(self):
        self.calls += 1


@pytest.fixture
def mock_llm(monkeypatch):
    """把 rag.query.OpenAI 换成替身，并确保 LLM_API_KEY 非空。"""
    import rag.query as q

    holder = {}

    def factory(*_args, **_kwargs):
        stub = StubLLM()
        holder["stub"] = stub
        return stub

    monkeypatch.setattr(q, "OpenAI", factory)
    monkeypatch.setattr(q, "LLM_API_KEY", "test-key")
    return holder


# ---------------------------------------------------------------- 合成语料

@pytest.fixture
def fake_corpus(isolated_dirs):
    """在隔离的 RAG_DATA_DIR 下造合成资料，返回该目录。"""
    data_dir = isolated_dirs["data"]
    sub = data_dir / "006" / "研究方法" / "扎根理论"
    sub.mkdir(parents=True, exist_ok=True)

    (sub / "sample.txt").write_text(
        "扎根理论的研究程序\n\n"
        "扎根理论强调从资料中生成理论。三级编码包括开放式编码、主轴编码和选择性编码。"
        "开放式编码对资料逐句贴标签；主轴编码建立范畴之间的联系；选择性编码提炼核心范畴。\n\n"
        "文献综述的写法\n\n"
        "文献综述要围绕研究问题组织，而不是简单罗列。",
        encoding="utf-8",
    )
    # index._collect_files 只认 SUPPORTED_EXTS，另造一个可解析的 .docx
    make_docx(
        sub / "sample.docx",
        [
            "扎根理论的研究程序",
            "扎根理论强调从资料中生成理论。三级编码包括开放式编码、主轴编码和选择性编码。",
            "开放式编码对资料逐句贴标签；主轴编码建立范畴之间的联系；选择性编码提炼核心范畴。",
        ],
    )
    return data_dir


def make_docx(path, paragraphs):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))
    return path


def make_pdf(path, text):
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path
