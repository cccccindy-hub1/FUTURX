"""名言库配置。所有路径与参数均可通过环境变量覆盖（或写入 .env）。

合集为整理的外部资料，默认不放进仓库：用 QUOTE_DOC 指向本机文件。
"""
import os
from pathlib import Path

from ..config import RAG_DIR

# ---- 输入 ----
QUOTE_DOC = Path(os.environ.get("QUOTE_DOC", RAG_DIR / "quote_data" / "习近平总书记关于数智的有关表述合集.docx"))

# ---- 输出（与 rag/chroma_db、rag/course_cache、rag/policy_cache、rag/case_cache 分开）----
QUOTE_CACHE_DIR = Path(os.environ.get("QUOTE_CACHE_DIR", RAG_DIR / "quote_cache"))

# ---- 向量库（独立的 Chroma 目录与集合，绝不读写论文库的 rag/chroma_db）----
QUOTE_STORE_DIR = Path(os.environ.get("QUOTE_STORE_DIR", RAG_DIR / "quote_store"))
QUOTE_COLLECTION = os.environ.get("QUOTE_COLLECTION", "xi_digital_quotes")
