"""判例库配置。所有路径与参数均可通过环境变量覆盖（或写入 .env）。

判例汇编为整理的外部资料，默认不放进仓库：用 CASE_DOC 指向本机文件。
"""
import os
from pathlib import Path

from ..config import RAG_DIR

# ---- 输入 ----
CASE_DOC = Path(os.environ.get("CASE_DOC", RAG_DIR / "case_data" / "数据法典型判例库（十章分类完整版）.docx"))

# ---- 输出（与 rag/chroma_db、rag/course_cache、rag/policy_cache 分开）----
CASE_CACHE_DIR = Path(os.environ.get("CASE_CACHE_DIR", RAG_DIR / "case_cache"))

# ---- 切块 ----
CASE_CARD_MAX_CHARS = int(os.environ.get("CASE_CARD_MAX_CHARS", os.environ.get("CASE_CARD_FALLBACK_CHARS", "800")))   # 卡片正文句末截断上限（卡片头不计）；兼容旧变量名
CASE_BODY_MIN = int(os.environ.get("CASE_BODY_MIN", "800"))                          # 正文块字数下限（太短的小节与相邻小节合并）
CASE_BODY_MAX = int(os.environ.get("CASE_BODY_MAX", "1200"))                         # 正文块字数上限（超出按段落 / 句子拆开）

CASE_CARD_FALLBACK_CHARS = CASE_CARD_MAX_CHARS      # 旧名，保留给已有引用

# ---- 向量库（独立的 Chroma 目录与集合，绝不读写论文库的 rag/chroma_db）----
CASE_STORE_DIR = Path(os.environ.get("CASE_STORE_DIR", RAG_DIR / "case_store"))
CASE_COLLECTION = os.environ.get("CASE_COLLECTION", "data_law_cases")
