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
CASE_CARD_FALLBACK_CHARS = int(os.environ.get("CASE_CARD_FALLBACK_CHARS", "800"))   # 没有要点类小节时，卡片取「本院认为」等约这么多字
CASE_BODY_MIN = int(os.environ.get("CASE_BODY_MIN", "800"))                          # 正文块字数下限（太短的小节与相邻小节合并）
CASE_BODY_MAX = int(os.environ.get("CASE_BODY_MAX", "1200"))                         # 正文块字数上限（超出按段落 / 句子拆开）
