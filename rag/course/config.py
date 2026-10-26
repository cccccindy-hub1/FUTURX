"""课程知识库配置。所有路径与参数均可通过环境变量覆盖（或写入 .env）。

教材原文受版权保护，默认不放进仓库：用 COURSE_TEXTBOOK 指向本机文件。
"""
import os
from pathlib import Path

from ..config import RAG_DIR

# ---- 输入 ----
COURSE_TEXTBOOK = Path(os.environ.get("COURSE_TEXTBOOK", RAG_DIR / "course_data" / "textbook.docx"))
# 教材版本号写入 chunk_id，换版本时 id 整体换代，旧映射不会张冠李戴
BOOK_VERSION = os.environ.get("COURSE_BOOK_VERSION", "0906")

# ---- 输出（与 rag/chroma_db、rag/cache 分开）----
COURSE_CACHE_DIR = Path(os.environ.get("COURSE_CACHE_DIR", RAG_DIR / "course_cache"))

# ---- 分块 ----
PARENT_MAX = int(os.environ.get("COURSE_PARENT_MAX", "1500"))  # 父块字符上限（超出按段落边界切）
CHILD_MAX = int(os.environ.get("COURSE_CHILD_MAX", "500"))     # 子块打包上限
CHILD_MIN = int(os.environ.get("COURSE_CHILD_MIN", "120"))     # 短于此的段落与相邻段合并
# 只有超过此长度的单段才按句切。书稿0906最长段 671 字，取 800 保证任何段落（含整段定义）都不被切开
PARA_SPLIT = int(os.environ.get("COURSE_PARA_SPLIT", "800"))
