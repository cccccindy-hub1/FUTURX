"""政策库配置。所有路径与参数均可通过环境变量覆盖（或写入 .env）。

汇编为整理的外部资料，默认不放进仓库：用 POLICY_DOC 指向本机文件。
"""
import os
from pathlib import Path

from ..config import RAG_DIR

# ---- 输入 ----
POLICY_DOC = Path(os.environ.get("POLICY_DOC", RAG_DIR / "policy_data" / "数据政策汇编.docx"))

# ---- 输出（与 rag/chroma_db、rag/course_cache 分开）----
POLICY_CACHE_DIR = Path(os.environ.get("POLICY_CACHE_DIR", RAG_DIR / "policy_cache"))

# ---- 切块 ----
POLICY_CHUNK_MAX = int(os.environ.get("POLICY_CHUNK_MAX", "1500"))      # 超过此长度的条/事项才拆开
POLICY_PART_TARGET = int(os.environ.get("POLICY_PART_TARGET", "1000"))  # 拆开时每片的目标长度
# 标题含这些词的文件不入政策库（数据字典类标准：逐字段定义，不是政策条文），逗号分隔
POLICY_EXCLUDE_TITLES = tuple(
    s.strip() for s in os.environ.get("POLICY_EXCLUDE_TITLES", "数据规范").split(",") if s.strip()
)
POLICY_DUP_JACCARD = float(os.environ.get("POLICY_DUP_JACCARD", "0.8"))     # 两文件正文相似度达到即视为重复
POLICY_DRAFT_JACCARD = float(os.environ.get("POLICY_DRAFT_JACCARD", "0.5"))  # 征求意见稿与正式稿的相似度门槛

# ---- 检索（agentic：BM25 初筛 → 模型选块 → 作答；不用向量库）----
POLICY_CANDIDATES = int(os.environ.get("POLICY_CANDIDATES", "15"))     # BM25 初筛最多取多少个块交给模型挑
POLICY_PICK_MAX = int(os.environ.get("POLICY_PICK_MAX", "6"))          # 模型最多选几个块
POLICY_CHUNK_MIN = int(os.environ.get("POLICY_CHUNK_MIN", "800"))      # 非条款式文件：累积到这么多字就收一块
POLICY_CHUNK_CAP = int(os.environ.get("POLICY_CHUNK_CAP", "1200"))     # 非条款式文件：单块上限
POLICY_ARTICLE_CAP = int(os.environ.get("POLICY_ARTICLE_CAP", "1800"))  # 单条超过此长度才按段拆开
