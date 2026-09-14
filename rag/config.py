"""全局配置。所有可调参数均支持通过环境变量覆盖（或写入项目根目录 .env）。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent          # /Users/guo/FUTURX
DATA_DIR = ROOT / "nlp_"                               # 原始资料目录
RAG_DIR = ROOT / "rag"
STORE_DIR = RAG_DIR / "chroma_db"                      # 向量库持久化目录

# 优先加载根目录 .env，其次 rag/.env
load_dotenv(ROOT / ".env")
load_dotenv(RAG_DIR / ".env")

# ---- 向量化 ----
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "nlp_degree_thesis")

# ---- 分块 ----
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))       # 每个 chunk 的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100")) # 相邻 chunk 重叠字符数

# ---- 检索 ----
TOP_K = int(os.environ.get("TOP_K", "4"))                   # 检索返回的片段数

# ---- 生成（OpenAI 兼容协议，默认指向 DeepSeek）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
