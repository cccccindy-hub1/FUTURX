"""全局配置。所有可调参数均支持通过环境变量覆盖（或写入项目根目录 .env）。"""
import os
from pathlib import Path

try:  # python-dotenv 缺失时仍可运行（配置改由真实环境变量注入）
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

ROOT = Path(__file__).resolve().parent.parent          # 项目根目录 FUTURX
RAG_DIR = Path(os.environ.get("RAG_DIR", ROOT / "rag"))

# 优先加载根目录 .env，其次 rag/.env
load_dotenv(ROOT / ".env")
load_dotenv(RAG_DIR / ".env")

# ---- 路径（均可用环境变量覆盖，默认值保持不变）----
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ROOT / "nlp_"))   # 原始资料目录
STORE_DIR = Path(os.environ.get("STORE_DIR", RAG_DIR / "chroma_db"))  # 向量库持久化目录
CACHE_DIR = Path(os.environ.get("CACHE_DIR", RAG_DIR / "cache"))      # 向量/索引缓存目录

# ---- 向量化 ----
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "nlp_degree_thesis")

# ---- 分块 ----
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))       # 每个 chunk 的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100")) # 相邻 chunk 重叠字符数

# ---- 检索 ----
TOP_K = int(os.environ.get("TOP_K", "4"))                   # 检索返回的片段数

# ---- 检索增强（默认关闭，保持原有 dense-only 行为；按需 opt-in）----
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "dense")  # dense | bm25 | hybrid
FETCH_K = int(os.environ.get("FETCH_K", "20"))              # 融合前的候选池大小
RRF_K = int(os.environ.get("RRF_K", "60"))                  # RRF 常数
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "8"))
MAX_DISTANCE = float(os.environ.get("MAX_DISTANCE", "0.0"))  # >0 时按余弦距离过滤片段

# ---- 生成（OpenAI 兼容协议，默认指向 DeepSeek）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
