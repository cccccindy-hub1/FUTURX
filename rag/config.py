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
SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOT_DIR", CACHE_DIR / "snapshots"))  # build 前的库快照
SNAPSHOT_KEEP = int(os.environ.get("SNAPSHOT_KEEP", "2"))     # 保留最近 N 份快照

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

# ---- 解析兜底：扫描版 PDF OCR（默认关闭；开启后文本少于阈值页才 OCR）----
OCR_ENABLED = os.environ.get("OCR_ENABLED", "false").lower() in ("1", "true", "yes")
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "50"))  # 单页文本少于该字符数视为扫描页
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))

# ---- 分块扩展：父子分块（默认关闭；子块入库检索，父块作生成上下文）----
PARENT_CHILD_ENABLED = os.environ.get("PARENT_CHILD_ENABLED", "false").lower() in ("1", "true", "yes")
PARENT_SIZE = int(os.environ.get("PARENT_SIZE", "1500"))       # 父块字符数上限
CONTEXT_EXPAND_ENABLED = os.environ.get("CONTEXT_EXPAND_ENABLED", "false").lower() in ("1", "true", "yes")

# ---- HTTP 对外接口（OpenAI 兼容层：POST /v1/chat/completions）----
# 逗号分隔的密钥列表；留空则不校验（仅建议在内网/本机时留空）。
API_KEYS = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
# 对外暴露的模型名，客户端 model 字段填它即可（实际生成模型由 LLM_MODEL 决定）。
API_MODEL_NAME = os.environ.get("API_MODEL_NAME", "ai-li-teacher")

# ---- 生成（OpenAI 兼容协议，默认指向 DeepSeek）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ---- 查询增强（需要 LLM；默认关闭）----
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").lower() in ("1", "true", "yes")
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "false").lower() in ("1", "true", "yes")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "") or LLM_MODEL  # 改写/HyDE 用的小模型，默认同 LLM_MODEL

# ---- data_benchmark 评测（默认从 <项目根>/data_benchmark 读取）----
BENCHMARK_DIR = Path(os.environ.get("BENCHMARK_DIR", ROOT / "data_benchmark"))
BENCHMARK_SEED = int(os.environ.get("BENCHMARK_SEED", "42"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.08"))         # 测试集占比下限（报告建议 7~8%）
MAX_TEST_RATIO = float(os.environ.get("MAX_TEST_RATIO", "0.10"))  # 测试集占比上限
# 按来源（CSV「来源姓名」，即提问专家）分组切分：同一来源不跨 train/test。
# 来源是整体进出的最小单元，超大来源（如占全库 16.9% 的单个专家）一旦进 test
# 就会把比例顶到 MAX_TEST_RATIO 之上，因此体量超过 MAX_SOURCE_SHARE 的来源
# 整户留在 train，不参与评测。
MAX_SOURCE_SHARE = float(os.environ.get("MAX_SOURCE_SHARE", "0.06"))
# 种子评估集：从全部数据按主题铺开取 k 条（不参与 train/test），用于检查
# 检索语料对 data_benchmark 主题的覆盖度。0 = 不生成。
PROBE_SIZE = int(os.environ.get("PROBE_SIZE", "60"))

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
