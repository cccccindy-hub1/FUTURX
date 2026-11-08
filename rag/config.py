"""全局配置。所有可调参数均支持通过环境变量覆盖（或写入项目根目录 .env）。"""
import hashlib
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

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

from .persona import PERSONA_CORE  # noqa: E402 - 角色卡不依赖配置，放在 load_dotenv 之后不改变加载顺序

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
_API_KEYS_INLINE = frozenset(k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip())
API_KEYS_FILE = os.environ.get("API_KEYS_FILE", "").strip()
# 密钥文件的重新读取间隔（秒）：0 = 每次都读（不建议），-1 = 只在启动时读一次。
# 按人发 key 时撤销/轮换是常态，重启服务要中断 30-60 秒；热加载把这件事变成「改文件即生效」。
API_KEYS_RELOAD_S = float(os.environ.get("API_KEYS_RELOAD_S", "5"))


def _read_keys_file(path: str) -> list[str]:
    """从文件读密钥：每行一个，# 开头为注释。"""
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


_load_api_keys_file = _read_keys_file        # 兼容旧引用


class ApiKeySet:
    """API key 集合，按文件内容哈希自动重载。

    用 set 而非 list：按人发 key 时（例如一个班一千人）每个请求都要查一次，
    列表是 O(n) 的字符串比较，集合是 O(1) 哈希查找。

    热加载是这里的关键：1000 人里必然有人泄露 key，而重启服务要中断所有人
    30-60 秒。改成「文件变了就重读」，撤销/新增/轮换都不再需要重启。
    """

    def __init__(self, inline: frozenset[str], path: str, reload_s: float) -> None:
        self._inline = inline
        self._path = Path(path).expanduser() if path else None
        self._reload_s = reload_s
        self._lock = threading.Lock()
        self._stamp: str | None = None                  # 文件内容哈希（见 load 里的说明）
        self._next_check = 0.0
        self._keys: frozenset[str] = frozenset()
        self.load(force=True)

    # -- 读取 --
    def load(self, *, force: bool = False) -> None:
        """按需重读文件；force=True 忽略节流。"""
        now = time.monotonic()
        # reload_s <= 0 表示每次都查，不能走节流：monotonic 在同一微秒内可能相等，
        # 用 `<` 比较会把「刚改完文件」的那次查询挡掉。
        if not force and self._reload_s > 0 and now < self._next_check:
            return
        with self._lock:
            self._next_check = now + max(0.0, self._reload_s)
            if self._reload_s < 0 and not force:
                return
            stamp, rows = None, []
            if self._path is not None:
                try:
                    raw = self._path.read_bytes()
                    # 用内容哈希而不是 mtime：文件系统时间戳精度有限，
                    # 「撤销一把再加一把」大小不变、mtime 也可能落在同一刻度，
                    # 只靠时间戳会漏掉改动。1000 把 key 约 48KB，哈希开销可忽略。
                    stamp = hashlib.sha256(raw).hexdigest()
                except OSError:
                    stamp = None
                if stamp is not None and stamp != self._stamp:
                    rows = [ln.split("#", 1)[0].strip() for ln in raw.decode("utf-8").splitlines()]
                    rows = [k for k in rows if k]
                elif stamp is not None:
                    return                              # 内容没变，什么都不用做
            else:
                rows = []
            self._stamp = stamp
            if stamp is None and self._path is not None:
                # 文件被删/不可读：不要静默退回空集合（那等于放开鉴权），保留上一次的集合。
                # 只在状态变化时打一次日志，否则每个请求都会刷屏。
                if self._stamp is not None:
                    logger.error("API_KEYS_FILE 读取失败：%s（沿用上一次的 %d 把 key）",
                                 self._path, len(self._keys))
                self._stamp = None
                return
            self._keys = frozenset(self._inline | frozenset(rows))

    # -- 对外 --
    def __contains__(self, item: object) -> bool:
        self.load()                                     # 每个请求走到这里，节流后才真的 stat
        return item in self._keys

    def __len__(self) -> int:
        self.load()
        return len(self._keys)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        self.load()
        return iter(self._keys)

    def __repr__(self) -> str:
        return f"ApiKeySet({len(self)} keys, file={self._path})"

    # -- 运维 --
    def reload(self) -> int:
        self.load(force=True)
        return len(self._keys)

    @property
    def path(self) -> Path | None:
        return self._path

    def file_keys(self) -> frozenset[str]:
        self.load()
        return frozenset(_read_keys_file(str(self._path))) if self._path else frozenset()


API_KEYS = ApiKeySet(_API_KEYS_INLINE, API_KEYS_FILE, API_KEYS_RELOAD_S)


def _check_api_key_config() -> None:
    """启动自检：显式配了 API_KEYS_FILE 却读不到/为空时拒绝启动。

    这一条防的是运维手滑：文件被误删或路径写错时，静默退回「空集合 = 不鉴权」
    意味着服务对公网完全敞开，且没人会收到告警。
    """
    if not API_KEYS_FILE:
        return
    p = Path(API_KEYS_FILE).expanduser()
    if not p.is_file():
        raise RuntimeError(f"API_KEYS_FILE 指向的文件不存在：{p}（为避免不鉴权，拒绝启动）")
    if not _read_keys_file(str(p)) and not _API_KEYS_INLINE:
        raise RuntimeError(f"API_KEYS_FILE 里没有任何 key：{p}（为避免不鉴权，拒绝启动）")


_check_api_key_config()

# 对外暴露的模型名，客户端 model 字段填它即可（实际生成模型由 LLM_MODEL 决定）。
API_MODEL_NAME = os.environ.get("API_MODEL_NAME", "ai-li-teacher")

# ---- 生成（OpenAI 兼容协议，默认指向 DeepSeek）----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
# 思考链开关：上游推理模型默认先「思考」再输出正文，但思考 token 不推给客户端，
# 表现为首字节要等 30s+。实测 deepseek-v4-pro 关闭后 TTFT 32.5s -> 2.7s。
# 取值 none / low / medium / high；置空表示不传该参数（沿用上游默认）。
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "none")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))  # 单次请求超时（秒）

# ---- 向量化批处理（rag.embedding）：把并发请求的单条 encode 合并成批 ----
# 批大小、批内总字数上限、攒批最多等多久（毫秒）、编码 worker 数。
# worker 数 × torch 线程数应接近核数：实测 8 核上 2 worker × 4 线程最稳。
EMBED_MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "32"))
EMBED_MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "6000"))
EMBED_BATCH_WAIT_MS = float(os.environ.get("EMBED_BATCH_WAIT_MS", "15"))
EMBED_WORKERS = max(1, int(os.environ.get("EMBED_WORKERS", "2")))
EMBED_TORCH_THREADS = int(os.environ.get("EMBED_TORCH_THREADS", "4")) or None  # 0 = 不动 torch 默认

# ---- 查询增强（需要 LLM；默认关闭）----
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").lower() in ("1", "true", "yes")
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "false").lower() in ("1", "true", "yes")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "") or LLM_MODEL  # 改写/HyDE 用的小模型，默认同 LLM_MODEL

# ---- 多轮对话（OpenAI 兼容层 /v1/chat/completions）----
# 带上 messages 里最多这么多轮历史（user+assistant 算一轮）交给模型；0 = 忽略历史（旧行为）。
CHAT_HISTORY_MAX_TURNS = max(0, int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "6")))
# 历史总字数上限：从最近一轮往前累加，超了就丢更早的轮次。只限轮数会让 6 个长问题把
# prompt 撑到两万字，直接推高首字时间（上游 100 并发下 p50 已是 46.7s）。0 = 不限。
CHAT_HISTORY_MAX_CHARS = max(0, int(os.environ.get("CHAT_HISTORY_MAX_CHARS", "8000")))
# 检索前先用历史把追问改写成独立问题（「第二种具体怎么算」→「数据资产估值的收益法具体怎么算」）。
CONTEXTUALIZE_QUERY_ENABLED = os.environ.get("CONTEXTUALIZE_QUERY_ENABLED", "true").lower() in ("1", "true", "yes")
# 改写这一步在首字之前，超时就退回原问题，不让追问整体变慢。
CONTEXTUALIZE_TIMEOUT = float(os.environ.get("CONTEXTUALIZE_TIMEOUT", "8"))

# ---- Contextual Retrieval（建库时 LLM 为每个 chunk 生成上下文，拼进向量化与 BM25 文本；默认关闭）----
# 开启后每个新 chunk 一次 LLM 调用（结果缓存在 CACHE_DIR/contextual_cache.json），见 rag/contextual.py
CONTEXTUAL_ENABLED = os.environ.get("CONTEXTUAL_ENABLED", "false").lower() in ("1", "true", "yes")
CONTEXTUAL_MODEL = os.environ.get("CONTEXTUAL_MODEL", "") or REWRITE_MODEL
CONTEXTUAL_DOC_MAX_CHARS = int(os.environ.get("CONTEXTUAL_DOC_MAX_CHARS", "6000"))  # 送给 LLM 的文档窗口上限
CONTEXTUAL_WORKERS = int(os.environ.get("CONTEXTUAL_WORKERS", "4"))                 # 并发调用数

# ---- 全局 LLM 路由（统一入口 rag.unified；一次调用、JSON 输出，复用 rag.course.llm.OpenAICompatClient）----
# 默认复用课程库的胜算云配置；API_BASE / KEY 都不设默认值，未配置时路由退化为规则判断。
ROUTER_MODEL = os.environ.get("ROUTER_MODEL") or "deepseek/deepseek-v4-pro"
ROUTER_API_BASE = (os.environ.get("ROUTER_API_BASE") or os.environ.get("COURSE_LLM_API_BASE", "")).strip()
ROUTER_API_KEY = (os.environ.get("ROUTER_API_KEY") or os.environ.get("COURSE_LLM_API_KEY", "")).strip()
ROUTER_REASONING = os.environ.get("ROUTER_REASONING", "disabled").strip()  # disabled / low / medium / high
ROUTER_TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT", "15"))             # 秒；超时走规则兜底
# 与知识库无关的问题（off_topic）不再拒答：直接用大模型自身的能力回答，不检索、不引用资料。
# 这两个值只是该路径的系统提示词与说明文案；拒答文案保留给不配置模型时的兜底。
ROUTER_REFUSAL_TEXT = os.environ.get("ROUTER_REFUSAL_TEXT") or (
    "抱歉，我只能回答以下范围的问题：①学位论文写作与指导；②《数据要素》课程知识点；③课程试题；"
    "④数据政策法规；⑤数据法律判例；⑥总书记关于数字领域的讲话名言。您的问题不在这些范围内，请换一个相关问题试试。"
)
GENERAL_MODEL = os.environ.get("GENERAL_MODEL", "") or ROUTER_MODEL      # 通用作答：超出知识库时用这个模型
GENERAL_REASONING = os.environ.get("GENERAL_REASONING", "disabled").strip()
GENERAL_TIMEOUT = float(os.environ.get("GENERAL_TIMEOUT", "60"))
GENERAL_SYSTEM = os.environ.get("GENERAL_SYSTEM") or (
    PERSONA_CORE + "\n本次问题不在团队知识库的范围内，请用你自己的知识直接、准确地回答。\n"
    "- 不声称内容来自本知识库，也不编造书名、页码、条文号、案号等具体出处；确实需要引用时说明是通用常识。\n"
    "- 不确定的地方直接说不确定，不要为了显得权威而硬凑。\n"
    "- 涉及事实、数字、法条、时效性信息时，提醒用户以权威来源为准。\n"
    "- 用中文简答，分点不超过 4 条；不要写开场白与结语。"
)

# ---- 原 RAG 自适应（rag.adaptive）：simple 一次检索；低置信度改写重查一次；complex 拆子问题 ----
# 重排最高分低于该值视为检索不可靠（bge-reranker-v2-m3 经 CrossEncoder 默认 sigmoid，分数 0~1；初始值未经评测校准）
ADAPTIVE_MIN_RERANK_SCORE = float(os.environ.get("ADAPTIVE_MIN_RERANK_SCORE", "0.3"))
# 未开重排且为纯稠密检索时，最近邻余弦距离高于该值视为不可靠；hybrid/bm25 且无重排时跳过纠错
ADAPTIVE_MAX_DISTANCE = float(os.environ.get("ADAPTIVE_MAX_DISTANCE", "0.6"))
ADAPTIVE_MAX_SUBQUESTIONS = max(1, min(3, int(os.environ.get("ADAPTIVE_MAX_SUBQUESTIONS", "3"))))
ADAPTIVE_MAX_ROUNDS = max(1, min(3, int(os.environ.get("ADAPTIVE_MAX_ROUNDS", "2"))))  # complex 检索总轮数硬上限
ADAPTIVE_INSUFFICIENT_TEXT = os.environ.get("ADAPTIVE_INSUFFICIENT_TEXT") or "资料不足：现有论文指导资料中没有找到足以回答该问题的内容，建议换个问法或补充资料。"

# ---- 试卷库（/auto/ask 的 exam 路由与课程回答后的基础题推荐）----
EXAM_PRACTICE_MAX = max(0, min(3, int(os.environ.get("EXAM_PRACTICE_MAX", "3"))))   # 课程回答附带的基础题数，0 = 不附
EXAM_NO_MATCH_TEXT = os.environ.get("EXAM_NO_MATCH_TEXT") or "试卷库中没有找到与该问题对应的题目，可以换个知识点或题型再试。"

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

# Active HTTP requests per process, including streaming responses.
MAX_CONCURRENT_REQUESTS = int(os.environ.get("RAG_MAX_CONCURRENT_REQUESTS", "10"))
