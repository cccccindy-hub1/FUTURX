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

# 本体数据根目录，只读。两种结构自动识别（见 ontology.OntologyPaths）：
#   9.11 版：9.11update/onto1、onto2、三层集合 + 第1-10章_concept总表_问题标注版.xlsx
#   11.2 版：onto1、onto2、三层集合 + 第1-11章_concept总表_问题标注版.xlsx + 书稿（1102合成版）.docx
#     （用 11.2 版时同时设 COURSE_BOOK_VERSION=1102，chunk_id 随之换代）
COURSE_ONTO_ROOT = Path(os.environ.get("COURSE_ONTO_ROOT", RAG_DIR / "course_data" / "ontology"))

# ---- 输出（与 rag/chroma_db、rag/cache 分开）----
COURSE_CACHE_DIR = Path(os.environ.get("COURSE_CACHE_DIR", RAG_DIR / "course_cache"))

# ---- 分块 ----
PARENT_MAX = int(os.environ.get("COURSE_PARENT_MAX", "1500"))  # 父块字符上限（超出按段落边界切）
CHILD_MAX = int(os.environ.get("COURSE_CHILD_MAX", "500"))     # 子块打包上限
CHILD_MIN = int(os.environ.get("COURSE_CHILD_MIN", "120"))     # 短于此的段落与相邻段合并
# 只有超过此长度的单段才按句切。书稿0906最长段 671 字，取 800 保证任何段落（含整段定义）都不被切开
PARA_SPLIT = int(os.environ.get("COURSE_PARA_SPLIT", "800"))


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip().lower() in ("1", "true", "yes", "on")


# ---- 问答管线（M3）：胜算云 DeepSeek V4 Pro，OpenAI 兼容协议 ----
# 与 Claude Code 的 ANTHROPIC_* 无关。API_BASE 不设默认值，未配置时 llm.OpenAICompatClient 会报错提示。
COURSE_LLM_API_BASE = os.environ.get("COURSE_LLM_API_BASE", "").strip()
COURSE_LLM_API_KEY = os.environ.get("COURSE_LLM_API_KEY", "").strip()
COURSE_LLM_MODEL = os.environ.get("COURSE_LLM_MODEL") or "deepseek/deepseek-v4-pro"
COURSE_SELECT_MODEL = os.environ.get("COURSE_SELECT_MODEL") or COURSE_LLM_MODEL
COURSE_FILTER_MODEL = os.environ.get("COURSE_FILTER_MODEL") or COURSE_LLM_MODEL
COURSE_ANSWER_MODEL = os.environ.get("COURSE_ANSWER_MODEL") or COURSE_LLM_MODEL
COURSE_LLM_NATIVE_TOOLS = _env_bool("COURSE_LLM_NATIVE_TOOLS", True)  # False：prompt 约定 JSON + 本地解析
# Step 1 模式：single = 本地候选 + 一次 LLM 调用（默认）；agent = 旧的多轮工具调用
COURSE_SELECT_MODE = (os.environ.get("COURSE_SELECT_MODE") or "single").strip().lower()
COURSE_SELECT_CANDIDATES = int(os.environ.get("COURSE_SELECT_CANDIDATES", "30"))   # single 模式候选概念数上限
# Step 3 筛选一次调用最多送入的子块数；超出时先留本体有证据的块，再按 BM25 分取前若干，丢弃数写入 trace
COURSE_FILTER_MAX_CHUNKS = int(os.environ.get("COURSE_FILTER_MAX_CHUNKS", "40"))
# 召回块数不超过此值时跳过筛选（省一次调用），全部直接进入作答；作答本身最多用 COURSE_ANSWER_MAX_CHUNKS 块，故默认取同值；0 = 总是筛选
COURSE_FILTER_SKIP_MAX = int(os.environ.get("COURSE_FILTER_SKIP_MAX", "15"))
COURSE_AGENT_MAX_STEPS = int(os.environ.get("COURSE_AGENT_MAX_STEPS", "7"))      # Step 1 最多 LLM 调用次数（含到上限后强制 finish 的那一次，即 4 = 最多 4 次选词调用）
# Step 1 快速路径（实验开关，默认关闭）：问题里只命中唯一、无歧义的概念别名时，不调用 LLM 直接选中，
# 并补入最多 COURSE_FAST_RELATED_MAX 个一跳关联概念（见 agent_select.fast_select）；0 = 不补关联概念
COURSE_SELECT_FAST_PATH = _env_bool("COURSE_SELECT_FAST_PATH", False)
COURSE_FAST_RELATED_MAX = int(os.environ.get("COURSE_FAST_RELATED_MAX", "5"))
COURSE_RECALL_MAX_CHARS = int(os.environ.get("COURSE_RECALL_MAX_CHARS", "12000"))  # Step 3 单批子块正文字符上限
SINK_THRESHOLD = float(os.environ.get("COURSE_SINK_THRESHOLD", "0.6"))           # 标题行证据下沉的覆盖率阈值
# Step 4 最多使用的子块数：先取筛选为 direct 的，再取 background，超出部分记入 trace
COURSE_ANSWER_MAX_CHUNKS = int(os.environ.get("COURSE_ANSWER_MAX_CHUNKS", "15"))
# 各步骤的思考模式（胜算云 reasoning 参数）：disabled / low / medium / high；留空则不发送，用模型默认
COURSE_REASONING = {
    "select": os.environ.get("COURSE_SELECT_REASONING", "disabled").strip(),
    "filter": os.environ.get("COURSE_FILTER_REASONING", "disabled").strip(),
    "answer": os.environ.get("COURSE_ANSWER_REASONING", "low").strip(),
}
# HTTP 接口（rag.course.api）单个问题的字符上限，防止超长输入放大 LLM 费用
COURSE_QUESTION_MAX_CHARS = int(os.environ.get("COURSE_QUESTION_MAX_CHARS", "500"))


# 兜底召回：用问题对教材子块做 BM25（jieba），取 top-k 与本体召回合并去重后再进入筛选；0 = 关闭（见 rag.course.fallback）
COURSE_BM25_TOPK = int(os.environ.get("COURSE_BM25_TOPK", "6"))
