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

# 本体数据根目录（含 9.11update/onto1、onto2、三层集合 与 第1-10章_concept总表_问题标注版.xlsx），只读
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
COURSE_AGENT_MAX_STEPS = int(os.environ.get("COURSE_AGENT_MAX_STEPS", "6"))      # Step 1 最多工具调用轮数
COURSE_RECALL_MAX_CHARS = int(os.environ.get("COURSE_RECALL_MAX_CHARS", "12000"))  # Step 3 单批子块正文字符上限
SINK_THRESHOLD = float(os.environ.get("COURSE_SINK_THRESHOLD", "0.6"))           # 标题行证据下沉的覆盖率阈值
# Step 4 最多使用的子块数：先取筛选为 direct 的，再取 background，超出部分记入 trace
COURSE_ANSWER_MAX_CHUNKS = int(os.environ.get("COURSE_ANSWER_MAX_CHUNKS", "15"))
# 各步骤的思考模式（胜算云 reasoning 参数）：disabled / low / medium / high；留空则不发送，用模型默认
COURSE_REASONING = {
    "select": os.environ.get("COURSE_SELECT_REASONING", "").strip(),
    "filter": os.environ.get("COURSE_FILTER_REASONING", "disabled").strip(),
    "answer": os.environ.get("COURSE_ANSWER_REASONING", "low").strip(),
}
# HTTP 接口（rag.course.api）单个问题的字符上限，防止超长输入放大 LLM 费用
COURSE_QUESTION_MAX_CHARS = int(os.environ.get("COURSE_QUESTION_MAX_CHARS", "500"))

# ---- 政策库（rag.course.policy）：数据政策汇编，按「一条完整的意思」切块 ----
# 汇编为整理的外部资料，默认不放进仓库：用 COURSE_POLICY_DOC 指向本机文件；输出到 COURSE_CACHE_DIR/policy
COURSE_POLICY_DOC = Path(os.environ.get("COURSE_POLICY_DOC", RAG_DIR / "course_data" / "数据政策汇编.docx"))
POLICY_CHUNK_MAX = int(os.environ.get("COURSE_POLICY_CHUNK_MAX", "1500"))      # 超过此长度的条/事项才拆开
POLICY_PART_TARGET = int(os.environ.get("COURSE_POLICY_PART_TARGET", "1000"))  # 拆开时每片的目标长度
# 标题含这些词的文件不入政策库（数据字典类标准：逐字段定义，不是政策条文），逗号分隔
POLICY_EXCLUDE_TITLES = tuple(
    s.strip() for s in os.environ.get("COURSE_POLICY_EXCLUDE_TITLES", "数据规范").split(",") if s.strip()
)
POLICY_DUP_JACCARD = float(os.environ.get("COURSE_POLICY_DUP_JACCARD", "0.8"))     # 两文件正文相似度达到即视为重复
POLICY_DRAFT_JACCARD = float(os.environ.get("COURSE_POLICY_DRAFT_JACCARD", "0.5"))  # 征求意见稿与正式稿的相似度门槛
