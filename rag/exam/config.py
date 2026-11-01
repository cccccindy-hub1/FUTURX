"""试卷库配置。所有路径与参数均可通过环境变量覆盖（或写入 .env）。

题库原文为课程内部资料，默认不放进仓库：用 EXAM_TEACHER_DOC / EXAM_STUDENT_DOC 指向本机文件。
教师版含学生版全部题干并附参考答案与评分要点，是解析的主来源；学生版只用于核对题目是否一一对应。
"""
import os
from pathlib import Path

from ..config import RAG_DIR
from ..course.config import COURSE_ONTO_ROOT

# ---- 输入 ----
EXAM_TEACHER_DOC = Path(os.environ.get("EXAM_TEACHER_DOC", RAG_DIR / "exam_data" / "teacher.docx"))
EXAM_STUDENT_DOC = Path(os.environ.get("EXAM_STUDENT_DOC", RAG_DIR / "exam_data" / "student.docx"))
# 学生知识集合（三层集合之一），用于把每道题挂到知识点上；默认取课程库本体目录下的同一份文件
EXAM_STUDENT_SET = Path(os.environ.get(
    "EXAM_STUDENT_SET", COURSE_ONTO_ROOT / "9.11update" / "三层集合" / "学生知识集合.xlsx"))
# 题库版本号写入 question_id，换版本时 id 整体换代
EXAM_BANK_VERSION = os.environ.get("EXAM_BANK_VERSION", "1031")

# ---- 输出（与 rag/chroma_db、rag/course_cache 分开）----
EXAM_CACHE_DIR = Path(os.environ.get("EXAM_CACHE_DIR", RAG_DIR / "exam_cache"))

# ---- 知识点挂接 ----
# 题干中字面出现、但未加引号的知识点，名称至少这么长才算（太短的如「数据」会挂到几乎所有题上）
EXAM_MENTION_MIN_CHARS = int(os.environ.get("EXAM_MENTION_MIN_CHARS", "4"))
# 每个概念最多挂接的知识点三元组数（按关系优先级取前若干）
EXAM_KP_PER_CONCEPT = int(os.environ.get("EXAM_KP_PER_CONCEPT", "6"))

# ---- 检索 ----
EXAM_SEARCH_TOPK = int(os.environ.get("EXAM_SEARCH_TOPK", "5"))
# BM25 最高分低于此值视为题库中没有对应的题（不硬凑）
EXAM_BM25_MIN_SCORE = float(os.environ.get("EXAM_BM25_MIN_SCORE", "4.0"))
# 同一次检索中，BM25 分不足最高分这一比例的题不要（避免拿弱相关的题凑够题数）
EXAM_BM25_REL = float(os.environ.get("EXAM_BM25_REL", "0.6"))
