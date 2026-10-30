"""政策库：数据政策汇编按「一条完整的意思」切块，供语义匹配。

与课程知识库（rag.course，《数据要素》教材 + 本体）分开：独立的配置（POLICY_*）、
命令行（python -m rag.policy.cli）和输出目录（POLICY_CACHE_DIR，默认 rag/policy_cache）。
只复用 rag.course 里读 docx XML 的底层函数。
"""
