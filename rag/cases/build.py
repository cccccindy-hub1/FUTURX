"""判例库建库：python -m rag.cases.build [--doc DOCX] [--rebuild] [--dry-run]

切块（如缺）+ 写入 rag/case_store（集合 data_law_cases）。绝不读写 rag/chroma_db；目录已存在且非空时拒绝，除非 --rebuild。
原始资料取 .env 的 CASE_DOC（默认 rag/case_data/ 下）。向量化只用本地 BGE-m3，调用大模型 0 次。逻辑见 rag.kbbuild。
"""
from ..kbbuild import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli("cases"))
