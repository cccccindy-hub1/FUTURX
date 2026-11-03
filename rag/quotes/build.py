"""名言库建库：python -m rag.quotes.build [--doc DOCX] [--rebuild] [--dry-run]

切块（如缺）+ 写入 rag/quote_store（集合 xi_digital_quotes）。绝不读写 rag/chroma_db；目录已存在且非空时拒绝，除非 --rebuild。
原始资料取 .env 的 QUOTE_DOC（默认 rag/quote_data/ 下）。向量化只用本地 BGE-m3，调用大模型 0 次。逻辑见 rag.kbbuild。
"""
from ..kbbuild import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli("quotes"))
