"""《数据要素》课程知识库（Ontology-Enhanced RAG）。

与论文指导知识库完全隔离：独立的数据路径、缓存目录与 collection，
不读写 rag/chroma_db，也不改变 rag.config 的任何默认值。
"""
