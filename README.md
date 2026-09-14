# FUTURX · 学科知识库（RAG）

面向高校数字治理科研团队的**数字分身（数字人）**项目，本仓库承载其中的**学位论文指导**场景知识库：将 `nlp_/` 中的学术写作与研究方法资料，构建为可语义检索、可问答的 RAG 知识库。

> 说明：`nlp_/` 原始资料默认不进入本仓库（见 `.gitignore`），需在本地自行放置后执行建库。

## 能力

- 多格式文档解析（PDF / doc / docx）
- 中文语义分块（标题感知 + 句子边界，带重叠）
- 本地向量化（BGE-m3）+ 向量库（Chroma，持久化）
- 语义检索 + 大模型生成回答（OpenAI 兼容协议）
- 机器标注 + 专家标注双轨预留接口

## 目录结构

```
rag/              核心代码（解析 / 分块 / 向量化 / 入库 / 检索 / 生成 / CLI）
docs/             建设规划文档
.env.example      配置模板（复制为 .env 填入真实值）
```

## 快速开始

```bash
# 1) 安装依赖
python3 -m venv .venv
.venv/bin/pip install -r rag/requirements.txt

# 2) 配置生成层（可选，不配则仅检索）
cp .env.example .env    # 填入 LLM_API_KEY 等

# 3) 建库（解析 → 分块 → 向量化 → 入库）
.venv/bin/python -m rag.cli build

# 4) 问答
.venv/bin/python -m rag.cli ask "扎根理论的三级编码是什么？"
```

## 文档

- 使用说明：`rag/README.md`
- 建设规划：`docs/RAG知识库建设规划.md`

## License

版权所有，保留所有权利（不设开源许可证）。
