# FUTURX · 学科知识库（RAG）

面向高校数字治理科研团队的**数字分身（数字人）**项目，本仓库承载其中的**学位论文指导**场景知识库：将 `nlp_/` 中的学术写作与研究方法资料，构建为可语义检索、可问答的 RAG 知识库。

> 说明：`nlp_/` 原始资料默认不进入本仓库（见 `.gitignore`），需在本地自行放置后执行建库。

## 能力

- 多格式文档解析（PDF / doc / docx / ppt / pptx）
- 中文语义分块（标题感知 + 句子边界，带重叠）
- 本地向量化（BGE-m3）+ 向量库（Chroma，持久化）
- 语义检索 + 大模型生成回答（OpenAI 兼容协议）
- 机器标注 + 专家标注双轨预留接口

## 目录结构

```
rag/              核心代码（解析 / 分块 / 向量化 / 入库 / 检索 / 生成 / CLI）
docs/             建设规划文档
requirements.txt  环境依赖文件
.env.example      配置模板（复制为 .env 填入真实值）
```

## 快速开始

### 1. 创建 Conda 环境并安装依赖

使用 **Conda**（推荐 Python 3.10）管理环境：

```bash
# 1) 创建并激活 conda 虚拟环境
conda create -n futurx python=3.10 -y
conda activate futurx

# 2) 安装项目依赖
pip install -r requirements.txt
```

> **平台说明**：
> - **Windows**：依赖中已包含 `pywin32`，可直接利用本地 Microsoft Word COM 组件批量解析 `.doc` 文件。
> - **Linux / macOS**：如需解析旧格式 `.doc` / `.ppt` 文件，建议安装系统级 `LibreOffice`。

### 2. 配置环境变量

```bash
# Linux / macOS
cp .env.example .env

# Windows (CMD)
copy .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

编辑 `.env` 文件，至少填入 `LLM_API_KEY`（如需大模型生成回答；若不配置则仅执行向量检索）。

### 3. 构建知识库

将待检索的资料按分类放置于 `nlp_/` 目录下，执行全量建库（解析 → 分块 → 向量化 → 入库）：

```bash
python -m rag.cli build
```

### 4. 问答与检索

```bash
# 基础问答
python -m rag.cli ask "扎根理论的三级编码是什么？"

# 指定检索 top_k 片段数
python -m rag.cli ask "学位论文开题阶段要完成什么？" -k 6

# 混合检索 + 重排序 + 类别过滤
python -m rag.cli ask "文献综述怎么写？" --mode hybrid --rerank --where category=002
```

### 5. 常驻服务（可选）

```bash
# 启动命令行交互式 REPL（模型与向量库常驻内存，免去反复加载开销）
python -m rag.cli serve

# 启动 Web 前端服务（FastAPI + 对话界面，默认访问 http://127.0.0.1:8000）
python -m rag.cli serve --http
```

### 6. 运行测试

```bash
pytest
```

## 文档

- 使用说明与高级特性：`rag/README.md`
- 建设规划与设计文档：`docs/RAG知识库建设规划.md`

## License

版权所有，保留所有权利（不设开源许可证）。
