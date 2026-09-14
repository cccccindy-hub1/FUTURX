# nlp_ 学位论文指导知识库（RAG）

把 `nlp_/` 里的论文写作 / 研究方法资料建成可检索、可问答的知识库。
这是「数字治理科研团队 · 数字人」项目中「学位论文指导」场景的知识库组件。

## 目录结构

```
rag/
  config.py      全局配置（可用环境变量覆盖）
  extract.py     解析层：PDF/doc/docx -> 纯文本 + 元数据
  chunk.py       分块层：标题感知 + 句子边界的中文分块
  index.py       入库层：BGE-m3 向量化 -> Chroma
  query.py       检索 + LLM 生成
  cli.py         命令行入口
chroma_db/       向量库持久化目录（build 后生成）
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r rag/requirements.txt
```

## 配置

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY（不填则只能检索、不能生成）
```

> 生成层走 OpenAI 兼容协议，`LLM_API_BASE` 换成任意兼容服务的地址即可
> （DeepSeek / OpenAI / 智谱 / 通义 / Moonshot …）。

## 使用

在项目根目录 `FUTURX` 下：

```bash
# 1) 建库（解析 + 分块 + 向量化 + 入库，全量重建）
.venv/bin/python -m rag.cli build

# 2) 问答（检索 + 生成）
.venv/bin/python -m rag.cli ask "扎根理论的研究程序是什么？"
.venv/bin/python -m rag.cli ask "学位论文开题阶段要完成什么？" -k 6

# 3) 常驻服务（模型/向量库只加载一次，反复问，消除每次 ~18s 重载）
.venv/bin/python -m rag.cli serve             # 交互式 REPL
.venv/bin/python -m rag.cli serve --http      # FastAPI 服务（默认 127.0.0.1:8000）
```

## 检索结果说明

- 每个 chunk 带 `source`（相对 nlp_ 的路径）、`category`（方法类别，如 `研究方法/扎根理论`）、
  `section`（章节标题）、`filename` 元数据，检索后可按类别/来源过滤或引用。
- 生成回答会标注 `[来源N]`，CLI 末尾也列出参考来源与余弦距离。

## 专家标注（预留接口）

当前为「机器标注」：自动切块 + 向量 + 上述元数据。要接入「专家标注」，可在这条通路上扩展：

1. **按 id 增删改**：chunk 的 id 形如 `文件名::chunkN`，可对 Chroma collection 直接
   `collection.update/delete/get`，实现人工改写原文、删除低质片段。
2. **打标签**：在 `index.py` 的 `metadatas` 里增加自定义字段（如 `expert_tag`、`quality`、
   `difficulty`），并提供一个人工标注入口写入这些字段；检索时用 `where=` 过滤。
3. **对齐方式**：专家标注数据可另存为 `rag/annotations.jsonl`（id -> 标签/修订），
   与机器向量库通过 id 对齐，不破坏自动建库流程。

## 已知限制

- 唯一一个 `.ppt`（`002/MPA学位论文选题规范与管理2015.ppt`）无法用当前工具解析，build 时会跳过。
  需要的话装 LibreOffice 后可补抽。
- 分块未记录 PDF 页码，后续如需「参见某页」级引用可再加页码元数据。
