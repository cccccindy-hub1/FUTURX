# nlp_ 学位论文指导知识库（RAG）

把 `nlp_/` 里的论文写作 / 研究方法资料建成可检索、可问答的知识库。
这是「数字治理科研团队 · 数字人」项目中「学位论文指导」场景的知识库组件。

## 目录结构

```
rag/
  config.py      全局配置（可用环境变量覆盖）
  extract.py     解析层：PDF/doc/docx/ppt/pptx -> 纯文本 + 元数据（跨平台，按能力探测解析器）
  chunk.py       分块层：标题感知 + 句子边界分块；可选父子分块
  index.py       入库层：BGE-m3 向量化 -> Chroma（+ 向量缓存 + 父块 sidecar）
  retrieval.py   检索层：稠密向量 / BM25 稀疏 / RRF 融合 / 重排序 / 元数据过滤
  enhance.py     查询增强：查询改写 + HyDE（均需 LLM）
  query.py       检索 + 生成（RAGService）
  benchmark.py   data_benchmark 切分：CSV -> 清洗 -> 按用户分组分层切分 train/test
  judge.py       打分：词法基线（char-F1 / ROUGE-L）+ LLM-as-judge
  eval.py        评估：检索指标（BM25 vs 稠密）与答案质量评测
  serve.py       常驻服务（REPL / HTTP）
  cli.py         命令行入口
  eval_data/     检索评估测试集（testset.json，含 answer_key）
  web/           前端页面（index.html / style.css / app.js，由 serve --http 托管在 /）
chroma_db/       向量库持久化目录（build 后生成）
cache/           向量缓存 / BM25 索引缓存 / 父块 sidecar（build 后生成）
```

## 安装

```bash
python -m venv .venv
.venv/bin/pip install -r rag/requirements.txt     # Windows: .venv/Scripts/pip
```

`.doc` / `.ppt` 是二进制旧格式，额外需要一个后端（二选一，按可用性自动选择）：

- **LibreOffice**（跨平台）：装上即可，程序会探测 `soffice`，或用 `SOFFICE_BIN` 指定路径；
- **Windows + Word**：安装 `pywin32`（已含在 requirements 的 win32 条件依赖里）后走 Word COM。

## 配置

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY（不填则只能检索、不能生成）
```

> 生成层走 OpenAI 兼容协议，`LLM_API_BASE` 换成任意兼容服务的地址即可
> （DeepSeek / OpenAI / 智谱 / 通义 / Moonshot …）。
> 全部可调项见 `.env.example`，**新增能力默认关闭**，保持原有「纯向量检索」行为。

## 使用

在项目根目录 `FUTURX` 下：

```bash
# 1) 建库（解析 + 分块 + 向量化 + 入库，全量重建）
.venv/bin/python -m rag.cli build

# 2) 问答（检索 + 生成）
.venv/bin/python -m rag.cli ask "扎根理论的研究程序是什么？"
.venv/bin/python -m rag.cli ask "学位论文开题阶段要完成什么？" -k 6
.venv/bin/python -m rag.cli ask "文献综述怎么写？" --mode hybrid --rerank --where category=002

# 3) 常驻服务（模型/向量库只加载一次，反复问，消除每次 ~18s 重载）
.venv/bin/python -m rag.cli serve             # 交互式 REPL
.venv/bin/python -m rag.cli serve --http      # FastAPI 服务（默认 127.0.0.1:8000）

# 4) 评估
.venv/bin/python -m rag.cli eval retrieval              # BM25 vs 稠密向量检索指标
.venv/bin/python -m rag.cli eval split                  # 切分 data_benchmark -> train/test
.venv/bin/python -m rag.cli eval answer --limit 20      # 答案质量评测（LLM-as-judge）
.venv/bin/python -m rag.cli eval answer --ablate        # 消融：dense/hybrid/rerank/改写 对比
```

启动 `serve --http` 后，浏览器打开 **http://127.0.0.1:8000/** 即前端对话界面：
支持提问、选择检索片段数（top_k）、查看带 `[来源N]` 的回答与可展开的参考来源卡片。

## 检索流水线

```
question
  │
  ├─(可选) 查询改写 / HyDE ── enhance.py ── 需 LLM
  │
  ├─ 稠密召回 (BGE-m3, Chroma) ─┐
  │                              ├─ RRF 融合 ─(可选) CrossEncoder 重排序 ─> top_k
  ├─ 稀疏召回 (BM25 + jieba) ───┘
  │
  ├─(可选) 父块展开 ── 命中子块替换为父块上下文
  │
  └─(可选) 余弦距离阈值过滤 ──> 拼 [来源N] 上下文 ──> LLM 生成
```

各开关的 CLI 参数与环境变量对应关系：

| 能力 | CLI | 环境变量（默认） |
|------|-----|------------------|
| 检索模式 | `--mode dense\|bm25\|hybrid` | `RETRIEVAL_MODE=dense` |
| 候选池大小 | `--fetch-k N` | `FETCH_K=20` |
| 重排序 | `--rerank` / `--no-rerank` | `RERANK_ENABLED=false` |
| 元数据过滤 | `--where KEY=VALUE`（可重复） | — |
| 查询改写 | `--rewrite` | `QUERY_REWRITE_ENABLED=false` |
| HyDE | `--hyde` | `HYDE_ENABLED=false` |
| 父块展开 | `--expand` | `CONTEXT_EXPAND_ENABLED=false` |

> **建库与查询的开关必须匹配**：父块 sidecar 只在 `PARENT_CHILD_ENABLED=true` 建库时生成，
> 此时才应使用 `--expand`。用默认参数建库、却加 `--expand` 会拿不到父块（自动退回子块原文，不报错）。

## 检索结果说明

- 每个 chunk 带 `source`（相对 nlp_ 的路径）、`category`（目录推导，如 `006/研究方法/扎根理论`）、
  `section`（章节标题）、`filename`、`ext` 元数据，检索后可按类别/来源过滤或引用。
- 开启父子分块时，子块额外带 `parent_id` / `parent_offset` / `parent_length`；父块全文不重复入库
  （见 `cache/<collection>.parents.json`），避免向量的重复冗余。
- 生成回答会标注 `[来源N]`，CLI 末尾也列出参考来源与余弦距离。

## 评测闭环

1. `eval split` 读取 `data_benchmark/*.csv`（目录可用 `BENCHMARK_DIR` 覆盖），
   按 `docs/data_benchmark_数据评估报告.md` 的建议做**按用户分组**的分层切分
   （同一 `来源姓名` 的问答不跨 train/test，避免相似问题泄漏），固定随机种子
   （`BENCHMARK_SEED=42`）保证可复现，结果落在 `data_benchmark/split/{train,test}.jsonl`。
2. `eval answer` 对 `test.jsonl` 逐条跑「检索 + 生成」，与标准答案比对：
   - 配了 `LLM_API_KEY`：用 LLM-as-judge（1~5 分 + 是否答对），输出严格 JSON；
   - 没配或解析失败：自动降级为词法基线（字符 bigram F1 映射到 5 分制）；
   - 无论走哪条路，都同时报告 `charF1` 与 `ROUGE-L`，便于横向对比。
3. `--ablate` 会跑 dense / hybrid / hybrid+rerank / hybrid+改写 的对照表。

> 评测要求测试问题对应的资料**已入库**（在 `nlp_/` 内），否则检索为空、分数无意义。
> 未找到 `data_benchmark/` 或 `split/test.jsonl` 时打印指引并优雅退出，不抛异常。

## 专家标注（预留接口）

当前为「机器标注」：自动切块 + 向量 + 上述元数据。要接入「专家标注」，可在这条通路上扩展：

1. **按 id 增删改**：chunk 的 id 形如 `路径哈希::文件名::chunkN`，可对 Chroma collection 直接
   `collection.update/delete/get`，实现人工改写原文、删除低质片段。
2. **打标签**：在 `index.py` 的 `metadatas` 里增加自定义字段（如 `expert_tag`、`quality`、
   `difficulty`），并提供一个人工标注入口写入这些字段；检索时用 `where=` 过滤。
3. **对齐方式**：专家标注数据可另存为 `rag/annotations.jsonl`（id -> 标签/修订），
   与机器向量库通过 id 对齐，不破坏自动建库流程。

## 已知限制

- **页码级引用缺失**：分块未记录 PDF 页码，精确到「参见某页」的引用待补。
- **`.ppt` / `.doc` 依赖外部后端**：未装 LibreOffice 且（非 Windows 或未装 Word）时会被跳过；
  `build` 会打印各格式的后端可用性，把静默跳过变成显式告警。
- **重排序需下载模型**：`--rerank` 首次会下载 `BAAI/bge-reranker-v2-m3`（约 2.3GB），
  因此默认不启用；CPU 上也会明显变慢。
- **查询改写 / HyDE / LLM 评审需 LLM key**：未配置时自动退回原查询 / 词法打分，不阻断流程。
- **评测的近似性**：`eval answer` 的答案质量分依赖 LLM 评委，与人工判断存在偏差；
  词法基线只看字面重合，长答复下会系统性偏低，适合做相对比较而非绝对结论。
