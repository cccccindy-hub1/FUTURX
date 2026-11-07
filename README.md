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

## HTTP parallel requests

`python -m rag.cli serve --http --host 127.0.0.1 --port 8000` accepts concurrent
requests to `/ask`, `/ask/stream`, and `/v1/chat/completions` (including SSE).
Set `RAG_MAX_CONCURRENT_REQUESTS=10` in `.env` to control the active-request limit
per process. Restart the server after changing it. Excess requests wait in a queue
(`RAG_QUEUE_TIMEOUT_S`, default 30s) and get HTTP 503 with `Retry-After: 1` only if
the wait times out. Health checks remain available at capacity.

Retrieval is no longer serialized: embeddings are merged into batches by
`rag/embedding.py` (tune with `EMBED_WORKERS` / `EMBED_TORCH_THREADS` /
`EMBED_MAX_CHARS`), Chroma and BM25 queries are read-only and run concurrently, and
LLM generation overlaps across requests. Timing data is local to each calling
thread. Streaming responses send the first SSE frame and the HTTP headers before
retrieval starts, so clients see the connection immediately. Upstream provider
rate limits can still limit throughput — measure them before blaming this layer.

## 多轮对话

`POST /v1/chat/completions` 会带上 `messages` 里之前的 user / assistant 轮次：
历史按原样插在系统提示之后交给模型（最多 `CHAT_HISTORY_MAX_TURNS` 轮、总字数至多
`CHAT_HISTORY_MAX_CHARS`，按整轮丢弃），调用方的 `system` 消息仍被忽略，服务端用自己的提示词。

追问（「第二种具体怎么算」）在检索前会先用历史改写成独立问题
（「数据资产估值的收益法具体怎么算」），改写只用于检索，交给模型的仍是原话。
改写走 `REWRITE_MODEL`、关闭思考、`CONTEXTUALIZE_TIMEOUT` 秒超时，失败即退回原问题。

## 会话存储与查询

每一轮问答按会话落 SQLite（`rag/conversation.py`，默认 `rag/conversation_db/conversations.sqlite3`，
已在 `.gitignore`）。这是**对话档案**；`rag/usage_log` 仍只管延迟 / token 计量，两者分工不重叠。

`/v1/chat/completions` 与 `/auto/ask`、`/auto/ask/stream` 都接受这三个扩展字段：

| 字段 | 说明 |
|---|---|
| `conversation_id` | 会话 id。不传则新建并回填；传了但不存在返回 404（避免拼错 id 导致历史悄悄丢失）|
| `store` | `false` = 本轮不落库（临时提问），默认 `true` |

响应体里的 `conversation_id` 与响应头 `X-Conversation-Id` 都会带回。同一请求重试按
`request_id` 去重，不会插两条。

管理接口（均按调用方 API key 隔离，别人的会话一律 404）：

```
GET    /v1/conversations                       会话列表（cursor 分页，archived 默认只看未归档）
POST   /v1/conversations                       新建（可选 title / 自定义 id）
GET    /v1/conversations/{id}                  详情（include_messages 默认 true）
PATCH  /v1/conversations/{id}                  改名 / 归档
DELETE /v1/conversations/{id}                  删除（软删，可恢复）
GET    /v1/conversations/{id}/messages         消息列表（after_seq 增量拉取，断线重连用）
DELETE /v1/conversations/{id}/messages         清空消息但保留会话
```

历史来源的优先级：客户端 `messages` 里有多轮时以客户端为准（OpenAI 兼容平台的常见做法），
只有一句本轮问题时用服务端存的历史——这样两种调用方式都不会把同一轮算两遍。

清理：`CONVERSATION_RETENTION_DAYS`（默认 365，0 = 永久），到期用
`python scripts/purge_conversations.py` 删除，建议 cron 每天跑一次。

完整的接口协议（字段、事件序列、错误码、典型调用流程）见
[`docs/API协议与交接.md`](docs/API协议与交接.md)。

### 接口冒烟测试

服务启动后，对本机实例跑一遍（基础部分不调用模型，几秒完成）：

```bash
export FUTURX_BASE=http://127.0.0.1:8000
export FUTURX_KEY=<.env 里 API_KEYS 的一把>

python scripts/smoke_api.py                          # 健康检查 + 会话管理 + 404 语义
python scripts/smoke_api.py --with-multi-turn        # 追加多轮对话与流式落库（真实调模型）
python scripts/smoke_api.py --with-concurrency       # 追加并发响应头跨度检查
```

退出码 0 表示全部通过。
