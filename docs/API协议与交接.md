# AI 李老师 · 接口协议与交接文档

对应代码：`rag/serve.py`（应用装配）、`rag/api.py`（OpenAI 兼容层）、`rag/unified.py`（统一入口）、
`rag/conversation_api.py`（会话管理）、`rag/conversation.py`（会话存储）。
文档对应提交：`eb5ee75`。

---

## 0. 一分钟版

| 你要做的 | 用哪个 |
|---|---|
| 通用对话（论文 / 课程 / 政策 / 判例 / 名言，自动路由） | `POST /auto/ask` |
| 同上但要流式 | `POST /auto/ask/stream`（SSE） |
| 只问论文写作指导 | `POST /v1/chat/completions`（OpenAI 兼容） |
| 接第三方 OpenAI 兼容平台 | `POST /v1/chat/completions` |
| 想保存 / 读取历史对话 | 上面三个都支持 `conversation_id`；读取用 `/v1/conversations*` |

**推荐做法**：客户端自己保存 `conversation_id`，每轮只发本轮问题，
历史由服务端从库里取；同时把本轮的 `user` 消息带在 `messages` 里（`/v1`）或 `question`（`/auto/ask`）。

---

## 1. 基础约定

### 1.1 服务地址与启动

```
python -m rag.cli serve --http --host 0.0.0.0 --port 8000
# 生产：systemctl restart futurx
```

- 单进程部署，进程内并发上限 `RAG_MAX_CONCURRENT_REQUESTS`（默认 100）。
- 超限先排队（`RAG_QUEUE_TIMEOUT_S`，默认 30s），等到名额就放行；等待超时才 503 + `Retry-After: 1`。

### 1.2 鉴权

```http
Authorization: Bearer <API_KEY>
```

- 由 `.env` 的 `API_KEYS`（逗号分隔）或 `API_KEYS_FILE`（每行一个 key，`#` 开头为注释）配置，
  两者同时存在取并集。**两者都为空时不校验**，该实例内所有调用视为同一匿名身份。
- 按人发 key（例如一个班上千个）时用 `API_KEYS_FILE`：1000 个 key 塞进 `.env` 一行约 48KB，
  既难维护也容易误提交。生成方式见 12.3。
- **改了密钥文件不用重启**：服务端按文件内容哈希检查，默认 5 秒内生效（`API_KEYS_RELOAD_S`）。
  撤销/新增/轮换都是「改文件即生效」，见 12.4。
- 校验用集合查找，千级 key 下每次请求仍是 O(1)。
- **启动自检**：配了 `API_KEYS_FILE` 但文件不存在或里面没有任何 key 时**拒绝启动**。
  这是为了防止「文件被误删 → 静默变成不鉴权」。
- 失败返回 `401`：

```json
{"detail": {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}}
```

> 注意：`/health` 不需要鉴权，其余接口都需要。

### 1.3 错误格式

**FastAPI 的 HTTPException 会包一层 `detail`**，非 2xx 响应统一形如：

```json
{"detail": {"error": {"message": "...", "type": "invalid_request_error", "code": "not_found"}}}
```

`type` 取值：`invalid_request_error` | `service_unavailable` | `upstream_error`。
流式响应开始后的错误不再用 HTTP 状态码，改为在流里发一个 `error` 事件。

### 1.4 状态码

| 码 | 场景 |
|---|---|
| 200 | 正常 |
| 400 | 请求体不合法（如缺 user 消息、question 为空） |
| 401 | API key 无效 |
| 404 | `conversation_id` 不存在或不属于当前 key |
| 409 | 新建会话时 id 已存在 |
| 422 | FastAPI 参数校验失败（自带格式，不是上面的 `error` 包裹） |
| 500 | 未预期异常 |
| 502 | 下游模型调用失败（只回类型名） |
| 503 | 未配置 LLM / 统一入口未就绪 / 并发排队超时 |

### 1.5 会话归属

会话按 API key 的指纹（sha256 前 8 位）归属。**A 的 key 查不到 B 的会话，一律 404**（不泄露存在性）。

---

## 2. `POST /auto/ask` —— 统一入口（推荐主用）

一次调用完成：路由 → 对应知识库检索 → 作答 → 追问建议。

### 2.1 请求

```json
{
  "question": "数据资产估值有哪些方法？",
  "top_k": 4,
  "include_trace": false,
  "conversation_id": "conv_xxx",
  "store": true
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `question` | string | 是 | — | 1–500 字。**这是路由、检索、作答的依据** |
| `top_k` | int | 否 | 服务端配置 | 检索片段数 |
| `include_trace` | bool | 否 | `true` | 是否返回调试信息。**生产建议 `false`**，可省流量 |
| `conversation_id` | string | 否 | — | 会话 id，见第 4 节 |
| `store` | bool | 否 | `true` | `false` = 本轮不落库 |

### 2.2 响应 `200`

```json
{
  "route": "course",
  "routes": ["course"],
  "complexity": "simple",
  "reason": "课程库命中",
  "count": 1,
  "chapter": null,
  "status": "ok",
  "answer": "……正文，引用形如 [1] [2]……",
  "sources": [
    {"n": 1, "chunk_id": "...", "path": "第3章 › 3.2 数据资产"}
  ],
  "followups": ["要不要来几道基础题？"],
  "conversation_id": "conv_xxx"
}
```

| 字段 | 说明 |
|---|---|
| `route` | 实际路线，见下表 |
| `routes` | 参与的库；组合问答时 >1，此时 `route == routes[0]` |
| `complexity` | `simple` / `complex`（仅论文路线有意义） |
| `status` | 见 2.3 |
| `answer` | 正文。`status != ok` 时也可能是固定文案 |
| `sources` | 引用列表，`n` 对应正文里的 `[n]`；结构随 route 略有不同（课程/判例/政策/名言） |
| `followups` | 追问建议，可能缺省；`off_topic` / `quote` / 非 ok 时为空 |
| `conversation_id` | 落库时才有 |
| `trace` | `include_trace=true` 时有，含路由耗时与各阶段明细，**不要用于业务逻辑** |

`route` 取值：

| route | 含义 |
|---|---|
| `course` | 课程教材（《数据要素》知识库） |
| `thesis` | 论文写作指导（另可含 `exam` 库的解题结果） |
| `policy` | 政策文件库 |
| `case` | 判例库 |
| `quote` | 讲话/名言库（原文照录，不生成） |
| `exam` | 试卷库（默认只给题目） |
| `off_topic` | 超出范围，固定拒答，不检索不调模型 |

组合（`routes` 有两个）仅在 `course` / `case` / `quote` / `policy` 之间发生，只调一次作答模型。

### 2.3 `status`

| status | 含义 | 客户端建议 |
|---|---|---|
| `ok` | 正常作答 | 展示 |
| `insufficient` | 资料不足 | 展示提示文案，可建议换个问法 |
| `refused` | 超出范围（`off_topic`） | 展示拒答文案 |
| `unavailable` | 某个库未加载 | 提示能力未部署 |
| `error` | 下游失败 | 可重试 |

### 2.4 示例

```bash
curl -s http://127.0.0.1:8000/auto/ask \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"数据资产估值有哪些方法？","include_trace":false}'
```

---

## 3. `POST /auto/ask/stream` —— SSE 流式

请求体与 `/auto/ask` 完全一致。响应 `Content-Type: text/event-stream`，每行 `data: <json>`，事件按顺序：

### 3.1 事件序列

```
route  →  (stage)*  →  (materials | delta)*  →  done  →  (followups)?
                           └─ 出错时以 error 取代/插入
```

- `route`：第一个事件，路由结果。**客户端可据此先渲染「正在查询 XX 库」**。
- `stage`：仅课程路线有，取材进度（`select` / `filter` / `combined`），只用于展示进度条。
- `materials`：分配引用序号时发出，`n` / `chunk_id` / `path`，可用于提前渲染引用卡片。
- `delta`：正文增量，`{"type":"delta","text":"..."}`，客户端拼接。
- `done`：最终结果。**字段与 `/auto/ask` 的非流式响应相同**（含完整 `answer`、`sources`、`status`）。
  课程/组合路线里 `done.answer` 可能做过引号校验，**应以 `done.answer` 覆盖拼接得到的文本**。
- `followups`：`done` 之后单独发，可能没有。
- `error`：`{"type":"error","error":"SomeErrorName"}`，只带异常类型名。

### 3.2 各事件示例

```json
{"type":"route","route":"course","routes":["course"],"complexity":"simple","count":1,"chapter":null,"reason":"...","router_s":0.42}
{"type":"stage","stage":"select","recalled":18}
{"type":"materials","n":1,"chunk_id":"...","path":"第3章 › 3.2"}
{"type":"delta","text":"数据资产估值主要有"}
{"type":"done","route":"course","routes":["course"],"complexity":"simple","status":"ok","answer":"……全文……","sources":[{"n":1,"chunk_id":"...","path":"..."}],"trace":{},"conversation_id":"conv_xxx"}
{"type":"followups","items":["要不要来几道基础题？"]}
```

非课程路线（政策、判例、名言、论文）不产生 `delta`，完整答案直接放在 `done` 里。

### 3.3 客户端注意

- 响应头无 `X-Accel-Buffering` 缓冲；反向代理需关掉 buffering。
- 断线：服务端会捕获断开，把这轮以 `status=partial`（`/v1`）或 `error=client_disconnect`（`/auto`）落库，不会留半截脏数据。
- 超时：上游模型慢时首字可能到 10s+，客户端读超时建议 **≥120s**。
- 末帧不是 `[DONE]`（这是 `/v1` 的约定），以 `done` / `error` 事件作为结束标志，然后流关闭。

---

## 4. `POST /v1/chat/completions` —— OpenAI 兼容层（仅论文库）

供第三方 OpenAI 兼容客户端接入。**只走论文 RAG（route 固定 `thesis`）**，
要课程 / 政策 / 判例 / 名言请用 `/auto/ask`。

### 4.1 请求

```json
{
  "model": "ai-li-teacher",
  "messages": [
    {"role": "user", "content": "数据资产估值有哪些方法？"},
    {"role": "assistant", "content": "主要有成本法、收益法、市场法。"},
    {"role": "user", "content": "第二种具体怎么算？"}
  ],
  "stream": false,
  "top_k": 4,
  "conversation_id": "conv_xxx",
  "store": true
}
```

| 字段 | 说明 |
|---|---|
| `model` | 任意值都接受，仅回显。真实模型见 `/v1/models` 的 `upstream_model` |
| `messages` | 标准 OpenAI 结构。`content` 支持字符串或 `[{type:"text",text:"..."}]` 数组 |
| `stream` | `true` 走 SSE |
| `temperature` | 接受但当前不影响作答 |
| `top_k` | 扩展：检索片段数 |
| `conversation_id` / `store` | 扩展：见第 5 节 |

**`messages` 的处理规则**（重要）：

- 最后一条非空 `user` 消息 = 本轮问题（检索与作答都用它）。
- 之前的 `user` / `assistant` 消息 = 客户端历史，按原样插入系统提示之后。
- `system` 消息**被忽略**，服务端用自己的提示词。
- 历史上限 `CHAT_HISTORY_MAX_TURNS`（默认 6 轮）与 `CHAT_HISTORY_MAX_CHARS`（默认 8000 字），按整轮丢弃。
- **历史与 `conversation_id` 同时存在时**：`messages` 里有多轮 → 以客户端为准；只有本轮一句 → 用服务端存的。

### 4.2 非流式响应

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1762500000,
  "model": "ai-li-teacher",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "……"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "sources": [{"source": "xxx.pdf", "section": "...", "distance": 0.21}],
  "conversation_id": "conv_xxx"
}
```

`usage` 的三个数当前恒为 0（未接 tokenizer）；`sources` 与 `conversation_id` 是扩展字段，标准客户端会忽略。

### 4.3 流式响应

标准 SSE，`data: {"choices":[{"delta":{...}}]}`，以 `data: [DONE]` 结束。

帧顺序有明确约定：

1. 立刻发首帧 `{"delta":{"role":"assistant","content":""}}` —— **此时检索还没开始**，用来让客户端尽快拿到响应头。
2. 逐段 `{"delta":{"content":"..."}}`。
3. 末帧 `finish_reason: "stop"`，**额外带 `sources` 与 `conversation_id`**。
4. `data: [DONE]`。

> 因此「首字时间」= 第 2 步第一帧，不是第 1 步。压测时别把 role 帧算进去。

### 4.4 其他 OpenAI 兼容接口

`GET /v1/models` —— 返回 `ai-li-teacher` 以及真实上游模型名。

---

## 5. 会话：两个问答入口的共同扩展

### 5.1 字段语义

| 字段 | 行为 |
|---|---|
| `conversation_id` 不传 | 新建会话，响应体 + `X-Conversation-Id` 头带回 id |
| `conversation_id` 传了且存在 | 追加到该会话，历史从库里取 |
| `conversation_id` 传了但不存在 | **404**，不静默新建（避免拼错 id 后历史悄悄丢失） |
| `store=false` | 本轮不落库、不取历史，响应里没有 `conversation_id` |

### 5.2 追问改写（服务端自动，客户端不用管）

当本轮问题像追问（短句，或含「那 / 它 / 第二种 / 再 / 继续」等），服务端会：
先用历史把问题改写成独立问句（`第二种具体怎么算？` → `市场法估值具体怎么算？`），
再用它去**路由、检索、作答**；改写只走 `REWRITE_MODEL`、关思考、8s 超时，失败自动退回原问题。

改写后的问题会存在消息的 `search_query` 字段里，排查「为什么答偏了」时直接看它。

### 5.3 一轮问答落库的内容

每次成功/部分成功的一轮，写两条消息（同一事务）：

| 字段 | user 消息 | assistant 消息 |
|---|---|---|
| `seq` | 2n-1 | 2n（会话内自增，重放与分页都靠它） |
| `content` | 本轮原话 | 回答正文（流式断开时是已生成的半截） |
| `route` | 路线 | 路线 |
| `status` | 同助手 | `ok` / `partial` / `error` |
| `search_query` | 改写后的检索问题 | — |
| `sources` | `[]` | 引用列表 |
| `latency_ms` | — | 本轮耗时 |
| `key_id` | 发起这次调用的 API key 指纹 | 同 user |
| `month` | `YYYY-MM`（本地时区） | 同 user |

`key_id` 在消息级也记一份，而不是只靠会话的归属：会话可能被迁移（老数据归置、换人接手），
但「这一轮当时是谁问的」不该跟着变。会话被 `reassign_key` 改归属时，只补 `key_id IS NULL`
的消息，不会覆盖后来新写入的记录。

重试同一请求按 `request_id` 去重，不会插两条。

---

## 6. 会话管理接口

均需鉴权，均按 key 隔离。

### 6.1 `GET /v1/conversations` 列表

| 参数 | 默认 | 说明 |
|---|---|---|
| `limit` | 20 | 1–100 |
| `cursor` | — | 上一页返回的 `next_cursor` 原样回传 |
| `archived` | `false` | `false` 只看未归档 / `true` 只看已归档 / `all` 全部（其它值 400） |

```json
{
  "object": "list",
  "data": [{
    "id": "conv_xxx", "object": "conversation", "title": "数据资产估值有哪些方法？",
    "created_at": 1762500000, "created_at_iso": "2026-11-07T20:00:00+0800",
    "updated_at": 1762500060, "updated_at_iso": "...",
    "turn_count": 3, "archived": false
  }],
  "next_cursor": "1762500060:conv_xxx",
  "has_more": true
}
```

### 6.2 `POST /v1/conversations` 新建

```json
{"title": "可选，≤200 字", "id": "可选，客户端自己生成，≤80 字"}
```

返回会话对象。id 已存在 → `409`。

> 一般不需要手动建：问答接口不传 `conversation_id` 会自动建。

### 6.3 `GET /v1/conversations/{id}` 详情

| 参数 | 默认 | 说明 |
|---|---|---|
| `include_messages` | `true` | 是否带消息 |
| `limit` | 100 | 消息条数 1–500 |

返回会话对象 + `messages: [...]`（结构见 6.5）。

### 6.4 `PATCH /v1/conversations/{id}` 改名 / 归档

```json
{"title": "新标题", "archived": true}
```

两个字段都可选，只传一个就只改一个。

### 6.5 `GET /v1/conversations/{id}/messages` 消息列表

| 参数 | 默认 | 说明 |
|---|---|---|
| `after_seq` | — | 只取 `seq >` 该值的消息，**断线重连用** |
| `limit` | 200 | 1–500 |

```json
{
  "object": "list",
  "data": [{
    "id": "...", "object": "conversation.message", "conversation_id": "conv_xxx",
    "seq": 1, "role": "user", "content": "...", "created_at": 1762500000,
    "route": "course", "status": "ok", "error": null,
    "search_query": "数据资产估值有哪些方法？",
    "sources": [], "latency_ms": null,
    "key_id": "c72f2e45", "month": "2026-11"
  }],
  "next_cursor": 2
}
```

`key_id` 是这一轮提问所用 API key 的指纹（sha256 前 8 位，**不是 key 原文**），
`month` 是 `YYYY-MM`（本地时区）。两者都按人/按月留档，重算统计不用扫时间戳。

`next_cursor` 是**最后一条的 seq**，下一轮直接当 `after_seq` 传。
分页判断：`len(data) == limit` 说明可能还有。

### 6.6 `DELETE /v1/conversations/{id}` 删除

软删除，`{"id":"...","object":"conversation.deleted","deleted":true}`。
删除后该 id 视同不存在（404），保留期内可人工恢复。

### 6.7 `DELETE /v1/conversations/{id}/messages` 清空消息

保留会话本身，`{"id":"...","object":"conversation.messages.cleared","deleted_messages":6}`。
适合「同一会话里开新话题」。

---

## 7. 其他接口（已存在，非本次改动）

| 接口 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}`，免鉴权 |
| `POST /ask` | 论文库单轮，`{question, top_k}` → `{answer, sources, contexts}`。无会话 |
| `POST /ask/stream` | 论文库单轮 SSE：`sources` → `delta*` → `done` |
| `POST /course/ask` | 课程库直连，支持 `dry_run` / `active_concepts` / `strategy` |
| `POST /exam/search`、`/exam/by_concepts`、`GET /exam/question/{id}` | 试卷库 |

`/ask`、`/ask/stream` **不支持会话**，新接入建议直接用 `/auto/ask`。

---

## 8. 典型调用流程

### 8.1 新会话 + 追问（推荐）

```
① POST /auto/ask        {"question":"数据资产估值有哪些方法？"}
   ← {"conversation_id":"conv_A", "answer":"...", ...}

② 客户端存 conv_A

③ POST /auto/ask        {"conversation_id":"conv_A", "question":"第二种具体怎么算？"}
   ← 服务端取 conv_A 历史 → 改写为「市场法具体怎么算」→ 路由 → 作答
```

### 8.2 第三方 OpenAI 客户端（自己管历史）

每轮把完整 `messages` 发上来，**不要**传 `conversation_id`，
服务端以客户端历史为准；这样最稳，不依赖服务端状态。

### 8.3 前端断线重连

```
GET /v1/conversations/{id}/messages?after_seq=<本地最后一条 seq>
```

拿到缺失的消息补齐，然后继续发新问题。

### 8.4 网页前端的历史会话

`rag/web/` 的页面已经接上会话，行为如下（改前端时按这个约束走）：

| 动作 | 行为 |
|---|---|
| 发第一句 | 服务端建会话，id 从响应头 `X-Conversation-Id` 取回，存 `localStorage`（按 API key 分开存） |
| 继续追问 | 请求体带 `conversation_id`，服务端取历史 → 改写 → 作答 |
| 刷新页面 | 会话 id 从 `localStorage` 恢复，侧栏从 `/v1/conversations` 重新拉 |
| 点侧栏某条 | `GET /v1/conversations/{id}` 取 `messages`，逐条回填；**回填的消息不带耗时行与追问胶囊** |
| 换访问密钥 | 清空对话区，历史列表按新 key 重新拉（会话按 key 隔离） |
| 新对话 | 只清掉当前会话 id，历史列表照旧 |
| 会话已被删 / 换了 key | `/auto/ask/stream` 返回 404 时自动丢弃本地 id 并提示重发 |

回填时引用序号靠 assistant 消息的 `sources` 重建（`[n]` → 路径），所以 `sources` 必须落库——
这也是 `conversation.py` 里 assistant 消息保留 `sources` 字段的原因。

---

## 9. 配置项（与本次功能相关）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CHAT_HISTORY_MAX_TURNS` | 6 | 最多带几轮历史；0 = 只答最后一句 |
| `CHAT_HISTORY_MAX_CHARS` | 8000 | 历史总字数上限，按整轮丢弃 |
| `CONTEXTUALIZE_QUERY_ENABLED` | true | 是否做追问改写 |
| `CONTEXTUALIZE_TIMEOUT` | 8 | 改写超时（秒），超时退回原问题 |
| `CONVERSATION_DB` | `rag/conversation_db/conversations.sqlite3` | 会话库位置 |
| `CONVERSATION_RETENTION_DAYS` | 365 | 保留天数；0 = 永久 |
| `EMBED_WORKERS` | 2 | 向量化 worker 数 |
| `EMBED_TORCH_THREADS` | 4 | 每个 worker 的 torch 线程数（两者乘积 ≈ 核数） |
| `RAG_MAX_CONCURRENT_REQUESTS` | 100 | 单进程并发上限 |

会话库清理（建议 cron 每天一次）：

```bash
0 4 * * * cd /home/FUTURX && /root/miniconda3/envs/futurx/bin/python scripts/purge_conversations.py
```

---

## 10. 已知限制

1. **`/v1/chat/completions` 只覆盖论文库**。要全库能力必须用 `/auto/ask`。
2. **`/auto/ask` 的 `question` 上限 500 字**。更长的输入（例如把整段材料贴进来）会被 422 拒绝。
3. **`/v1` 的 `usage` 恒为 0**，未接 tokenizer。
4. **并发瓶颈已不在本服务**：100 并发短问题直连上游供应商 p50 已 46.7s，本服务侧首字 p50 约 10s。
5. **会话检索不跨轮**：检索只用本轮（或改写后的）问题，历史仅作为作答上下文。多跳问题需要用户自己说清楚。
6. **`/ask`、`/ask/stream` 不支持会话**，且是论文库单轮接口，保留只为兼容旧前端。

---

## 11. 交接：代码地图

| 文件 | 职责 |
|---|---|
| `rag/serve.py` | 应用装配：注册各 router、并发中间件、`/ask`、`/health` |
| `rag/api.py` | OpenAI 兼容层：`/v1/chat/completions`、`/v1/models`、鉴权、历史上限与合并 |
| `rag/unified.py` | 统一入口：路由分发、各库作答、`/auto/ask`、`/auto/ask/stream`、会话接入 |
| `rag/conversation.py` | 会话存储（SQLite，WAL，线程本地连接） |
| `rag/conversation_api.py` | 会话管理 REST |
| `rag/enhance.py` | 追问改写（兼容 OpenAI SDK 与 `OpenAICompatClient`） |
| `rag/embedding.py` | 并发向量化批处理 |
| `rag/retrieval.py` | 检索 / BM25 / 重排 |
| `rag/query.py` | 论文库 `RAGService`、提示词组装 |
| `rag/web/` | 网页前端：`index.html` 布局、`app.js` 会话与历史回填、`style.css` |
| `scripts/purge_conversations.py` | 过期会话清理 |
| `scripts/gen_api_keys.py` | 生成 API key（支持写入密钥文件，千级批量） |
| `scripts/smoke_api.py` | 接口冒烟测试 |

改接口时优先看 `rag/api.py` 的 `ChatCompletionRequest` 与 `rag/unified.py` 的 `AutoAskRequest`，
两个请求模型就是对外契约。

---

## 12. 运维

### 12.1 前端

页面在 `GET /`（静态资源挂 `/static`）。改 `rag/web/` 下的文件**不需要重启**，
刷新浏览器即可（FastAPI 静态目录是运行时读取的）。

### 12.2 会话库

`rag/conversation_db/conversations.sqlite3`（可用 `CONVERSATION_DB` 改），已进 `.gitignore`。
备份直接复制文件即可，但建议在服务空闲时做（WAL 模式下还需一并复制 `-wal`/`-shm`）。

### 12.3 批量发 key（千级）

```bash
# 生成 1000 个 key 到单独文件（推荐），并输出 学号→key 的 json/csv 分配表
python scripts/gen_api_keys.py --count 1000 --keys-file /etc/futurx/api_keys.txt

# 已经拿到学号名单时按顺序分配
python scripts/gen_api_keys.py --count 1000 --keys-file /etc/futurx/api_keys.txt \
    --students "$(paste -sd, students.txt)"

# 先看看会生成什么，不落盘
python scripts/gen_api_keys.py --count 1000 --keys-file /etc/futurx/api_keys.txt --dry-run
```

然后：

```bash
echo 'API_KEYS_FILE=/etc/futurx/api_keys.txt' >> .env
systemctl restart futurx        # 只有第一次要重启，之后改文件即可
```

- 密钥文件权限自动设为 `600`，内容每行一个 key。
- 分配表写到 `logs/api_keys_<时间>.json` 与 `.csv`（`logs/` 已忽略），**这两份是唯一能反查
  学号与 key 对应关系的地方**，务必在分发前妥善保存（注意：表里是明文 key）。
- 追加生成时是「已有 + 新增」，不会重复造 key。

### 12.4 撤销 / 轮换（热加载，不重启）

服务端按**文件内容哈希**检查密钥文件，默认每 5 秒一次（`API_KEYS_RELOAD_S`），
改动后 5 秒内生效，**不需要重启，不中断正在进行的对话**。

```bash
# 按学号撤销（查 logs/api_keys_*.json 找到对应的 key）
python scripts/gen_api_keys.py revoke --student 2023101 --keys-file /etc/futurx/api_keys.txt

# 按明文撤销
python scripts/gen_api_keys.py revoke --key sk-futurx-xxxx --keys-file /etc/futurx/api_keys.txt

# 按指纹撤销（日志里看到的 key_id，只知道 8 位也能撤）
python scripts/gen_api_keys.py revoke --key-id c72f2e45 --keys-file /etc/futurx/api_keys.txt

# 先看看会删什么
python scripts/gen_api_keys.py revoke --student 2023101 --keys-file /etc/futurx/api_keys.txt --dry-run
```

改动前自动备份为 `api_keys.txt.bak.<时间戳>`。

> **注意**：50 分钟内改完不重启也能生效，但 `API_KEYS_RELOAD_S=-1` 可关掉热加载
> （只在启动时读一次）；`0` 表示每个请求都检查（仅调试用）。

### 12.5 会话归属与 key 的关系

会话与消息都记 `key_id`（key 的 sha256 前 8 位），不记 key 原文。因此：

- 轮换一把 key（删旧建新）= 换了一个人，**旧会话看不到**。
- 想让学生换 key 后仍看到历史，只能保持 key 不变，或人工改库里 `key_id`。
- 撤销一把 key 后，它创建的会话仍归属那个 `key_id`，**无人能再访问**（可 `reassign_key` 或删除）。

配 `API_KEYS` 之前产生的匿名会话（`key_id` 为 NULL）谁都打不开，可用存储层函数归置：

```python
from rag import conversation as c
c.reassign_key(src_key_id=None, dst_key_id="c72f2e45", dry_run=True)   # 先看数量
c.reassign_key(src_key_id=None, dst_key_id="c72f2e45")
```

### 12.6 按人 / 按月审计

`messages` 表存 `key_id` 与 `month`（YYYY-MM），走 `ix_msg_key_month` 索引：

```python
from rag import conversation as c
c.monthly_stats()                    # 全部人、全部月份
c.monthly_stats(key_id="c72f2e45")   # 某一把 key 的逐月用量
```

返回每行 `{key_id, month, messages, turns, avg_latency_ms}`。
`turns` 是提问轮数（`messages` 的一半，含失败轮）。

---

## 13. 变更记录

- `eb5ee75`：多轮对话（历史 + 追问改写）+ 会话持久化（`/v1/conversations*`）+
  检索去全局锁与向量化批处理 + 流式先回头再检索。
- `6c1c15a`：接口协议文档、冒烟脚本；`archived` 支持 `all`。
- 前端：历史会话侧栏 + 历史回填；每轮带 `conversation_id`（此前每轮都新建会话，多轮在页面上不生效）。
- 鉴权：`API_KEYS` 改 `frozenset`，新增 `API_KEYS_FILE` 支持千级 key。
