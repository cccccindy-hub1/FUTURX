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

会话按 **(API key 指纹, 调用方 uuid)** 归属。**A 查不到 B 的会话，一律 404**（不泄露存在性）。

- 传了 `uuid`（问答接口的请求体字段，或 `X-User-Uuid` / `X-User-Id` 请求头）：只看到自己的会话。
  这是**多个学生共用一把 key** 时的正确用法。
- 没传 `uuid`：只看到「没有人员标识」的老会话，**不会串到带 uuid 的会话里**；同一 key 下多个无身份调用方之间共享。
- 一人一 key 时 key 本身已经隔离，`uuid` 可以不传。

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
  "store": true,
  "uuid": "stu-0001",
  "user_name": "张三"
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `question` | string | 是 | — | 1–500 字。**这是路由、检索、作答的依据** |
| `top_k` | int | 否 | 服务端配置 | 检索片段数 |
| `include_trace` | bool | 否 | `true` | 是否返回调试信息。**生产建议 `false`**，可省流量 |
| `conversation_id` | string | 否 | — | 会话 id，见第 4 节 |
| `store` | bool | 否 | `true` | `false` = 本轮不落库 |
| `uuid` | string | 否 | — | 调用方人员唯一标识，记进问答记录的 `user_id` 列。优先于 `X-User-Uuid` / `X-User-Id` |
| `user_name` | string | 否 | — | 提问者姓名，中文可用（限 32 字），记进 `user_name` 列。优先于 `X-User-Name` 头 |

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
| `meta` | 问助手自身（能做什么 / 知识库有哪些内容 / 我之前问过什么）→ 不检索不调模型 |
| `off_topic` | 超出知识库范围 → 不检索，直接用大模型自身知识作答（`GENERAL_*`） |

组合（`routes` 有两个）仅在 `course` / `case` / `quote` / `policy` 之间发生，只调一次作答模型。
`meta` 与 `exam`、`off_topic`、`thesis` 一样只能单独出现。

**`meta` 与 `off_topic` 的边界**：打招呼（「你好」「在吗」「hello」）、问助手能力
（「你能做什么」「介绍一下你自己」「你的知识库有哪些内容」），以及问自己之前问过什么
（「我之前都问过你哪些内容」「我们刚才聊了什么」）都归 `meta`，不拒答。
拒答一句「你好」会让人以为系统坏了。`meta` 有三种回答：

| 问法 | 回答 |
|---|---|
| 纯打招呼 | 一句欢迎语 + 两三个可问的例子（不甩长清单） |
| 问能力 / 知识库 | 按**实际加载的库**列出能力清单（缺数据的库不列），末尾给引导 |
| 问我之前问过什么 | 按问答记录（`qa_log`）列出**这个人**问过的问题，新的在前 |

`meta` 零检索、零模型调用（`trace.meta.llm_calls == 0`），不消耗额度也不会答偏。
「你好，我想问数据确权」这类带真问题的，按真问题走，不会整句当成打招呼。

**「我之前问过什么」的取数口径**：按 `user_id` 查 `qa_log`（就是 5.4 节那个 `uuid`），
所以调用方每次都要带同一个 `uuid` —— 不带就退回 `anon-<key指纹>`，只能看到同样没带
uuid 的记录。带 `X-Session-Id`（或会话 `conversation_id`）时只看本次对话，不带则看这个人全部提问。
最多列 20 条、同一句重复问只列一次，`trace.meta.recall` 里有 `count` / `total` / `user_id`。

| 情况 | 回答 |
|---|---|
| 有记录 | 「此前/本次对话一共问过 N 个问题」+ 编号列表（问题 + 日期） |
| 没有记录 | 「此前还没有提问记录。直接问我就行，例如『什么是数据确权』。」 |

### 超出知识库的问题不再拒答

路由判成 `off_topic`（与六个知识库都无关）时，**不再返回拒答文案**，而是不检索、直接用大模型自身知识作答：

```json
{"route": "off_topic", "status": "ok", "general": true,
 "answer": "……", "sources": [],
 "trace": {"general": {"configured": true, "model": "...", "llm_calls": 1}}}
```

| 情况 | 行为 |
|---|---|
| 配了 `ROUTER_API_BASE` / `ROUTER_API_KEY`（默认复用） | 用 `GENERAL_MODEL` 作答，`status=ok`、`general=true`、`sources=[]` |
| 未配置，或调用失败 | 退回原拒答文案，`status=refused`，`trace.general` 里给出原因 |
| 流式（`/auto/ask/stream`） | 逐字推 `delta`，`done` 带完整正文（通用问答也要几十秒，不能等整段写完） |

`sources` 恒为空、且系统提示词禁止编造页码 / 条文号 / 案号：库外回答**不能伪装成有据可查**。
客户端可用 `general: true` 把它和知识库答案区分展示（例如加一句「以下内容不来自课程知识库」）。

配置项（都可不填，默认沿用路由那套）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `GENERAL_MODEL` | 同 `ROUTER_MODEL` | 通用作答模型 |
| `GENERAL_REASONING` | `disabled` | 思考模式 |
| `GENERAL_TIMEOUT` | `60` | 超时秒数 |
| `GENERAL_SYSTEM` | 内置提示词 | 通用作答的系统提示词 |

### 2.3 `status`

| status | 含义 | 客户端建议 |
|---|---|---|
| `ok` | 正常作答 | 展示 |
| `insufficient` | 资料不足 | 展示提示文案，可建议换个问法 |
| `refused` | 未配置通用作答模型时的兜底拒答（`off_topic`） | 展示拒答文案 |
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
  "store": true,
  "uuid": "stu-0001",
  "user_name": "张三"
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
| `uuid` / `user_name` | 扩展：调用方人员标识与提问者姓名，只用于永久问答记录，见 5.4 |

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

### 5.4 `uuid` / `user_name`：把每轮问答记到具体的人

两个入口（`/auto/ask`、`/auto/ask/stream`、`/v1/chat/completions`、`/ask`）都接受这两个字段，
它们**只用于永久问答记录（`data/qa_log.sqlite`），不参与鉴权、路由与检索**，传错也不会改变回答。

| 入参 | 落库位置 | 说明 |
|---|---|---|
| `uuid` | `qa_log.user_id` | 调用方自己的人员标识（学号、工号、业务库主键都行）。只允许 `[A-Za-z0-9_-]`，最长 64；不合法或没传时退回 `user` / `X-User-Id`，再没有就记 `anon-<key指纹>` |
| `user_name` | `qa_log.user_name` | 提问者姓名，中文可用，去空白后最长 32 字；不传记空串 |

优先级（从高到低）：

```
uuid（请求体） → X-User-Uuid（请求头） → user（OpenAI 标准字段） → X-User-Id（请求头） → anon-<key_id>
user_name（请求体） → X-User-Name（请求头） → ''
```

**请求头里的中文要百分号编码**。HTTP 头只能放 latin-1 字节，直接发 UTF-8 的中文到服务端会变成乱码；
服务端对 `X-User-Name` 做了还原（能按 UTF-8 解就按 UTF-8 解，也接受 `%E5%BC%A0%E4%B8%89` 这种编码）。
**最省事的做法是把姓名放在请求体里**，请求体是 JSON，直接写中文即可。

```python
# 推荐：两个字段都放请求体
requests.post("http://<服务器>:8000/auto/ask",
              json={"question": "数据确权为什么难？", "uuid": "stu-0001", "user_name": "张三"},
              headers={"Authorization": "Bearer <key>"})

# 只能改请求头时（例如第三方 OpenAI 兼容平台的自定义 header）：
#   X-User-Uuid: stu-0001
#   X-User-Name: %E5%BC%A0%E4%B8%89
```

同一个人的多次提问只要 `uuid` 一致，在库里就能按 `user_id` 聚合；姓名只是给人看的标签，
以调用方传入的为准，服务端不做实名校验（同一个人两次传不同姓名会各记各的）。

**`uuid` 同时决定会话归属**，不只是记录字段：多轮对话、历史列表、读消息都按 (key, uuid) 过滤。
共用一把 key 时，学生 A 拿不到学生 B 的 `conversation_id` 也读不到 B 的会话（一律 404）。
因此**换 uuid = 换人**：uuid 变了，之前的历史就看不到了；客户端要把 uuid 持久化（微信小程序可用
`openid`，或首次生成后存本地），不要每次请求重新生成。

**兼容性**：两个字段都可以不传。老调用方继续用 `user` / `X-User-Id`，记录行为与改动前完全一致；
老库（已存在的 `data/qa_log.sqlite`）在服务启动后首次写入时自动补 `user_name` 列，旧行取空串，**不重建表、不丢数据**。

查与导出：

```bash
sqlite3 data/qa_log.sqlite "select user_name, user_id, count(*) from qa_log group by user_id order by 3 desc limit 20"
python scripts/export_qa.py --out qa.csv --user-id stu-0001        # 按 uuid 筛
python scripts/export_qa.py --out qa.csv --user-name 张三          # 按姓名筛（精确匹配）
```

### 5.5 角色卡与学习者画像（`rag/persona.py`、`rag/profile.py`）

**角色卡**：人设、能答范围、红线、开场白、示例对话只在 `rag/persona.py` 写一份，六个作答提示词
（课程 / 政策 / 组合 / 判例 / 论文 / 库外）都从它 import。改人设只改这个文件，不会再出现
「课程助教」「论文指导助手」「学术数字人」三个人设并存。

| 常量 | 用途 |
|---|---|
| `PERSONA_CORE` / `BOUNDARY` / `RED_LINES` | 身份 + 能答范围 + 跨库红线，拼成 `PERSONA_BLOCK` 放进各库 system |
| `GREETING_SHORT` | 纯打招呼的短回复（`meta` 路由） |
| `EXAMPLE_DIALOGUE` | 多轮示例对话，示范「接话」；追加在课程作答提示词末尾 |

红线只写跨库共用的四条（不编出处、不把常识说成原文、只依据材料时资料不足就明说、不写套话免责）；
各库专属红线（判例不编案号、政策标注征求意见稿、讲话原文不得改写）仍在各库自己的提示词里。

**作答格式按问法分流**（`rag/prompt_rules.ANSWER_STRUCTURE`）：定义 / 是非 / 单概念说明不分点，
150–250 字连着说；列举、关系、区别、原因、作用类才用「总—分—总」。此前一律要求总—分—总 + 400 字 + 最多 4 点，
每道题长得一模一样，学生读到的是模板而不是回答。作答思考模式保持 `low`（`COURSE_ANSWER_REASONING`），
设成 `disabled` 后回答会明显变浅——`tests/test_stream_and_models.py` 会因此失败。

**`meta` 的六种问法**（都不检索、不调模型）：

| 问法 | 回答 |
|---|---|
| 纯打招呼（「你好」） | 一句欢迎语 + 两三个例子 |
| 确认 / 收尾（「好的 了解了」「谢谢」「嗯嗯」） | 一句短的；有上文时带上上一轮的具体概念（「数据资产在考试里怎么考」） |
| 问刚才怎么答的（「你的回答链路是怎样的」） | 讲本轮实际路径，不背能力清单 |
| **问我是谁（「我是谁」「你记得我是谁」）** | 回学生档案：自述身份 + 最近问过的主题 + 交过的题（见 `trace.meta.whoami`）；没记录就如实说没记录 |
| 问之前问过什么 | 按 `qa_log` 列本人提问（见 `trace.meta.recall`） |
| 问能力 / 知识库 | 按实际加载的库列完整清单 |

「我是谁」与「你是谁」必须分开：前者问学生自己，后者问助手。此前「我是谁」被判成 meta 的兜底分支，
答成了「我是 AI 李老师…我能回答以下范围内」（`qa_log` #128/#131/#132）。

「好的，那什么是数据确权」这类带真问题的不算确认语，按真问题走。确认语的下一步只给类型不给概念的旧写法
（「接着可以问「这个概念怎么考」」）已废：概念从上一轮提问里抽，抽不出（如「出三道判断题」这类整句请求）
才退回类型文案。

**学习者画像**（`data/profile.sqlite`，可用 `PROFILE_DB` 覆盖）：

- 记什么：**自述的身份**（学校 / 学历 / 专业 / 学科 / 论文阶段 / 论文题目，`learner_profile`）、课程管线选中的概念
  （名称、次数、最近时间）、判例 / 试题命中的章、**做题的判分结果**
  （`learner_mastery`：题号、章、概念、1–5 分、是否答对）。判分只在学生真的交了答案时发生，不记推断。
- 怎么用：有记录时把一句「【这个学生的情况】内蒙古大学，硕士生，公共管理专业；论文题目：…；之前问过：…；
  做过题但还没答对：…」追加到作答 system 末尾；**没有记录时提示词逐字不变**，不会出现「我记得你」
  这类编造的亲近感。提示词里明确要求「按他的专业和阶段举例选材」「他说过的题目要按原题理解，不要改写」。
- 只取问过 **两次以上** 的概念：课程管线一次会选出好几个概念，全列出来等于把词表塞进提示词。
  「还没答对」只来自 `correct=0` 的判分记录，同一道题重做答对即消失，不做时间衰减。
- 每人上限 8 个概念、5 章、50 条判分记录，超出按最近时间淘汰；写失败只记 warning，不影响回答。
- 按 `uuid` 记（与问答记录同一口径）；没有 `uuid` 的调用方记在 key 指纹下。

**学生自述身份**（`rag/student.py`）：识别「我是本科生 编辑出版专业（新闻传播大类）」「论文在选题阶段」
这类自述，写进 `learner_profile`，之后每次作答都当作已知条件。**只认明确说法，不猜**——认不出就什么都不记。

- **纯自述**（没有追问、长度短、不带要处理的内容）→ 回一句「记住了：…，接下来想先解决哪一件？」，
  `route=meta`、零检索零调用。此前这句会被判成 `off_topic`，模型当问题答，讲了一段考研方向；
  紧接着问选题，回答里全是公共管理的例子（实测，`qa_log` #97/#98）。
- **带内容的句子不算纯自述**：线上实测「公共管理研究生专硕，妇幼保健院门诊患者满意度现状及提升对策研究
  ——以内蒙古某妇幼保健院为例」满足「短 + 有身份字段」，被整句吞成「记住了：硕士生」，题目一个字都没进回答。
  现在句子里出现论文题 / 研究对象 / 写作要求（`rag/student.py` 的 `_SUBSTANCE`、`_TITLE_RE`）就只记录身份、
  按原问题作答。
- **学校与题目分开记**：`learner_profile` 有 `school` 与 `title` 两列。校名先从句子摘出再解析专业，
  否则「我是内蒙古大学公共管理专硕」会把专业记成「内蒙古大学公共管理」。题目只认显式说法
  （「论文题目是…」「在做…的研究」），以及整句就是一个题目的情形（以「研究 / 分析 / 调查 / 对策」收尾、
  可带「——以…为例」），不从句子中间猜——猜错一次，之后每轮都会引用一个学生没写过的题目。
  老库由 `_migrate()` 补列，不需要手工改表。
- **「我是谁」会念出学校和题目**：「我记得的你是：内蒙古大学，硕士生，公共管理专业。你在做的论文题目是：…」。
- **自述带真问题**（「我是本科生，编辑出版专业，想问选题怎么定」）→ 只记录，按真问题正常作答。

**论文库不再套用课程格式**（`rag/query.py`）：`_SYSTEM_PROMPT` 不再拼 `ANSWER_STRUCTURE`，
改为「先读懂问题、再决定怎么答」：一句话能说清就一句说完，需要并列才分点，不硬凑小标题。
同时写明「资料偏公共管理 / 新闻传播，不代表学生自己的学科」——论文库的资料是团队那批 MPA 文档，
不写这句，模型会把公共管理的例子套到编辑出版的学生身上。
**追问也按路由分学科**（`rag/followups.py`）：论文类用学术写作角度，不再出现「考试怎么考」。

```bash
sqlite3 data/profile.sqlite "select user_id, concept, hits from learner_concept order by last_at desc limit 20"
sqlite3 data/profile.sqlite "select user_id, question_id, score, correct from learner_mastery order by last_at desc limit 20"
sqlite3 data/profile.sqlite "select user_id, level, major, field, stage, school, title from learner_profile"
```

**学生作答判分**（`rag/grade.py` + `POST /exam/grade`）：把学生答案与题库参考答案交给 `rag/judge.py`
（1–5 分 + `correct`，LLM 评审；模型未配置时降级为词法比对），判完回写 `learner_mastery`。
判分失败（题不存在 / 空答案 / 这道题没有参考答案）返回对应错误码，**不伪装成 0 分**——
把「没判成」显示成「答错了」会让学生以为自己会的东西不会。网页的每道题下方有「先自己答一句」的输入框，
判分结果就地显示，参考答案仍在「查看答案」里折叠对照。

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
| `POST /exam/grade` | 学生作答判分：`{question_id, answer}` → 1–5 分 + `correct` + 参考答案；判完回写画像 |

`/ask`、`/ask/stream` **不支持会话**，新接入建议直接用 `/auto/ask`。

---

## 8. 典型调用流程

### 8.1 新会话 + 追问（推荐）

```
① POST /auto/ask        {"question":"数据资产估值有哪些方法？","uuid":"stu-0001","user_name":"张三"}
   ← {"conversation_id":"conv_A", "answer":"...", ...}

② 客户端存 conv_A（同一个 uuid 才会被认成同一个人）

③ POST /auto/ask        {"conversation_id":"conv_A", "question":"第二种具体怎么算？",
                         "uuid":"stu-0001","user_name":"张三"}
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

`rag/web/` 的页面已经接上会话，行为如下（改前端时按这个约束走）。
网页把浏览器里长期保存的随机标识同时放在 `X-User-Id` 与 `X-User-Uuid` 上，所以浏览器之间互看不到历史
（同一个浏览器不填姓名也能一直看到自己之前的问答）。

| 动作 | 行为 |
|---|---|
| 发第一句 | 服务端建会话，id 从响应头 `X-Conversation-Id` 取回，存 `localStorage`（按 API key 分开存） |
| 继续追问 | 请求体带 `conversation_id`，服务端取历史 → 改写 → 作答 |
| 刷新页面 | 会话 id 从 `localStorage` 恢复，侧栏从 `/v1/conversations` 重新拉 |
| 点侧栏某条 | `GET /v1/conversations/{id}` 取 `messages`，逐条回填；**回填的消息不带耗时行与追问胶囊** |
| 换访问密钥 | 清空对话区，历史列表按新 key 重新拉（会话按 key + uuid 隔离） |
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
| `rag/persona.py` | 角色卡：人设 / 能答范围 / 跨库红线 / 开场白 / 多轮示例对话的唯一来源 |
| `rag/prompt_rules.py` | 作答格式（按问法分流）与 few-shot 样例，各库共用 |
| `rag/profile.py` | 学习者画像（`data/profile.sqlite`）：问过的概念 / 章 / 判分结果 |
| `rag/grade.py` | 学生作答判分：接题库参考答案，用 `rag/judge.py` 打分并回写画像 |
| `rag/student.py` | 学生自述身份识别：学历 / 专业 / 学科 / 论文阶段（只认明确说法，不猜） |
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

按人看提问量用问答记录表（`qa_log`，按调用方传的 `uuid` / `user_name` 聚合，见 5.4）：

```sql
-- 每人每天问了多少
SELECT date(created_at) d, user_name, user_id, count(*) n
FROM qa_log GROUP BY d, user_id ORDER BY d DESC, n DESC;

-- 某个人问了什么
SELECT created_at, route, substr(input_prompt,1,50) q
FROM qa_log WHERE user_id = 'stu-0001' ORDER BY id DESC LIMIT 20;
```

---

## 13. 变更记录

- `eb5ee75`：多轮对话（历史 + 追问改写）+ 会话持久化（`/v1/conversations*`）+
  检索去全局锁与向量化批处理 + 流式先回头再检索。
- `6c1c15a`：接口协议文档、冒烟脚本；`archived` 支持 `all`。
- 前端：历史会话侧栏 + 历史回填；每轮带 `conversation_id`（此前每轮都新建会话，多轮在页面上不生效）。
- 鉴权：`API_KEYS` 改 `frozenset`，新增 `API_KEYS_FILE` 支持千级 key。
- 问答记录按人归属：四个问答接口新增可选入参 `uuid` / `user_name`（也可用 `X-User-Uuid` / `X-User-Name` 头），
  `uuid` 记进 `user_id` 列，姓名单独一列 `user_name`；老库自动补列，旧调用方与旧数据都不受影响。
- 会话归属改为 (key_id, user_id)：共用一把 key 时学生之间互不可见（列表、详情、消息、追问、删除全部校验）；
  `conversations.user_id` 由迁移就地补列，老会话作为「无人员标识」仍可见；网页同时带 `X-User-Uuid`。
- 新增 `meta` 路由：问助手自身（能做什么 / 知识库有哪些内容）与打招呼都不再拒答，
  按实际加载的库返回能力说明，零检索、零模型调用。
- `off_topic`（超出知识库）改为用大模型自身知识作答（`GENERAL_*`、`general=true`、流式逐字推）；
  未配置模型时退回原拒答文案。
- `meta` 增加「我之前问过什么」：按 `user_id` 查 `qa_log` 列出这个人的提问（可只看本次对话），
  零检索、零模型调用；没带 `uuid` 的调用方只看到 `anon-<key指纹>` 下的记录。
- 人设收敛到 `rag/persona.py` 单一来源（课程 / 政策 / 组合 / 判例 / 论文 / 库外六处 system 共用），
  补多轮示例对话；`meta` 增加「确认 / 收尾」与「你刚才怎么答的」两类问法，不再一律甩能力清单。
- 新增学习者画像 `rag/profile.py`（`data/profile.sqlite`）：记问过的概念与章，有记录时拼一句
  关系层进作答提示词；不记对错、不做推断，没有记录时提示词逐字不变。

### 2026-11-09 第二次（回答「死板」的修正 + 判分闭环）

- 作答思考模式从 `disabled` 改回 `low`（`COURSE_ANSWER_REASONING`）：关闭思考后回答明显变浅、
  句间跳跃，是「死板」的首要原因；`.env.example` 原本就写明不要设成 disabled。
- 作答格式改为**按问法分流**（`rag/prompt_rules.ANSWER_STRUCTURE`）：定义 / 是非 / 单概念说明不分点、
  150–250 字；列举 / 关系 / 区别 / 原因 / 作用类才用总—分—总。样例同步换成「一、不分点」「二、分点」两条，
  `scripts/check_answer_format.py` 按问题判断是否要求分点（拿不到问题按保守口径）。
- 确认语的下一步带上上一轮的具体概念（「数据资产在考试里怎么考」），并跳过更早的确认语往回找真正的问题；
  抽不出概念（「出三道判断题」这类整句请求）时退回类型文案。
- 新增 `rag/grade.py` 与 `POST /exam/grade`：学生作答按题库参考答案判分（`rag/judge.py`，1–5 分 + correct），
  判完回写 `learner_mastery`；画像增加「做过题但还没答对」一句，重做答对即消失。判分失败给错误码，不伪装成 0 分。
- 网页每道题下方加「先自己答一句」的判分入口，结果就地显示。

### 2026-11-09 第三次（理解用户，而不是套格式）

学生实测：「我是本科生 编辑出版专业」→ 紧接着问「什么算好选题」，得到的全是公共管理的例子；
纯大模型直接问反而更贴题。逐条复现后确认，问题不在「总—分—总」，而在三处：

- **自述身份被当成问题答**：「我是本科生 编辑出版专业（新闻传播大类）」被判成 `off_topic`，
  模型把它当「介绍一下这个专业」，讲了一段考研方向；这段信息也没有被记住，下一问仍按公共管理的
  学生来答。新增 `rag/student.py` + `learner_profile` 表：认出学历 / 专业 / 学科 / 阶段，
  纯自述回一句确认（`route=meta`，零调用），带真问题的只记录。
- **论文库仍套课程格式**：`_SYSTEM_PROMPT` 拼 `ANSWER_STRUCTURE`（总分总 + 400 字 + 最多 4 点），
  咨询类问题也被压成同一外形。改为问题驱动：先判断学生卡在哪一步，结构服务于内容；
  并写明资料偏公共管理 / 新闻传播，**不代表学生自己的学科**，不能把资料里的学科当成学生的学科。
- **追问学科错配**：追问提示词写死「《数据要素》课程的助教」，角度含「考试怎么考」，于是论文类问题
  也会追问「考试会怎么考选题设计」。改为按路由选提示词，论文用学术写作角度。

线上验证：自报背景 → 「记住了：本科生，编辑出版专业（新闻传播）。接下来想先解决哪一件？」；
再问好选题 → 553 字、三点、全是编辑出版的例子（少儿社 / 大学社、直播售书、选题策划）；
追问变成「文献要读多少才算摸清选题？」「怎么判断资料能不能拿到？」。

### 2026-11-09 第四次（两个答非所问的 bug）

学生实测反馈「还是没抓住我的意思」，按其对话复现出两处硬 bug：

- **长句被吞成一句「记住了」**：「公共管理研究生专硕，妇幼保健院门诊患者满意度现状及提升对策研究
  ——以内蒙古某妇幼保健院为例」满足 `only_intro` 的「短 + 有身份字段」，被整句当成纯自述，
  回答只有「记住了：硕士生」，题目一个字都没进去。修：句子里出现论文题 / 研究对象 / 写作要求
  （`_SUBSTANCE`、`_TITLE_RE`）就不算纯自述，只记录身份、按原问题作答。
  同时补上「研究生专硕」这类说法的专业识别（此前只认到「公共管理研究生」）。
- **「我是谁」答成助手介绍**：路由提示词里 `meta` 只有「问助手能力」，没有「问我是谁」，
  于是「我是谁」走兜底分支，回了「我是 AI 李老师…我能回答以下范围内」（`qa_log` #128/#131/#132）。
  修：新增 `looks_whoami` 与 `_whoami`，回学生档案（身份 + 最近主题 + 交过的题），零模型调用；
  没记录就如实说没记录，并说明怎么说才能被记住。「你是谁」仍问助手，两者不混。

未采纳的两条（与学生讨论后）：不增加「独立一轮 LLM 理解用户」——多一次调用、多约两秒延迟，
只是把「猜」挪个位置，改成结构化的身份 / 阶段状态更划算；论文指导场景也暂不接联网查询，
文献类问题联网容易返回二手解读，破坏引用的可核查性。

### 2026-11-09 第五次（能看到用户多少信息）

学生问「现在可以看到用户的所有信息了吗」。核了一遍，答案是：身份与题目此前只记了一半。

- **校名被并进专业名**：「我是内蒙古大学公共管理专硕」把专业记成「内蒙古大学公共管理」，
  之后每轮提示词都带着这个错名字。现在先把校名摘出来单独存 `school`，再解析专业。
- **论文题目根本没收**：此前只有「学历 / 专业 / 学科 / 阶段」四个字段，学生说过的题目一律没存，
  于是问「这个题目怎么样」时模型得从历史里重新找。新增 `title` 列，只认显式说法
  （「论文题目是…」「在做…的研究」）或整句就是题目的情形，不从句子中间猜。
  老库由 `_migrate()` 自动补列（`CREATE TABLE IF NOT EXISTS` 不会给已存在的表加字段）。
- 「我是谁」现在会念出学校和题目：「我记得的你是：内蒙古大学，硕士生，公共管理专业。
  你在做的论文题目是：妇幼保健院门诊患者满意度现状及提升对策研究。」

仍然**看不到**的（有意不记）：学生的回答对错以外的能力判断、情绪、身份推断、导师姓名、时间节点。
这些要么没有依据，要么属于隐私，记下来只会让模型顺着编。
