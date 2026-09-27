# RAG 知识库建设规划

> 项目：数字治理科研团队 · 数字人（数字分身）
> 本模块：**学位论文指导** 场景的知识库（数据源为 `nlp_/`）
> 文档状态：已落地实现（代码见 `rag/`），部分后续项见「路线图」
> 最近更新：2026-09-27（对齐实际代码：跨平台解析、混合检索/重排序、父子分块、评测闭环）

---

## 1. 项目背景与定位

这是一个面向高校数字治理科研团队的**学术数字人**项目，核心价值是把教授的知识、研究方法和指导能力数字化，服务于学生培养、科研推理和学术协作。

### 四大核心需求场景

| 场景 | 目标用户 | 核心诉求 |
|------|----------|----------|
| 研究项目解释 | 学生 | 解释研究项目工作内容，降低理解门槛 |
| 前沿热点推理 | 研究团队 | 推理数字治理领域研究前沿与热点，辅助科研决策 |
| **学位论文指导** | 研究生 | 提供规范的学位论文指导，保障学术质量 |
| 学术沟通渠道 | 其他老师 / 相关智能体 | 学术沟通渠道，跨主体学术交流 |

### 功能映射

- **研究功能**（解释、推理、指导）→ 覆盖场景 1 / 2 / 3
- **分身功能**（交流、语言、情绪）→ 覆盖场景 4，使数字人具备拟人化交互能力

### 本模块边界

`nlp_/` 目录内容对应 **场景 3「学位论文指导」**，本规划围绕把它建成可检索、可问答的 RAG 知识库展开。其余场景（1/2/4）为后续扩展，架构预留接口。

---

## 2. 知识库范围：nlp_ 现状盘点

共 **68 个内容文件**（另有 4 个 `.DS_Store` 忽略）：

| 类型 | 数量 | 解析方式 |
|------|------|----------|
| PDF | 44 | pymupdf（均为文本型，无需 OCR） |
| `.doc` | 17 | Windows + Word 走 COM（pywin32）；否则 LibreOffice `soffice` 转换 |
| `.docx` | 6 | python-docx |
| `.ppt` | 1 | LibreOffice `soffice` 转换；未安装则跳过（预留） |

> 解析后端按**能力探测**自动选择（`extract.parser_status()`），并在 `build` 时打印可用性，
> 缺后端不再是静默失败。

内容主题（按目录）：

- `001` 典型问题 & 期望回答颗粒度
- `002` 论文写作 / 文献综述 / 选题 / 开题报告
- `003` 答辩点评与问答集合（可作评估测试集）
- `006` 研究方法（扎根理论 / 案例研究 / 访谈方法）+ 学位论文规范
- `008` 学位法、研究生学科专业简介与学位基本要求

---

## 3. 已确认的技术决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 部署形态 | 本地先试运行，后续上服务器 | 模块化分层，便于「检索+生成」层独立服务化 |
| 运行环境 | 跨平台（Windows / macOS / Linux） | 解析层按能力探测选择后端，不绑定单一 OS 工具 |
| 向量化模型 | 本地 `BAAI/bge-m3` | 离线、免费、中文效果好，1024 维，cosine 距离 |
| 稀疏检索 | BM25（rank_bm25 + jieba） | 与稠密向量经 RRF 融合；缺失 jieba 时退化字符 bigram |
| 检索增强 | 默认全关，opt-in | 混合检索 / 重排序 / 查询改写 / HyDE / 父子分块互不耦合 |
| 生成模型 | 接大模型生成 | OpenAI 兼容协议，`LLM_API_KEY/BASE/MODEL` 由用户环境变量注入 |
| 向量库 | Chroma（本地文件式持久化） | 轻量、可迁移、支持元数据过滤 |
| 评测 | data_benchmark 金标准 + LLM-as-judge | 按用户分组切分避免泄漏；无 LLM 时降级词法基线 |
| 标注 | 先机器标注，专家标注留接口 | 机器自动切块+向量+元数据；预留人工增删改/打标签通路 |

---

## 4. 总体架构

```
nlp_/  (68 文件)
   │
   ▼
[1] 解析层 extract.py     PDF→pymupdf / docx→python-docx / pptx→python-pptx
   │                      doc→Word COM 或 LibreOffice / ppt→LibreOffice（缺后端则跳过）
   │   统一为 {text, source, category, ext}
   ▼
[2] 分块层 chunk.py       标题感知 + 句子边界分块（600 字，重叠 100），保留章节标题
   │                     可选「父子分块」：子块（600）入库、父块（1500）存 sidecar
   │   → (chunk_text, section[, parent_id/parent_offset/parent_length])
   ▼
[3] 向量化层              BGE-m3（normalize，cosine）→ 1024 维向量
   │
   ▼
[4] 入库层 index.py       Chroma 持久化(rag/chroma_db) + 唯一主键 + 元数据
   │                       向量结果缓存 rag/cache（内容未变时跳过向量化）
   ▼
[5] 检索层 retrieval.py   稠密召回 + BM25 稀疏召回 → RRF 融合
   │                       →（可选）CrossEncoder 重排序 → top-k；支持元数据过滤
   │                       可选查询增强：改写 / HyDE（enhance.py，需 LLM）
   │                       可选父块展开：命中子块 → 替换为父块上下文
   ▼
[6] 生成层 query.py       OpenAI 兼容 LLM（env 配置），回答标注[来源N]
   │
   ▼
[7] 评测层 eval.py        检索指标（BM25 vs 稠密）+ 答案质量（LLM-as-judge / 词法基线）
                          测试集来自 benchmark.py 切分 data_benchmark
```

### 代码目录

```
rag/
  config.py      全局配置（环境变量可覆盖）
  extract.py     解析层（跨平台，能力探测）
  chunk.py       分块层（含父子分块）
  retrieval.py   检索层（混合检索 / RRF / 重排序 / 元数据过滤）
  enhance.py     查询增强（改写 / HyDE）
  index.py       向量化 + 入库 + 缓存 + 父块 sidecar
  query.py       检索 + 生成
  benchmark.py   data_benchmark 清洗与切分
  judge.py       打分（词法 + LLM 评委）
  eval.py        评估入口
  serve.py       常驻服务（REPL / HTTP）
  cli.py         命令行入口（build / ask / serve / eval）
  README.md      使用说明
chroma_db/       向量库（build 后生成）
cache/           向量缓存 / BM25 索引 / 父块 sidecar（build 后生成）
```

---

## 5. 详细流程（分阶段）

### 5.1 解析层（extract.py）

- `.pdf` → `pymupdf` 逐页抽取文本（无 pymupdf 时回退 LibreOffice）。
- `.docx` → `python-docx`（正文段落 + 表格单元格）。
- `.pptx` → `python-pptx`（逐页文本框）。
- `.doc` → Windows 且装了 Word+pywin32 时走 COM 自动化；否则用 LibreOffice 转 `.docx` 再解析。
- `.ppt` → 用 LibreOffice 转 `.pptx` 再解析。
- 缺后端时抛出带原因的 `UnsupportedFormat`，由 `index._parse` 捕获并记入跳过列表；
  `build` 开头会打印 `parser_status()`，显式告知哪些格式不可用。
- 单文件失败不影响全局（`try/except` 兜底）。
- 输出元数据：`source`（相对 nlp_ 路径）、`category`（目录推导，如 `006/研究方法/扎根理论`）、`ext`。

### 5.2 分块层（chunk.py）

- 先按空行拆段落，再逐行处理。
- **标题识别**：数字/汉字序号（`1.`、`一、`、`（一）`）、`第X章`、常见章节词（摘要/引言/文献综述/研究方法/结论/参考文献…）。标题作为其后内容的 `section` 元数据。
- **句子边界切分**：按 `。！？；` 累积到 `CHUNK_SIZE`（默认 600 字），超长无标点文本硬切兜底。
- **重叠**：相邻 chunk 保留 `CHUNK_OVERLAP`（默认 100 字）尾部，避免语义被截断
  （代码内强制 `overlap < chunk_size`，防止切分不推进）。
- 每个 chunk 携带 `section`，回答时可定位到具体小节。
- **父子分块（可选，`PARENT_CHILD_ENABLED`）**：父块（`PARENT_SIZE`，默认 1500）先按段落打包，
  再在其内部切子块（`CHUNK_SIZE`/`CHUNK_OVERLAP`）。子块文本前置 `〔parent_id〕` 标记后入向量库，
  并携带 `parent_id` / `parent_offset` / `parent_length`；父块全文只写入 `cache/<collection>.parents.json`。
  优点是子块检索精度高、生成时又能拿到父块完整上下文，且向量库不存冗余长文。

### 5.3 向量化层

- `BAAI/bge-m3`，本地加载，`normalize_embeddings=True`。
- 检索距离采用 cosine。
- 全量批量编码（默认 batch 32，可在后续调优）。

### 5.4 入库层（index.py）

- 写入 Chroma `PersistentClient`，collection `nlp_degree_thesis`。
- **唯一主键**：`相对路径哈希10位::文件名::chunkN` —— 解决重名文件冲突（如 `002/论文写作/文献综述写作方法_法学.doc` 与 `002/文献综述的写法及范文/文献综述写作方法_法学.doc`）。
- 元数据：`source / filename / category / section / ext`。
- **向量缓存**：embedding 结果落盘到 `rag/cache/`，以「文件集合 + 模型 + 分块参数」的哈希为签名；内容未变时重建直接复用，跳过向量化。
- 全量重建（delete → create → add），幂等。

### 5.5 检索层（retrieval.py + enhance.py）

- **稠密召回**：查询向量化 → Chroma `query` 取 `FETCH_K`（默认 20）候选。
- **稀疏召回**：BM25（rank_bm25 + jieba 分词）取 `FETCH_K` 候选；BM25 索引落盘缓存
  （`cache/<collection>.bm25.pkl`，按 id 集合指纹判定是否复用）。
- **融合**：`RRF`（`score(d)=Σ 1/(RRF_K+rank_i(d))`，默认 `RRF_K=60`）合并两路排名，取 top-k（默认 4）。
- **重排序（可选）**：对融合后的候选做 CrossEncoder（`bge-reranker-v2-m3`）重排，
  懒加载模型、默认关闭；CPU 环境下需自行开启。
- **元数据过滤**：`where={"category": ...}`，支持 `{"$in": [...]}` / `{"$eq": ...}`；
  稠密路由 Chroma 原生支持，稀疏路在内存里按同一条件过滤。
- **查询增强（可选，需 LLM）**：查询改写（口语 → 检索友好术语串）/ HyDE（先生成假设答案再检索）；
  无 LLM 或调用失败时原样返回，不影响主流程。
- **父块展开（可选）**：命中子块后按 `parent_offset`/`parent_length` 从 sidecar 取出父块片段，
  替换送进生成层的上下文（检索排序仍基于子块）。

`mode` 三选一：`dense`（默认，与旧行为完全一致）/ `bm25` / `hybrid`（RRF 融合）。

### 5.6 生成层（query.py）

- OpenAI 兼容客户端，`LLM_API_BASE` 默认指向 DeepSeek，可切换 OpenAI / 智谱 / 通义等。
- System 提示词约束：**仅依据给定资料回答，标注 `[来源N]`，资料不足时明确说明**。
- 未设置 `LLM_API_KEY` 时降级为「纯检索」模式，只返回相关原文片段。

### 5.7 评测层（benchmark.py / judge.py / eval.py）

分两条线：

1. **检索指标**（`eval retrieval`）：用 `eval_data/testset.json` 的改写问题 +
   `answer_key` 定位 gold chunk，对比 BM25 与 BGE-m3 的 Recall@k / MRR@k / nDCG@k。
2. **答案质量**（`eval split` → `eval answer`）：
   - `split`：清洗 `data_benchmark/*.csv`（去 BOM、首尾 `...`、过滤 <5 字提问 / <20 字答复、去重），
     **按 `来源姓名` 用户分组** + 主题分层抽样切分 train/test，固定种子（`BENCHMARK_SEED=42`），
     落到 `data_benchmark/split/{train,test}.jsonl`。
   - `answer`：对测试集逐条跑「检索 + 生成」，用 LLM-as-judge（1~5 分 + 是否答对，严格 JSON）
     与标准答案比对；无 LLM 或解析失败时降级为词法基线（字符 bigram F1 → 5 分制）。
     同时报告 `charF1` 与字符级 `ROUGE-L`。
   - `--ablate` 输出 dense / hybrid / hybrid+rerank / hybrid+改写 的对照表。

**为什么按用户分组切分**：`data_benchmark` 前 5 名用户贡献约一半样本，
同一用户问题高度同质，随机切分会让相似问题跨 train/test 泄漏、虚高评测分
（详见 `docs/data_benchmark_数据评估报告.md`）。

---

## 6. 专家标注（双轨）预留接口

当前为「机器标注」：自动切块 + 向量 + 元数据。要接入「专家标注」，沿以下通路扩展：

1. **按 id 增删改**：chunk id 形如 `<hash>::<文件名>::chunkN`，可对 Chroma collection 直接 `update / delete / get`，实现人工改写原文、删除低质片段。
2. **打标签**：在 `index.py` 的 `metadatas` 增加自定义字段（如 `expert_tag`、`quality`、`difficulty`），提供人工标注入口写入；检索时用 `where=` 过滤。
3. **对齐方式**：专家标注数据另存 `rag/annotations.jsonl`（id → 标签/修订），与机器向量库通过 id 对齐，不破坏自动建库流程。

---

## 7. 当前进展

- ✅ 依赖环境搭建（venv + pymupdf / sentence-transformers / chromadb / openai / python-dotenv）。
- ✅ 解析、分块、向量化、入库、检索、生成六层代码全部实现。
- ✅ 分块逻辑、解析逻辑、ID 唯一性均已验证（68 文件 → 5133 chunk → 5133 唯一 ID）。
- ✅ 已修复重名文件导致的 chunk ID 冲突；已加入向量缓存。
- ✅ 解析层跨平台化：`.docx`/`.pptx` 走 python-docx/python-pptx，`.doc` 走 Word COM 或 LibreOffice，
  后端可用性在 `build` 时显式打印（不再静默跳过）。
- ✅ 检索增强落地（默认关闭）：混合检索（BM25 + 稠密 + RRF）、CrossEncoder 重排序、
  元数据过滤、查询改写 / HyDE。
- ✅ 父子分块落地（默认关闭）：子块入库 + 父块 sidecar + 查询期上下文展开。
- ✅ 评测闭环落地：data_benchmark 按用户分组切分 + LLM-as-judge（含词法降级）+ 消融对比。
- ✅ 离线测试覆盖解析/分块/检索/切分/打分（`pytest`，需要模型与数据的用例自动跳过）。
- ⏳ 全量建库执行中（BGE-m3 向量化约 30–40 分钟，视机器而定）。

---

## 8. 使用方式

```bash
# 建库（首次或资料变更后）
.venv/bin/python -m rag.cli build

# 问答（需先在 .env 配 LLM_API_KEY；不配则纯检索）
.venv/bin/python -m rag.cli ask "扎根理论的三级编码是什么？"
.venv/bin/python -m rag.cli ask "开题报告要写哪些内容？" -k 6

# 开启检索增强（按需）
.venv/bin/python -m rag.cli ask "文献综述怎么写？" --mode hybrid --rerank
.venv/bin/python -m rag.cli ask "文献综述怎么写？" --where category=002

# 常驻服务 / 评测
.venv/bin/python -m rag.cli serve --http
.venv/bin/python -m rag.cli eval split
.venv/bin/python -m rag.cli eval answer --ablate
```

---

## 9. 后续路线图

| 阶段 | 事项 | 说明 |
|------|------|------|
| ✅ 已完成 | 检索质量评估 | `eval retrieval`：BM25 vs BGE-m3 的 Recall/MRR/nDCG |
| ✅ 已完成 | 重排序 Rerank | `--rerank`：bge-reranker-v2-m3，懒加载、默认关闭 |
| ✅ 已完成 | 评测闭环 | data_benchmark 切分 + LLM-as-judge + 消融（`eval split`/`eval answer`） |
| ✅ 已完成 | 跨平台运行 | 解析层按能力探测后端，去掉 macOS `textutil` 依赖 |
| 近期 | 专家标注入口 | 打通 `annotations.jsonl` + 人工增删改通路 |
| 近期 | 检索参数调优 | 用评测闭环扫 `CHUNK_SIZE` / `FETCH_K` / `RRF_K` / rerank 组合 |
| 中期 | 服务器部署 | 「检索+生成」层封装为 FastAPI 服务，向量库迁移到服务器 |
| 中期 | 页码级引用 | PDF 分块携带页码，支持「参见某页」级回答 |
| 中期 | 增量重建 | 目前 build 是全量重建；改为按文件 mtime/哈希做增量索引 |
| 远期 | 其余三场景 | 项目解释 / 前沿推理 / 学术沟通 各自知识库接入同一框架 |
| 远期 | 语义近重复去重 | 报告建议的 embedding 聚类聚簇，切分时同簇不跨池 |

---

## 10. 风险与限制

- **向量化耗时**：BGE-m3 本地向量化 5133 个 chunk 约 30–40 分钟（视机器）；已用向量缓存缓解，
  但仍是**全量重建**，资料变更后的增量重建待做。
- **`.ppt` / `.doc` 解析**：依赖外部后端（LibreOffice，或 Windows 上的 Word + pywin32）；
  缺后端时对应文件被跳过，`build` 会打印后端可用性。
- **重排序成本**：`--rerank` 首次下载约 2.3GB 模型，CPU 推理明显变慢，故默认关闭。
- **无页码元数据**：当前 chunk 不含 PDF 页码，精确引用待补。
- **生成依赖外部 LLM**：需用户提供 API key；纯本地离线生成需部署开源模型（vLLM/Ollama）。
- **评测依赖 LLM 评委**：`eval answer` 的判分带模型主观性；词法基线只看字面重合，
  在长答复上系统性偏低——应作相对比较，不宜作绝对结论。
- **建库/查询开关耦合**：父块 sidecar 只在开启 `PARENT_CHILD_ENABLED` 建库时生成；
  建库与查询开关不一致时会静默退回子块原文。
