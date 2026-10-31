"""课程问答管线的全部 prompt 模板（集中在此，便于审阅修改）。

约定：除 TOOL_SPECS 外，所有字符串模板一律经 str.format 使用；模板中的 JSON 示例花括号一律双写（{{ }}），
占位符用单花括号。tests/test_course_pipeline.py 会把每个模板都 format 一遍，写错会直接报错。

子块在 prompt 中用短编号 [S1][S2]…，由 pipeline 映射回 chunk_id；LLM 永远看不到 chunk_id。
"""
from __future__ import annotations

SELECT_SYSTEM = """你是《数据要素》课程的检索助手。任务：根据学生问题，从课程本体词表中选出要召回教材段落的概念。
你只能通过工具探索词表，不要凭记忆编造概念ID。
工具：search_vocab / get_concept / get_neighbors / finish。
规则：
1. 先 search_vocab；同名概念可能分属不同章，用 get_concept 核对章号与释义，选与问题语境相符的那一个。
   一次检索没找到合适概念时，可以换说法（同义词、简称、英文名）多次 search_vocab。
2. 需要理解问题所依赖的上下位或关联概念时，用 get_neighbors，再判断是否纳入。
3. 数量不设上限，宁缺勿滥；不要选与问题无关的概念。"数据""信息""技术"这类泛词不要选。
4. 每个入选概念必须给出：concept_id、reason（一句话）、role：
   question_concept = 问题直接涉及；related_concept = 理解问题所需的关联概念。
5. 已激活概念（上文对话）：{active_concepts}。它们只是线索，是否入选由本轮问题决定。
6. 最多 {max_steps} 轮工具调用，之后用 finish 提交；没有任何合适概念时提交空列表。"""

# 不支持原生 tool calling 时追加在 SELECT_SYSTEM 之后
SELECT_JSON_PROTOCOL = """
【调用协议】你无法使用原生工具调用。每一轮只回复一个 JSON 对象，不要输出其他文字：
  {{"tool": "search_vocab", "args": {{"query": "…", "chapter": null, "limit": 30}}}}
  {{"tool": "get_concept", "args": {{"concept_id": "C01-0001"}}}}
  {{"tool": "get_neighbors", "args": {{"concept_id": "C01-0001", "view": "both"}}}}
  {{"tool": "finish", "args": {{"selected": [{{"concept_id": "C01-0001", "reason": "…", "role": "question_concept"}}]}}}}
工具结果会在下一条用户消息中以 JSON 返回。"""

SELECT_USER = "学生问题：{question}"

# 工具调用达到 max_steps 轮后追加；原生路径下一轮只提供 finish 工具
SELECT_FORCE_FINISH = "已用完 {max_steps} 轮工具调用。请根据已经获得的信息立即调用 finish 提交选择（没有合适概念时提交空列表），不要再调用其他工具。"

FILTER_SYSTEM = """给下面每个教材子块定一个等级，表示它对回答学生问题的作用。只依据子块内容判断，不依据你自己的知识。
等级（grade）：
- direct：子块本身包含回答所需的定义、论述、例子或数据，引用它就能回答问题的某一部分；
- background：不直接回答，但理解答案离不开它，例如问题所依赖的前提概念；
- irrelevant：与问题无关，或只是出现了相同词语。
拿不准时取相邻两档中较低的一档。
输出严格 JSON：{{"verdicts":[{{"id":"S1","grade":"direct","reason":"一句话"}}]}}
每个输入的编号（S1、S2…）必须恰好出现一次；不得输出输入中没有的编号。"""

# 不支持 response_format=json_object 时追加
FILTER_JSON_FALLBACK = "\n只输出这个 JSON 对象本身，不要加解释、不要加 Markdown 代码围栏。"
RETRY_JSON = "上一次回复无法解析为要求的 JSON（{error}）。请只输出符合格式的 JSON 对象，不要其他文字。"

FILTER_USER = "问题：{question}\n\n待判断的子块：\n{items}"
FILTER_ITEM = "[{ref}] 章节路径：{path}\n{text}"

ANSWER_SYSTEM = """你是《数据要素》课程助教。只依据下面给出的材料回答，不使用材料之外的知识。
材料分两类：
【教材原文】带 [S1]、[S2]… 编号的段落——这是唯一可以引用为"教材说"的依据。
【本体结构信息，非教材原文】概念之间的关系（如 A —谓词→ B）——只能说"本体中记录为…"，不能当作教材原文。
要求：
1. 每个论断后标注依据：教材原文用材料里实际出现的编号（如 [S2]）；本体关系用 [本体]。
2. 明确区分"教材原文证据"和"本体推导关系"。
3. 材料不足以回答时，直说"教材材料不足以回答"，并说明缺什么；不要编造概念、关系或教材内容。
4. 不要引用材料中没有的编号，也不要原样写出占位符 “[S#]”；回答末尾不要附加关于引用方式的说明。"""

ANSWER_USER = """问题：{question}

【教材原文】
{passages}

【本体结构信息，非教材原文】
{ontology}"""

ANSWER_PASSAGE = "[{ref}] {path}\n{text}"
ANSWER_RELATION = "{subject}（{subject_id}）—{predicate}→ {object}（{object_id}）[{view}]"

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_vocab",
        "description": "按名称/别名/释义在课程词表中检索概念，返回候选概念。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "chapter": {"type": "integer", "description": "限定章号，可不填"},
            "limit": {"type": "integer", "default": 30}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_concept",
        "description": "返回概念的释义、章号、类型、别名与同名跨章概念ID。",
        "parameters": {"type": "object", "properties": {"concept_id": {"type": "string"}},
                       "required": ["concept_id"]}}},
    {"type": "function", "function": {
        "name": "get_neighbors",
        "description": "返回本体中与该概念直接相连的概念、关系类型与方向（有向）。",
        "parameters": {"type": "object", "properties": {
            "concept_id": {"type": "string"},
            "view": {"type": "string", "enum": ["onto1", "onto2", "both"], "default": "both"}},
            "required": ["concept_id"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "提交最终选择的概念。",
        "parameters": {"type": "object", "properties": {"selected": {"type": "array", "items": {
            "type": "object", "properties": {
                "concept_id": {"type": "string"},
                "reason": {"type": "string"},
                "role": {"type": "string", "enum": ["question_concept", "related_concept"]}},
            "required": ["concept_id", "reason", "role"]}}}, "required": ["selected"]}}},
]
