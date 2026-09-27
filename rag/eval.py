"""检索质量评估：对比 BM25 与 BGE-m3 稠密向量，计算 Recall@k / MRR@k / nDCG@k。

用法（在项目根目录 FUTURX 下）：
  .venv/bin/python -m rag.eval

评估专用依赖：.venv/bin/pip install rank_bm25 jieba

说明：
- 测试集来自 nlp_/001、003 的问答对，问题已改写（paraphrase）以规避「原文问题
  与答案同处一个 chunk 导致的自匹配泄漏」，让检索靠语义而非字面命中答案。
- gold chunk = 包含 answer_key 的入库 chunk（因 001 与 003/问答集合 内容相同，
  每题通常有 2 个 gold chunk，属正常）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .config import COLLECTION_NAME, EMBED_MODEL, STORE_DIR

TEST_FILE = Path(__file__).resolve().parent / "eval_data" / "testset.json"
KS = [1, 3, 5, 10]


def load_testset() -> list[dict]:
    return json.loads(TEST_FILE.read_text(encoding="utf-8"))


def load_corpus():
    import chromadb

    client = chromadb.PersistentClient(path=str(STORE_DIR))
    col = client.get_collection(COLLECTION_NAME)
    data = col.get(include=["documents"])
    return list(data["ids"]), list(data["documents"])


def find_gold(testset, docs, ids):
    """gold[qid] = 包含 answer_key 的 chunk id 列表。"""
    gold = {}
    for item in testset:
        key = item["answer_key"]
        gold[item["id"]] = [cid for cid, d in zip(ids, docs) if key in d]
    return gold


def _metrics(gold_ids, ranked_ids):
    g = set(gold_ids)
    if not g:
        return 0.0, 0.0, 0.0
    hits = [i for i, cid in enumerate(ranked_ids, 1) if cid in g]
    recall = len(hits) / len(g)
    mrr = 1.0 / hits[0] if hits else 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i in hits)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), len(ranked_ids)) + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return recall, mrr, ndcg


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def run() -> None:
    testset = load_testset()
    ids, docs = load_corpus()
    gold = find_gold(testset, docs, ids)

    print(f"语料 chunk 数：{len(docs)}")
    print(f"测试问题数：{len(testset)}\n")

    print("gold chunk 覆盖检查：")
    for item in testset:
        n = len(gold[item["id"]])
        flag = "" if n else "  <-- 未命中，需修正 answer_key"
        print(f"  {item['id']:>4}  找到 {n} 个 gold chunk{flag}")

    # ---- BM25 ----
    try:
        from rank_bm25 import BM25Okapi
        import jieba
    except ImportError:
        print("\n缺少评估依赖，请先安装：.venv/bin/pip install rank_bm25 jieba")
        return

    tokenize = jieba.lcut
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    # ---- Dense (BGE-m3) ----
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    col = client.get_collection(COLLECTION_NAME)

    top_n = KS[-1]
    bm25_ranks, dense_ranks = {}, {}
    for item in testset:
        qid, q = item["id"], item["question"]

        scores = bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
        bm25_ranks[qid] = [ids[i] for i in order]

        emb = model.encode(q, normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=[emb], n_results=top_n)
        dense_ranks[qid] = res["ids"][0]

    # ---- 汇总指标 ----
    print("\n指标对比（BM25  vs  BGE-m3 稠密向量）：")
    print(f"{'k':>3} | {'Recall@k':>15} | {'MRR@k':>12} | {'nDCG@k':>12}")
    print(f"{'':>3} | {'BM25':>7}{'Dense':>8} | {'BM25':>6}{'Dense':>6} | {'BM25':>6}{'Dense':>6}")
    for k in KS:
        bm, dn = [], []
        for item in testset:
            qid = item["id"]
            bm.append(_metrics(gold[qid], bm25_ranks[qid][:k]))
            dn.append(_metrics(gold[qid], dense_ranks[qid][:k]))
        br = tuple(_mean(x[i] for x in bm) for i in range(3))
        dr = tuple(_mean(x[i] for x in dn) for i in range(3))
        print(f"{k:>3} | {br[0]:>7.3f}{dr[0]:>8.3f} | {br[1]:>6.3f}{dr[1]:>6.3f} | {br[2]:>6.3f}{dr[2]:>6.3f}")

    # ---- 逐题：首个命中排名 ----
    print("\n逐题首个命中排名（- 表示 top10 未命中）：")
    print(f"{'id':>4} | {'BM25':>6} | {'Dense':>6} | 问题")
    for item in testset:
        qid = item["id"]
        g = set(gold[qid])
        bm_hit = next((i for i, cid in enumerate(bm25_ranks[qid], 1) if cid in g), None)
        dn_hit = next((i for i, cid in enumerate(dense_ranks[qid], 1) if cid in g), None)
        print(f"{qid:>4} | {str(bm_hit) if bm_hit else '-':>6} | {str(dn_hit) if dn_hit else '-':>6} | {item['question']}")


def run_benchmark(ablate: bool = False, limit: int = 0) -> None:
    """基于 data_benchmark/test.jsonl 的答案质量评测（LLM-as-judge + 词法基线）。

    data_benchmark 缺失时打印指引并优雅退出，不抛异常。
    注意：需先 `python -m rag.cli eval --split` 生成切分，且要求各测试问题
    对应的资料已入库（否则检索结果为空，分数无意义）。
    """
    from pathlib import Path

    from .benchmark import BENCHMARK_DIR, read_jsonl
    from .config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL
    from .judge import char_f1, judge_answer, lexical_score, mean, rouge_l

    test_path = Path(BENCHMARK_DIR) / "split" / "test.jsonl"
    if not test_path.exists():
        print(f"未找到测试集 {test_path}")
        print("请先运行：python -m rag.cli eval --split")
        return

    tests = read_jsonl(test_path)
    if limit:
        tests = tests[:limit]
    if not tests:
        print("测试集为空。")
        return
    print(f"测试集：{len(tests)} 条（来自 {test_path}）\n")

    llm = None
    if LLM_API_KEY:
        from openai import OpenAI

        llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    else:
        print("（未设置 LLM_API_KEY，仅跑词法基线；LLM 评审已跳过）\n")

    configs = [("hybrid", False)]
    if ablate:
        configs = [
            ("dense", False),
            ("hybrid", False),
            ("hybrid", True),
            ("hybrid", False, True),   # + 查询改写
            ("hybrid", True, True),    # rerank + 改写
        ]

    print(f"{'配置':<28} | {'n':>3} | {'judge(5)':>8} | {'答对率':>7} | {'charF1':>7} | {'ROUGE-L':>7}")
    print("-" * 86)
    for cfg in configs:
        mode, rerank = cfg[0], cfg[1]
        rewrite = cfg[2] if len(cfg) > 2 else False
        label = f"mode={mode},rerank={rerank}" + (",rewrite=1" if rewrite else "")
        scores, corrects, f1s, rls = [], [], [], []
        for it in tests:
            try:
                ans, _sources, _docs = _answer_once(it.question, mode, rerank, rewrite)
            except Exception as e:
                print(f"  [跳过] {it.id}: {e}")
                continue
            pred = ans or ""
            f1s.append(char_f1(pred, it.answer))
            rls.append(rouge_l(pred, it.answer))
            if llm is not None:
                jr = judge_answer(it.question, pred, it.answer, llm, LLM_MODEL)
                scores.append(jr.score)
                corrects.append(1.0 if jr.correct else 0.0)
            else:
                lr = lexical_score(pred, it.answer)
                scores.append(lr.score)
                corrects.append(1.0 if lr.correct else 0.0)
        n = len(f1s)
        print(
            f"{label:<28} | {n:>3} | {mean(scores):>8.3f} | {mean(corrects):>7.3f} | "
            f"{mean(f1s):>7.3f} | {mean(rls):>7.3f}"
        )


def _answer_once(question: str, mode: str, rerank: bool, rewrite: bool):
    """单次问答（每次新建服务会重复加载模型；大规模评测建议用 serve）。"""
    from .query import RAGService

    svc = RAGService(mode=mode, rerank=rerank, rewrite=rewrite)
    return svc.answer(question)


if __name__ == "__main__":
    run()
