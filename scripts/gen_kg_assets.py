"""离线生成网页用的两份静态数据（词云 + 知识图谱），源：纯粹知识集合.xlsx（三层集合之一）。

输出（默认写到 rag/web/data/）：
  wordcloud.json  词云：只含 {"w": 关键词, "s": 权重(1-5 档)}，60–80 个；不含定义。
  graph.json      知识图谱（部分展示）：按章节组织；已在这里完成打码——
                  · 每章只保留约 30% 最核心（关联最多）的概念名称，其余节点只有占位（无名称）；
                  · 不含任何定义 / 公理 / 原文；关系只有类型（取枚举，不含具体内容）与连线；
                  · 完整数据不写入任何网页文件。前端拿不到被打码的名称，CSS 遮挡不是保护手段。
用法：python scripts/gen_kg_assets.py --xlsx 纯粹知识集合.xlsx [--out rag/web/data] [--words 72] [--reveal 0.3] [--dry-run]
同一份 xlsx 重复运行结果完全一致（不使用随机数）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.course.ontology import load_student_set  # noqa: E402

# ---- 哪些关系是“结构 / 出处”性质，不代表概念之间的知识联系：不计入关联数，也不画线 ----
STRUCTURAL = {"hasSourceSpan", "extractedFrom", "hasSource", "instanceOf", "hasDefinition", "hasLearningTag",
              "recommendedFor", "answersGuidingQuestion", "hasKnowledgePoint", "hasSubsection"}
CASE_HINT = re.compile(r"(诉|案|法院|纠纷|发布|典型案例|判决|裁定)")
CHAPTER_HEAD = re.compile(r"^第[一二三四五六七八九十\d]+章")
# 不适合当“知识关键词”展示的：章节小结类、序号开头、过泛的单词、模块名
BAD_PREFIX = re.compile(r"^[（(\d一二三四五六七八九十]+[）).、]")
GENERIC = {"数据", "小结", "总结", "概述", "引言", "本章小结", "背景", "意义", "原则", "方法", "特点", "作用", "问题", "要素", "技术", "信息"}
BAD_SUFFIX = ("模块", "小结", "难量化")
NAME_MIN, NAME_MAX = 2, 14
MIN_PER_CH, MAX_PER_CH = 3, 22          # 词云：每章至少 3 个、至多 22 个
GRAPH_PER_CHAPTER = 70                  # 图谱：每章最多保留的节点数（按关联度取前几名），否则第 4 章（约六成概念）把整张图染成一个颜色
CHAPTER_NAMES = {1: "数据要素", 2: "数据权属", 3: "数据价值", 4: "数据政策", 5: "数据产品", 6: "数据市场",
                 7: "数据交易", 8: "数据安全", 9: "隐私保护", 10: "数据治理", 11: "数据工程师"}
# 5 档字号对应的权重阈值在生成时按分位数决定（见 tiers）


def clean_name(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def is_concept_name(name: str) -> bool:
    """适合当“知识关键词”展示的名称：长度合适，不是章节标题、政策全称（书名号）、案例名。"""
    if not (NAME_MIN <= len(name) <= NAME_MAX):
        return False
    if name.startswith(("《", "“", "\"")) or "《" in name or CHAPTER_HEAD.match(name) or CASE_HINT.search(name):
        return False
    if "：" in name or ":" in name or BAD_PREFIX.match(name) or name in GENERIC or name.endswith(BAD_SUFFIX):
        return False
    return True


def chapter_no(label: str) -> int:
    m = re.search(r"(\d+)", label or "")
    return int(m.group(1)) if m else 0


def load(xlsx: Path) -> list[dict]:
    rows = load_student_set(xlsx)
    for r in rows:
        r["主体"], r["客体"], r["关系"] = clean_name(r.get("主体", "")), clean_name(r.get("客体", "")), (r.get("关系") or "").strip()
        r["_ch"] = chapter_no(r.get("章节", ""))
    return rows


def build(rows: list[dict], n_words: int = 72, reveal: float = 0.3, per_chapter: int = GRAPH_PER_CHAPTER) -> tuple[dict, dict, dict]:
    """返回 (wordcloud, graph, report)。report 只用于打印统计，不写进网页。"""
    subjects = {r["主体"] for r in rows if r["主体"]}
    concepts = {s for s in subjects if is_concept_name(s)}
    # 概念出现的章节（取出现次数最多的章为“所属章”）；关联数 = 与其它概念之间的非结构性关系数（作主体或客体）
    ch_count: dict[str, Counter] = defaultdict(Counter)
    deg: Counter = Counter()
    appear: Counter = Counter()
    edges: Counter = Counter()          # (a, b, rel) -> 次数
    for r in rows:
        s, o, rel, ch = r["主体"], r["客体"], r["关系"], r["_ch"]
        if s in concepts:
            ch_count[s][ch] += 1
            appear[s] += 1
        if o in concepts:
            ch_count[o][ch] += 1
        if rel in STRUCTURAL or s == o:
            continue
        if s in concepts and o in concepts:
            deg[s] += 1
            deg[o] += 1
            a, b = sorted((s, o))
            edges[(a, b, rel)] += 1
    home = {c: max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0] for c, cnt in ch_count.items()}
    home = {c: ch for c, ch in home.items() if 1 <= ch <= 11}
    nodes = [c for c in concepts if c in home]
    # 重要性 = 关联数 + 0.25 × 出现次数 + 0.5 × 跨章数；平手按名称排序，保证可复现
    score = {c: deg[c] + 0.25 * appear[c] + 0.5 * (len(ch_count[c]) - 1) for c in nodes}
    order = sorted(nodes, key=lambda c: (-score[c], c))

    # ---------- 词云 ----------
    # 第四章占了全部概念的六成：不做配额的话词云会被“数据政策”一章占满。每章先保底取前 MIN_PER_CH 个，其余名额按全局重要性补满。
    chap_order: dict[int, list[str]] = defaultdict(list)
    for c in order:
        if deg[c] >= 1:
            chap_order[home[c]].append(c)
    words: list[str] = []
    for ch in sorted(chap_order):
        words += chap_order[ch][:MIN_PER_CH]
    words = words[:n_words]
    for c in order:
        if len(words) >= n_words:
            break
        if c not in words and deg[c] >= 1 and sum(home[w] == home[c] for w in words) < MAX_PER_CH:
            words.append(c)
    words.sort(key=lambda c: (-score[c], c))
    ws = [score[w] for w in words]
    lo, hi = min(ws), max(ws)
    cuts = _tier_cuts(ws, 5)
    cloud = {"words": [{"w": w, "s": _tier(score[w], cuts)} for w in words]}

    # ---------- 图谱（打码在这里完成）----------
    by_ch: dict[int, list[str]] = defaultdict(list)
    for c in order:
        if len(by_ch[home[c]]) < per_chapter:             # order 已按重要性降序：每章只留前 per_chapter 个
            by_ch[home[c]].append(c)
    kept = {c for cs in by_ch.values() for c in cs}
    order = [c for c in order if c in kept]
    revealed: set[str] = set()
    for ch, cs in by_ch.items():
        k = max(1, round(len(cs) * reveal))
        revealed.update(cs[:k])                    # cs 已按重要性降序：取关联最多的前 30%
    index = {c: i for i, c in enumerate(order)}
    # 边：只保留两端都在 nodes 里的；同一对节点只留一条（取出现最多的关系类型）；关系类型保留英文枚举名
    pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (a, b, rel), n in edges.items():
        if a in index and b in index:
            pair[(a, b)][rel] += n
    g_nodes = []
    for c in order:
        node = {"i": index[c], "c": home[c], "d": deg[c]}
        if c in revealed:
            node["n"] = c                          # 只有被公开的节点才带名称
        g_nodes.append(node)
    g_edges = []
    for (a, b), cnt in sorted(pair.items()):
        rel = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        g_edges.append([index[a], index[b], rel])
    totals = {"chapters": len(by_ch), "concepts": len(order), "relations": len(g_edges),
              "shown_names": len(revealed), "shown_ratio": round(len(revealed) / max(len(order), 1), 3),
              "concepts_total": len(nodes), "per_chapter_cap": per_chapter}
    graph = {"chapters": {str(k): CHAPTER_NAMES.get(k, f"第{k}章") for k in sorted(by_ch)}, "nodes": g_nodes, "edges": g_edges, "totals": totals}
    report = {"concepts_total": len(concepts), "concepts_in_graph": len(order), "words": len(words), "word_tiers": dict(Counter(w["s"] for w in cloud["words"])),
              "score_range": [round(lo, 2), round(hi, 2)], "edges": len(g_edges), "revealed": len(revealed), "ratio": totals["shown_ratio"],
              "per_chapter": {str(k): {"concepts": len(v), "named": sum(1 for c in v if c in revealed)} for k, v in sorted(by_ch.items())},
              "top_words": words[:12]}
    return cloud, graph, report


def _tier_cuts(vals: list[float], n: int) -> list[float]:
    s = sorted(vals)
    return [s[min(len(s) - 1, int(len(s) * i / n))] for i in range(1, n)]


def _tier(v: float, cuts: list[float]) -> int:
    t = 1
    for c in cuts:
        if v >= c:
            t += 1
    return min(t, len(cuts) + 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "rag" / "web" / "data"))
    ap.add_argument("--words", type=int, default=72)
    ap.add_argument("--reveal", type=float, default=0.3)
    ap.add_argument("--per-chapter", type=int, default=GRAPH_PER_CHAPTER)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cloud, graph, report = build(load(Path(a.xlsx)), a.words, a.reveal, a.per_chapter)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.dry_run:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("wordcloud.json", cloud), ("graph.json", graph)):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("written", out / name, (out / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
