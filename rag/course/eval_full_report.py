"""完整评测的汇总（纯函数，可离线测试）：关键句命中（仅本体 vs 本体+BM25）、BM25 块的筛选等级、引用情况。"""
from __future__ import annotations

SMALL = ("r11", "r12", "r13")


def _hit(target_chunks: set[str], ids: set[str]) -> bool:
    return bool(target_chunks & ids)


def summarize(rows: list[dict], gold: list[dict]) -> dict:
    by_id = {g["id"]: g for g in gold}
    per_q, groups = [], {"r11-r13": [0, 0, 0], "r14-r20": [0, 0, 0], "all": [0, 0, 0]}   # [仅本体, 本体+BM25, 总数]
    bm_grades = {"direct": 0, "background": 0, "irrelevant": 0, "ungraded": 0}
    bm_only_cited = 0
    for r in rows:
        if r.get("error") or "recalled" not in r:
            per_q.append({"id": r["id"], "error": r.get("error")})
            continue
        g = by_id[r["id"]]
        onto = {x["chunk_id"] for x in r["recalled"] if x["source"] in ("ontology", "both")}
        both = {x["chunk_id"] for x in r["recalled"]}
        hits = []
        for m in g["must_recall"]:
            t = set(m["chunks"])
            ho, hb = _hit(t, onto), _hit(t, both)
            hits.append({"key": m["key"], "ontology_only": ho, "ontology_plus_bm25": hb})
            grp = "r11-r13" if r["id"] in SMALL else "r14-r20"
            for k in (grp, "all"):
                groups[k][0] += int(ho); groups[k][1] += int(hb); groups[k][2] += 1
        bm_only = [x for x in r["recalled"] if x["source"] == "bm25"]
        for x in bm_only:
            bm_grades[x["grade"] if x["grade"] in bm_grades else "ungraded"] += 1
        cited_bm_only = [c for c in r["citations"] if c in {x["chunk_id"] for x in bm_only}]
        bm_only_cited += len(cited_bm_only)
        per_q.append({"id": r["id"], "hits": hits, "recalled": len(r["recalled"]),
                      "sources": {s: sum(x["source"] == s for x in r["recalled"]) for s in ("ontology", "both", "bm25")},
                      "bm25_only_grades": {gr: sum(x["grade"] == gr for x in bm_only) for gr in ("direct", "background", "irrelevant")},
                      "citations": len(r["citations"]), "bm25_only_cited": len(cited_bm_only),
                      "fallback": r["fallback"]})
    rate = lambda a, n: round(a / n, 3) if n else None
    return {"hit_rate": {k: {"ontology_only": f"{v[0]}/{v[2]}", "ontology_plus_bm25": f"{v[1]}/{v[2]}",
                             "ontology_only_rate": rate(v[0], v[2]), "ontology_plus_bm25_rate": rate(v[1], v[2])}
                         for k, v in groups.items()},
            "bm25_only_chunk_grades": bm_grades, "bm25_only_chunks_cited_in_answers": bm_only_cited, "per_question": per_q}
