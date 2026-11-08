"""新旧题库对比（只读）：各题型题数、带标记题数、题干 / 答案变化、question_id 稳定性、知识点挂接率。

用法：python scripts/exam_bank_diff.py --old 旧教师版.docx --new 新教师版.docx [--student-set 学生知识集合.xlsx] [--concept-table 总表.xlsx] [--examples 5]
不写任何文件，不调用模型。
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.exam.bank import norm, parse_bank  # noqa: E402


def _key(q):
    return (q.chapter, q.qtype, q.number)


def _plain(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def compare(old_path, new_path, examples: int = 5) -> dict:
    qo, so = parse_bank(old_path)
    qn, sn = parse_bank(new_path)
    o, n = {_key(q): q for q in qo}, {_key(q): q for q in qn}
    common = sorted(set(o) & set(n))
    stem_changed = [k for k in common if norm(o[k].stem) != norm(n[k].stem)]
    ans_changed = [k for k in common if _plain(o[k].answer) != _plain(n[k].answer)]
    key_ans_changed = [k for k in common if o[k].qtype in ("choice", "judge") and o[k].answer != n[k].answer]
    opt_changed = [k for k in common if o[k].qtype == "choice" and o[k].options != n[k].options]
    analysis_changed = [k for k in common if _plain(o[k].analysis) != _plain(n[k].analysis)]
    id_same = sum(o[k].question_id == n[k].question_id for k in common)

    def sim(a, b):
        return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)
    ex = [{"题": f"第{k[0]}章 {o[k].qtype_cn} 第{k[2]}题", "旧": o[k].stem[:70], "新": n[k].stem[:70], "相似度": sim(norm(o[k].stem), norm(n[k].stem))}
          for k in stem_changed[:: max(1, len(stem_changed) // max(examples, 1))][:examples]]
    ans_ex = [{"题": f"第{k[0]}章 {o[k].qtype_cn} 第{k[2]}题", "旧答案": o[k].answer[:40], "新答案": n[k].answer[:40]} for k in key_ans_changed[:examples]]
    by_ch = Counter(k[0] for k in stem_changed)
    return {
        "题数": {"旧": len(qo), "新": len(qn), "旧各题型": dict(Counter(q.qtype for q in qo)), "新各题型": dict(Counter(q.qtype for q in qn))},
        "标记": {"旧": so["flags"], "新": sn["flags"], "旧未解析段": so["unparsed_count"], "新未解析段": sn["unparsed_count"],
                 "新声明题数不符": sn["count_mismatch"]},
        "按（章,题型,题号）对齐": {"共同": len(common), "只在旧": len(set(o) - set(n)), "只在新": len(set(n) - set(o))},
        "题干改了": {"道数": len(stem_changed), "按章": dict(sorted(by_ch.items())), "例子": ex},
        "选项改了(选择题)": len(opt_changed),
        "答案变了": {"选择/判断题的答案字母或正误变了": len(key_ans_changed), "全部题型（忽略空白）": len(ans_changed), "例子": ans_ex},
        "解析/评分要点文字改了": len(analysis_changed),
        "question_id": {"沿用同一编号规则时相同": id_same, "共同题数": len(common), "说明": "id 含版本号与题干哈希；题干改了的题 id 必然不同"},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args(argv)
    print(json.dumps(compare(args.old, args.new, args.examples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
