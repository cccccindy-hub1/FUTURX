"""检查 eval result json 的回答格式：总起 / 分点 / 收束 / 字数 / 追问 / 样例复用。

用法：
    python scripts/check_answer_format.py rag/eval_data/course_e2e_1102_flash_v2.json
    python scripts/check_answer_format.py <file> [<file> ...] [--min-points 2]

输出逐题表、通过率、字数分布（min/median/max）、追问合格率与样例复用标记。
判定口径与 rag/prompt_rules.ANSWER_STRUCTURE 一致：
- 总起：首个非空行是完整句子（≥12 字）且不以分点符号开头；
- 分点：2–4 条以 - / * / 1. / （1） / 一、 开头的行；>4 条判 FAIL（与 prompt 硬上限一致）；
- 收束：最后一个非分点段落，且不是分点行本身；
- 字数：剥离引用标记（[1][2] 与 [来源N]）后统计；≤400 PASS，401–500 WARN，>500 FAIL；
- 追问：2–4 条、每条 ≤25 字、以问号结尾（route=exam 记 N/A，剔除出分母）；
- 样例复用：与 prompt_rules 四条样例连续 ≥12 字相同即标记（样例只示范结构，不得抄原句）。

压缩格式例外：≤150 字的答案不要求分点（降级为警告），但总起与收束仍需存在。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import prompt_rules  # noqa: E402

MAX_CHARS = 400
WARN_CHARS = 500                        # 401–500 警告，>500 失败
MAX_POINTS = 4                          # 与 prompt 硬上限一致：>4 条即 FAIL
SHORT_ANSWER = 150                      # ≤ 此长度，分点要求降级为警告
SAMPLE_OVERLAP = 12                     # 与样例连续相同的字数阈值
_BULLET = re.compile(r"^\s*(?:[-*+·]|\d{1,2}[.)、]|[（(]\d{1,2}[）)]|[一二三四五六七八九十]+[、.])")
_SENT_END = re.compile(r"[。！？；]")
# 引用标记：[1] [12] [来源1] [来源 3]，以及误留的 [S1]（作答提示词要求不出现）
_CITE = re.compile(r"[\[【]\s*(?:来源\s*\d+|[Ss]\d{1,3}|\d{1,3})\s*[\]】]")

# 四条样例：两个常量拼接后按样例标题切开，取「回答：」之后的正文作为比对语料
_SAMPLES = re.split(r"【样例[一二三四五六七八九十]+·",
                    prompt_rules.COURSE_SAMPLES + "\n" + prompt_rules.THESIS_SAMPLES)[1:]
_SAMPLE_BODIES = [re.sub(r"\s+", "", s.split("回答：", 1)[1]) for s in _SAMPLES if "回答：" in s]


def strip_citations(text: str) -> str:
    """剥离引用标记：字数与结构判定都按剥离后的正文算。"""
    return _CITE.sub("", text or "")


def _lines(text: str) -> list[str]:
    return [x.strip() for x in (text or "").split("\n") if x.strip()]


def _longest_common(a: str, b: str) -> int:
    """最长公共子串长度（样例量级很小，直接 DP）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, bc in enumerate(b, 1):
            if ch == bc:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def sample_overlap(answer: str) -> int:
    """与四条样例相比，最长连续相同字数（忽略空白与引用标记）。"""
    body = re.sub(r"\s+", "", strip_citations(answer))
    return max((_longest_common(body, s) for s in _SAMPLE_BODIES), default=0)


def analyze(answer: str, followups: list[str] | None = None, route: str = "") -> dict:
    raw = (answer or "").strip()
    text = strip_citations(raw).strip()
    lines = _lines(text)
    bullets = [x for x in lines if _BULLET.match(x)]
    prose = [x for x in lines if not _BULLET.match(x)]
    chars = len(text)

    head = prose[0] if prose else ""
    opening = bool(head) and len(head) >= 12 and not _BULLET.match(head) and bool(_SENT_END.search(head))
    points = len(bullets)
    # 收束：最后一段非分点文字，且与总起不是同一段
    tail = prose[-1] if prose else ""
    closing = bool(tail) and tail != head and len(tail) >= 8 and bool(_SENT_END.search(tail))

    warnings: list[str] = []
    problems: list[str] = []
    if MAX_CHARS < chars <= WARN_CHARS:
        warnings.append(f"字数偏高({chars})")
    elif chars > WARN_CHARS:
        problems.append(f"超字数({chars}>{WARN_CHARS})")
    if points < 2:
        if chars <= SHORT_ANSWER:
            warnings.append("短答未分点(允许)")
        else:
            problems.append(f"分点不足({points})")
    if points > MAX_POINTS:
        problems.append(f"分点过多({points}>{MAX_POINTS})")
    if not opening:
        problems.append("缺总起")
    if not closing:
        problems.append("缺收束")

    reused = sample_overlap(raw)
    if reused >= SAMPLE_OVERLAP:
        warnings.append(f"疑似复用样例({reused}字)")

    fu = [str(x) for x in (followups or []) if str(x).strip()]
    fu_ok = None if route == "exam" else (             # route=exam 走题库模板，记 N/A 不进分母
        2 <= len(fu) <= 4
        and all(len(x) <= 25 for x in fu)
        and all(x.rstrip().endswith(("？", "?")) for x in fu)
    )
    if fu_ok is False:
        problems.append("追问不合格")

    return {"chars": chars, "points": points, "opening": opening, "closing": closing,
            "followups": len(fu), "followup_ok": fu_ok, "reused": reused,
            "problems": problems, "warnings": warnings, "ok": not problems}


def _rows(doc) -> list[dict]:
    """兼容两种形状：{rows:[...]} 与裸数组。"""
    if isinstance(doc, dict):
        return doc.get("rows") or []
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查回答是否符合总—分—总格式")
    ap.add_argument("files", nargs="+", help="eval result json 路径")
    ap.add_argument("--min-points", type=int, default=2, help="最少分点数（默认 2）")
    args = ap.parse_args(argv)

    total = passed = 0
    all_chars: list[int] = []
    fu_checked = fu_passed = 0
    reused_rows: list[str] = []
    for path in args.files:
        p = Path(path)
        if not p.is_file():
            print(f"[skip] {p} 不存在")
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        rows = _rows(doc)
        print(f"\n=== {p.name}  ({len(rows)} 条)")
        print(f"{'id':6} {'字数':>5} {'分点':>4} {'总起':>4} {'收束':>4} {'追问':>4}  结论")
        for r in rows:
            answer = r.get("answer") or ""
            if not answer.strip():
                continue
            fu = r.get("followups") or []
            route = r.get("route") or ""
            m = analyze(answer, fu, route)
            total += 1
            passed += 1 if m["ok"] else 0
            all_chars.append(m["chars"])
            if m["followup_ok"] is not None:
                fu_checked += 1
                fu_passed += 1 if m["followup_ok"] else 0
            if m["reused"] >= SAMPLE_OVERLAP:
                reused_rows.append(f"{r.get('id') or r.get('question', '')}（{m['reused']} 字）")
            verdict = "PASS" if m["ok"] else "FAIL: " + "、".join(m["problems"])
            if m["warnings"]:
                verdict += "  (~" + "、".join(m["warnings"]) + ")"
            fu_cell = "N/A" if m["followup_ok"] is None else str(m["followups"])
            print(f"{str(r.get('id') or r.get('question', ''))[:6]:6} {m['chars']:>5} {m['points']:>4} "
                  f"{'Y' if m['opening'] else 'N':>4} {'Y' if m['closing'] else 'N':>4} {fu_cell:>4}  {verdict}")

    if total:
        print(f"\n通过率：{passed}/{total} = {passed / total * 100:.1f}%")
        print(f"字数分布：min {min(all_chars)} / median {statistics.median(all_chars):.0f} / max {max(all_chars)}")
        print(f"追问合格率：{fu_passed}/{fu_checked} = "
              f"{(fu_passed / fu_checked * 100) if fu_checked else 0:.1f}%"
              + ("（route=exam 记 N/A，已剔除出分母）" if fu_checked < total else ""))
        print("样例复用标记：" + ("、".join(reused_rows) if reused_rows else "无"))
    else:
        print("\n没有可检查的回答（rows 为空或所有 answer 为空）")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
