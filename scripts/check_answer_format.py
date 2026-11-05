"""检查 eval result json 的回答格式：总起 / 分点 / 收束 / 字数 / 追问。

用法：
    python scripts/check_answer_format.py rag/eval_data/course_e2e_1102_flash_v2.json
    python scripts/check_answer_format.py <file> [<file> ...] [--min-points 2]

输出逐题表 + 通过率。判定口径与 rag/prompt_rules.ANSWER_STRUCTURE 一致：
- 总起：首个非空行是完整句子（≥12 字）且不以分点符号开头；
- 分点：2–4 条以 - / * / 1. / （1） / 一、 开头的行；
- 收束：最后一个非分点段落，且不是分点行本身；
- 字数：≤400（含标点）；
- 追问：2–4 条、每条 ≤25 字、以问号结尾。

压缩格式例外：≤150 字的答案不要求分点（降级为警告），但总起与收束仍需存在。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_CHARS = 400
SHORT_ANSWER = 150                      # ≤ 此长度，分点要求降级为警告
_BULLET = re.compile(r"^\s*(?:[-*+·]|\d{1,2}[.)、]|[（(]\d{1,2}[）)]|[一二三四五六七八九十]+[、.])")
_SENT_END = re.compile(r"[。！？；]")


def _lines(text: str) -> list[str]:
    return [x.strip() for x in (text or "").split("\n") if x.strip()]


def analyze(answer: str, followups: list[str] | None = None) -> dict:
    text = (answer or "").strip()
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

    warn: list[str] = []
    if points < 2:
        if chars <= SHORT_ANSWER:
            warn.append("短答未分点(允许)")
        elif not opening and not closing:
            warn.append("无分点且无总起收束")
    if points > 4:
        warn.append(f"分点过多({points})")

    problems: list[str] = []
    if chars > MAX_CHARS:
        problems.append(f"超字数({chars}>{MAX_CHARS})")
    if not opening:
        problems.append("缺总起")
    if not closing:
        problems.append("缺收束")
    if points < 2 and chars > SHORT_ANSWER:
        problems.append(f"分点不足({points})")

    fu = followups or []
    fu_len_ok = all(len(x) <= 25 for x in fu)
    fu_q_ok = all(x.rstrip().endswith(("？", "?")) for x in fu)
    if fu:
        if not (2 <= len(fu) <= 4):
            problems.append(f"追问条数({len(fu)})")
        if not fu_len_ok:
            problems.append("追问超长")
        if not fu_q_ok:
            problems.append("追问非疑问句")

    return {"chars": chars, "points": points, "opening": opening, "closing": closing,
            "followups": len(fu), "problems": problems, "warnings": warn,
            "ok": not problems}


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
            m = analyze(answer, fu)
            total += 1
            passed += 1 if m["ok"] else 0
            verdict = "PASS" if m["ok"] else "FAIL: " + "、".join(m["problems"])
            if m["warnings"]:
                verdict += "  (~" + "、".join(m["warnings"]) + ")"
            print(f"{str(r.get('id') or r.get('question', ''))[:6]:6} {m['chars']:>5} {m['points']:>4} "
                  f"{'Y' if m['opening'] else 'N':>4} {'Y' if m['closing'] else 'N':>4} {m['followups']:>4}  {verdict}")

    if total:
        print(f"\n通过率：{passed}/{total} = {passed / total * 100:.1f}%")
    else:
        print("\n没有可检查的回答（rows 为空或所有 answer 为空）")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
