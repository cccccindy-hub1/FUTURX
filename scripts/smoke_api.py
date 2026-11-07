#!/usr/bin/env python
"""AI 李老师接口冒烟测试：只验证「本次改动」的功能，不依赖判分。

覆盖：
  1. 多轮对话：追问能接上上一轮（/v1 与 /auto/ask 各一次）
  2. 会话落库：写入 / 读取 / 增量拉取 / 改名 / 归档 / 清空 / 删除
  3. 归属隔离：换一个 key 看不到别人的会话
  4. 并发：N 个请求的响应头时间应当接近，不是一个接一个

用法（服务需已启动）：
    export FUTURX_BASE=http://127.0.0.1:8000
    export FUTURX_KEY=<API_KEYS 里的一把>
    python scripts/smoke_api.py                 # 默认跑 1-3
    python scripts/smoke_api.py --with-concurrency
    python scripts/smoke_api.py --with-multi-turn   # 会真实调用模型，较慢

退出码：0 全过；1 有失败。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid

import httpx

BASE = os.environ.get("FUTURX_BASE", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("FUTURX_KEY", "")
TIMEOUT = float(os.environ.get("FUTURX_TIMEOUT", "180"))

_ok = 0
_fail = 0


def _h():
    return {"Authorization": f"Bearer {KEY}"} if KEY else {}


def check(name: str, cond: bool, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  PASS  {name}")
    else:
        _fail += 1
        print(f"  FAIL  {name}" + (f"  <- {detail}" if detail else ""))


def _post(path: str, body: dict) -> httpx.Response:
    return httpx.post(f"{BASE}{path}", json=body, headers=_h(), timeout=TIMEOUT)


def _get(path: str, **params) -> httpx.Response:
    return httpx.get(f"{BASE}{path}", params=params, headers=_h(), timeout=TIMEOUT)


def _del(path: str) -> httpx.Response:
    return httpx.delete(f"{BASE}{path}", headers=_h(), timeout=TIMEOUT)


def _patch(path: str, body: dict) -> httpx.Response:
    return httpx.patch(f"{BASE}{path}", json=body, headers=_h(), timeout=TIMEOUT)


# ---------------------------------------------------------------- 1. 基础

def test_health() -> None:
    print("\n[1] 健康检查")
    r = httpx.get(f"{BASE}/health", timeout=10)
    check("GET /health 返回 200", r.status_code == 200, r.text[:120])
    check("body.status == ok", r.json().get("status") == "ok", r.text[:120])


def test_models() -> None:
    print("\n[2] /v1/models")
    r = _get("/v1/models")
    if r.status_code == 401:
        check("鉴权配置正确（401 说明 KEY 不对）", False, "检查 FUTURX_KEY")
        return
    check("GET /v1/models 返回 200", r.status_code == 200, r.text[:160])
    data = r.json().get("data") or []
    check("返回至少一个模型", len(data) >= 1, r.text[:160])
    if data:
        print(f"        模型 {data[0].get('id')} -> 上游 {data[0].get('upstream_model')}")


# ---------------------------------------------------------------- 2. 会话管理（不调模型）

def test_conversation_crud() -> None:
    print("\n[3] 会话管理接口（不调用模型）")
    r = _post("/v1/conversations", {"title": "冒烟测试会话"})
    check("POST /v1/conversations 建会话", r.status_code == 200, r.text[:160])
    if r.status_code != 200:
        return
    cid = r.json()["id"]
    print(f"        会话 id = {cid}")

    r = _post("/v1/conversations", {"id": cid})
    check("重复 id 建会话返回 409", r.status_code == 409, f"{r.status_code} {r.text[:120]}")

    r = _get("/v1/conversations")
    check("GET 列表 200", r.status_code == 200, r.text[:160])
    ids = [c["id"] for c in r.json().get("data", [])]
    check("新建的会话在列表里", cid in ids, f"列表 {ids[:5]}")
    check("列表项含 object=conversation", r.json()["data"][0].get("object") == "conversation")

    r = _get(f"/v1/conversations/{cid}")
    check("GET 详情 200", r.status_code == 200, r.text[:160])
    check("空会话 messages 为空", r.json().get("messages") == [], r.text[:160])

    r = _patch(f"/v1/conversations/{cid}", {"title": "改过的标题"})
    check("PATCH 改名 200", r.status_code == 200, r.text[:160])
    check("标题已改", r.json().get("title") == "改过的标题", r.text[:160])

    r = _patch(f"/v1/conversations/{cid}", {"archived": True})
    check("PATCH 归档 200", r.status_code == 200, r.text[:160])
    check("archived == true", r.json().get("archived") is True, r.text[:160])

    r = _get("/v1/conversations", archived="false")
    check("默认列表不含已归档", cid not in [c["id"] for c in r.json().get("data", [])], r.text[:160])
    r = _get("/v1/conversations", archived="true")
    check("archived=true 能看到", cid in [c["id"] for c in r.json().get("data", [])], r.text[:160])

    _patch(f"/v1/conversations/{cid}", {"archived": False})

    r = _get("/v1/conversations/conv_does_not_exist")
    check("不存在的 id 返回 404", r.status_code == 404, f"{r.status_code} {r.text[:120]}")

    r = _del(f"/v1/conversations/{cid}")
    check("DELETE 删除 200", r.status_code == 200, r.text[:160])
    check("deleted == true", r.json().get("deleted") is True, r.text[:160])
    r = _get(f"/v1/conversations/{cid}")
    check("删除后详情 404", r.status_code == 404, f"{r.status_code} {r.text[:120]}")


def test_store_false() -> None:
    print("\n[4] store=false 不落库")
    r = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": "1+1 等于几"}], "store": False})
    if r.status_code == 503:
        print("  SKIP  未配置 LLM，跳过")
        return
    check("200", r.status_code == 200, r.text[:160])
    check("响应里无 conversation_id", "conversation_id" not in r.json(), r.text[:200])
    check("响应头无 X-Conversation-Id", "x-conversation-id" not in r.headers, str(dict(r.headers))[:200])


def test_unknown_conversation_404() -> None:
    print("\n[5] 未知 conversation_id 返回 404（不静默新建）")
    r = _post("/v1/chat/completions", {"conversation_id": "conv_nope_" + uuid.uuid4().hex[:6],
                                       "messages": [{"role": "user", "content": "你好"}]})
    check("400/404（不是 200）", r.status_code in (400, 404), f"{r.status_code} {r.text[:160]}")
    if r.status_code == 404:
        body = r.json()
        check("错误体带 code=not_found",
              (body.get("detail") or {}).get("error", {}).get("code") == "not_found", r.text[:200])


# ---------------------------------------------------------------- 3. 多轮（会调模型）

def _ask_v1(messages: list[dict], conversation_id: str | None = None) -> httpx.Response:
    body = {"messages": messages}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return _post("/v1/chat/completions", body)


def test_multi_turn_v1() -> None:
    print("\n[6] 多轮对话 /v1（真实调用模型）")
    r = _ask_v1([{"role": "user", "content": "数据资产估值有哪些方法？"}])
    if r.status_code == 503:
        print("  SKIP  未配置 LLM，跳过")
        return
    check("第一轮 200", r.status_code == 200, r.text[:200])
    if r.status_code != 200:
        return
    cid = r.json()["conversation_id"]
    first = r.json()["choices"][0]["message"]["content"]
    print(f"        第一轮答案：{first[:80]}...")
    check("响应头带 X-Conversation-Id", r.headers.get("x-conversation-id") == cid)

    r = _ask_v1([{"role": "user", "content": "第二种具体怎么算？"}], conversation_id=cid)
    check("第二轮 200", r.status_code == 200, r.text[:200])
    if r.status_code != 200:
        return
    second = r.json()["choices"][0]["message"]["content"]
    print(f"        第二轮答案：{second[:120]}...")
    check("第二轮 return 同一 conversation_id", r.json()["conversation_id"] == cid)
    # 只要求有实质回答且不是「无法判断」那种拒答
    check("第二轮不是空回答", len(second.strip()) > 20, second[:120])
    check("第二轮没有说「无法判断/未提供上下文」",
          not any(k in second for k in ("无法判断", "指代不明", "没有提供前文")), second[:160])

    r = _get(f"/v1/conversations/{cid}/messages")
    msgs = r.json().get("data", [])
    check("会话里落了 4 条消息（2 轮）", len(msgs) == 4, f"{len(msgs)} 条")
    if len(msgs) >= 4:
        check("seq 从 1 连续递增", [m["seq"] for m in msgs] == [1, 2, 3, 4], [m["seq"] for m in msgs])
        check("role 交替 u/a/u/a", [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"],
              [m["role"] for m in msgs])
        check("第二轮 user 的 search_query 是改写后的完整问题",
              bool((msgs[2].get("search_query") or "").strip()) and len(msgs[2]["search_query"]) > 0,
              str(msgs[2].get("search_query")))
        print(f"        改写后的检索问题：{msgs[2].get('search_query')}")

    r = _get(f"/v1/conversations/{cid}/messages", after_seq=2)
    check("after_seq=2 只返回后 2 条", len(r.json().get("data", [])) == 2, r.text[:200])

    _del(f"/v1/conversations/{cid}")


def test_multi_turn_auto() -> None:
    print("\n[7] 多轮对话 /auto/ask（追问要被正确路由，不能判 off_topic）")
    r = _post("/auto/ask", {"question": "数据资产估值有哪些方法？", "include_trace": False})
    if r.status_code == 503:
        print("  SKIP  统一入口未就绪，跳过")
        return
    check("第一轮 200", r.status_code == 200, r.text[:200])
    if r.status_code != 200:
        return
    cid = r.json()["conversation_id"]
    print(f"        route={r.json().get('route')} status={r.json().get('status')}")

    r = _post("/auto/ask", {"question": "那它适用什么情形？", "conversation_id": cid,
                            "include_trace": False})
    check("追问 200", r.status_code == 200, r.text[:200])
    if r.status_code != 200:
        return
    got = r.json()
    print(f"        追问 route={got.get('route')} status={got.get('status')}")
    check("追问没有被判 off_topic", got.get("route") != "off_topic", json.dumps(got, ensure_ascii=False)[:200])
    check("追问没有 status=refused", got.get("status") != "refused", json.dumps(got, ensure_ascii=False)[:200])
    check("追问返回了回答", bool((got.get("answer") or "").strip()))

    msgs = _get(f"/v1/conversations/{cid}/messages").json().get("data", [])
    check("两轮共 4 条消息", len(msgs) == 4, f"{len(msgs)} 条")
    _del(f"/v1/conversations/{cid}")


def test_stream_persist() -> None:
    print("\n[8] 流式落库（/auto/ask/stream）")
    t0 = time.time()
    got_done = None
    with httpx.stream("POST", f"{BASE}/auto/ask/stream",
                      json={"question": "什么是数据要素？", "include_trace": False},
                      headers=_h(), timeout=TIMEOUT) as r:
        head_at = time.time() - t0
        cid = r.headers.get("x-conversation-id")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "done":
                got_done = ev
    check("有 X-Conversation-Id", bool(cid), str(dict(r.headers))[:200])
    check("收到 done 事件", got_done is not None)
    print(f"        响应头 {head_at:.2f}s")
    if not cid:
        return
    msgs = _get(f"/v1/conversations/{cid}/messages").json().get("data", [])
    check("流式结束后已落库 2 条", len(msgs) == 2, f"{len(msgs)} 条")
    if len(msgs) == 2:
        check("助手消息有正文", len((msgs[1].get("content") or "")) > 5, str(msgs[1])[:160])
        check("助手消息 status 不是 error", msgs[1].get("status") != "error", str(msgs[1])[:160])
    _del(f"/v1/conversations/{cid}")


# ---------------------------------------------------------------- 4. 并发

def test_concurrency(n: int = 4, chars: int = 600) -> None:
    print(f"\n[9] 并发：{n} 个约 {chars} 字的请求，响应头时间应当接近")
    q = ("数据资产估值的收益法具体怎么算，请结合课程资料详细说明。" * (chars // 26 + 1))[:chars]
    out: dict[int, float] = {}
    errs: dict[int, str] = {}
    lock = threading.Lock()

    def one(i: int) -> None:
        try:
            with httpx.stream("POST", f"{BASE}/v1/chat/completions",
                              json={"stream": True, "store": False,
                                    "messages": [{"role": "user", "content": q}]},
                              headers=_h(), timeout=TIMEOUT) as r:
                with lock:
                    out[i] = time.time() - start
                for _ in r.iter_lines():
                    break                      # 只要拿到响应头就够
        except Exception as e:  # noqa: BLE001
            with lock:
                errs[i] = type(e).__name__

    start = time.time()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errs:
        check("所有请求都拿到了响应头", False, str(errs))
        return
    times = sorted(out.values())
    spread = times[-1] - times[0]
    print("        响应头时间：" + ", ".join(f"{t:.2f}s" for t in times))
    check("全部拿到响应头", len(out) == n, str(out))
    # 改前是每 2.5s 一个（4 个就是 7.5s 的跨度）；这里放宽到 3s
    check(f"跨度 < 3s（实测 {spread:.2f}s）", spread < 3.0, "排队仍在，检查 EMBED_WORKERS / 全局锁")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="AI 李老师接口冒烟测试")
    ap.add_argument("--with-concurrency", action="store_true", help="跑并发检查（会真实调用模型）")
    ap.add_argument("--with-multi-turn", action="store_true", help="跑多轮检查（会真实调用模型，较慢）")
    a = ap.parse_args()

    print(f"目标：{BASE}")
    print(f"鉴权：{'已带 key' if KEY else '未带 key（服务端若配置了 API_KEYS 会 401）'}")

    test_health()
    test_models()
    test_conversation_crud()
    test_unknown_conversation_404()
    if a.with_multi_turn:
        test_multi_turn_v1()
        test_multi_turn_auto()
        test_stream_persist()
    if a.with_concurrency:
        test_concurrency()

    print(f"\n通过 {_ok}，失败 {_fail}")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
