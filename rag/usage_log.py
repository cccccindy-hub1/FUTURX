"""调用日志（需求5）：每个请求一行 JSONL，按天落盘，供统计与延迟分析。

- 目录：RAG_LOG_DIR，默认 <项目根>/logs/usage；文件名 YYYYMMDD.jsonl（本地时间）；
- 记录：request_id、起止时间与耗时、key_id（API key 的 sha256 前 8 位，不落原 key）、
  问题、route/routes/complexity、status、回答全文、追问、来源、分阶段耗时、
  llm_calls（step/model/tokens/latency）、流式首字耗时、错误类型、来源（http/eval）；
- 线程安全：模块内一把锁；单条记录一次写入，不会与其他线程交错；
- 异常隔离：所有对外函数整体 try/except，任何失败只记 logging 警告，绝不影响主请求；
- 保留：RAG_LOG_RETENTION_DAYS（默认 90）天，写盘时每天检查清理一次。

LLM 调用明细由 rag.course.llm.OpenAICompatClient 主动上报（record_llm），
当前请求靠 contextvar 传递；后台线程用 bind() 带上上下文。
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "logs" / "usage"
RETENTION_DEFAULT = 90
STAGE_KEYS = ("router", "select", "filter", "answer", "followup", "prematch")
_TEST_ENV = "PYTEST_CURRENT_TEST"


def _default_dir() -> Path:
    """测试进程（pytest）默认写临时目录：测试用的是假答案，混进真实日志会污染统计。"""
    if os.environ.get(_TEST_ENV):
        import tempfile

        return Path(tempfile.gettempdir()) / "futurx_usage_test"
    return DEFAULT_DIR

_current: contextvars.ContextVar["RequestLog | None"] = contextvars.ContextVar("rag_usage_log", default=None)
_write_lock = threading.Lock()
_last_cleanup_day: str = ""


def log_dir() -> Path:
    """日志目录；每次调用都读环境变量，便于测试与运行期改路径。"""
    return Path(os.environ.get("RAG_LOG_DIR") or _default_dir())


def retention_days() -> int:
    try:
        return max(0, int(os.environ.get("RAG_LOG_RETENTION_DAYS", RETENTION_DEFAULT)))
    except ValueError:
        return RETENTION_DEFAULT


def key_id(api_key: str | None) -> str | None:
    """API key 的 sha256 前 8 位（不可逆，不落原值）；空 key 返回 None。"""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def key_id_from_header(authorization: str | None) -> str | None:
    """从 Authorization: Bearer <key> 取 key（只用于算指纹）。"""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return key_id(token.strip())


def _ms(seconds) -> float | None:
    try:
        return round(float(seconds) * 1000, 1)
    except (TypeError, ValueError):
        return None


class RequestLog:
    """一个请求的累积器。finish() 只写一次；之后的 record_llm 之类调用被忽略。"""

    def __init__(self, question: str, origin: str = "eval", key_id_value: str | None = None):
        self.request_id = uuid.uuid4().hex
        self.ts_start = time.time()
        self._t0 = time.perf_counter()
        self.question = (question or "")[:2000]
        self.origin = origin
        self.key_id = key_id_value
        self.llm_calls: list[dict] = []
        self.stages: dict[str, float | None] = {}
        self.time_to_first_delta_ms: float | None = None
        self.pending_out: dict = {}                     # 流式：done 时的最终结果，供生成器收尾时落盘
        self.route_name: str | None = None              # 流式：route 事件先到、done 可能因断开而不来
        self.routes: list[str] = []
        self.complexity: str | None = None
        self.locks: dict[str, dict] = {}                # 检索锁：{name: {lock_wait_ms, retrieval_ms, locks}}
        self._first_delta_at: float | None = None
        self._done = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 采集
    def record_llm(self, step: str, model: str, prompt_tokens: int = 0,
                   completion_tokens: int = 0, latency_s: float = 0.0) -> None:
        item = {"step": step, "model": model, "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0), "latency_ms": _ms(latency_s) or 0.0}
        with self._lock:
            if not self._done:
                self.llm_calls.append(item)

    def stage(self, name: str, seconds: float) -> None:
        with self._lock:
            if not self._done:
                self.stages[name] = _ms(seconds)

    def note_lock(self, name: str, wait_ms: float, hold_ms: float) -> None:
        """累加一次锁等待/持有时长（同一请求内多次检索会累加，locks 计次数）。"""
        with self._lock:
            if self._done:
                return
            cur = self.locks.setdefault(name, {"lock_wait_ms": 0.0, "retrieval_ms": 0.0, "locks": 0})
            cur["lock_wait_ms"] = round(cur["lock_wait_ms"] + wait_ms, 1)
            cur["retrieval_ms"] = round(cur["retrieval_ms"] + hold_ms, 1)
            cur["locks"] += 1

    def first_delta(self) -> None:
        """流式首段正文到达（重复调用只记第一次）。"""
        with self._lock:
            if not self._done and self._first_delta_at is None:
                self._first_delta_at = time.perf_counter()
                self.time_to_first_delta_ms = _ms(self._first_delta_at - self._t0)

    # ------------------------------------------------------------ 落盘
    def finish(self, out: dict | None = None, error: str | None = None) -> dict | None:
        with self._lock:
            if self._done:
                return None
            self._done = True
            record = self._build(out or {}, error)
        try:
            _append(record)
        except Exception as e:  # noqa: BLE001 - 日志失败绝不影响主请求
            log.warning("usage_log 写入失败（%s），本条丢弃", type(e).__name__)
        return record

    def _build(self, out: dict, error: str | None) -> dict:
        trace = out.get("trace") or {}
        routes = out.get("routes") or self.routes
        if not routes:
            fallback_route = out.get("route") or self.route_name
            routes = [fallback_route] if fallback_route else []
        stages = {k: None for k in STAGE_KEYS}
        stages.update(self._trace_stages(trace))
        stages.update({k: v for k, v in self.stages.items() if v is not None})
        answer = out.get("answer") or ""
        sources = out.get("sources") or []
        return {
            "request_id": self.request_id,
            "ts_start": round(self.ts_start, 3),
            "ts_end": round(time.time(), 3),
            "duration_ms": _ms(time.perf_counter() - self._t0),
            "key_id": self.key_id,
            "question": self.question,
            "route": out.get("route") or self.route_name,
            "routes": routes,
            "complexity": out.get("complexity") or self.complexity,
            "status": out.get("status") or ("error" if error else None),
            "answer": answer,
            "answer_chars": len(answer),
            "followups": list(out.get("followups") or []),
            "sources": _brief_sources(sources),
            "stages": stages,
            "locks": {k: dict(v) for k, v in self.locks.items()},
            "llm_calls": list(self.llm_calls),
            "time_to_first_delta_ms": self.time_to_first_delta_ms,
            "error": error,
            "origin": self.origin,
        }

    @staticmethod
    def _trace_stages(trace: dict) -> dict:
        """从 trace 里取 router / prematch / select / filter / answer 的耗时（ms）。"""
        got: dict[str, float | None] = {}
        router = trace.get("router") or {}
        if router.get("latency_s") is not None:
            got["router"] = _ms(router["latency_s"])
        prematch = trace.get("prematch") or {}
        if prematch.get("ms") is not None:
            got["prematch"] = _ms(float(prematch["ms"]) / 1000)
        by_step = ((trace.get("course") or {}).get("totals") or {}).get("by_step") or {}
        for k in ("select", "filter", "answer"):
            if k in by_step:
                got[k] = _ms(by_step[k].get("latency_s", 0))
        return got


def _brief_sources(sources: list) -> list[dict]:
    """来源只留可检索的标识，避免把整段原文带进日志（体积与隐私）。"""
    out = []
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        out.append({k: s.get(k) for k in ("n", "kind", "chunk_id", "path", "title", "source", "section")
                    if s.get(k) is not None})
    return out


def _append(record: dict) -> None:
    path = log_dir()
    path.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    day = time.strftime("%Y%m%d")
    with _write_lock:
        with open(path / f"{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        _maybe_cleanup(path, day)


def _maybe_cleanup(path: Path, day: str) -> None:
    """每天最多检查一次；删除超过保留天数的 YYYYMMDD.jsonl。"""
    global _last_cleanup_day
    if _last_cleanup_day == day:
        return
    _last_cleanup_day = day
    keep = retention_days()
    if keep <= 0:
        return
    cutoff = time.time() - keep * 86400
    for p in path.glob("*.jsonl"):
        try:
            if p.stem.isdigit() and time.mktime(time.strptime(p.stem, "%Y%m%d")) < cutoff:
                p.unlink()
        except Exception:  # noqa: BLE001 - 清理失败不影响写入
            continue


# ---------------------------------------------------------------- 对外入口

class _Recorder:
    def __init__(self, rec: RequestLog, token):
        self.rec, self._token = rec, token

    def __getattr__(self, name):                      # 让 with 绑定物直接代理 record_llm/stage/first_delta/finish
        return getattr(self.rec, name)

    def __enter__(self) -> RequestLog:
        return self.rec

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._token.var.reset(self._token)
        except Exception:  # noqa: BLE001 - 生成器的 finally 可能落在别的线程
            pass
        if exc is not None and not issubclass(exc_type, GeneratorExit):
            self.rec.finish(error=exc_type.__name__)
        return False


def record(question: str, origin: str = "eval", key_id_value: str | None = None) -> _Recorder:
    """开启一条记录：with usage_log.record(q, "http", kid) as rec: ... rec.finish(out)。"""
    rec = RequestLog(question, origin, key_id_value)
    try:
        token = _current.set(rec)
    except Exception:  # noqa: BLE001
        token = None
    return _Recorder(rec, token)


def current() -> RequestLog | None:
    try:
        return _current.get()
    except Exception:  # noqa: BLE001
        return None


def record_llm(step: str, model: str, prompt_tokens: int = 0,
               completion_tokens: int = 0, latency_s: float = 0.0) -> None:
    """供 LLM 客户端上报一次调用；无当前记录时静默无操作。"""
    try:
        rec = current()
        if rec is not None:
            rec.record_llm(step, model, prompt_tokens, completion_tokens, latency_s)
    except Exception:  # noqa: BLE001
        pass


def stage(name: str, seconds: float) -> None:
    try:
        rec = current()
        if rec is not None:
            rec.stage(name, seconds)
    except Exception:  # noqa: BLE001
        pass


def note_lock(name: str, wait_ms: float, hold_ms: float) -> None:
    """检索锁埋点：排队等待与持锁时长分别累加（无当前记录时静默无操作）。"""
    try:
        rec = current()
        if rec is not None:
            rec.note_lock(name, wait_ms, hold_ms)
    except Exception:  # noqa: BLE001
        pass


def first_delta() -> None:
    try:
        rec = current()
        if rec is not None:
            rec.first_delta()
    except Exception:  # noqa: BLE001
        pass


def bind(fn):
    """把当前 context 带进后台线程：threading.Thread(target=usage_log.bind(fn))。"""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.run(fn, *a, **kw)


def write(record: dict) -> None:
    """直接写一条完整记录（serve/api 层补轻量记录时用）。异常只记日志。"""
    try:
        _append(record)
    except Exception as e:  # noqa: BLE001
        log.warning("usage_log 写入失败（%s），本条丢弃", type(e).__name__)


def simple_record(question: str, *, origin: str = "http", key_id_value: str | None = None,
                  route: str = "thesis", status: str = "ok", answer: str = "",
                  sources: list | None = None, started: float | None = None,
                  error: str | None = None, stages: dict | None = None,
                  request_id: str | None = None,
                  time_to_first_delta_ms: float | None = None) -> dict:
    """轻量记录：/ask 与 /v1/chat/completions 这类不经统一入口的路径。"""
    t0 = started if started is not None else time.time()
    rec = {
        "request_id": request_id or uuid.uuid4().hex, "ts_start": round(t0, 3), "ts_end": round(time.time(), 3),
        "duration_ms": _ms(time.time() - t0), "key_id": key_id_value, "question": (question or "")[:2000],
        "route": route, "routes": [route], "complexity": None, "status": status,
        "answer": answer or "", "answer_chars": len(answer or ""), "followups": [],
        "sources": _brief_sources(sources or []), "stages": {**{k: None for k in STAGE_KEYS}, **(stages or {})},
        "locks": {}, "llm_calls": [], "time_to_first_delta_ms": time_to_first_delta_ms, "error": error, "origin": origin,
    }
    write(rec)
    return rec


__all__ = ["RequestLog", "record", "current", "record_llm", "stage", "note_lock", "first_delta", "bind",
           "key_id", "key_id_from_header", "log_dir", "retention_days", "simple_record", "write"]
