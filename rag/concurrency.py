"""Bound active HTTP work, including the entire lifetime of SSE responses.

超限不再立即 503：先在队列里等待（默认 30 秒），等到名额就放行；等待超时才回 503 并带 Retry-After。
"""
from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse

DEFAULT_QUEUE_TIMEOUT = 30.0
DEFAULT_THREADPOOL = 140


class RequestConcurrencyMiddleware:
    # /auto/ask 与 /auto/ask/stream：统一入口，一题最多 4 次付费 LLM 调用（路由 + 课程三步），同样计入并发上限；
    # 流式的名额持有到响应完整结束（见 __call__ 的 finally）
    _PATHS = {"/ask", "/ask/stream", "/v1/chat/completions", "/auto/ask", "/auto/ask/stream"}

    def __init__(self, app, max_requests: int, queue_timeout: float | None = None):
        if max_requests < 1:
            raise ValueError("RAG_MAX_CONCURRENT_REQUESTS must be at least 1")
        self.app = app
        self._slots = asyncio.Semaphore(max_requests)
        self._max = max_requests
        self._queue_timeout = (float(os.environ.get("RAG_QUEUE_TIMEOUT_S", DEFAULT_QUEUE_TIMEOUT))
                               if queue_timeout is None else queue_timeout)

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") not in self._PATHS):
            await self.app(scope, receive, send)
            return
        # 满员时排队等待而不是立刻拒绝：客户端看到的是更长的等待，而不是可重试的 503。
        # 等待不占工作线程（asyncio.Semaphore 的等待者是协程）。
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=self._queue_timeout)
        except asyncio.TimeoutError:
            response = JSONResponse(
                {"error": {"message": "Server is busy; retry shortly.",
                           "type": "service_unavailable", "code": "server_busy"}},
                status_code=503, headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            # ASGI returns after streaming finishes or a disconnect/error.
            self._slots.release()


def raise_threadpool_limit(limit: int | None = None) -> int:
    """把 anyio 默认线程池上限提到 limit（默认取 RAG_THREADPOOL_LIMIT，否则 140）。

    同步 handler 每请求占一个工作线程，流式 handler 更是全程占用；上限必须 ≥ 并发信号量，
    否则拿到名额的请求仍会排在默认的 40 个线程后面。必须在事件循环里调用。
    """
    from anyio import to_thread

    target = limit if limit is not None else int(os.environ.get("RAG_THREADPOOL_LIMIT", DEFAULT_THREADPOOL))
    limiter = to_thread.current_default_thread_limiter()
    limiter.total_tokens = max(int(target), 1)
    return limiter.total_tokens


__all__ = ["RequestConcurrencyMiddleware", "raise_threadpool_limit"]
