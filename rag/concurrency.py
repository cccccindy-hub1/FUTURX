"""Bound active HTTP work, including the entire lifetime of SSE responses."""
from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse


class RequestConcurrencyMiddleware:
    _PATHS = {"/ask", "/ask/stream", "/v1/chat/completions"}

    def __init__(self, app, max_requests: int):
        if max_requests < 1:
            raise ValueError("RAG_MAX_CONCURRENT_REQUESTS must be at least 1")
        self.app = app
        self._slots = asyncio.Semaphore(max_requests)

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") not in self._PATHS):
            await self.app(scope, receive, send)
            return
        # No blocking worker-thread waits: reject overload while keeping health
        # checks and existing streams responsive. No await between the capacity
        # check and the immediately successful acquisition on this event loop.
        if self._slots.locked():
            response = JSONResponse(
                {"error": {"message": "Server is busy; retry shortly.",
                           "type": "service_unavailable", "code": "server_busy"}},
                status_code=503, headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return
        await self._slots.acquire()
        try:
            await self.app(scope, receive, send)
        finally:
            # ASGI returns after streaming finishes or a disconnect/error.
            self._slots.release()
