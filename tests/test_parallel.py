"""Concurrent HTTP, stream lifetime, shared retrieval, and timing regression tests."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, RLock, local
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import rag.serve as serve
import rag.query as query
from rag.concurrency import RequestConcurrencyMiddleware


def _body(path, question):
    if path.startswith('/v1/'):
        return {'messages': [{'role': 'user', 'content': question}],
                'stream': path.endswith('/stream')}
    return {'question': question}


def _auth_headers():
    """从项目根 .env 读 API_KEYS 的第一个 key，运行时取值，不硬编码。

    conftest 的 autouse fixture 会把 rag.api.API_KEYS 清空，让本文件的无鉴权用例照常跑通；
    但 /ask 与 /ask/stream 已于 2026-11-05 补鉴权依赖，请求带真实 key 才与线上一致。
    读不到时返回空 dict，请求不带 Authorization（等价于改动前的行为）。
    """
    env = Path(__file__).resolve().parent.parent / '.env'
    try:
        for line in env.read_text(encoding='utf-8').splitlines():
            if line.startswith('API_KEYS='):
                keys = line.split('=', 1)[1].strip()
                key = keys.split(',')[0].strip() if keys else ''
                return {'Authorization': 'Bearer ' + key} if key else {}
    except OSError:
        pass
    return {}


@pytest.mark.parametrize('path,stream', [('/ask', False), ('/ask/stream', True),
                                        ('/v1/chat/completions', False),
                                        ('/v1/chat/completions', True)])
def test_http_answers_overlap_and_stay_separate(monkeypatch, path, stream):
    barrier = Barrier(2, timeout=5)

    class ParallelService:
        def answer(self, q, k):
            barrier.wait()
            return 'answer:' + q, [{'source': q}], [q]

        def open_stream(self, q, k):
            def deltas():
                barrier.wait()
                yield 'answer:' + q
            return [q], [{'source': q}], deltas()

    monkeypatch.setattr(serve, 'RAGService', lambda **kw: ParallelService())
    monkeypatch.setattr(serve, 'MAX_CONCURRENT_REQUESTS', 2)
    monkeypatch.setattr('rag.api.API_KEYS', [])
    with TestClient(serve.build_app()) as client, ThreadPoolExecutor(2) as pool:
        def send(q):
            body = _body(path, q)
            if path.startswith('/v1/'):
                body['stream'] = stream
            return client.post(path, json=body, headers=_auth_headers())
        futures = [pool.submit(send, q) for q in ['request_alpha', 'request_beta']]
        for q, future in zip(['request_alpha', 'request_beta'], futures):
            response = future.result(timeout=10)
            assert response.status_code == 200
            assert 'answer:' + q in response.text
            other = 'request_beta' if q == 'request_alpha' else 'request_alpha'
            assert other not in response.text


@pytest.mark.parametrize('stream', [False, True])
def test_limit_held_until_response_finishes_and_health_stays_available(monkeypatch, stream):
    started, release = Event(), Event()

    class GatedService:
        def wait(self):
            started.set()
            assert release.wait(5)

        def answer(self, q, k):
            self.wait()
            return q, [], []

        def open_stream(self, q, k):
            def deltas():
                self.wait()
                yield q
            return [], [], deltas()

    monkeypatch.setattr(serve, 'RAGService', lambda **kw: GatedService())
    monkeypatch.setattr(serve, 'MAX_CONCURRENT_REQUESTS', 1)
    with TestClient(serve.build_app()) as client, ThreadPoolExecutor(1) as pool:
        path = '/ask/stream' if stream else '/ask'
        headers = _auth_headers()
        first = pool.submit(client.post, path, json={'question': 'first'}, headers=headers)
        try:
            assert started.wait(5)
            assert client.get('/health').status_code == 200
            busy = client.post('/ask', json={'question': 'overload'}, headers=headers)
            assert busy.status_code == 503
            assert busy.headers['retry-after'] == '1'
            assert busy.json()['error']['code'] == 'server_busy'
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
        assert client.post('/ask', json={'question': 'after'}, headers=headers).status_code == 200


@pytest.mark.parametrize('failure', ['exception', 'cancel'])
def test_capacity_released_after_error_or_disconnect(failure):
    async def run():
        started = asyncio.Event()

        async def app(scope, receive, send):
            if scope.get('fail'):
                started.set()
                if failure == 'cancel':
                    await asyncio.Event().wait()
                raise RuntimeError('synthetic failure')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'ok'})

        limiter = RequestConcurrencyMiddleware(app, max_requests=1)
        async def receive():
            return {'type': 'http.disconnect'}
        messages = []
        async def send(msg):
            messages.append(msg)
        scope = {'type': 'http', 'method': 'POST', 'path': '/ask'}
        task = asyncio.create_task(limiter({**scope, 'fail': True}, receive, send))
        await started.wait()
        if failure == 'cancel':
            task.cancel()
        with pytest.raises(asyncio.CancelledError if failure == 'cancel' else RuntimeError):
            await task
        await limiter(scope, receive, send)
        assert messages[0]['status'] == 200
    asyncio.run(run())


def test_parallel_answers_keep_their_own_timings(monkeypatch):
    svc = object.__new__(query.RAGService)
    svc._request_state = local()
    ticks = local()
    barrier = Barrier(2, timeout=5)
    svc.prepare = lambda q, k: ([], [], [{'role': 'user', 'content': q}])

    class LLM:
        def create(self, **kwargs):
            barrier.wait()
            q = kwargs['messages'][0]['content']
            return iter([SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=q))])])
    llm = LLM()
    svc.llm = SimpleNamespace(chat=SimpleNamespace(completions=llm))
    monkeypatch.setattr(query.time, 'perf_counter', lambda: next(ticks.values))

    def answer(unit):
        ticks.values = iter([0, unit, 2*unit, 3*unit, 4*unit])
        ans, _, _ = svc.answer(str(unit))
        barrier.wait()  # Both calls must finish before either inspects timing.
        return ans, svc.timing.copy()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(answer, unit) for unit in [1, 10]]
        for unit, future in zip([1, 10], futures):
            ans, timing = future.result(timeout=10)
            assert ans == str(unit)
            assert timing['retrieval'] == unit
            assert timing['llm_ttft'] == unit
            assert timing['llm_generation'] == unit
            assert timing['llm_total'] == 2*unit


def test_retrieval_shared_state_is_guarded():
    import time
    svc = object.__new__(query.RAGService)
    svc._retrieval_lock = RLock()
    svc._retrieval_query = lambda q: q
    svc.fetch_k = 20
    svc.where = None
    svc.expand_context = False
    active = maximum = 0
    guard = Lock()

    def search(q, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(active, maximum)
        time.sleep(0.01)
        with guard:
            active -= 1
        return q
    svc.retriever = SimpleNamespace(search=search)
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(svc.retrieve_result, ['a', 'b', 'c', 'd'])) == ['a', 'b', 'c', 'd']
    assert maximum == 1


def test_invalid_concurrency_limit_rejected():
    with pytest.raises(ValueError, match='at least 1'):
        RequestConcurrencyMiddleware(None, max_requests=0)
