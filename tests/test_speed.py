"""Compare sequential and concurrent real HTTP requests on the same 20 questions.

Run: python tests/test_speed.py
Overrides: SPEED_N_PROMPTS=20 SPEED_CONCURRENCY=10 SPEED_SEED=42
Reports: speed_test_report.txt (full replies), speed_test_report.json (aggregates).
Loads one temporary loopback server; excludes startup and retrieval warm-up from
both phases. Sequential runs first, so upstream caching/load can affect comparison.
This calls the real LLM: 20 questions x 2 phases = 40 requests by default.
"""
from __future__ import annotations

import json
import math
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median

_SYS_PATH_ROOT = Path(__file__).resolve().parents[1]
if str(_SYS_PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYS_PATH_ROOT))

import pytest

from rag.benchmark import build_items, load_raw
from rag.config import BENCHMARK_SEED, LLM_TIMEOUT, MAX_CONCURRENT_REQUESTS, ROOT

SPEED_N_PROMPTS = int(os.environ.get('SPEED_N_PROMPTS', '20'))
SPEED_CONCURRENCY = int(os.environ.get('SPEED_CONCURRENCY', '10'))
SPEED_SEED = int(os.environ.get('SPEED_SEED', str(BENCHMARK_SEED)))
SPEED_REPORT_PATH = Path(os.environ.get('SPEED_REPORT_PATH', ROOT / 'speed_test_report.txt'))


def _pick_prompts(n: int, seed: int) -> list:
    items = build_items(load_raw())
    return random.Random(seed).sample(items, min(n, len(items)))


@contextmanager
def _benchmark_server(items):
    import uvicorn
    from rag.serve import build_app

    started = time.perf_counter()
    app = build_app()
    init_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    app.state.service.retrieve_result(items[0].question)
    warm_seconds = time.perf_counter() - warm_started
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    try:
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError('Temporary HTTP server did not start')
            time.sleep(0.05)
        print(f'SERVER_READY init={init_seconds:.3f}s warmup={warm_seconds:.3f}s limit={MAX_CONCURRENT_REQUESTS}', flush=True)
        yield f'http://127.0.0.1:{port}', init_seconds, warm_seconds
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError('Temporary HTTP server did not stop')
        print('SERVER_STOPPED', flush=True)


def _http_request(base_url: str, index: int, item) -> dict:
    row = {'index': index, 'question': item.question, 'answer': '', 'error': None,
           'status': None, 'total': None}
    started = time.perf_counter()
    row['started_at'] = started
    try:
        body = json.dumps({'question': item.question}, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(base_url + '/ask', data=body,
                                         headers={'Content-Type': 'application/json'}, method='POST')
        # This is loopback traffic; do not route it through environment proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=max(180, LLM_TIMEOUT * 3)) as response:
            row['status'] = response.status
            result = json.loads(response.read())
        answer = result.get('answer')
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Response has no nonempty answer')
        row['answer'] = answer
    except urllib.error.HTTPError as error:
        row['status'] = error.code
        row['error'] = f'HTTPError: {error.code} {error.reason}'
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    finally:
        row['ended_at'] = time.perf_counter()
        row['total'] = row['ended_at'] - started
    return row


def _summarize(rows: list[dict], wall_seconds: float, workers: int) -> dict:
    successful = [row for row in rows if row['error'] is None]
    latencies = sorted(row['total'] for row in successful)
    events = [(row['started_at'], 1) for row in rows] + [(row['ended_at'], -1) for row in rows]
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return {
        'workers': workers, 'requests': len(rows), 'successful': len(successful),
        'errors': len(rows) - len(successful), 'wall_seconds': wall_seconds,
        'requests_per_second': len(successful) / wall_seconds if wall_seconds > 0 else 0,
        'mean_latency_seconds': mean(latencies) if latencies else None,
        'median_latency_seconds': median(latencies) if latencies else None,
        'p95_latency_seconds': latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        'min_latency_seconds': min(latencies) if latencies else None,
        'max_latency_seconds': max(latencies) if latencies else None,
        'peak_client_in_flight': peak,
        'answer_characters': sum(len(row['answer']) for row in successful),
    }


def _run_phase(base_url: str, items: list, workers: int, label: str, report) -> dict:
    print(f'PHASE_START {label} workers={workers} questions={len(items)}', flush=True)
    report.write(f'\n=== {label}: workers={workers} ===\n')
    report.flush()
    rows = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_http_request, base_url, i, item) for i, item in enumerate(items, 1)]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            rendered = (f"[{label}] #{row['index']} HTTP={row['status']} total={row['total']:.3f}s\n"
                        f"Question (full):\n{row['question']}\nAnswer (full):\n{row['answer']}\n")
            if row['error']:
                rendered += f"[ERROR] {row['error']}\n"
            report.write(rendered + '\n')
            report.flush()
            print(rendered, flush=True)
            print(f'PROGRESS {label} {len(rows)}/{len(items)}', flush=True)
    # End at the final HTTP completion, excluding report printing after it.
    wall_seconds = max(row['ended_at'] for row in rows) - started
    result = _summarize(rows, wall_seconds, workers)
    report.write('PHASE_SUMMARY ' + json.dumps({'phase': label, **result}) + '\n')
    report.flush()
    print('PHASE_DONE ' + json.dumps({'phase': label, **result}), flush=True)
    return result


def _comparison(sequential: dict, concurrent: dict) -> dict:
    comparable = (sequential['errors'] == concurrent['errors'] == 0
                  and sequential['requests'] == concurrent['requests']
                  and sequential['requests'] > 0)
    speedup = sequential['wall_seconds'] / concurrent['wall_seconds'] if comparable else None
    return {
        'comparable': comparable, 'batch_speedup': speedup,
        'batch_time_reduction_percent': (1 - concurrent['wall_seconds'] / sequential['wall_seconds']) * 100 if comparable else None,
        'throughput_increase_percent': (concurrent['requests_per_second'] / sequential['requests_per_second'] - 1) * 100 if comparable else None,
    }


def run_speed_test(n: int = SPEED_N_PROMPTS, seed: int = SPEED_SEED,
                   report_path: Path = SPEED_REPORT_PATH,
                   concurrency: int = SPEED_CONCURRENCY) -> dict:
    if n < 1 or concurrency < 1:
        raise ValueError('Question count and concurrency must be positive')
    if concurrency > MAX_CONCURRENT_REQUESTS:
        raise ValueError('SPEED_CONCURRENCY exceeds RAG_MAX_CONCURRENT_REQUESTS')
    items = _pick_prompts(n, seed)
    if not items:
        raise ValueError('No benchmark questions available')
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime('%Y-%m-%d %H:%M:%S%z')
    with _benchmark_server(items) as (base_url, init, warmup):
        with report_path.open('w', encoding='utf-8') as report:
            report.write(f'FUTURX sequential vs concurrent HTTP benchmark\nStarted: {started_at}\n'
                         f'Questions per phase: {len(items)} | seed: {seed} | concurrent workers: {concurrency}\n'
                         f'Server initialization: {init:.3f}s | retrieval warm-up: {warmup:.3f}s\n'
                         'Startup/warm-up excluded. Same questions in both phases; sequential first.\n'
                         'Latency starts when a worker sends a request; excludes time queued in the client pool.\n'
                         'No client retries. Full responses below are in completion order; # preserves question identity.\n')
            report.flush()
            sequential = _run_phase(base_url, items, 1, 'sequential', report)
            concurrent = _run_phase(base_url, items, concurrency, 'concurrent', report)
            result = {'n': len(items), 'seed': seed, 'server_init_time': init,
                      'retrieval_warmup_seconds': warmup, 'started_at': started_at,
                      'sequential': sequential, 'concurrent': concurrent,
                      **_comparison(sequential, concurrent)}
    # Keep a prompt-free machine-readable summary for monitoring/comparison.
    report_path.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    details = report_path.read_text(encoding='utf-8')
    temporary = report_path.with_name(report_path.name + '.tmp')
    temporary.write_text('AGGREGATE SUMMARY\n' + json.dumps(result, indent=2)
                         + '\n\n' + details, encoding='utf-8')
    temporary.replace(report_path)
    print('COMPARISON ' + json.dumps(result), flush=True)
    return result


def main() -> None:
    result = run_speed_test()
    print(f'Report: {SPEED_REPORT_PATH}', flush=True)
    if not result['comparable']:
        raise SystemExit(1)


@pytest.mark.skipif(os.environ.get('RUN_SPEED_TEST') != '1',
                    reason='Loads real models and calls the real LLM; requires RUN_SPEED_TEST=1')
def test_speed():
    result = run_speed_test()
    assert result['n'] > 0
    assert result['comparable']


if __name__ == '__main__':
    main()
