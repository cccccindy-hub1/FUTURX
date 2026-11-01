"""Offline regression tests for the HTTP speed comparison runner."""
from contextlib import contextmanager
from io import StringIO
from threading import Barrier, Lock
from types import SimpleNamespace
import json
import time

import pytest

from tests import test_speed as speed


def test_same_questions_run_sequentially_then_with_ten_workers(monkeypatch, tmp_path, capsys):
    items = [SimpleNamespace(question=f'question-{i}') for i in range(20)]
    monkeypatch.setattr(speed, '_pick_prompts', lambda n, seed: items)
    monkeypatch.setattr(speed, 'MAX_CONCURRENT_REQUESTS', 10)
    barrier, lock = Barrier(10, timeout=5), Lock()
    calls = []

    @contextmanager
    def fake_server(selected):
        assert selected is items
        yield 'http://synthetic', 3.0, 1.0
    monkeypatch.setattr(speed, '_benchmark_server', fake_server)

    def request(base_url, index, item):
        started = time.perf_counter()
        with lock:
            calls.append(item.question)
            concurrent_phase = len(calls) > 20
        if concurrent_phase:
            barrier.wait()
        time.sleep(0.005)
        ended = time.perf_counter()
        return {'index': index, 'question': item.question, 'answer': 'full reply\n' + item.question,
                'error': None, 'status': 200, 'started_at': started, 'ended_at': ended, 'total': ended-started}
    monkeypatch.setattr(speed, '_http_request', request)
    path = tmp_path / 'report.txt'
    result = speed.run_speed_test(n=20, concurrency=10, report_path=path)
    assert calls[:20] == [item.question for item in items]
    assert sorted(calls[20:]) == sorted(calls[:20])
    assert result['sequential']['peak_client_in_flight'] == 1
    assert result['concurrent']['peak_client_in_flight'] == 10
    assert result['sequential']['successful'] == result['concurrent']['successful'] == 20
    assert result['comparable']
    report = path.read_text()
    assert report.count('Answer (full):') == 40
    assert report.count('full reply\nquestion-19') == 2
    summary = json.loads(path.with_suffix('.json').read_text())
    assert summary['n'] == 20 and summary['concurrent']['workers'] == 10
    assert 'question-19' not in path.with_suffix('.json').read_text()


def test_speedup_calculation_and_failures_are_not_hidden():
    sequential = {'requests': 20, 'errors': 0, 'wall_seconds': 200, 'requests_per_second': 0.1}
    concurrent = {'requests': 20, 'errors': 0, 'wall_seconds': 40, 'requests_per_second': 0.5}
    result = speed._comparison(sequential, concurrent)
    assert result['batch_speedup'] == 5
    assert result['batch_time_reduction_percent'] == 80
    assert result['throughput_increase_percent'] == 400
    result = speed._comparison(sequential, {**concurrent, 'errors': 1})
    assert not result['comparable'] and result['batch_speedup'] is None


def test_summary_counts_only_successful_requests_in_throughput():
    rows = [{'started_at': 0, 'ended_at': 2, 'total': 2, 'error': None, 'answer': 'ok'},
            {'started_at': 0.5, 'ended_at': 1.5, 'total': 1, 'error': 'HTTP 503', 'answer': ''}]
    result = speed._summarize(rows, 2, 10)
    assert result['successful'] == result['errors'] == 1
    assert result['requests_per_second'] == 0.5
    assert result['mean_latency_seconds'] == 2
    assert result['peak_client_in_flight'] == 2


@pytest.mark.parametrize('n,concurrency', [(0, 10), (20, 0), (20, 11)])
def test_invalid_benchmark_configuration_rejected(monkeypatch, n, concurrency):
    monkeypatch.setattr(speed, 'MAX_CONCURRENT_REQUESTS', 10)
    with pytest.raises(ValueError):
        speed.run_speed_test(n=n, concurrency=concurrency)
