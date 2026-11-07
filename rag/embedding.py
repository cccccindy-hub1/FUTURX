"""向量化的批处理调度器：把并发的单条 encode 合并成一批。

背景：RAGService 以前在整段检索外面套一把全局锁，每次请求单独调用
`model.encode(单条文本)`。BGE-m3 在 CPU 上单条要 90ms（20 字）到 2.7s（2600 字），
100 个并发请求就变成一条一条排队、最后一条要等 30 秒；而且逐条 encode 无法
利用批处理的吞吐优势（实测 100 条 20 字：逐条 9.6s，成批 2.2s）。

做法：一个后台线程收集待编码文本，按「条数 ≤ max_batch」且「总字数 ≤ max_chars」
组成一批，一次 `model.encode(list)`，再把结果分发给各自的等待方。
队列为空时立即编码（单条请求不多等）；有积压时最多等 max_wait_ms 让更多请求入批。

按字数封顶而不是按条数：2600 字的查询一条就占满预算，不会被塞进大 batch
（实测 4 条 2600 字塞一批要 11s，两条一批各 5s，后者首字更早）。
"""
from __future__ import annotations

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


class _Job:
    __slots__ = ("text", "done", "vector", "error")

    def __init__(self, text: str):
        self.text = text
        self.done = threading.Event()
        self.vector: list[float] | None = None
        self.error: BaseException | None = None


class BatchedEmbedder:
    """线程安全的单条编码入口；内部成批调用底层 SentenceTransformer。

    多个编码 worker（默认 2 个）各自取批：长问题只凑成一对就能开跑，短问题能一次吃满 32 条。
    注意 worker 数要和 torch 线程数相乘不超过核数，否则会像实测那样严重反向（8 线程 × 4 并发
    处理 4 条 2600 字要 148s）；见 tune_torch_threads。
    """

    def __init__(self, model, *, max_batch: int = 32, max_chars: int = 6000,
                 max_wait_ms: float = 15.0, workers: int = 1):
        self.model = model
        self.max_batch = max(1, int(max_batch))
        self.max_chars = max(1, int(max_chars))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._carry = [None] * max(1, int(workers))      # 每个 worker 各自的「留到下一批」任务
        self._batches = 0
        self._count_lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._run, args=(i,), name=f"embed-batcher-{i}", daemon=True)
            for i in range(max(1, int(workers)))
        ]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- 对外
    def encode(self, text: str) -> list[float]:
        """阻塞到这条文本编码完成；失败时抛出底层异常。"""
        job = _Job(text)
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.vector or []

    @property
    def batch_calls(self) -> int:
        return getattr(self, "_batches", 0)

    # ---------------------------------------------------------------- 内部
    def _take_batch(self, worker: int) -> list[_Job]:
        first = self._carry[worker] if self._carry[worker] is not None else self._queue.get()
        self._carry[worker] = None
        batch = [first]
        chars = len(first.text)
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                job = self._queue.get(timeout=timeout) if timeout else self._queue.get_nowait()
            except queue.Empty:
                break
            if chars + len(job.text) > self.max_chars:
                self._carry[worker] = job               # 超预算：留到下一批（不塞回队尾，保持先来先服务）
                break
            batch.append(job)
            chars += len(job.text)
        return batch

    def _run(self, worker: int = 0) -> None:
        while True:
            batch = self._take_batch(worker)
            try:
                vectors = self.model.encode([j.text for j in batch], normalize_embeddings=True)
                rows = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
                if len(batch) == 1 and isinstance(rows, list) and rows and not isinstance(rows[0], (list, tuple)):
                    rows = [rows]                      # 兼容只返回单条向量的替身模型
                for job, row in zip(batch, rows):
                    job.vector = list(row)
                    job.done.set()
                with self._count_lock:
                    self._batches += 1
            except BaseException as e:  # noqa: BLE001 - 逐条回传失败，绝不让调度线程退出
                logger.warning("批量向量化失败（%s），本批 %d 条按失败返回", type(e).__name__, len(batch))
                for job in batch:
                    job.error = e
                    job.done.set()
