"""Dynamic micro-batching scheduler for a blocking inference function.

Collects concurrent ``submit(payload)`` calls into batches (up to
``max_batch_size``, or until ``max_wait_ms`` elapses after the first item),
runs the blocking ``infer_fn`` for the whole batch in a single executor thread
so the event loop stays responsive, then resolves each caller's future with its
slice of the result list.

Exactly one batch runs at a time: ``infer_fn`` is invoked only from the single
scheduler loop and awaited to completion before the next batch is collected, so
a non-thread-safe engine (e.g. vLLM) is never called concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class _Job:
    __slots__ = ("payload", "future")

    def __init__(self, payload: Any, future: "asyncio.Future[Any]") -> None:
        self.payload = payload
        self.future = future


class MicroBatchScheduler:
    """Start-once/stop-once micro-batching scheduler for a blocking inference function."""

    def __init__(
        self,
        infer_fn: Callable[[list[Any]], list[Any]],
        max_batch_size: int = 8,
        max_wait_ms: float = 8.0,
        fast_flush_ms: float = 2.0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if fast_flush_ms < 0:
            raise ValueError("fast_flush_ms must be >= 0")
        self._infer_fn = infer_fn
        self._max_batch = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._fast_flush = min(fast_flush_ms, max_wait_ms) / 1000.0
        self._queue: "asyncio.Queue[_Job]" = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._last_batch_size = 0

    async def start(self) -> None:
        if self._stopping:
            raise RuntimeError("MicroBatchScheduler cannot be restarted after stop()")
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        # Drain anything still queued so callers don't hang on `await submit()`.
        # A batch that was mid-inference when the cancel landed has already been
        # failed by the loop's CancelledError handler.
        cancelled = asyncio.CancelledError("scheduler stopped")
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(cancelled)

    async def submit(self, payload: Any) -> Any:
        """Enqueue ``payload`` and await its batched result."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_Job(payload, fut))
        return await fut

    async def _collect_batch(self) -> list[_Job]:
        first = await self._queue.get()
        batch = [first]
        loop = asyncio.get_running_loop()
        start = loop.time()
        full_deadline = start + self._max_wait
        # A lone submission is flushed after fast_flush_ms rather than max_wait_ms,
        # unless the previous batch had more than one item or more work is already
        # queued (pressure), in which case the full window is used to fill the batch.
        pressure = self._last_batch_size > 1 or self._queue.qsize() > 0
        deadline = full_deadline if pressure else start + self._fast_flush
        while len(batch) < self._max_batch:
            timeout = deadline - loop.time()
            if timeout <= 0:
                if len(batch) > 1 and deadline < full_deadline:
                    deadline = full_deadline
                    continue
                break
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if len(batch) > 1 and deadline < full_deadline:
                    deadline = full_deadline
                    continue
                break
            batch.append(job)
            if deadline < full_deadline:
                deadline = full_deadline
        self._last_batch_size = len(batch)
        return batch

    async def _run_loop(self) -> None:
        """The ONLY caller of ``infer_fn``: collect -> infer (executor) -> scatter."""
        loop = asyncio.get_running_loop()
        while not self._stopping:
            batch = await self._collect_batch()
            payloads = [j.payload for j in batch]
            try:
                results = await loop.run_in_executor(None, self._infer_fn, payloads)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"infer_fn returned {len(results)} results for {len(batch)} jobs"
                    )
                for job, res in zip(batch, results):
                    if not job.future.done():
                        job.future.set_result(res)
            except asyncio.CancelledError:
                # Shutdown landed mid-batch: resolve in-flight callers so they
                # don't hang on `await submit()`, then propagate the cancel.
                self._fail_batch(batch, asyncio.CancelledError("scheduler stopped"))
                raise
            except Exception as e:  # whole-batch failure -> fail every job in it
                logger.exception("batch inference failed: %s", e)
                self._fail_batch(batch, e)

    @staticmethod
    def _fail_batch(batch: list[_Job], exc: BaseException) -> None:
        for job in batch:
            if not job.future.done():
                job.future.set_exception(exc)
