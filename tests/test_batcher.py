"""MicroBatchScheduler: collect submissions into batches, scatter results."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from lightnav.serving.batcher import MicroBatchScheduler


async def test_batches_concurrent_submissions():
    seen_batch_sizes: list[int] = []

    def infer(payloads: list[int]) -> list[int]:
        seen_batch_sizes.append(len(payloads))
        return [p * 10 for p in payloads]

    sched = MicroBatchScheduler(infer, max_batch_size=8, max_wait_ms=20)
    await sched.start()
    try:
        results = await asyncio.gather(*[sched.submit(i) for i in range(5)])
    finally:
        await sched.stop()

    assert results == [0, 10, 20, 30, 40]
    assert max(seen_batch_sizes) > 1  # at least one real batch formed


async def test_respects_max_batch_size():
    seen: list[int] = []

    def infer(payloads: list[int]) -> list[int]:
        seen.append(len(payloads))
        return list(payloads)

    sched = MicroBatchScheduler(infer, max_batch_size=3, max_wait_ms=50)
    await sched.start()
    try:
        await asyncio.gather(*[sched.submit(i) for i in range(7)])
    finally:
        await sched.stop()

    assert all(b <= 3 for b in seen)


async def test_single_submission_uses_fast_flush_instead_of_full_wait():
    def infer(payloads: list[int]) -> list[int]:
        return list(payloads)

    sched = MicroBatchScheduler(infer, max_batch_size=8, max_wait_ms=80)
    await sched.start()
    t0 = asyncio.get_running_loop().time()
    try:
        assert await sched.submit(1) == 1
    finally:
        await sched.stop()
    elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000.0

    assert elapsed_ms < 40.0


async def test_error_isolated_per_batch():
    def infer(payloads: list[int]) -> list[int]:
        raise RuntimeError("boom")

    sched = MicroBatchScheduler(infer, max_batch_size=4, max_wait_ms=10)
    await sched.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await sched.submit(1)
        # The loop survives a failed batch: the next submission still runs.
        sched._infer_fn = lambda payloads: list(payloads)
        assert await sched.submit(2) == 2
    finally:
        await sched.stop()


async def test_result_count_mismatch_fails_the_batch():
    def infer(payloads: list[int]) -> list[int]:
        return payloads[:-1]

    sched = MicroBatchScheduler(infer, max_batch_size=1, max_wait_ms=1)
    await sched.start()
    try:
        with pytest.raises(RuntimeError, match="results"):
            await sched.submit(1)
    finally:
        await sched.stop()


async def test_infer_fn_never_overlaps_itself():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def infer(payloads: list[int]) -> list[int]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return list(payloads)

    sched = MicroBatchScheduler(infer, max_batch_size=1, max_wait_ms=1)
    await sched.start()
    try:
        results = await asyncio.gather(*[sched.submit(i) for i in range(4)])
    finally:
        await sched.stop()

    assert results == [0, 1, 2, 3]
    assert max_active == 1


async def test_inflight_caller_resolved_on_stop():
    """stop() mid-batch must resolve in-flight futures with CancelledError, not hang."""
    in_infer = threading.Event()
    release = threading.Event()

    def infer(payloads: list[int]) -> list[int]:
        in_infer.set()  # signal: batch is running in the executor
        release.wait(timeout=2.0)  # block so stop() lands mid-batch
        return list(payloads)

    sched = MicroBatchScheduler(infer, max_batch_size=4, max_wait_ms=5)
    await sched.start()
    task = asyncio.create_task(sched.submit(1))

    # Wait (off the event loop) until infer is actually running in the pool.
    await asyncio.get_running_loop().run_in_executor(None, in_infer.wait, 2.0)

    await sched.stop()  # cancels the loop task mid-executor
    release.set()  # let the dangling infer thread finish cleanly

    # The caller must be resolved with CancelledError, NOT hang.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)


async def test_stop_drains_queued_callers():
    """A submit still queued (never collected) when stop() lands must not hang."""
    block = threading.Event()

    def infer(payloads):
        block.wait(timeout=2.0)  # park the GPU stage so later submits stay queued
        return list(payloads)

    sched = MicroBatchScheduler(infer, max_batch_size=1, max_wait_ms=1)
    await sched.start()
    first = asyncio.create_task(sched.submit(0))  # gets collected + parked in the executor
    await asyncio.sleep(0.02)
    queued = asyncio.create_task(sched.submit(1))  # never collected (executor busy)
    await asyncio.sleep(0.02)

    await sched.stop()
    block.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(queued, timeout=1.0)


async def test_start_after_stop_raises():
    sched = MicroBatchScheduler(lambda payloads: list(payloads), max_batch_size=1, max_wait_ms=1)
    await sched.start()
    await sched.stop()
    with pytest.raises(RuntimeError):
        await sched.start()


async def test_start_is_idempotent_while_running():
    sched = MicroBatchScheduler(lambda payloads: list(payloads), max_batch_size=1, max_wait_ms=1)
    await sched.start()
    try:
        task = sched._task
        await sched.start()
        assert sched._task is task
        assert await sched.submit(3) == 3
    finally:
        await sched.stop()


def test_constructor_validates_arguments():
    infer = list
    with pytest.raises(ValueError):
        MicroBatchScheduler(infer, max_batch_size=0)
    with pytest.raises(ValueError):
        MicroBatchScheduler(infer, max_wait_ms=-1.0)
    with pytest.raises(ValueError):
        MicroBatchScheduler(infer, fast_flush_ms=-1.0)
