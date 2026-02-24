"""Tests for mmsafe._internal.concurrency module."""

from __future__ import annotations

import asyncio

import pytest

from mmsafe._internal.concurrency import bounded_gather


class TestBoundedGather:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self) -> None:
        async def make_val(i: int) -> int:
            return i * 2

        tasks = [lambda i=i: make_val(i) for i in range(5)]
        results = await bounded_gather(tasks, max_concurrency=5)
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_preserves_result_order(self) -> None:
        """Results should be in the same order as input tasks regardless of
        execution order."""

        async def delayed(val: int, delay: float) -> int:
            await asyncio.sleep(delay)
            return val

        # Task 0 finishes last, task 2 finishes first
        tasks = [
            lambda: delayed(0, 0.03),
            lambda: delayed(1, 0.02),
            lambda: delayed(2, 0.01),
        ]
        results = await bounded_gather(tasks, max_concurrency=3)
        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_returns_exceptions_inline(self) -> None:
        """Failed tasks return the exception object instead of raising."""

        async def ok() -> int:
            return 42

        async def fail() -> int:
            raise ValueError("boom")

        tasks = [ok, fail, ok]
        results = await bounded_gather(tasks, max_concurrency=3)
        assert results[0] == 42
        assert isinstance(results[1], ValueError)
        assert results[2] == 42

    @pytest.mark.asyncio
    async def test_respects_max_concurrency(self) -> None:
        """Concurrent count should never exceed the limit."""
        active = 0
        peak = 0

        async def tracked() -> int:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return 1

        tasks = [tracked for _ in range(10)]
        results = await bounded_gather(tasks, max_concurrency=3)
        assert peak <= 3
        assert all(r == 1 for r in results)
