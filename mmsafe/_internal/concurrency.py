"""Concurrency utilities for bounded parallel execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


async def bounded_gather(
    tasks: Sequence[Callable[[], Awaitable[T]]],
    *,
    max_concurrency: int = 5,
) -> list[T | BaseException]:
    """Execute async callables with bounded concurrency.

    Args:
        tasks: Sequence of zero-arg async callables.
        max_concurrency: Maximum simultaneous executions.

    Returns:
        List of results in the same order as input tasks.
        Failed tasks return the exception instead of raising.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[T | BaseException] = [Exception("not started")] * len(tasks)

    async def _run(index: int, func: Callable[[], Awaitable[T]]) -> None:
        async with semaphore:
            try:
                results[index] = await func()
            except BaseException as exc:
                results[index] = exc

    async with asyncio.TaskGroup() as tg:
        for i, task in enumerate(tasks):
            tg.create_task(_run(i, task))

    return results
