"""Exponential backoff retry logic for API calls."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from mmsafe._internal.logging import get_logger

log = get_logger("retry")

T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async callable to retry.
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        jitter: Add randomized jitter to prevent thundering herd.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The result of the successful call.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                log.warning("All %d attempts exhausted: %s", max_attempts, exc)
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())  # noqa: S311

            log.info(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                max_attempts,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    if last_exc is None:
        msg = "retry_with_backoff: unreachable — no attempts were made"
        raise RuntimeError(msg)
    raise last_exc
