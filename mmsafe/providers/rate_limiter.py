"""Token-bucket rate limiter for API providers."""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter.

    Ensures API calls stay within the provider's rate limit by
    controlling the rate at which tokens (permits) are issued.
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        """Initialize rate limiter.

        Args:
            rate: Tokens per second to add to the bucket.
            burst: Maximum bucket capacity (burst allowance).
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = (1.0 - self._tokens) / self._rate

            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @classmethod
    def from_rpm(cls, requests_per_minute: int) -> TokenBucketRateLimiter:
        """Create a rate limiter from requests-per-minute specification."""
        rate = requests_per_minute / 60.0
        burst = max(1, requests_per_minute // 10)
        return cls(rate=rate, burst=burst)
