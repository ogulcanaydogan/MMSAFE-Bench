"""Tests for mmsafe._internal.retry module."""

from __future__ import annotations

import asyncio

import pytest

from mmsafe._internal.retry import retry_with_backoff


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        call_count = 0

        async def ok() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await retry_with_backoff(ok, max_attempts=3, base_delay=0.01)
        assert result == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = await retry_with_backoff(
            flaky,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_all_attempts(self) -> None:
        async def always_fail() -> str:
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError, match="permanent"):
            await retry_with_backoff(
                always_fail,
                max_attempts=2,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        call_count = 0

        async def type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            await retry_with_backoff(
                type_error,
                max_attempts=3,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_respects_max_attempts(self) -> None:
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                always_fail,
                max_attempts=4,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )
        assert call_count == 4
