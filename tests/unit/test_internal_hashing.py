"""Tests for mmsafe._internal.hashing utilities."""

from __future__ import annotations

from mmsafe._internal.hashing import hash_content, hash_dict, short_id


class TestHashContent:
    def test_string_input(self) -> None:
        digest = hash_content("hello")
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex

    def test_bytes_input(self) -> None:
        digest = hash_content(b"hello")
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_string_and_bytes_match(self) -> None:
        """Same content in str vs bytes should produce the same hash."""
        assert hash_content("hello") == hash_content(b"hello")

    def test_deterministic(self) -> None:
        assert hash_content("same") == hash_content("same")

    def test_different_inputs_differ(self) -> None:
        assert hash_content("a") != hash_content("b")


class TestHashDict:
    def test_deterministic(self) -> None:
        d = {"key": "value", "num": 42}
        assert hash_dict(d) == hash_dict(d)

    def test_key_order_independent(self) -> None:
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert hash_dict(d1) == hash_dict(d2)

    def test_different_dicts_differ(self) -> None:
        assert hash_dict({"x": 1}) != hash_dict({"x": 2})


class TestShortId:
    def test_default_length(self) -> None:
        sid = short_id("test")
        assert len(sid) == 8

    def test_custom_length(self) -> None:
        sid = short_id("test", length=12)
        assert len(sid) == 12

    def test_deterministic(self) -> None:
        assert short_id("foo") == short_id("foo")

    def test_different_inputs_differ(self) -> None:
        assert short_id("a") != short_id("b")
