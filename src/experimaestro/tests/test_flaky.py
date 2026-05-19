"""Tests for the retry-on-flake helper.

These tests lock in the contracts that matter:
- `pytest.fail()` (which raises a BaseException-derived `Failed`) IS
  retried — historically the default `catch=Exception` missed it.
- A successful retry yields a passing test (the outer wrapper does not
  re-raise the earlier attempt's exception).
- After `max_attempts` consecutive failures, the last exception is
  re-raised.
"""

from __future__ import annotations

import asyncio

import pytest

from experimaestro.tests._flaky import retry_on_flake


class TestSyncRetry:
    def test_retries_on_pytest_fail(self):
        calls = []

        @retry_on_flake(max_attempts=3, delay=0)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                pytest.fail("simulated CI flake")
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3

    def test_retries_on_assertion_error(self):
        calls = []

        @retry_on_flake(max_attempts=2, delay=0)
        def flaky():
            calls.append(1)
            if len(calls) < 2:
                assert False, "boom"
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 2

    def test_reraises_after_max_attempts(self):
        @retry_on_flake(max_attempts=2, delay=0)
        def always_fails():
            pytest.fail("real bug")

        with pytest.raises(pytest.fail.Exception):
            always_fails()


class TestAsyncRetry:
    def test_retries_on_pytest_fail(self):
        calls = []

        @retry_on_flake(max_attempts=3, delay=0)
        async def flaky():
            calls.append(1)
            if len(calls) < 3:
                pytest.fail("simulated CI flake")
            return "ok"

        assert asyncio.run(flaky()) == "ok"
        assert len(calls) == 3
