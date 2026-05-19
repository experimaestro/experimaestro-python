"""Helpers for handling tests that have flaked in CI.

Two tools, used together:

- :func:`retry_on_flake` — decorator that re-runs a sync or async test
  function on failure, dumping all thread stacks between attempts. Use
  it sparingly, only on tests that have actually flaked in CI; if a
  test is broken, the retry will hide it.

- A session-level autouse fixture (in ``conftest.py``) arms
  :func:`faulthandler.dump_traceback_later` so a test that hangs past
  ``XPM_TEST_THREAD_DUMP_AFTER`` seconds prints every thread's stack to
  stderr *before* pytest-timeout fires. Pure diagnostic; no behaviour
  change.
"""

from __future__ import annotations

import asyncio
import faulthandler
import functools
import logging
import sys
import time
from typing import Any, Callable

import pytest

logger = logging.getLogger("xpm.tests.flaky")


# pytest.fail() raises a Failed exception that extends BaseException
# (not Exception) precisely so user code can't accidentally swallow it.
# A retry-on-flake decorator legitimately needs to catch it.
_DEFAULT_CATCH: tuple[type[BaseException], ...] = (Exception, pytest.fail.Exception)


def dump_thread_stacks(prefix: str = "Thread dump") -> None:
    """Write every Python thread's stack to stderr."""
    sys.stderr.write(f"\n=== {prefix} ===\n")
    sys.stderr.flush()
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        logger.exception("Could not dump thread stacks")
    sys.stderr.write("=== end thread dump ===\n")
    sys.stderr.flush()


def retry_on_flake(
    max_attempts: int = 3,
    *,
    delay: float = 0.5,
    catch: tuple[type[BaseException], ...] = _DEFAULT_CATCH,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Re-run a test up to ``max_attempts`` times on failure.

    Works on both sync and ``async def`` test functions. On each failed
    attempt, dumps thread state to stderr (helpful when the failure is
    a hang surfaced by pytest-timeout) and waits ``delay`` seconds
    before retrying.

    Use only on tests that have a documented history of CI flakes —
    silencing genuine bugs with retries is worse than failing loudly.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exc: BaseException | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except catch as e:
                        last_exc = e
                        logger.warning(
                            "Test attempt %d/%d failed: %s",
                            attempt,
                            max_attempts,
                            e,
                        )
                        dump_thread_stacks(
                            f"After attempt {attempt}/{max_attempts} of {func.__name__}"
                        )
                        if attempt == max_attempts:
                            raise
                        await asyncio.sleep(delay)
                # Unreachable, but keeps type-checkers happy
                raise last_exc  # type: ignore[misc]

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except catch as e:
                    last_exc = e
                    logger.warning(
                        "Test attempt %d/%d failed: %s",
                        attempt,
                        max_attempts,
                        e,
                    )
                    dump_thread_stacks(
                        f"After attempt {attempt}/{max_attempts} of {func.__name__}"
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return sync_wrapper

    return decorator
