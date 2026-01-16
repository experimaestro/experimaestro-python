"""Tests for process utilities, particularly exit code handling."""

import asyncio
import subprocess
import sys

import pytest

from experimaestro.utils.psutils import (
    _has_kqueue,
    _has_pidfd,
    aio_wait_pid,
)

# Mark all tests in this module as connector tests (utilities)
pytestmark = [pytest.mark.connectors]


@pytest.mark.parametrize("exit_code", [0, 1, 2, 42, 127])
def test_aio_wait_pid_exit_codes(exit_code: int):
    """Test that aio_wait_pid correctly returns process exit codes.

    This test ensures that our async process waiting implementation
    (whether using kqueue on macOS, pidfd on Linux, or polling fallback)
    correctly reports the exit code of child processes.

    This was a regression: kqueue on macOS requires NOTE_EXITSTATUS flag
    to get the exit status in the data field.
    """

    async def run_test():
        # Start a process that exits with the specified code
        p = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.exit({exit_code})"]
        )
        pid = p.pid

        # Wait using our async implementation
        actual_code = await aio_wait_pid(pid)

        return actual_code

    actual = asyncio.run(run_test())
    assert actual == exit_code, (
        f"Expected exit code {exit_code}, got {actual}. "
        f"kqueue={_has_kqueue()}, pidfd={_has_pidfd()}"
    )


@pytest.mark.skipif(not _has_kqueue(), reason="kqueue not available")
@pytest.mark.parametrize("exit_code", [0, 1, 42])
def test_kqueue_exit_codes(exit_code: int):
    """Test kqueue-specific exit code handling on macOS/BSD.

    This test specifically tests the kqueue implementation to catch
    regressions in NOTE_EXITSTATUS handling.
    """
    from experimaestro.utils.psutils import _aio_wait_kqueue

    async def run_test():
        p = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.exit({exit_code})"]
        )
        pid = p.pid
        actual_code = await _aio_wait_kqueue(pid)
        return actual_code

    actual = asyncio.run(run_test())
    assert actual == exit_code, f"kqueue returned {actual} instead of {exit_code}"


@pytest.mark.skipif(not _has_pidfd(), reason="pidfd not available (Linux 5.3+)")
@pytest.mark.parametrize("exit_code", [0, 1, 42])
def test_pidfd_exit_codes(exit_code: int):
    """Test pidfd-specific exit code handling on Linux."""
    from experimaestro.utils.psutils import _aio_wait_pidfd

    async def run_test():
        p = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.exit({exit_code})"]
        )
        pid = p.pid
        actual_code = await _aio_wait_pidfd(pid)
        return actual_code

    actual = asyncio.run(run_test())
    assert actual == exit_code, f"pidfd returned {actual} instead of {exit_code}"
