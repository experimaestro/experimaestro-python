"""Unit tests for token locking mechanism

Tests the CounterToken condition variable-based synchronization
without requiring full scheduler integration.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
import time
import hashlib

from experimaestro.tokens import CounterToken
from experimaestro.locking import LockError
from experimaestro.ipc import AsyncEventBridge

# Mark all tests in this module as token tests
pytestmark = [pytest.mark.anyio, pytest.mark.tokens]


@pytest.fixture(autouse=True)
def reset_async_bridge():
    """Reset AsyncEventBridge singleton and tokens before and after each test.

    This ensures tests don't interfere with each other's event handling.
    Note: EventLoopThread is not reset as it's the central event loop.
    """
    AsyncEventBridge.reset()
    # Reset token registry to avoid event loop binding issues across tests
    CounterToken.TOKENS.clear()
    yield
    AsyncEventBridge.reset()
    CounterToken.TOKENS.clear()


class MockIdentifier:
    """Mock identifier with hex() method."""

    def __init__(self, value: str):
        self._hex = hashlib.sha256(value.encode()).hexdigest()

    def hex(self):
        return self._hex


class MockXPM:
    """Mock __xpm__ object."""

    def __init__(self, name: str):
        self.identifier = type("Identifier", (), {"main": MockIdentifier(name)})()


class MockConfig:
    """Mock config object."""

    def __init__(self, name: str):
        self.__xpm__ = MockXPM(name)


def create_mock_job(name: str, tmpdir: str):
    """Create a mock job with all required attributes and PID file."""
    import os

    class MockJob:
        task_id = "mock-task"
        config = MockConfig(name)

        @property
        def identifier(self):
            return f"mock-job-{name}"

        @property
        def basepath(self):
            return Path(tmpdir) / name

    job = MockJob()

    # Create job directory and PID file with valid process data
    job.basepath.mkdir(parents=True, exist_ok=True)
    pid_path = job.basepath.with_suffix(".pid")
    # Use current process PID so it's valid and running
    pid_path.write_text(json.dumps({"type": "local", "pid": os.getpid()}))

    return job


async def test_token_acquire_release():
    """Test basic token acquire and release"""
    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-basic", Path(tmpdir) / "token", count=1)

        job = create_mock_job("1", tmpdir)

        # Create dependency
        dep = token.dependency(1)
        dep.target = job

        # Should be able to acquire
        lock = await dep.aio_lock(timeout=1.0)
        assert lock is not None
        assert token.available == 0

        # Release
        await lock.aio_release()
        assert token.available == 1


async def test_token_blocking():
    """Test that acquiring blocks when no tokens available"""
    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-blocking", Path(tmpdir) / "token", count=1)

        job1 = create_mock_job("1", tmpdir)
        job2 = create_mock_job("2", tmpdir)

        dep1 = token.dependency(1)
        dep1.target = job1

        dep2 = token.dependency(1)
        dep2.target = job2

        # Acquire with first dependency
        lock1 = await dep1.aio_lock(timeout=0.5)
        assert token.available == 0

        # Second acquire should timeout
        start = time.time()
        with pytest.raises(LockError, match="Timeout"):
            await dep2.aio_lock(timeout=0.5)
        elapsed = time.time() - start
        assert 0.4 < elapsed < 0.7  # Should timeout around 0.5s

        # Release first lock
        await lock1.aio_release()
        assert token.available == 1

        # Now second should succeed
        lock2 = await dep2.aio_lock(timeout=0.5)
        assert lock2 is not None
        await lock2.aio_release()


async def test_token_notification():
    """Test that condition notification wakes up waiting tasks"""
    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-notify", Path(tmpdir) / "token", count=1)

        job1 = create_mock_job("1", tmpdir)
        job2 = create_mock_job("2", tmpdir)

        dep1 = token.dependency(1)
        dep1.target = job1

        dep2 = token.dependency(1)
        dep2.target = job2

        # Acquire with first dependency
        lock1 = await dep1.aio_lock(timeout=0.5)

        # Start second acquisition in background
        async def acquire_second():
            lock = await dep2.aio_lock(timeout=5.0)  # Long timeout
            return lock

        task = asyncio.create_task(acquire_second())

        # Give it time to start waiting
        await asyncio.sleep(0.1)

        # Release first lock - should notify waiting task
        start = time.time()
        await lock1.aio_release()

        # Second task should complete quickly (not timeout)
        lock2 = await task
        elapsed = time.time() - start

        assert lock2 is not None
        assert elapsed < 1.0  # Should wake up immediately, not wait 5s
        await lock2.aio_release()


async def test_token_multiple_waiting():
    """Test multiple tasks waiting for tokens"""
    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-multiple", Path(tmpdir) / "token", count=1)

        # Acquire the token
        job1 = create_mock_job("1", tmpdir)
        dep1 = token.dependency(1)
        dep1.target = job1
        lock1 = await dep1.aio_lock(timeout=0.5)

        # Start multiple waiting tasks
        acquired_order = []

        async def acquire_task(name):
            job = create_mock_job(name, tmpdir)
            dep = token.dependency(1)
            dep.target = job
            lock = await dep.aio_lock(timeout=10.0)
            acquired_order.append(name)
            await asyncio.sleep(0.05)  # Hold briefly
            await lock.aio_release()

        tasks = [
            asyncio.create_task(acquire_task("2")),
            asyncio.create_task(acquire_task("3")),
            asyncio.create_task(acquire_task("4")),
        ]

        # Give tasks time to start waiting
        await asyncio.sleep(0.1)

        # Release first lock
        await lock1.aio_release()

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # All tasks should have acquired the lock
        assert len(acquired_order) == 3
        assert set(acquired_order) == {"2", "3", "4"}


async def test_token_timeout_zero():
    """Test that timeout=0 waits indefinitely"""
    from experimaestro.dynamic import ResourcePoller

    # Reset ResourcePoller to ensure clean state from any previous tests
    ResourcePoller.reset()

    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-timeout-zero", Path(tmpdir) / "token", count=1)

        job1 = create_mock_job("1", tmpdir)
        job2 = create_mock_job("2", tmpdir)

        dep1 = token.dependency(1)
        dep1.target = job1

        dep2 = token.dependency(1)
        dep2.target = job2

        # Acquire with first
        lock1 = await dep1.aio_lock(timeout=0.5)

        # Track task state for better error messages
        task_started = asyncio.Event()
        task_exception = None

        # Start waiting with timeout=0 (infinite)
        async def acquire_infinite():
            nonlocal task_exception
            task_started.set()
            try:
                return await dep2.aio_lock(timeout=0)  # Should wait forever
            except Exception as e:
                task_exception = e
                raise

        task = asyncio.create_task(acquire_infinite())

        # Wait for task to start
        await task_started.wait()

        # Give the task time to enter the waiting state
        # Use multiple short sleeps to check task state more frequently
        # Extended wait time (2 seconds) for slower CI machines
        for _ in range(20):
            await asyncio.sleep(0.1)
            if task.done():
                # Task completed unexpectedly - get error info
                if task_exception:
                    pytest.fail(
                        f"Task completed unexpectedly with exception: {task_exception}"
                    )
                else:
                    # Try to get result (might raise)
                    try:
                        result = task.result()
                        pytest.fail(
                            f"Task completed unexpectedly with result: {result}"
                        )
                    except Exception as e:
                        pytest.fail(f"Task completed unexpectedly with error: {e}")

        # Task should still be waiting after 2 seconds
        assert not task.done(), "Task should be waiting indefinitely with timeout=0"

        # Release first lock
        await lock1.aio_release()

        # Now task should complete quickly
        lock2 = await asyncio.wait_for(task, timeout=5.0)
        assert lock2 is not None
        await lock2.aio_release()


async def test_orphaned_lock_detection():
    """Test detection of orphaned locks (no PID, scheduler crashed)"""
    from experimaestro.tokens import TokenLockFile

    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-orphaned", Path(tmpdir) / "token", count=10)

        job1 = create_mock_job("1", tmpdir)

        # Create an orphaned lock file (no process info)
        lock_path = token.jobs_folder / "orphaned.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Write lock file without process info (simulating scheduler crash)
        with lock_path.open("w") as f:
            json.dump({"job_uri": str(job1.basepath), "information": {"count": 2}}, f)

        # Make the file old (31 seconds ago)
        old_time = time.time() - 31
        import os

        os.utime(lock_path, (old_time, old_time))

        # Load the lock file
        lock_file = TokenLockFile(lock_path, token)

        # But should be detected as orphaned
        is_orphaned = await lock_file.async_resolve_orphaned(min_age_seconds=30)
        assert is_orphaned, "Lock file without PID should be detected as orphaned"


async def test_orphaned_lock_too_recent():
    """Test that recent locks without PID are not considered orphaned"""
    from experimaestro.tokens import TokenLockFile

    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-recent", Path(tmpdir) / "token", count=10)

        job1 = create_mock_job("1", tmpdir)

        # Create a lock file without process info
        lock_path = token.jobs_folder / "recent.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with lock_path.open("w") as f:
            json.dump({"job_uri": str(job1.basepath), "information": {"count": 1}}, f)

        # Make the file recent (10 seconds ago)
        recent_time = time.time() - 10
        import os

        os.utime(lock_path, (recent_time, recent_time))

        # Load the lock file
        lock_file = TokenLockFile(lock_path, token)

        # Should NOT be orphaned (too recent)
        is_orphaned = await lock_file.async_resolve_orphaned(min_age_seconds=30)
        assert not is_orphaned, (
            "Recent lock file without PID should not be considered orphaned"
        )


async def test_orphaned_lock_crash_scenario():
    """Test orphaned lock detection when scheduler crashes in aio_job_started.

    Scenario:
    1. Job acquires a token lock
    2. Scheduler crashes in aio_job_started before writing process info
    3. Another job tries to acquire the same token
    4. Should detect orphaned lock and clean it up
    """
    from experimaestro.dynamic import ResourcePoller
    import os

    # Reset ResourcePoller to ensure clean state
    ResourcePoller.reset()

    with tempfile.TemporaryDirectory() as tmpdir:
        token = CounterToken("test-crash", Path(tmpdir) / "token", count=1)

        job1 = create_mock_job("1", tmpdir)
        job2 = create_mock_job("2", tmpdir)

        # Manually create an orphaned lock file (simulating crash after acquire)
        lock_path = token.jobs_folder / f"{job1.task_id}@{job1.identifier}.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Write lock file without process info (simulates crash before aio_job_started)
        lock_data = {"job_uri": str(job1.basepath), "information": {"count": 1}}
        with lock_path.open("w") as f:
            json.dump(lock_data, f)

        # Make the lock file old enough to be considered orphaned
        old_time = time.time() - 35  # 35 seconds old
        os.utime(lock_path, (old_time, old_time))

        # Force token to re-read state from disk
        async with token.ipc_lock:
            token._ipc_update()

        # Verify the lock file is in cache and accounted for
        assert len(token.cache) == 1, "Lock file should be in cache"
        assert token.available == 0, "Token should not be available"

        # Now try to acquire the token with a second job
        dep2 = token.dependency(1)
        dep2.target = job2

        # This should detect the orphaned lock and clean it up
        lock2 = await dep2.aio_lock(timeout=5.0)

        assert lock2 is not None, "Should acquire lock after cleaning orphaned lock"

        # Verify the orphaned lock was removed
        assert not lock_path.exists(), "Orphaned lock file should be removed"

        # Verify the new lock exists
        lock_path2 = token.jobs_folder / f"{job2.task_id}@{job2.identifier}.json"
        assert lock_path2.exists(), "New lock file should exist"

        # Cleanup
        await lock2.aio_release()


# NOTE: Removed test_orphaned_lock_waits_for_ipc_lock
# The test was trying to verify intra-process serialization with IPC locks,
# but file locks (used for IPC) are per-process, not per-coroutine.
# The orphaned lock detection is primarily for inter-process scenarios
# (detecting crashes from other scheduler processes), which is tested
# in test_orphaned_lock_crash_scenario.
