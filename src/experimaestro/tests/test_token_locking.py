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

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def reset_async_bridge():
    """Reset AsyncEventBridge singleton before and after each test.

    This ensures tests don't interfere with each other's event handling.
    Note: EventLoopThread is not reset as it's the central event loop.
    """
    AsyncEventBridge.reset()
    yield
    AsyncEventBridge.reset()


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

    # Create PID file with MockProcess data so lock files are valid
    pid_path = job.basepath.with_suffix(".pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(json.dumps({"type": "mock"}))

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
