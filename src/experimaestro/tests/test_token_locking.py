"""Unit tests for token locking mechanism

Tests the CounterToken condition variable-based synchronization
without requiring full scheduler integration.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
import time
import hashlib

from experimaestro.tokens import CounterToken
from experimaestro.locking import LockError

pytestmark = pytest.mark.anyio


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
    """Create a mock job with all required attributes."""

    class MockJob:
        task_id = "mock-task"
        config = MockConfig(name)

        @property
        def identifier(self):
            return f"mock-job-{name}"

        @property
        def basepath(self):
            return Path(tmpdir) / name

    return MockJob()


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
        lock.release()
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
        lock1.release()
        assert token.available == 1

        # Now second should succeed
        lock2 = await dep2.aio_lock(timeout=0.5)
        assert lock2 is not None
        lock2.release()


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
        lock1.release()

        # Second task should complete quickly (not timeout)
        lock2 = await task
        elapsed = time.time() - start

        assert lock2 is not None
        assert elapsed < 1.0  # Should wake up immediately, not wait 5s
        lock2.release()


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
            lock.release()

        tasks = [
            asyncio.create_task(acquire_task("2")),
            asyncio.create_task(acquire_task("3")),
            asyncio.create_task(acquire_task("4")),
        ]

        # Give tasks time to start waiting
        await asyncio.sleep(0.1)

        # Release first lock
        lock1.release()

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # All tasks should have acquired the lock
        assert len(acquired_order) == 3
        assert set(acquired_order) == {"2", "3", "4"}


async def test_token_timeout_zero():
    """Test that timeout=0 waits indefinitely"""
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

        # Start waiting with timeout=0 (infinite)
        async def acquire_infinite():
            return await dep2.aio_lock(timeout=0)  # Should wait forever

        task = asyncio.create_task(acquire_infinite())

        # Give it time to start waiting
        await asyncio.sleep(0.1)

        # Wait a bit more - task should still be waiting
        await asyncio.sleep(0.5)
        assert not task.done()

        # Release first lock
        lock1.release()

        # Now task should complete
        lock2 = await asyncio.wait_for(task, timeout=2.0)
        assert lock2 is not None
        lock2.release()
