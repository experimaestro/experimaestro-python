"""Tests for async event handling in dynamic resources.

Tests AsyncEventBridge and async event handling in TrackedDynamicResource.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from experimaestro.ipc import AsyncEventBridge, AsyncFileSystemEventHandler

pytestmark = pytest.mark.anyio


@pytest.fixture
def reset_async_bridge():
    """Reset AsyncEventBridge singleton before and after each test."""
    AsyncEventBridge.reset()
    yield
    AsyncEventBridge.reset()


async def test_async_event_bridge_singleton(reset_async_bridge):
    """Test AsyncEventBridge singleton pattern."""
    bridge1 = AsyncEventBridge.instance()
    bridge2 = AsyncEventBridge.instance()
    assert bridge1 is bridge2


async def test_async_event_bridge_set_loop(reset_async_bridge):
    """Test setting the event loop."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()

    bridge.set_loop(loop)
    assert bridge._loop is loop


async def test_async_event_bridge_register_handler(reset_async_bridge):
    """Test registering and unregistering handlers."""
    bridge = AsyncEventBridge.instance()

    events_received = []

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path, kwargs))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Register handler
        unregister = bridge.register_handler(path, handler)

        # Verify handler is registered
        path_str = str(path.absolute())
        assert path_str in bridge._handlers
        assert handler in bridge._handlers[path_str]

        # Unregister handler
        unregister()

        # Verify handler is removed
        assert path_str not in bridge._handlers or handler not in bridge._handlers.get(
            path_str, []
        )


async def test_async_event_bridge_post_event(reset_async_bridge):
    """Test posting events to registered handlers."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)

    events_received = []
    event_received = asyncio.Event()

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path, kwargs))
        event_received.set()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister = bridge.register_handler(path, handler)

        # Post event
        bridge.post_event(path, "created", str(path / "test.txt"), is_directory=False)

        # Wait for event to be processed
        await asyncio.wait_for(event_received.wait(), timeout=2.0)

        assert len(events_received) == 1
        assert events_received[0][0] == "created"
        assert events_received[0][1] == str(path / "test.txt")
        assert events_received[0][2] == {"is_directory": False}

        unregister()


async def test_async_event_bridge_no_loop(reset_async_bridge):
    """Test that events are dropped when no loop is set."""
    bridge = AsyncEventBridge.instance()
    # Don't set loop

    events_received = []

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister = bridge.register_handler(path, handler)

        # Post event - should be dropped since no loop is set
        bridge.post_event(path, "created", str(path / "test.txt"))

        # Give time for any processing (should not happen)
        await asyncio.sleep(0.1)

        assert len(events_received) == 0

        unregister()


async def test_async_event_bridge_multiple_handlers(reset_async_bridge):
    """Test multiple handlers for the same path."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)

    events1 = []
    events2 = []
    both_received = asyncio.Event()
    count = 0

    async def handler1(event_type: str, src_path: str, **kwargs):
        nonlocal count
        events1.append(event_type)
        count += 1
        if count >= 2:
            both_received.set()

    async def handler2(event_type: str, src_path: str, **kwargs):
        nonlocal count
        events2.append(event_type)
        count += 1
        if count >= 2:
            both_received.set()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister1 = bridge.register_handler(path, handler1)
        unregister2 = bridge.register_handler(path, handler2)

        # Post event
        bridge.post_event(path, "modified", str(path / "test.txt"))

        # Wait for both handlers to be called
        await asyncio.wait_for(both_received.wait(), timeout=2.0)

        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0] == "modified"
        assert events2[0] == "modified"

        unregister1()
        unregister2()


async def test_async_file_system_event_handler(reset_async_bridge):
    """Test AsyncFileSystemEventHandler bridges events correctly."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)

    events_received = []
    event_received = asyncio.Event()

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path, kwargs))
        event_received.set()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister = bridge.register_handler(path, handler)

        # Create the handler
        fs_handler = AsyncFileSystemEventHandler(path, bridge)

        # Simulate a file creation event
        class MockEvent:
            is_directory = False
            src_path = str(path / "new_file.txt")

        fs_handler.on_created(MockEvent())

        # Wait for event
        await asyncio.wait_for(event_received.wait(), timeout=2.0)

        assert len(events_received) == 1
        assert events_received[0][0] == "created"
        assert events_received[0][1] == str(path / "new_file.txt")

        unregister()


async def test_async_file_system_event_handler_moved(reset_async_bridge):
    """Test AsyncFileSystemEventHandler handles moved events with dest_path."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)

    events_received = []
    event_received = asyncio.Event()

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path, kwargs))
        event_received.set()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister = bridge.register_handler(path, handler)

        fs_handler = AsyncFileSystemEventHandler(path, bridge)

        # Simulate a file move event
        class MockMoveEvent:
            is_directory = False
            src_path = str(path / "old_name.txt")
            dest_path = str(path / "new_name.txt")

        fs_handler.on_moved(MockMoveEvent())

        # Wait for event
        await asyncio.wait_for(event_received.wait(), timeout=2.0)

        assert len(events_received) == 1
        assert events_received[0][0] == "moved"
        assert events_received[0][1] == str(path / "old_name.txt")
        assert events_received[0][2]["dest_path"] == str(path / "new_name.txt")

        unregister()


async def test_async_file_system_event_handler_ignores_directories(reset_async_bridge):
    """Test that directory events are ignored."""
    bridge = AsyncEventBridge.instance()
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)

    events_received = []

    async def handler(event_type: str, src_path: str, **kwargs):
        events_received.append((event_type, src_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        unregister = bridge.register_handler(path, handler)

        fs_handler = AsyncFileSystemEventHandler(path, bridge)

        # Simulate directory events
        class MockDirEvent:
            is_directory = True
            src_path = str(path / "subdir")

        fs_handler.on_created(MockDirEvent())
        fs_handler.on_deleted(MockDirEvent())
        fs_handler.on_modified(MockDirEvent())

        # Give time for any processing
        await asyncio.sleep(0.1)

        # No events should have been received (directories are ignored)
        assert len(events_received) == 0

        unregister()
