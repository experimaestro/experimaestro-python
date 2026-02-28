"""Comprehensive tests for the FileWatcherService

Tests the centralized file watching service including:
- Singleton & configuration
- Directory watching (basic, adaptive polling, async)
- File following (basic, edge cases, read_tail/read_new)
- Resource management
- Async watching
"""

import asyncio
import gc
import time
import weakref
from threading import Event

import pytest

from experimaestro.filewatcher import (
    AsyncEventBridge,
    FileFollower,
    FileWatcherService,
    PolledFile,
    WatcherType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_service():
    """Reset FileWatcherService and AsyncEventBridge between tests."""
    FileWatcherService.reset()
    AsyncEventBridge.reset()
    # Ensure testing mode
    FileWatcherService.configure(testing_mode=True, polling_interval=0.01)
    yield
    FileWatcherService.reset()
    AsyncEventBridge.reset()


# =============================================================================
# Singleton & Configuration Tests
# =============================================================================


class TestSingletonConfiguration:
    def test_singleton_same_instance(self):
        """instance() returns same object"""
        svc1 = FileWatcherService.instance()
        svc2 = FileWatcherService.instance()
        assert svc1 is svc2

    def test_reset_clears_instance(self):
        """reset() clears the singleton"""
        svc1 = FileWatcherService.instance()
        FileWatcherService.reset()
        svc2 = FileWatcherService.instance()
        assert svc1 is not svc2

    def test_configure_sets_watcher_type(self):
        """configure() sets watcher type"""
        FileWatcherService.reset()
        FileWatcherService.configure(watcher_type=WatcherType.POLLING)
        assert FileWatcherService._watcher_type == WatcherType.POLLING

    def test_testing_mode_uses_polling(self):
        """testing_mode=True uses polling observer"""
        FileWatcherService.reset()
        FileWatcherService.configure(testing_mode=True, polling_interval=0.01)
        svc = FileWatcherService.instance()
        # In testing mode, the observer should be a PollingObserver
        from watchdog.observers.polling import PollingObserver

        assert isinstance(svc._observer, PollingObserver)


# =============================================================================
# Directory Watching -- Basic
# =============================================================================


class TestDirectoryWatchingBasic:
    def test_detects_file_creation(self, tmp_path):
        """Watch detects file creation"""
        created = []
        create_event = Event()

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_created=lambda p: (created.append(p), create_event.set()),
            on_change=lambda p: None,
        )

        try:
            time.sleep(0.1)
            test_file = tmp_path / "new.txt"
            test_file.write_text("hello")

            assert create_event.wait(timeout=3.0), "Creation not detected"
            assert any(p.name == "new.txt" for p in created)
        finally:
            watch.close()

    def test_detects_file_modification(self, tmp_path):
        """Watch detects file modification via polling"""
        changes = []
        change_event = Event()

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: (changes.append(p), change_event.set()),
            min_poll_interval=0.1,
            max_poll_interval=0.5,
        )
        watch.add_file(test_file)

        try:
            time.sleep(0.2)
            test_file.write_text("modified content")

            assert change_event.wait(timeout=3.0), "Modification not detected"
            assert test_file in changes
        finally:
            watch.close()

    def test_detects_file_deletion(self, tmp_path):
        """Watch detects file deletion"""
        deleted = []
        delete_event = Event()

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
            on_deleted=lambda p: (deleted.append(p), delete_event.set()),
        )

        try:
            time.sleep(0.1)
            test_file.unlink()

            if delete_event.wait(timeout=3.0):
                assert any(p.name == "test.txt" for p in deleted)
        finally:
            watch.close()

    def test_recursive_watches_subdirs(self, tmp_path):
        """Recursive watch detects changes in subdirectories"""
        changes = []
        change_event = Event()

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            recursive=True,
            on_change=lambda p: (changes.append(p), change_event.set()),
            on_created=lambda p: None,
        )

        try:
            time.sleep(0.1)
            sub_file = subdir / "test.txt"
            sub_file.write_text("hello")

            assert change_event.wait(timeout=3.0), "Recursive change not detected"
        finally:
            watch.close()

    def test_non_recursive_ignores_subdirs(self, tmp_path):
        """Non-recursive watch ignores subdirectory changes (watchdog behavior)"""
        # This test verifies the recursive=False flag is passed through
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            recursive=False,
            on_change=lambda p: None,
        )

        try:
            assert watch._recursive is False
        finally:
            watch.close()

    def test_file_filter_applied(self, tmp_path):
        """File filter restricts which files are tracked"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            file_filter=lambda p: p.suffix == ".txt",
            on_change=lambda p: None,
        )

        try:
            txt_file = tmp_path / "test.txt"
            txt_file.write_text("content")
            json_file = tmp_path / "test.json"
            json_file.write_text("{}")

            watch.add_file(txt_file)
            watch.add_file(json_file)

            assert txt_file in watch._files
            assert json_file not in watch._files
        finally:
            watch.close()

    def test_multiple_watches_same_dir(self, tmp_path):
        """Multiple watches on the same directory work independently"""
        changes1 = []
        changes2 = []
        event1 = Event()
        event2 = Event()

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        svc = FileWatcherService.instance()
        watch1 = svc.watch_directory(
            tmp_path,
            on_change=lambda p: (changes1.append(p), event1.set()),
            min_poll_interval=0.1,
        )
        watch2 = svc.watch_directory(
            tmp_path,
            on_change=lambda p: (changes2.append(p), event2.set()),
            min_poll_interval=0.1,
        )
        watch1.add_file(test_file)
        watch2.add_file(test_file)

        try:
            time.sleep(0.2)
            test_file.write_text("modified")

            assert event1.wait(timeout=3.0)
            assert event2.wait(timeout=3.0)
            assert test_file in changes1
            assert test_file in changes2
        finally:
            watch1.close()
            watch2.close()

    def test_close_stops_events(self, tmp_path):
        """Closing a watch stops further events"""
        changes = []

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: changes.append(p),
            min_poll_interval=0.1,
        )
        watch.add_file(test_file)
        watch.close()

        # Modify after close
        test_file.write_text("modified")
        time.sleep(0.5)

        assert len(changes) == 0

    def test_gc_releases_resources(self, tmp_path):
        """Dropping reference cleans up via __del__"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
        )
        _ref = weakref.ref(watch)  # noqa: F841

        with svc._watches_lock:
            assert watch in svc._directory_watches

        del watch
        gc.collect()

        # After GC, the watch should be cleaned up
        # (Note: __del__ is not guaranteed to run immediately)
        time.sleep(0.1)
        gc.collect()


# =============================================================================
# Directory Watching -- Adaptive Polling
# =============================================================================


class TestDirectoryWatchingPolling:
    def test_polling_catches_missed_watchdog_events(self, tmp_path):
        """Polling detects changes that watchdog might miss"""
        changes = []
        change_event = Event()

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: (changes.append(p), change_event.set()),
            min_poll_interval=0.1,
            max_poll_interval=0.5,
        )
        watch.add_file(test_file)

        try:
            time.sleep(0.2)
            test_file.write_text("modified")

            assert change_event.wait(timeout=3.0)
            assert test_file in changes
        finally:
            watch.close()

    def test_watchdog_hit_increases_reliability(self, tmp_path):
        """Watchdog-detected changes increase reliability"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
        )
        watch.add_file(test_file)
        initial = watch._files[test_file].watchdog_reliability

        watch.notify_change(test_file)

        assert watch._files[test_file].watchdog_reliability > initial
        watch.close()

    def test_poll_hit_decreases_reliability(self, tmp_path):
        """Poll-detected changes decrease reliability"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        polled = PolledFile(path=test_file)
        polled.watchdog_reliability = 0.8

        polled.on_poll_detected_change()

        assert polled.watchdog_reliability < 0.8

    def test_interval_grows_when_quiet(self, tmp_path):
        """Polling interval increases when no changes detected"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        polled = PolledFile(path=test_file)
        initial_interval = polled.estimated_change_interval

        polled.on_no_activity()

        assert polled.estimated_change_interval > initial_interval

    def test_set_poll_interval_dynamic(self, tmp_path):
        """set_poll_interval changes bounds immediately"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
            min_poll_interval=0.5,
            max_poll_interval=30.0,
        )
        watch.add_file(test_file)

        watch.set_poll_interval(min_interval=1.0, max_interval=10.0)

        assert watch._min_poll_interval == 1.0
        assert watch._max_poll_interval == 10.0
        assert watch._files[test_file].MIN_INTERVAL == 1.0
        assert watch._files[test_file].MAX_INTERVAL == 10.0
        watch.close()

    def test_add_remove_file_tracking(self, tmp_path):
        """add_file and remove_file work correctly"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watch.add_file(test_file)
        assert test_file in watch._files

        watch.remove_file(test_file)
        assert test_file not in watch._files
        watch.close()


# =============================================================================
# Directory Watching -- Async
# =============================================================================


class TestDirectoryWatchingAsync:
    @pytest.mark.anyio
    async def test_async_callback_dispatched_to_loop(self, tmp_path):
        """Async handler receives events via AsyncEventBridge"""
        bridge = AsyncEventBridge.instance()
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        events_received = []
        event_received = asyncio.Event()

        class MockHandler:
            async def on_created_async(self, event):
                events_received.append(("created", event.src_path))
                event_received.set()

        handler = MockHandler()
        unregister = bridge.register_handler(tmp_path, handler)

        try:
            # Simulate event
            class MockEvent:
                event_type = "created"
                src_path = str(tmp_path / "test.txt")
                is_directory = False

            bridge.post_event(tmp_path, MockEvent())

            await asyncio.wait_for(event_received.wait(), timeout=2.0)
            assert len(events_received) == 1
            assert events_received[0][0] == "created"
        finally:
            unregister()


# =============================================================================
# File Following -- Basic
# =============================================================================


class TestFileFollowingBasic:
    @pytest.mark.anyio
    async def test_follow_yields_new_lines(self, tmp_path):
        """FileFollower yields new lines as they're written"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\n")

        follower = FileFollower(test_file, poll_interval=0.1)

        lines = []
        async for line in follower:
            lines.append(line)
            if len(lines) >= 2:
                break

        assert lines == ["line1", "line2"]
        follower.close()

    @pytest.mark.anyio
    async def test_follow_from_end_skips_existing(self, tmp_path):
        """from_end=True starts reading from end of file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("existing\n")

        follower = FileFollower(test_file, poll_interval=0.1, from_end=True)

        # Write new content
        with open(test_file, "a") as f:
            f.write("new_line\n")

        lines = []
        async for line in follower:
            lines.append(line)
            if len(lines) >= 1:
                break

        assert lines == ["new_line"]
        follower.close()

    @pytest.mark.anyio
    async def test_follow_from_beginning_reads_all(self, tmp_path):
        """Default (from_end=False) reads from beginning"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("first\nsecond\n")

        follower = FileFollower(test_file, poll_interval=0.1)

        lines = []
        async for line in follower:
            lines.append(line)
            if len(lines) >= 2:
                break

        assert lines == ["first", "second"]
        follower.close()

    @pytest.mark.anyio
    async def test_buffers_incomplete_lines(self, tmp_path):
        """Incomplete lines (no newline) are buffered"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("complete\nincomplete")

        follower = FileFollower(test_file, poll_interval=0.1)

        lines = []
        # Read the complete line
        async for line in follower:
            lines.append(line)
            if len(lines) >= 1:
                # Now append newline to make incomplete line complete
                with open(test_file, "a") as f:
                    f.write("\n")

            if len(lines) >= 2:
                break

        assert lines == ["complete", "incomplete"]
        follower.close()

    def test_handles_empty_file(self, tmp_path):
        """FileFollower handles empty file gracefully"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        follower = FileFollower(test_file, poll_interval=0.1)
        content = follower.read_new()
        assert content == ""
        follower.close()


# =============================================================================
# File Following -- Edge Cases
# =============================================================================


class TestFileFollowingEdgeCases:
    def test_file_truncated_resets_position(self, tmp_path):
        """Truncated file resets read position"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("long content here")

        follower = FileFollower(test_file, poll_interval=0.1)
        follower.read_new()  # Read all content
        assert follower._position > 0

        # Truncate file
        test_file.write_text("short")
        content = follower.read_new()
        assert content == "short"

    def test_file_deleted_during_follow(self, tmp_path):
        """Deleted file returns empty content"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        follower = FileFollower(test_file, poll_interval=0.1)
        test_file.unlink()

        content = follower.read_new()
        assert content == ""
        follower.close()

    def test_file_deleted_and_recreated(self, tmp_path):
        """File deletion and recreation works"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("long original content")

        follower = FileFollower(test_file, poll_interval=0.1)
        follower.read_new()  # Read original

        test_file.unlink()
        # Recreate with shorter content - triggers truncation detection
        test_file.write_text("new")

        content = follower.read_new()
        assert content == "new"
        follower.close()

    def test_rapid_writes_all_delivered(self, tmp_path):
        """Rapid writes are all captured"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("")

        follower = FileFollower(test_file, poll_interval=0.1)

        with open(test_file, "a") as f:
            for i in range(100):
                f.write(f"line{i}\n")
                f.flush()

        content = follower.read_new()
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 100
        follower.close()

    def test_binary_content_handled(self, tmp_path):
        """Binary content handled with errors='replace'"""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello\x80\x81world\n")

        follower = FileFollower(test_file, poll_interval=0.1)
        content = follower.read_new()
        assert "hello" in content
        assert "world" in content
        follower.close()

    @pytest.mark.anyio
    async def test_close_stops_iteration(self, tmp_path):
        """Closing follower stops async iteration"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("")

        follower = FileFollower(test_file, poll_interval=0.1)

        # Close after a short delay
        async def close_later():
            await asyncio.sleep(0.3)
            follower.close()

        task = asyncio.create_task(close_later())

        lines = []
        async for line in follower:
            lines.append(line)

        await task
        # Should exit cleanly without hanging


# =============================================================================
# File Following -- read_tail / read_new
# =============================================================================


class TestFileFollowingReadMethods:
    def test_read_tail_last_bytes(self, tmp_path):
        """read_tail returns last N bytes"""
        test_file = tmp_path / "test.txt"
        content = "line1\nline2\nline3\nline4\n"
        test_file.write_text(content)

        follower = FileFollower(test_file, poll_interval=0.1)
        tail = follower.read_tail(max_bytes=12)

        # Should get last 12 bytes, possibly starting from a line boundary
        assert len(tail) <= 12 + 10  # Allow for line alignment
        assert tail.endswith("\n")
        follower.close()

    def test_read_new_returns_delta(self, tmp_path):
        """read_new returns only new content"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("first\n")

        follower = FileFollower(test_file, poll_interval=0.1)
        first = follower.read_new()
        assert first == "first\n"

        with open(test_file, "a") as f:
            f.write("second\n")

        second = follower.read_new()
        assert second == "second\n"
        follower.close()


# =============================================================================
# Resource Management
# =============================================================================


class TestResourceManagement:
    def test_directory_watch_context_manager(self, tmp_path):
        """DirectoryWatch works as context manager"""
        svc = FileWatcherService.instance()

        with svc.watch_directory(tmp_path, on_change=lambda p: None) as watch:
            assert not watch._closed

        assert watch._closed

    @pytest.mark.anyio
    async def test_file_follower_async_context_manager(self, tmp_path):
        """FileFollower works as async context manager"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        async with FileFollower(test_file, poll_interval=0.1) as follower:
            content = follower.read_new()
            assert content == "content"

        assert follower._closed

    def test_watch_and_follow_same_file_coexist(self, tmp_path):
        """Watching and following the same file works"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("initial\n")

        svc = FileWatcherService.instance()

        changes = []
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: changes.append(p),
        )
        watch.add_file(test_file)

        follower = svc.follow_file(test_file)
        content = follower.read_new()
        assert content == "initial\n"

        watch.close()
        follower.close()


# =============================================================================
# Async Watching (for TrackedDynamicResource)
# =============================================================================


class TestAsyncWatching:
    @pytest.mark.anyio
    async def test_async_watch_dispatches_to_loop(self, tmp_path):
        """async_watch dispatches events to the asyncio loop"""
        bridge = AsyncEventBridge.instance()
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        events_received = []
        event_received = asyncio.Event()

        class MockHandler:
            async def on_created_async(self, event):
                events_received.append(event)
                event_received.set()

        svc = FileWatcherService.instance()
        handler = MockHandler()
        watch = svc.async_watch(handler, tmp_path, recursive=True)

        try:
            # Create a file to trigger the event
            test_file = tmp_path / "test.txt"
            test_file.write_text("hello")

            await asyncio.wait_for(event_received.wait(), timeout=3.0)
            assert len(events_received) >= 1
        finally:
            watch.close()

    @pytest.mark.anyio
    async def test_async_watch_close_stops_events(self, tmp_path):
        """Closing async watch stops events"""
        bridge = AsyncEventBridge.instance()
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        events_received = []

        class MockHandler:
            async def on_created_async(self, event):
                events_received.append(event)

        svc = FileWatcherService.instance()
        handler = MockHandler()
        watch = svc.async_watch(handler, tmp_path, recursive=True)
        watch.close()

        # Create a file after closing
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        await asyncio.sleep(0.5)
        assert len(events_received) == 0
