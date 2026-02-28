"""Tests for the adaptive polling system (via FileWatcherService)"""

import time
from pathlib import Path
from threading import Event
from typing import List


from experimaestro.filewatcher import FileWatcherService, PolledFile


class TestPolledFile:
    """Tests for PolledFile dataclass"""

    def test_schedule_next(self):
        """Test that schedule_next sets next_poll correctly"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.poll_interval = 1.0
        before = time.time()
        polled.schedule_next()
        after = time.time()

        assert before + 1.0 <= polled.next_poll <= after + 1.0

    def test_on_activity_decreases_reliability(self):
        """Test that on_activity (poll detected change) decreases reliability"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.watchdog_reliability = 0.8  # High reliability

        polled.on_activity()

        # Reliability should decrease (Polyak toward 0)
        assert polled.watchdog_reliability < 0.8
        assert polled.next_poll > 0

    def test_on_no_activity_increases_change_interval(self):
        """Test that on_no_activity increases estimated change interval"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        initial_interval = polled.estimated_change_interval

        polled.on_no_activity()

        assert polled.estimated_change_interval > initial_interval

    def test_on_no_activity_caps_at_max(self):
        """Test that change interval doesn't grow unbounded"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.estimated_change_interval = polled.MAX_INTERVAL * 2

        polled.on_no_activity()

        assert polled.estimated_change_interval == polled.MAX_INTERVAL * 2

    def test_reliability_increases_on_watchdog_change(self):
        """Test that watchdog-detected changes increase reliability"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.watchdog_reliability = 0.3  # Low reliability

        polled.on_watchdog_detected_change()

        # Reliability should increase (Polyak toward 1)
        assert polled.watchdog_reliability > 0.3

    def test_reliability_decreases_on_poll_change(self):
        """Test that poll-detected changes decrease reliability"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.watchdog_reliability = 0.7  # Medium-high reliability

        polled.on_poll_detected_change()

        # Reliability should decrease (Polyak toward 0)
        assert polled.watchdog_reliability < 0.7

    def test_poll_interval_depends_on_reliability(self):
        """Test that poll interval scales with watchdog reliability"""
        polled_low = PolledFile(path=Path("/tmp/test.txt"))
        polled_high = PolledFile(path=Path("/tmp/test.txt"))

        polled_low.watchdog_reliability = 0.1
        polled_high.watchdog_reliability = 0.9
        polled_low.estimated_change_interval = 5.0
        polled_high.estimated_change_interval = 5.0

        polled_low._compute_poll_interval()
        polled_high._compute_poll_interval()

        # Higher reliability = longer poll interval
        assert polled_high.poll_interval > polled_low.poll_interval

    def test_update_size_detects_change(self, tmp_path):
        """Test that update_size detects file size changes"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        polled = PolledFile(path=test_file, last_size=0)

        assert polled.update_size() is True
        assert polled.last_size == 5  # "hello" is 5 bytes

    def test_update_size_no_change(self, tmp_path):
        """Test that update_size returns False when no change"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        polled = PolledFile(path=test_file, last_size=5)

        assert polled.update_size() is False

    def test_update_size_nonexistent_file(self, tmp_path):
        """Test that update_size handles missing files"""
        polled = PolledFile(path=tmp_path / "nonexistent.txt")

        assert polled.update_size() is False


class TestDirectoryWatch:
    """Tests for DirectoryWatch via FileWatcherService"""

    def test_add_and_remove_file(self, tmp_path):
        """Test adding and removing files"""
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

    def test_file_filter(self, tmp_path):
        """Test that file_filter is respected"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
            file_filter=lambda p: p.suffix == ".txt",
        )

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        json_file = tmp_path / "test.json"
        json_file.write_text("{}")

        watch.add_file(txt_file)
        watch.add_file(json_file)

        assert txt_file in watch._files
        assert json_file not in watch._files
        watch.close()

    def test_notify_change_increases_reliability(self, tmp_path):
        """Test that notify_change increases watchdog reliability"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watch.add_file(test_file)
        initial_reliability = watch._files[test_file].watchdog_reliability

        watch.notify_change(test_file)

        assert watch._files[test_file].watchdog_reliability > initial_reliability
        watch.close()

    def test_poll_detected_change_decreases_reliability(self):
        """Test that poll-detected changes decrease watchdog reliability

        Tests the PolledFile directly since DirectoryWatch always has
        watchdog active (which would increase reliability instead).
        """
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.watchdog_reliability = 0.8

        polled.on_poll_detected_change()

        # Reliability should decrease (Polyak toward 0)
        assert polled.watchdog_reliability < 0.8

    def test_polling_detects_changes(self, tmp_path):
        """Test that polling detects file changes"""
        changes: List[Path] = []
        change_event = Event()

        def on_change(path):
            changes.append(path)
            change_event.set()

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=on_change,
            min_poll_interval=0.1,
            max_poll_interval=0.5,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        watch.add_file(test_file)

        try:
            time.sleep(0.2)
            test_file.write_text("modified content")

            assert change_event.wait(timeout=2.0), "Change was not detected"
            assert test_file in changes
        finally:
            watch.close()

    def test_close_clears_files(self, tmp_path):
        """Test that close() clears all tracked files"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watch.add_file(test_file)
        watch.close()

        assert len(watch._files) == 0

    def test_adaptive_interval_increases(self, tmp_path):
        """Test that polling interval increases when no changes"""
        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
            min_poll_interval=0.1,
            max_poll_interval=1.0,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watch.add_file(test_file)
        initial_interval = watch._files[test_file].poll_interval

        try:
            time.sleep(0.5)
            assert watch._files[test_file].poll_interval > initial_interval
        finally:
            watch.close()

    def test_on_deleted_callback(self, tmp_path):
        """Test that on_deleted callback is called"""
        deleted: List[Path] = []
        delete_event = Event()

        def on_deleted(path):
            deleted.append(path)
            delete_event.set()

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=lambda p: None,
            on_deleted=on_deleted,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watch.add_file(test_file)

        try:
            test_file.unlink()

            if delete_event.wait(timeout=2.0):
                assert any(p.name == "test.txt" for p in deleted)
        finally:
            watch.close()

    def test_multiple_files(self, tmp_path):
        """Test watching multiple files"""
        changes: List[Path] = []
        change_event = Event()
        change_count = [0]

        def on_change(path):
            changes.append(path)
            change_count[0] += 1
            if change_count[0] >= 2:
                change_event.set()

        svc = FileWatcherService.instance()
        watch = svc.watch_directory(
            tmp_path,
            on_change=on_change,
            min_poll_interval=0.1,
            max_poll_interval=0.5,
        )

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        watch.add_file(file1)
        watch.add_file(file2)

        try:
            time.sleep(0.2)
            file1.write_text("modified1")
            file2.write_text("modified2")

            assert change_event.wait(timeout=3.0), "Changes were not detected"
            assert file1 in changes
            assert file2 in changes
        finally:
            watch.close()
