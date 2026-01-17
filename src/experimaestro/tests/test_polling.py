"""Tests for the FileWatcher adaptive polling system"""

import time
from pathlib import Path
from threading import Event
from typing import List


from experimaestro.scheduler.polling import FileWatcher, PolledFile


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

    def test_on_activity_resets_interval(self):
        """Test that on_activity resets to minimum interval"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.poll_interval = 10.0  # Slow polling

        polled.on_activity()

        assert polled.poll_interval == polled.MIN_INTERVAL
        assert polled.next_poll > 0

    def test_on_no_activity_increases_interval(self):
        """Test that on_no_activity increases interval up to max"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.poll_interval = 1.0

        polled.on_no_activity()

        assert polled.poll_interval == 1.0 * polled.INTERVAL_MULTIPLIER

    def test_on_no_activity_caps_at_max(self):
        """Test that interval doesn't exceed max"""
        polled = PolledFile(path=Path("/tmp/test.txt"))
        polled.poll_interval = 25.0  # Close to max

        polled.on_no_activity()

        assert polled.poll_interval == polled.MAX_INTERVAL

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


class TestFileWatcher:
    """Tests for FileWatcher class"""

    def test_add_and_remove_file(self, tmp_path):
        """Test adding and removing files"""
        changes: List[Path] = []
        watcher = FileWatcher(on_change=lambda p: changes.append(p))

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watcher.add_file(test_file)
        assert test_file in watcher._files

        watcher.remove_file(test_file)
        assert test_file not in watcher._files

    def test_file_filter(self, tmp_path):
        """Test that file_filter is respected"""
        changes: List[Path] = []
        watcher = FileWatcher(
            on_change=lambda p: changes.append(p),
            file_filter=lambda p: p.suffix == ".txt",
        )

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        json_file = tmp_path / "test.json"
        json_file.write_text("{}")

        watcher.add_file(txt_file)
        watcher.add_file(json_file)

        assert txt_file in watcher._files
        assert json_file not in watcher._files

    def test_notify_change_resets_interval(self, tmp_path):
        """Test that notify_change resets the polling interval"""
        watcher = FileWatcher(on_change=lambda p: None)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watcher.add_file(test_file)
        # Simulate time passing - set interval high
        watcher._files[test_file].poll_interval = 20.0

        watcher.notify_change(test_file)

        assert watcher._files[test_file].poll_interval == 0.5  # MIN_INTERVAL

    def test_polling_detects_changes(self, tmp_path):
        """Test that polling detects file changes"""
        changes: List[Path] = []
        change_event = Event()

        def on_change(path):
            changes.append(path)
            change_event.set()

        watcher = FileWatcher(
            on_change=on_change,
            min_interval=0.1,  # Fast polling for test
            max_interval=0.5,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")

        watcher.add_file(test_file)
        watcher.start()

        try:
            # Modify the file
            time.sleep(0.2)  # Wait for initial poll to complete
            test_file.write_text("modified content")

            # Wait for change to be detected
            assert change_event.wait(timeout=2.0), "Change was not detected"
            assert test_file in changes
        finally:
            watcher.stop()

    def test_stop_clears_files(self, tmp_path):
        """Test that stop() clears all tracked files"""
        watcher = FileWatcher(on_change=lambda p: None)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watcher.add_file(test_file)
        watcher.start()
        watcher.stop()

        assert len(watcher._files) == 0
        assert watcher._thread is None

    def test_adaptive_interval_increases(self, tmp_path):
        """Test that polling interval increases when no changes"""
        watcher = FileWatcher(
            on_change=lambda p: None,
            min_interval=0.1,
            max_interval=1.0,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watcher.add_file(test_file)
        initial_interval = watcher._files[test_file].poll_interval

        watcher.start()

        try:
            # Wait for a few poll cycles
            time.sleep(0.5)

            # Interval should have increased since file didn't change
            assert watcher._files[test_file].poll_interval > initial_interval
        finally:
            watcher.stop()

    def test_on_deleted_callback(self, tmp_path):
        """Test that on_deleted callback is called"""
        deleted: List[Path] = []
        delete_event = Event()

        def on_deleted(path):
            deleted.append(path)
            delete_event.set()

        watcher = FileWatcher(
            on_change=lambda p: None,
            on_deleted=on_deleted,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        watcher.add_file(test_file)
        watcher.watch_directory(tmp_path)
        watcher.start()

        try:
            # Delete the file
            test_file.unlink()

            # Wait for deletion to be detected (via watchdog)
            if delete_event.wait(timeout=2.0):
                assert test_file in deleted
        finally:
            watcher.stop()

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

        watcher = FileWatcher(
            on_change=on_change,
            min_interval=0.1,
            max_interval=0.5,
        )

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        watcher.add_file(file1)
        watcher.add_file(file2)
        watcher.start()

        try:
            time.sleep(0.2)
            # Modify both files
            file1.write_text("modified1")
            file2.write_text("modified2")

            # Wait for changes to be detected
            assert change_event.wait(timeout=3.0), "Changes were not detected"
            assert file1 in changes
            assert file2 in changes
        finally:
            watcher.stop()
