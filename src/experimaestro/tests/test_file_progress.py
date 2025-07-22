"""Tests for the file-based progress tracking system"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from experimaestro.progress import (
    ProgressEntry,
    ProgressFileWriter,
    ProgressFileReader,
    FileBasedProgressReporter,
)


class TestProgressEntry:
    """Test ProgressEntry dataclass"""

    def test_to_dict(self):
        entry = ProgressEntry(
            timestamp=1234567890.0, level=1, progress=0.5, desc="Test description"
        )

        expected = {
            "timestamp": 1234567890.0,
            "level": 1,
            "progress": 0.5,
            "desc": "Test description",
        }

        assert entry.to_dict() == expected

    def test_from_dict(self):
        data = {
            "timestamp": 1234567890.0,
            "level": 1,
            "progress": 0.5,
            "desc": "Test description",
        }

        entry = ProgressEntry.from_dict(data)

        assert entry.timestamp == 1234567890.0
        assert entry.level == 1
        assert entry.progress == 0.5
        assert entry.desc == "Test description"

    def test_from_dict_minimal(self):
        data = {"timestamp": 1234567890.0, "level": 0, "progress": 1.0}

        entry = ProgressEntry.from_dict(data)

        assert entry.timestamp == 1234567890.0
        assert entry.level == 0
        assert entry.progress == 1.0
        assert entry.desc is None


class TestProgressFileWriter:
    """Test ProgressFileWriter class"""

    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)
            writer = ProgressFileWriter(task_path)

            assert writer.progress_dir.exists()
            assert writer.progress_dir == task_path / ".experimaestro"

    def test_write_single_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)
            writer = ProgressFileWriter(task_path)

            writer.write_progress(0, 0.5, "Test progress")

            # Check file was created
            progress_file = writer.progress_dir / "progress-0000.jsonl"
            assert progress_file.exists()

            # Check symlink was created
            latest_link = writer.progress_dir / "progress-latest.jsonl"
            assert latest_link.exists()
            assert latest_link.is_symlink()
            assert latest_link.resolve().name == progress_file.name

    def test_write_multiple_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)
            writer = ProgressFileWriter(task_path)

            # Write multiple progress entries
            writer.write_progress(0, 0.1, "Step 1")
            writer.write_progress(0, 0.5, "Step 2")
            writer.write_progress(1, 0.3, "Substep")
            writer.write_progress(0, 1.0, "Complete")

            progress_file = writer.progress_dir / "progress-0000.jsonl"
            assert progress_file.exists()

            # Read and verify entries
            lines = progress_file.read_text().strip().split("\n")
            assert len(lines) == 4

            # Check first entry
            entry1 = json.loads(lines[0])
            assert entry1["level"] == 0
            assert entry1["progress"] == 0.1
            assert entry1["desc"] == "Step 1"

    def test_file_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)
            # Set small max entries for testing rotation
            writer = ProgressFileWriter(task_path, max_entries_per_file=2)

            # Write 3 entries to trigger rotation
            writer.write_progress(0, 0.1, "Entry 1")
            writer.write_progress(0, 0.2, "Entry 2")
            writer.write_progress(0, 0.3, "Entry 3")  # Should trigger rotation

            # Check both files exist
            file1 = writer.progress_dir / "progress-0000.jsonl"
            file2 = writer.progress_dir / "progress-0001.jsonl"

            assert file1.exists()
            assert file2.exists()

            # Check file1 has 2 entries
            lines1 = file1.read_text().strip().split("\n")
            assert len(lines1) == 2

            # Check file2 has 1 entry
            lines2 = file2.read_text().strip().split("\n")
            assert len(lines2) == 1

            # Check symlink points to latest file
            latest_link = writer.progress_dir / "progress-latest.jsonl"
            assert latest_link.resolve().name == file2.name

    def test_resume_from_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Create first writer and write some entries
            writer1 = ProgressFileWriter(task_path, max_entries_per_file=2)
            writer1.write_progress(0, 0.1, "Entry 1")
            writer1.write_progress(0, 0.2, "Entry 2")

            # Create second writer (simulating restart)
            writer2 = ProgressFileWriter(task_path, max_entries_per_file=2)

            # Should resume from existing state
            assert writer2.current_file_index == 0
            assert writer2.current_file_entries == 2

            # Writing one more should trigger rotation
            writer2.write_progress(0, 0.3, "Entry 3")

            file2 = writer2.progress_dir / "progress-0001.jsonl"
            assert file2.exists()


class TestProgressFileReader:
    """Test ProgressFileReader class"""

    def test_read_entries_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Write some test data
            writer = ProgressFileWriter(task_path)
            writer.write_progress(0, 0.5, "Test")
            writer.write_progress(1, 0.3, "Nested")

            # Read it back
            reader = ProgressFileReader(task_path)
            progress_file = reader.get_progress_files()[0]
            entries = list(reader.read_entries(progress_file))

            assert len(entries) == 2
            assert entries[0].level == 0
            assert entries[0].progress == 0.5
            assert entries[0].desc == "Test"
            assert entries[1].level == 1
            assert entries[1].progress == 0.3
            assert entries[1].desc == "Nested"

    def test_read_all_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Write entries across multiple files
            writer = ProgressFileWriter(task_path, max_entries_per_file=2)
            writer.write_progress(0, 0.1, "Entry 1")
            writer.write_progress(0, 0.2, "Entry 2")
            writer.write_progress(0, 0.3, "Entry 3")  # Triggers rotation
            writer.write_progress(0, 0.4, "Entry 4")

            # Read all entries
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            assert len(entries) == 4
            assert entries[0].desc == "Entry 1"
            assert entries[1].desc == "Entry 2"
            assert entries[2].desc == "Entry 3"
            assert entries[3].desc == "Entry 4"

    def test_read_latest_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Write many entries
            writer = ProgressFileWriter(task_path, max_entries_per_file=3)
            for i in range(10):
                writer.write_progress(0, i / 10.0, f"Entry {i}")

            # Read latest 5 entries
            reader = ProgressFileReader(task_path)
            latest = reader.read_latest_entries(5)

            assert len(latest) == 5
            # Should be entries 5-9 in chronological order
            assert latest[0].desc == "Entry 5"
            assert latest[4].desc == "Entry 9"

    def test_get_current_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Write progress for multiple levels
            writer = ProgressFileWriter(task_path)
            writer.write_progress(0, 0.1, "Level 0 start")
            writer.write_progress(1, 0.5, "Level 1 progress")
            writer.write_progress(0, 0.5, "Level 0 update")
            writer.write_progress(1, 1.0, "Level 1 complete")
            writer.write_progress(0, 1.0, "Level 0 complete")

            # Get current progress
            reader = ProgressFileReader(task_path)
            current = reader.get_current_progress()

            assert len(current) == 2
            assert current[0].progress == 1.0
            assert current[0].desc == "Level 0 complete"
            assert current[1].progress == 1.0
            assert current[1].desc == "Level 1 complete"

    def test_get_latest_file_via_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Write some entries
            writer = ProgressFileWriter(task_path, max_entries_per_file=2)
            writer.write_progress(0, 0.1, "Entry 1")
            writer.write_progress(0, 0.2, "Entry 2")
            writer.write_progress(0, 0.3, "Entry 3")  # Triggers rotation

            # Get latest file
            reader = ProgressFileReader(task_path)
            latest_file = reader.get_latest_file()

            expected_file = task_path / ".experimaestro" / "progress-0001.jsonl"
            assert latest_file.name == expected_file.name

    def test_no_progress_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reader = ProgressFileReader(task_path)

            assert reader.get_progress_files() == []
            assert reader.get_latest_file() is None
            assert list(reader.read_all_entries()) == []
            assert reader.read_latest_entries(10) == []
            assert reader.get_current_progress() == {}


class TestFileBasedProgressReporter:
    """Test FileBasedProgressReporter class"""

    def test_set_progress_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reporter = FileBasedProgressReporter(task_path)
            reporter.set_progress(0.5, 0, "Test progress")

            # Verify file was written
            progress_file = task_path / ".experimaestro" / "progress-0000.jsonl"
            assert progress_file.exists()

            # Read and verify content
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            assert len(entries) == 1
            assert entries[0].level == 0
            assert entries[0].progress == 0.5
            assert entries[0].desc == "Test progress"

    def test_set_progress_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reporter = FileBasedProgressReporter(task_path)

            # First progress
            reporter.set_progress(0.5, 0, "Test")

            # Small change (should not write)
            reporter.set_progress(0.505, 0, "Test")

            # Larger change (should write)
            reporter.set_progress(0.6, 0, "Test")

            # Read entries
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            # Should only have 2 entries (first and third)
            assert len(entries) == 2
            assert entries[0].progress == 0.5
            assert entries[1].progress == 0.6

    def test_set_progress_description_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reporter = FileBasedProgressReporter(task_path)

            # Same progress, different description
            reporter.set_progress(0.5, 0, "Description 1")
            reporter.set_progress(0.5, 0, "Description 2")

            # Read entries
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            # Should have both entries due to description change
            assert len(entries) == 2
            assert entries[0].desc == "Description 1"
            assert entries[1].desc == "Description 2"

    def test_eoj_writes_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reporter = FileBasedProgressReporter(task_path)
            reporter.set_progress(1.0, 0, "Complete")
            reporter.eoj()

            # Read entries
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            assert len(entries) == 2
            assert entries[0].level == 0
            assert entries[0].progress == 1.0
            assert entries[1].level == -1  # EOJ marker
            assert entries[1].progress == 1.0
            assert entries[1].desc == "EOJ"

    def test_multiple_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            reporter = FileBasedProgressReporter(task_path)

            # Progress at different levels
            reporter.set_progress(0.1, 0, "Main task")
            reporter.set_progress(0.5, 1, "Subtask")
            reporter.set_progress(0.3, 2, "Sub-subtask")
            reporter.set_progress(0.5, 0, "Main task update")

            # Read current progress
            reader = ProgressFileReader(task_path)
            current = reader.get_current_progress()

            assert len(current) == 3
            assert current[0].progress == 0.5
            assert current[0].desc == "Main task update"
            assert current[1].progress == 0.5
            assert current[1].desc == "Subtask"
            assert current[2].progress == 0.3
            assert current[2].desc == "Sub-subtask"


class TestIntegrationWithNotifications:
    """Test integration with existing notification system"""

    @patch("experimaestro.taskglobals.Env.instance")
    def test_progress_function_writes_to_file(self, mock_env):
        """Test that the progress() function writes to file system"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir)

            # Mock the task environment
            mock_env.return_value.taskpath = task_path
            mock_env.return_value.slave = False

            # Import and call progress function
            from experimaestro.notifications import progress

            progress(0.5, level=0, desc="Test progress")

            # Verify file was written
            progress_file = task_path / ".experimaestro" / "progress-0000.jsonl"
            assert progress_file.exists()

            # Read and verify
            reader = ProgressFileReader(task_path)
            entries = list(reader.read_all_entries())

            assert len(entries) == 1
            assert entries[0].level == 0
            assert entries[0].progress == 0.5
            assert entries[0].desc == "Test progress"


if __name__ == "__main__":
    pytest.main([__file__])
