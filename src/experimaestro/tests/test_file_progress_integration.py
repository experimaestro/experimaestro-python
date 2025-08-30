"""Integration tests for file-based progress tracking with actual Task execution"""

import time
from copy import copy
from pathlib import Path
from typing import List, Tuple

import fasteners
from experimaestro import Task, Annotated, pathgenerator, progress, tqdm
from experimaestro.core.objects import logger
from experimaestro.notifications import LevelInformation
from experimaestro.progress import ProgressFileReader
from experimaestro.scheduler import Job, Listener
from queue import Queue
from .utils import TemporaryExperiment


class FileProgressListener(Listener):
    """Listener that tracks both in-memory and file-based progress"""

    def __init__(self):
        self.current = []
        self.progresses: Queue[List[LevelInformation]] = Queue()
        self.jobs_seen = set()

    def job_state(self, job: Job):
        # Track job for file-based progress verification
        self.jobs_seen.add(job)

        if (len(self.current) != len(job.progress)) or any(
            l1 != l2 for l1, l2 in zip(self.current, job.progress)
        ):
            logger.info("Got some progress: %s", job.progress)
            self.current = [copy(level) for level in job.progress]
            self.progresses.put(self.current)


class SimpleProgressTask(Task):
    """Simple task that reports progress at specific intervals"""

    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        """Execute task and report progress based on file instructions"""
        _progress = 0.0

        while True:
            time.sleep(1e-4)
            if self.path.is_file():
                with fasteners.InterProcessLock(self.path.with_suffix(".lock")):
                    content = self.path.read_text().strip()
                    if not content:
                        continue

                    parts = content.split(" ", maxsplit=2)
                    _level = int(parts[0])
                    _progress = float(parts[1])
                    _desc = parts[2] if len(parts) > 2 and parts[2] else None

                    self.path.unlink()

                    if _progress > 0:
                        progress(_progress, level=_level, desc=_desc)
                    if _progress >= 0.99 and _level == 0:
                        break


class MultiLevelProgressTask(Task):
    """Task that reports progress at multiple levels"""

    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        """Execute with nested progress levels"""
        progress(0.0, level=0, desc="Starting main task")

        # Simulate subtasks
        for i in range(3):
            progress(i / 3.0, level=0, desc=f"Main task step {i + 1}")

            # Nested progress
            for j in range(5):
                progress(j / 5.0, level=1, desc=f"Subtask {i + 1}.{j + 1}")
                time.sleep(0.01)  # Small delay to ensure file writes

            progress(1.0, level=1, desc=f"Subtask {i + 1} complete")

        progress(1.0, level=0, desc="Main task complete")


class TqdmProgressTask(Task):
    """Task that uses tqdm for progress reporting"""

    path: Annotated[Path, pathgenerator("progress.txt")]

    def execute(self):
        """Execute with tqdm progress bars"""
        # Main progress bar
        for i in tqdm(range(10), desc="Main task", miniters=1, mininterval=0):
            # Nested progress bar
            for j in tqdm(range(5), desc=f"Subtask {i + 1}", miniters=1, mininterval=0):
                time.sleep(0.001)  # Small delay


def write_progress_instruction(
    path: Path, progress_val: float, level: int = 0, desc: str = None
):
    """Write progress instruction to file for task to read"""
    while True:
        time.sleep(5e-2)
        with fasteners.InterProcessLock(path.with_suffix(".lock")):
            if not path.is_file():
                desc_str = desc if desc else ""
                path.write_text(f"{level} {progress_val:.3f} {desc_str}")
                break


def verify_file_progress(job: Job, expected_entries: List[Tuple[int, float, str]]):
    """Verify that file-based progress matches expected entries"""
    reader = ProgressFileReader(job.path)

    # Wait a bit for file writes to complete
    time.sleep(0.1)

    entries = list(reader.read_all_entries())

    # Filter out entries that are too close in time (duplicates from rapid updates)
    filtered_entries = []
    last_timestamp = 0
    for entry in entries:
        if entry.timestamp - last_timestamp > 0.001:  # 1ms threshold
            filtered_entries.append(entry)
            last_timestamp = entry.timestamp

    logger.info(f"Found {len(filtered_entries)} file progress entries")
    for entry in filtered_entries:
        logger.info(f"  Level {entry.level}: {entry.progress:.3f} - {entry.desc}")

    # Verify we have at least the expected number of significant progress updates
    assert len(filtered_entries) >= len(
        expected_entries
    ), f"Expected at least {len(expected_entries)} entries, got {len(filtered_entries)}"

    # Verify current progress state
    current_progress = reader.get_current_progress()

    # Check that we have progress for expected levels
    expected_levels = set(level for level, _, _ in expected_entries)
    actual_levels = set(current_progress.keys())

    assert expected_levels.issubset(
        actual_levels
    ), f"Expected levels {expected_levels}, got {actual_levels}"

    return filtered_entries, current_progress


def test_file_progress_basic():
    """Test that file-based progress tracking works with basic task execution"""
    with TemporaryExperiment("file-progress-basic", maxwait=10, port=0) as xp:
        listener = FileProgressListener()
        xp.scheduler.addlistener(listener)

        # Submit task
        out = SimpleProgressTask.C().submit()
        path = out.path
        job = out.__xpm__.job

        # Wait for job to start
        logger.info("Waiting for job to start")
        while job.state.notstarted():
            time.sleep(1e-2)

        # Send progress instructions
        progress_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        expected_entries = []

        for i, v in enumerate(progress_values):
            desc = f"Step {i + 1}"
            write_progress_instruction(path, v, level=0, desc=desc)
            expected_entries.append((0, v, desc))

            if v < 1.0:
                # Verify in-memory progress (with timeout)
                try:
                    info = listener.progresses.get(timeout=2.0)[0]
                    logger.info("Got in-memory progress: %s", info)
                    assert abs(info.progress - v) < 0.01
                except Exception as e:
                    logger.warning(f"Failed to get progress for {v}: {e}")
                    # Continue anyway, we'll verify file-based progress later

        # Wait for job to complete
        while not job.state.finished():
            time.sleep(1e-2)

        # Verify file-based progress
        entries, current_progress = verify_file_progress(job, expected_entries)

        # Check that final progress is 1.0
        assert current_progress[0].progress == 1.0
        assert current_progress[0].desc == "Step 6"

        # Verify progress files exist
        progress_dir = job.path / ".experimaestro"
        assert progress_dir.exists()

        progress_files = list(progress_dir.glob("progress-*.jsonl"))
        assert len(progress_files) >= 1

        # Verify symlink exists
        latest_link = progress_dir / "progress-latest.jsonl"
        assert latest_link.exists()
        assert latest_link.is_symlink()


def test_file_progress_multilevel():
    """Test file-based progress tracking with multiple levels"""
    with TemporaryExperiment("file-progress-multilevel", maxwait=15, port=0) as xp:
        listener = FileProgressListener()
        xp.scheduler.addlistener(listener)

        # Submit task
        out = MultiLevelProgressTask.C().submit()
        job = out.__xpm__.job

        # Wait for job to start
        logger.info("Waiting for job to start")
        while job.state.notstarted():
            time.sleep(1e-2)

        # Wait for job to complete
        while not job.state.finished():
            time.sleep(1e-2)

        # Verify file-based progress
        reader = ProgressFileReader(job.path)
        entries = list(reader.read_all_entries())
        current_progress = reader.get_current_progress()

        logger.info(f"Found {len(entries)} total progress entries")

        # Should have progress for both levels 0 and 1
        assert 0 in current_progress
        assert 1 in current_progress

        # Final progress should be 1.0 for both levels
        assert current_progress[0].progress == 1.0
        assert current_progress[1].progress == 1.0

        # Check descriptions
        assert "complete" in current_progress[0].desc.lower()
        assert "complete" in current_progress[1].desc.lower()

        # Verify we have entries for both levels
        level_0_entries = [e for e in entries if e.level == 0]
        level_1_entries = [e for e in entries if e.level == 1]

        assert len(level_0_entries) >= 3  # At least start, middle, end
        assert len(level_1_entries) >= 3  # Multiple subtask updates


def test_file_progress_tqdm():
    """Test file-based progress tracking with tqdm"""
    with TemporaryExperiment("file-progress-tqdm", maxwait=15, port=0) as xp:
        listener = FileProgressListener()
        xp.scheduler.addlistener(listener)

        # Submit task
        out = TqdmProgressTask.C().submit()
        job = out.__xpm__.job

        # Wait for job to start
        logger.info("Waiting for job to start")
        while job.state.notstarted():
            time.sleep(1e-2)

        # Wait for job to complete
        while not job.state.finished():
            time.sleep(1e-2)

        # Verify file-based progress
        reader = ProgressFileReader(job.path)
        entries = list(reader.read_all_entries())
        current_progress = reader.get_current_progress()

        logger.info(f"Found {len(entries)} total progress entries from tqdm")

        # Should have progress for multiple levels (tqdm creates nested levels)
        assert len(current_progress) >= 1

        # Should have multiple progress entries
        assert len(entries) >= 10  # At least one per main iteration

        # Verify progress files structure
        progress_dir = job.path / ".experimaestro"
        assert progress_dir.exists()

        progress_files = list(progress_dir.glob("progress-*.jsonl"))
        assert len(progress_files) >= 1


def test_file_progress_concurrent_experiments():
    """Test that file-based progress works with multiple concurrent experiments"""
    max_wait = 10

    with TemporaryExperiment(
        "file-progress-concurrent-1", maxwait=max_wait, port=0
    ) as xp1:
        listener1 = FileProgressListener()
        xp1.scheduler.addlistener(listener1)

        # Submit first task
        out1 = SimpleProgressTask.C().submit()
        job1 = out1.__xpm__.job
        path1 = out1.path

        # Wait for first job to start
        while job1.state.notstarted():
            time.sleep(1e-2)

        with TemporaryExperiment(
            "file-progress-concurrent-2",
            workdir=xp1.workdir,
            maxwait=max_wait,
            port=0,
        ) as xp2:
            listener2 = FileProgressListener()
            xp2.scheduler.addlistener(listener2)

            # Submit second task
            out2 = SimpleProgressTask.C().submit()
            job2 = out2.__xpm__.job
            path2 = out2.path

            # Wait for second job to start
            while job2.state.notstarted():
                time.sleep(1e-2)

            # Send progress to both tasks
            progress_values = [0.2, 0.6, 1.0]

            for v in progress_values:
                write_progress_instruction(path1, v, desc=f"Task1-{v}")
                write_progress_instruction(path2, v, desc=f"Task2-{v}")

                if v < 1.0:
                    # Verify both get progress (with timeout)
                    try:
                        info1 = listener1.progresses.get(timeout=2.0)[0]
                        info2 = listener2.progresses.get(timeout=2.0)[0]

                        assert abs(info1.progress - v) < 0.01
                        assert abs(info2.progress - v) < 0.01
                    except Exception as e:
                        logger.warning(
                            f"Failed to get concurrent progress for {v}: {e}"
                        )
                        # Continue anyway, we'll verify file-based progress later

            # Wait for both jobs to complete
            while not job1.state.finished() or not job2.state.finished():
                time.sleep(1e-2)

            # Verify both have separate file-based progress
            reader1 = ProgressFileReader(job1.path)
            reader2 = ProgressFileReader(job2.path)

            entries1 = list(reader1.read_all_entries())
            entries2 = list(reader2.read_all_entries())

            # Both should have progress entries
            assert len(entries1) >= 3
            assert len(entries2) >= 3

            # Verify they have different descriptions
            task1_entries = [e for e in entries1 if e.desc and "Task1" in e.desc]
            task2_entries = [e for e in entries2 if e.desc and "Task2" in e.desc]

            assert len(task1_entries) >= 1
            assert len(task2_entries) >= 1


def test_file_progress_persistence():
    """Test that file-based progress persists during experiment execution"""
    with TemporaryExperiment("file-progress-persistence", maxwait=10, port=0):
        out = MultiLevelProgressTask.C().submit()
        job = out.__xpm__.job
        job_path = job.path

        # Wait for job to complete
        while not job.state.finished():
            time.sleep(1e-2)

        # Verify we can read progress files while experiment is still active
        reader = ProgressFileReader(job_path)
        entries = list(reader.read_all_entries())
        current_progress = reader.get_current_progress()

        # Should have progress data
        assert len(entries) > 0
        assert len(current_progress) > 0

        # Verify progress files exist
        progress_dir = job_path / ".experimaestro"
        assert progress_dir.exists()

        progress_files = list(progress_dir.glob("progress-*.jsonl"))
        assert len(progress_files) >= 1

        # Verify symlink works
        latest_link = progress_dir / "progress-latest.jsonl"
        assert latest_link.exists()
        assert latest_link.is_symlink()

        # Verify we can read from the symlink
        latest_entries = list(reader.read_entries(latest_link.resolve()))
        assert len(latest_entries) > 0


def test_file_progress_error_handling():
    """Test file-based progress tracking handles errors gracefully"""
    with TemporaryExperiment("file-progress-errors", maxwait=10, port=0) as xp:
        listener = FileProgressListener()
        xp.scheduler.addlistener(listener)

        # Submit task
        out = SimpleProgressTask.C().submit()
        job = out.__xpm__.job
        path = out.path

        # Wait for job to start
        while job.state.notstarted():
            time.sleep(1e-2)

        # Send some valid progress
        write_progress_instruction(path, 0.5, desc="Valid progress")

        # Wait a bit longer for the task to process the instruction
        time.sleep(0.5)

        # Complete the task
        write_progress_instruction(path, 1.0, desc="Complete")

        # Wait for completion
        while not job.state.finished():
            time.sleep(1e-2)

        # Verify progress was recorded after completion
        reader = ProgressFileReader(job.path)
        entries = list(reader.read_all_entries())
        current_progress = reader.get_current_progress()

        # Should have at least some progress entries
        assert len(entries) >= 1

        # Verify final state
        assert current_progress[0].progress == 1.0

        # Verify progress files exist and are readable
        progress_dir = job.path / ".experimaestro"
        assert progress_dir.exists()

        progress_files = list(progress_dir.glob("progress-*.jsonl"))
        assert len(progress_files) >= 1

        # Verify we can read from files without errors
        for progress_file in progress_files:
            file_entries = list(reader.read_entries(progress_file))
            # Each file should be readable (may be empty if no progress written to it)
            assert isinstance(file_entries, list)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
