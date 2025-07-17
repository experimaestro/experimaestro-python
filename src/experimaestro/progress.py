"""File-based progress tracking system for experimaestro tasks."""

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Iterator, Dict, Any
from datetime import datetime, timedelta
import fcntl
import os

from .utils import logger

DEFAULT_MAX_ENTRIES_PER_FILE = 10_000


@dataclass
class ProgressEntry:
    """A single progress entry in the JSONL file"""

    timestamp: float
    level: int
    progress: float
    desc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressEntry":
        """Create from dictionary"""
        return cls(**data)


class StateFile:
    """Represents the state file for progress tracking.
    Checks if the state must be written based on time and progress changes.
    By default, it writes every second or when progress changes significantly (>1%)"""

    def __init__(self, filename: Path):
        self.filename = filename
        self.state: Dict[int, ProgressEntry] = {}

        # Write threshold to avoid too frequent writes
        self._time_threshold = timedelta(seconds=1.0)
        self._last_write_time: datetime = datetime.now()
        # Minimum progress change to trigger write
        self._progress_threshold = 0.01
        self._last_write_progress: Optional[Dict[int, float]] = None

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def _allow_write(self) -> bool:
        """Check if the state should be written based on time and progress changes.
        Allows writing if:
        - BOTH: More than 1 second has passed since last write
        - AND: Progress has changed significantly (>1%)
        - OR: All entries are done (progress >= 1.0)"""
        time_check = datetime.now() - self._last_write_time > self._time_threshold
        progress_check = self._last_write_progress is None or any(
            abs(entry.progress - self._last_write_progress.get(entry.level, 0.0))
            > self._progress_threshold
            for entry in self.state.values()
        )
        all_entries_done = all(entry.progress >= 1.0 for entry in self.state.values())
        return all_entries_done or (time_check and progress_check)

    def write(self, force: bool = False):
        """Write the current state to the file."""
        if self._allow_write() or force:
            with open(self.filename, "w") as f:
                json.dump({k: v.to_dict() for k, v in self.state.items()}, f)
            self._last_write_time = datetime.now()
            self._last_write_progress = {k: v.progress for k, v in self.state.items()}

    def update(self, entry: ProgressEntry):
        self.state[entry.level] = entry

    def load(self):
        """Load the state from the file"""
        if self.filename.exists():
            with self.filename.open("r") as f:
                try:
                    data = json.load(f)
                    self.state = {
                        int(k): ProgressEntry.from_dict(v) for k, v in data.items()
                    }
                except (json.JSONDecodeError, IOError):
                    logger.warning(f"Failed to load state from {self.filename}")

    def read(self) -> Dict[int, ProgressEntry]:
        """Read the state from the file"""
        self.load()
        return self.state

    # flush on exit
    def __del__(self):
        """Ensure state is written on exit"""
        try:
            self.write(force=True)
        except Exception as e:
            logger.error(f"Failed to write state on exit: {e}")


class ProgressFileWriter:
    # TODO: Implement buffering and flushing

    def __init__(
        self, task_path: Path, max_entries_per_file: int = DEFAULT_MAX_ENTRIES_PER_FILE
    ):
        self.task_path = task_path
        self.progress_dir = task_path / ".experimaestro"
        self.max_entries_per_file = max_entries_per_file
        self.current_file_index = 0
        self.current_file_entries = 0
        self.lock = threading.Lock()

        # Ensure directory exists
        self.progress_dir.mkdir(exist_ok=True)

        # State is the latest entry per level
        self.state = StateFile(self.progress_dir / "progress_state.json")

        # Find the latest file index
        self._find_latest_file()

    def _find_latest_file(self):
        """Find the latest progress file and entry count"""
        progress_files = list(self.progress_dir.glob("progress-*.jsonl"))
        if not progress_files:
            self.current_file_index = 0
            self.current_file_entries = 0
            return

        # Sort by file index
        max_index = None
        for f in progress_files:
            try:
                index = int(f.stem.split("-")[1])
                if max_index is None or index > max_index:
                    max_index = index
            except (ValueError, IndexError):
                continue

        if max_index is not None:
            self.current_file_index = max_index
            # Count entries in current file
            current_file = self._get_current_file_path()
            if current_file.exists():
                with current_file.open("r") as f:
                    self.current_file_entries = sum(1 for _ in f.readlines())
            else:
                self.current_file_entries = 0
        else:
            self.current_file_index = 0
            self.current_file_entries = 0

    def _get_current_file_path(self) -> Path:
        """Get path to current progress file"""
        return self.progress_dir / f"progress-{self.current_file_index:04d}.jsonl"

    def _get_latest_symlink_path(self) -> Path:
        """Get path to latest progress symlink"""
        return self.progress_dir / "progress-latest.jsonl"

    def _rotate_file_if_needed(self):
        """Create new file if current one is full"""
        if self.current_file_entries >= self.max_entries_per_file:
            self.current_file_index += 1
            self.current_file_entries = 0
            logger.debug(f"Rotating to new progress file: {self.current_file_index}")

    def _update_latest_symlink(self):
        """Update symlink to point to latest file"""
        current_file = self._get_current_file_path()
        latest_symlink = self._get_latest_symlink_path()

        # Remove existing symlink
        if latest_symlink.exists() or latest_symlink.is_symlink():
            latest_symlink.unlink()

        # Create new symlink
        latest_symlink.symlink_to(current_file.name)

    def write_progress(self, level: int, progress: float, desc: Optional[str] = None):
        """Write a progress entry to the file

        Args:
            level: Progress level (0 is top level)
            progress: Progress value between 0.0 and 1.0
            desc: Optional description
        """
        with self.lock:
            # Eventually rotate internal state if needed
            self._rotate_file_if_needed()

            entry = ProgressEntry(
                timestamp=time.time(), level=level, progress=progress, desc=desc
            )
            self.state.update(entry)
            self.state.write(force=level == -1)  # Force write on EOJ

            current_file = self._get_current_file_path()

            # Write with file locking for concurrent access
            with current_file.open("a") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(entry.to_dict()) + "\n")
                    f.flush()  # Flush the file buffer
                    os.fsync(f.fileno())  # Ensure data is written to disk
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self.current_file_entries += 1
            self._update_latest_symlink()

            logger.debug(
                f"Progress written: level={level}, progress={progress}, desc={desc}"
            )

    def __del__(self):
        """Ensure state is written on exit"""
        try:
            self.state.write(force=True)
        except Exception as e:
            logger.error(f"Failed to write state on exit: {e}")


class ProgressFileReader:
    """Reads progress entries from JSONL files"""

    def __init__(self, task_path: Path):
        """Initialize progress file reader

        Args:
            task_path: Path to the task directory
        """
        self.task_path = task_path
        self.progress_dir = task_path / ".experimaestro"
        self.max_entries_per_file: Optional[int] = None
        self.state = StateFile(self.progress_dir / "progress_state.json")

    def get_progress_files(self) -> List[Path]:
        """Get all progress files sorted by index"""
        if not self.progress_dir.exists():
            return []

        progress_files = list(self.progress_dir.glob("progress-*.jsonl"))

        # Filter out symlinks to avoid duplicates
        progress_files = [f for f in progress_files if not f.is_symlink()]

        # Sort by file index
        # Alternatively, we could simply sort by filename
        def get_index(path: Path) -> int:
            try:
                return int(path.stem.split("-")[1])
            except (ValueError, IndexError):
                return 0

        return sorted(progress_files, key=get_index)

    def get_latest_file(self) -> Optional[Path]:
        """Get the latest progress file via symlink"""
        latest_symlink = self.progress_dir / "progress-latest.jsonl"
        if latest_symlink.exists() and latest_symlink.is_symlink():
            return latest_symlink.resolve()

        # Fallback to finding latest manually
        files = self.get_progress_files()
        return files[-1] if files else None

    def read_entries(self, file_path: Path) -> Iterator[ProgressEntry]:
        """Read progress entries from a file

        Args:
            file_path: Path to progress file

        Yields:
            ProgressEntry objects
        """
        if not file_path.exists():
            return

        try:
            with file_path.open("r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                yield ProgressEntry.from_dict(data)
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Invalid JSON in progress file {file_path}: {e}"
                                )
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            logger.warning(f"Could not read progress file {file_path}: {e}")

    def read_all_entries(self) -> Iterator[ProgressEntry]:
        """Read all progress entries from all files in order

        Yields:
            ProgressEntry objects in chronological order
        """
        logger.warning("Reading all progress entries, this may be slow for large jobs.")
        for file_path in self.get_progress_files():
            yield from self.read_entries(file_path)

    def read_latest_entries(self, count: Optional[int] = None) -> List[ProgressEntry]:
        """Read the latest N progress entries"""
        entries = []

        # Read files in reverse order to get latest entries first
        files = self.get_progress_files()
        # Fetch the max length of files, in lines
        if files and count is None:
            # Fetch the number of entries in the first file
            # This is the most likely to be the longest file
            count = sum(1 for _ in self.read_entries(files[0]))
        if count is None:
            count = DEFAULT_MAX_ENTRIES_PER_FILE

        for file_path in reversed(files):
            file_entries = list(self.read_entries(file_path))
            entries.extend(reversed(file_entries))

            if len(entries) >= count:
                break

        # Return latest entries in chronological order
        return list(reversed(entries[:count]))

    def get_current_progress(
        self, count: Optional[int] = None
    ) -> Dict[int, ProgressEntry]:
        """Get the current progress for each level"""
        logger.warning(
            "Reading current progress from progress logs, this may be slow for large jobs."
        )
        return {entry.level: entry for entry in self.read_latest_entries(count)}

    def get_current_state(self) -> Optional[Dict[int, ProgressEntry]]:
        """Fetch the latest progress entry from the state file"""
        current_state = self.state.read()
        return current_state or self.get_current_progress()

    def is_done(self) -> bool:
        """Check if the task is done by looking for a special 'done' file.
        Fallback to checking for end-of-job (EOJ) entries."""

        task_name = self.task_path.parent.stem.split(".")[-1]
        job_done_file = self.task_path / f"{task_name}.done"
        if job_done_file.exists() and job_done_file.is_file():
            return True

        # Check if any progress file has a level -1 entry indicating EOJ
        return any(entry.level == -1 for entry in self.read_all_entries())


class FileBasedProgressReporter:
    """File-based progress reporter that replaces the socket-based Reporter"""

    def __init__(self, task_path: Path):
        """Initialize file-based progress reporter

        Args:
            task_path: Path to the task directory
        """
        self.task_path = task_path
        self.writer = ProgressFileWriter(task_path)
        self.current_progress = {}  # level -> (progress, desc)
        self.lock = threading.Lock()

    def set_progress(self, progress: float, level: int = 0, desc: Optional[str] = None):
        """Set progress for a specific level

        Args:
            progress: Progress value between 0.0 and 1.0
            level: Progress level (0 is top level)
            desc: Optional description
        """
        with self.lock:
            # Check if progress has changed significantly
            current = self.current_progress.get(level, (None, None))
            if (
                current[0] is None
                or abs(progress - current[0]) > 0.01
                or desc != current[1]
            ):
                self.current_progress[level] = (progress, desc)
                self.writer.write_progress(level, progress, desc)

    def eoj(self):
        """End of job notification"""
        with self.lock:
            # Write a special end-of-job marker
            self.writer.write_progress(-1, 1.0, "EOJ")
