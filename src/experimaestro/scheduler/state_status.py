"""Filesystem-based state tracking for experiments

This module provides event and status file handling for tracking experiment state
without using a database. It replaces the SQLite/peewee-based state tracking.

Key components:
- Event dataclasses: Serializable events for JSONL event files
- EventWriter/EventReader: Base classes for event I/O
- JobEventWriter: Job-specific event handling
- ExperimentEventWriter/ExperimentEventReader: Experiment-specific event handling

File structure:
- workspace/.events/experiments/{experiment-id}/events-{count}.jsonl
- workspace/.events/jobs/{task-id}/event-{job-id}-{count}.jsonl
- workspace/experiments/{experiment-id}/{run-id}/status.json
- workspace/jobs/{task-id}/{job-id}/.experimaestro/information.json
"""

import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseExperiment

logger = logging.getLogger("xpm.state_status")

# Status file version
STATUS_VERSION = 1


# =============================================================================
# Hardlink Support Utilities
# =============================================================================


def supports_hardlinks(path: Path) -> bool:
    """Check if the filesystem at path supports hardlinks

    Creates temporary test files to verify hardlink support. Useful for
    determining whether to use hardlinks for event file archiving.

    Args:
        path: Directory to test for hardlink support

    Returns:
        True if hardlinks are supported, False otherwise
    """
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".hardlink_test"
    test_link = path / ".hardlink_test_link"
    try:
        # Clean up any leftover test files
        if test_link.exists():
            test_link.unlink()
        if test_file.exists():
            test_file.unlink()

        # Create test file and hardlink
        test_file.touch()
        os.link(test_file, test_link)

        # Verify it's actually a hardlink (same inode)
        success = test_file.stat().st_ino == test_link.stat().st_ino

        # Clean up
        test_link.unlink()
        test_file.unlink()
        return success
    except (OSError, AttributeError):
        # Clean up on failure
        try:
            if test_link.exists():
                test_link.unlink()
            if test_file.exists():
                test_file.unlink()
        except OSError:
            pass
        return False


def safe_link_or_copy(src: Path, dst: Path, use_hardlinks: bool = True) -> bool:
    """Create hardlink if supported, otherwise copy the file

    Args:
        src: Source file path
        dst: Destination file path
        use_hardlinks: If True, attempt hardlink first; if False, always copy

    Returns:
        True if hardlink was used, False if file was copied
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if use_hardlinks:
        try:
            # Remove destination if it exists (can't hardlink over existing file)
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return True
        except OSError:
            pass

    # Fall back to copy
    shutil.copy2(src, dst)
    return False


# =============================================================================
# Event System
# =============================================================================

# Registry for event deserialization (auto-populated by __init_subclass__)
EVENT_TYPES: dict[str, type["EventBase"]] = {}


@dataclass
class EventBase:
    """Base class for all events

    Events are lightweight - they carry only IDs, not object references.
    Use StateProvider to fetch actual objects (BaseJob, BaseExperiment, etc.)
    when needed.

    Subclasses are automatically registered in EVENT_TYPES by their class name.
    JSON serialization/deserialization is handled transparently via event_type field.
    """

    timestamp: float = field(default_factory=time.time)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register by class name
        EVENT_TYPES[cls.__name__] = cls

    @property
    def event_type(self) -> str:
        """Event type derived from class name"""
        return self.__class__.__name__

    def to_json(self) -> str:
        """Serialize event to JSON string"""
        d = asdict(self)
        d["event_type"] = self.event_type
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "EventBase":
        """Deserialize event from dictionary"""
        event_type = d.get("event_type")
        event_class = EVENT_TYPES.get(event_type, EventBase)
        # Filter to only known fields for the event class
        valid_fields = {f for f in event_class.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return event_class(**filtered)

    @classmethod
    def get_class(cls, name: str) -> "type[EventBase] | None":
        """Get an EventBase subclass by class name"""
        return EVENT_TYPES.get(name)


# -----------------------------------------------------------------------------
# Event Base Classes (for filtering)
# -----------------------------------------------------------------------------


@dataclass
class JobEventBase(EventBase):
    """Base class for job-related events (have job_id)"""

    job_id: str = ""


@dataclass
class ExperimentEventBase(EventBase):
    """Base class for experiment-related events (have experiment_id)"""

    experiment_id: str = ""


@dataclass
class ServiceEventBase(ExperimentEventBase):
    """Base class for service-related events (have service_id)"""

    service_id: str = ""


# -----------------------------------------------------------------------------
# Supporting Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ProgressLevel:
    """Progress information for a single level"""

    level: int = 0
    progress: float = 0.0
    desc: Optional[str] = None

    def to_dict(self) -> dict:
        return {"level": self.level, "progress": self.progress, "desc": self.desc}

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressLevel":
        return cls(
            level=d.get("level", 0),
            progress=d.get("progress", 0.0),
            desc=d.get("desc"),
        )


@dataclass
class JobTag:
    """A job tag (key-value pair)"""

    key: str
    value: str


# -----------------------------------------------------------------------------
# Job Events
# -----------------------------------------------------------------------------


@dataclass
class JobSubmittedEvent(JobEventBase, ExperimentEventBase):
    """Event: Job was submitted to the scheduler

    Fired when a job is added to an experiment run.
    This is both a job event and an experiment event.
    """

    task_id: str = ""
    run_id: str = ""
    transient: int = 0
    tags: list[JobTag] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class JobStateChangedEvent(JobEventBase):
    """Event: Job state changed

    Fired when a job's state changes (scheduled, running, done, error, etc.)
    Timestamps are stored as ISO format strings for JSON serialization.
    """

    state: str = ""
    failure_reason: Optional[str] = None
    submitted_time: Optional[str] = None  # ISO format timestamp
    started_time: Optional[str] = None  # ISO format timestamp
    ended_time: Optional[str] = None  # ISO format timestamp
    exit_code: Optional[int] = None
    retry_count: int = 0
    progress: list[ProgressLevel] = field(default_factory=list)


@dataclass
class JobProgressEvent(JobEventBase):
    """Event: Job progress update

    Written by the running job process to report progress.
    """

    level: int = 0
    progress: float = 0.0
    desc: Optional[str] = None


# -----------------------------------------------------------------------------
# Experiment Events
# -----------------------------------------------------------------------------


@dataclass
class ExperimentUpdatedEvent(ExperimentEventBase):
    """Event: Experiment was created or updated"""

    pass


@dataclass
class RunUpdatedEvent(ExperimentEventBase):
    """Event: Experiment run was created or updated"""

    run_id: str = ""


@dataclass
class RunCompletedEvent(ExperimentEventBase):
    """Event: Experiment run completed"""

    run_id: str = ""
    status: str = "completed"
    ended_at: str = ""


# -----------------------------------------------------------------------------
# Service Events
# -----------------------------------------------------------------------------


@dataclass
class ServiceAddedEvent(ServiceEventBase):
    """Event: Service was added to the experiment"""

    run_id: str = ""
    description: str = ""
    service_class: str = ""
    state_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStateChangedEvent(ServiceEventBase):
    """Event: Service state changed (STOPPED, STARTING, RUNNING, STOPPING)"""

    run_id: str = ""
    state: str = ""


# =============================================================================
# Event Writer Classes
# =============================================================================


class EventWriter(ABC):
    """Base class for writing events to JSONL files

    Events are written to {events_dir}/events-{count}.jsonl
    Uses line buffering so each event is flushed immediately after write.

    Supports automatic file rotation when files exceed MAX_EVENT_FILE_SIZE.
    When rotating, subclasses should override _on_rotate() to update status
    files before the new event file is created.

    Supports proactive hardlinking: when a permanent_dir is set and hardlinks
    are supported, a hardlink is created to permanent storage immediately when
    the event file is opened. This ensures events are written to both locations
    simultaneously and no data is lost if the process crashes.
    """

    # Maximum events per file before rotation (~100 bytes/event, targeting ~128KB)
    MAX_EVENTS_PER_FILE = 1280

    def __init__(self, initial_count: int = 0, permanent_dir: Path | None = None):
        """Initialize event writer

        Args:
            initial_count: Starting event file count for rotation
            permanent_dir: Optional permanent storage directory for archiving.
                If set and hardlinks are supported, events are written to both
                temporary and permanent locations via hardlink.
        """
        self._count = initial_count
        self._file = None
        self._permanent_dir = permanent_dir
        self._events_in_current_file = 0
        self._use_hardlinks: bool | None = None  # None = not checked yet
        self._hardlink_created_for_count: int | None = (
            None  # Track which file has hardlink
        )

    @property
    @abstractmethod
    def events_dir(self) -> Path:
        """Get the directory where events are written"""
        ...

    @property
    def permanent_dir(self) -> Path | None:
        """Get the permanent storage directory"""
        return self._permanent_dir

    def _check_hardlink_support(self) -> bool:
        """Check and cache hardlink support"""
        if self._use_hardlinks is None:
            if self._permanent_dir:
                self._permanent_dir.mkdir(parents=True, exist_ok=True)
                self._use_hardlinks = supports_hardlinks(self._permanent_dir)
            else:
                self._use_hardlinks = False
        return self._use_hardlinks

    def _get_event_file_path(self) -> Path:
        return self.events_dir / f"events-{self._count}.jsonl"

    def _get_permanent_event_file_path(self) -> Path:
        """Get path for permanent event file"""
        if self._permanent_dir is None:
            raise ValueError("No permanent directory configured")
        return self._permanent_dir / f"event-{self._count}.jsonl"

    def write_event(self, event: EventBase) -> None:
        """Write an event to the current event file

        If permanent storage is configured and hardlinks are supported,
        creates a hardlink immediately when the file is first opened.
        """
        if self._file is None:
            self.events_dir.mkdir(parents=True, exist_ok=True)
            # Use line buffering (buffering=1) so each line is flushed automatically
            self._file = self._get_event_file_path().open("a", buffering=1)

            # Create hardlink to permanent storage immediately if supported
            if self._check_hardlink_support() and self._permanent_dir:
                temp_path = self._get_event_file_path()
                perm_path = self._get_permanent_event_file_path()
                try:
                    perm_path.parent.mkdir(parents=True, exist_ok=True)
                    if not perm_path.exists():
                        os.link(temp_path, perm_path)
                        self._hardlink_created_for_count = self._count
                        logger.debug("Created hardlink %s -> %s", temp_path, perm_path)
                except FileExistsError:
                    pass  # Already linked
                except OSError as e:
                    logger.warning("Failed to create hardlink: %s", e)

        self._file.write(event.to_json() + "\n")
        self._events_in_current_file += 1

        # Check if rotation is needed
        if self._events_in_current_file >= self.MAX_EVENTS_PER_FILE:
            new_count = self._count + 1
            # Call hook to update status file before rotation
            self._on_rotate(new_count)
            # Then rotate the file
            self.rotate(new_count)

    def _on_rotate(self, new_count: int) -> None:
        """Hook called before rotation - subclasses override to update status

        This is called BEFORE the new event file is created, allowing subclasses
        to update status files (with the new events_count) first.

        Args:
            new_count: The new event file count after rotation
        """
        pass  # Base implementation does nothing

    def flush(self) -> None:
        """Flush the current event file to disk"""
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the current event file"""
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

    def rotate(self, new_count: int) -> None:
        """Rotate to a new event file (called after status file update)"""
        self.close()
        self._count = new_count
        self._events_in_current_file = 0

    def cleanup(self) -> None:
        """Delete all event files in this directory (temporary files only)"""
        self.close()
        for i in range(self._count + 1):
            path = self.events_dir / f"events-{i}.jsonl"
            if path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete event file %s: %s", path, e)

    def archive_events(self) -> None:
        """Archive events to permanent storage (called on completion)

        For each temp file:
        - If permanent file exists (hardlink already created): just delete temp
        - If permanent file doesn't exist: move temp to permanent
        """
        self.close()

        if not self._permanent_dir:
            return

        self._permanent_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self._count + 1):
            temp_path = self._get_temp_event_file_path(i)
            perm_path = self._permanent_dir / f"event-{i}.jsonl"

            if not temp_path.exists():
                continue

            if perm_path.exists():
                # Permanent file exists (hardlink) - just delete temp
                try:
                    temp_path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete temp file %s: %s", temp_path, e)
            else:
                # No permanent file - move temp to permanent
                try:
                    shutil.move(str(temp_path), str(perm_path))
                except OSError as e:
                    logger.warning("Failed to archive %s: %s", temp_path, e)

    def _get_temp_event_file_path(self, count: int) -> Path:
        """Get path for temporary event file at a specific count"""
        return self.events_dir / f"events-{count}.jsonl"


class JobEventWriter(EventWriter):
    """Writes events to job event files

    Events are stored in: {workspace}/.events/jobs/{task_id}/event-{job_id}-{count}.jsonl
    Permanent storage: {job_path}/.experimaestro/events/event-{count}.jsonl
    """

    def __init__(
        self,
        workspace_path: Path,
        task_id: str,
        job_id: str,
        initial_count: int = 0,
        job_path: Path | None = None,
    ):
        """Initialize job event writer

        Args:
            workspace_path: Path to workspace directory
            task_id: Task identifier (groups events by task type)
            job_id: Job identifier (unique hash for this job instance)
            initial_count: Starting event file count for rotation
            job_path: Optional job directory path for permanent storage
        """
        # Permanent storage: job_path/.experimaestro/events/
        permanent_dir = job_path / ".experimaestro" / "events" if job_path else None
        super().__init__(initial_count, permanent_dir)
        self.workspace_path = workspace_path
        self.task_id = task_id
        self.job_id = job_id
        self.job_path = job_path
        self._events_dir = workspace_path / ".events" / "jobs" / task_id

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    def _get_event_file_path(self) -> Path:
        """Get the path for job event file with job_id in filename"""
        return self.events_dir / f"event-{self.job_id}-{self._count}.jsonl"

    def _get_temp_event_file_path(self, count: int) -> Path:
        """Get path for temporary event file at a specific count"""
        return self.events_dir / f"event-{self.job_id}-{count}.jsonl"


class ExperimentEventWriter(EventWriter):
    """Writes events to experiment event files

    Events are stored in: {workspace}/.events/experiments/{experiment_id}/events-{count}.jsonl
    Permanent storage: {run_dir}/events/event-{count}.jsonl
    """

    def __init__(
        self,
        experiment: "BaseExperiment",
        workspace_path: Path,
        initial_count: int = 0,
    ):
        """Initialize experiment event writer

        Args:
            experiment: The experiment (BaseExperiment) to write events for
            workspace_path: Path to workspace directory
            initial_count: Starting event file count for rotation
        """
        from experimaestro.scheduler.interfaces import BaseExperiment

        assert isinstance(experiment, BaseExperiment), (
            f"experiment must be a BaseExperiment, got {type(experiment)}"
        )
        # Permanent storage: run_dir/events/
        run_dir = experiment.run_dir
        permanent_dir = run_dir / "events" if run_dir else None
        super().__init__(initial_count, permanent_dir)
        self.experiment = experiment
        self.workspace_path = workspace_path
        self._events_dir = (
            workspace_path / ".events" / "experiments" / experiment.experiment_id
        )

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def run_dir(self) -> Path | None:
        return self.experiment.run_dir

    def init_status(self) -> None:
        """Initialize status.json for a new run

        Uses the experiment's write_status() method to write the initial status.
        """
        # Ensure run directory exists
        run_dir = self.experiment.run_dir
        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Write initial status using experiment's write_status method
            self.experiment.write_status()

    def create_symlink(self) -> None:
        """Create/update symlink to current run directory

        The symlink is created at:
        .events/experiments/{experiment_id}/current -> run_dir
        """
        run_dir = self.experiment.run_dir
        if run_dir is None:
            return

        # Ensure the experiment events directory exists
        self._events_dir.mkdir(parents=True, exist_ok=True)

        # Handle legacy: if experiment_id path is a symlink (old format), remove it
        # Check both old .experimaestro and current .events paths
        for events_base in [".experimaestro", ".events"]:
            experiments_dir = self.workspace_path / events_base / "experiments"
            old_symlink = experiments_dir / self.experiment_id
            if old_symlink.is_symlink():
                old_symlink.unlink()

        # Create symlink inside the experiment directory
        symlink = self._events_dir / "current"

        # Compute relative path from symlink location to run_dir
        try:
            rel_path = os.path.relpath(run_dir, self._events_dir)
        except ValueError:
            # On Windows, relpath fails for paths on different drives
            rel_path = str(run_dir)

        # Remove existing symlink if present
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()

        symlink.symlink_to(rel_path)

    def _on_rotate(self, new_count: int) -> None:
        """Update experiment status with new events_count before rotation

        This ensures the status file is updated BEFORE the new event file is
        created, so listeners reading the status know which file to read.

        Note: We update self._count first so experiment.events_count returns
        the new value when write_status() serializes it.
        """
        # Update count first (experiment.events_count delegates to us)
        self._count = new_count
        self.experiment.write_status()


# =============================================================================
# Event Reader Class
# =============================================================================

# Callback types for event watching
# (entity_id, event) - entity_id is job_id or experiment_id
EntityEventCallback = Callable[[str, EventBase], None]
# (entity_id,) - called when event files are deleted
EntityDeletedCallback = Callable[[str], None]
# Extracts entity_id from path
EntityIdExtractor = Callable[[Path], Optional[str]]


def default_entity_id_extractor(path: Path) -> Optional[str]:
    """Default: entity_id is the parent directory name

    Path format: {base_dir}/{entity_id}/events-{count}.jsonl
    """
    return path.parent.name


def job_entity_id_extractor(path: Path) -> Optional[str]:
    """For jobs: entity_id (job_id) is extracted from the filename

    Path format: {base_dir}/{task_id}/event-{job_id}-{count}.jsonl
    """
    # Job ID is extracted from the filename
    # e.g., .events/jobs/my.task/event-abc123-0.jsonl -> abc123
    name = path.name
    if not name.startswith("event-"):
        return None
    # Remove "event-" prefix and ".jsonl" suffix
    # Format: event-{job_id}-{count}.jsonl
    rest = name[6:]  # Remove "event-"
    if rest.endswith(".jsonl"):
        rest = rest[:-6]  # Remove ".jsonl"
    # Now rest is "{job_id}-{count}", split on last "-" to get job_id
    parts = rest.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0]
    return None


# Resolver type: given an entity_id, returns the permanent storage path
PermanentStorageResolver = Callable[[str], Path]


@dataclass
class WatchedDirectory:
    """Configuration for a directory to watch for events

    Attributes:
        path: Temporary events directory (.events/...)
        entity_id_extractor: Function to extract entity ID from event file path
        glob_pattern: Pattern for matching event files
        permanent_storage_resolver: Optional function that returns permanent storage
            path for an entity. Used for hardlink archiving and deletion recovery.
    """

    path: Path
    entity_id_extractor: EntityIdExtractor = field(
        default_factory=lambda: default_entity_id_extractor
    )
    glob_pattern: str = "*/events-*.jsonl"
    # For archiving and deletion handling:
    permanent_storage_resolver: PermanentStorageResolver | None = None


class EventReader:
    """Generic reader for events from JSONL files

    Watches multiple directories with configurable entity ID extraction.

    Supports:
    - One-shot reading: read_events_since_count()
    - Incremental reading: read_new_events() - tracks file positions
    - File watching: start_watching(), stop_watching() - uses watchdog
    - Buffered mode: buffer events during initialization, flush after
    """

    def __init__(self, directories: list[WatchedDirectory]):
        """Initialize event reader

        Args:
            directories: List of directories to watch with their configurations
        """
        self.directories = directories
        # For incremental reading (live monitoring)
        self._file_positions: dict[Path, int] = {}
        # For file watching (using ipcom)
        self._watches: list[Any] = []
        self._handler: Any = None
        self._event_callbacks: list[EntityEventCallback] = []
        self._deleted_callbacks: list[EntityDeletedCallback] = []
        # Buffering mode: queue events instead of forwarding immediately
        self._buffering = False
        self._event_buffer: list[tuple[str, EventBase]] = []
        self._deleted_buffer: list[str] = []
        # Cache for events_count from status.json (entity_dir -> count)
        self._events_count_cache: dict[Path, int] = {}

    def _extract_entity_id(self, path: Path) -> Optional[str]:
        """Extract entity ID from event file path"""
        dir_config = self._find_dir_config(path)
        if dir_config:
            return dir_config.entity_id_extractor(path)
        return None

    def _find_dir_config(self, path: Path) -> WatchedDirectory | None:
        """Find the WatchedDirectory config that contains the given path"""
        for dir_config in self.directories:
            try:
                path.relative_to(dir_config.path)
                return dir_config
            except ValueError:
                continue
        return None

    def _get_entity_events_count(self, entity_dir: Path) -> int:
        """Get events_count from status.json in the entity directory

        Returns the events_count from status.json, or 1 if not found.
        Caches the result and updates when status.json is modified.
        """
        status_path = entity_dir / "status.json"

        # Check if status.json exists
        if not status_path.exists():
            return 1

        # Read status.json for events_count
        try:
            with status_path.open("r") as f:
                status = json.load(f)
                events_count = status.get("events_count", 1)
                self._events_count_cache[entity_dir] = events_count
                return events_count
        except (OSError, json.JSONDecodeError):
            # Fall back to cached value or default
            return self._events_count_cache.get(entity_dir, 1)

    def get_all_event_files(self) -> list[Path]:
        """Get all event files across all directories, sorted by modification time"""
        all_files = []
        for dir_config in self.directories:
            if dir_config.path.exists():
                all_files.extend(dir_config.path.glob(dir_config.glob_pattern))
        return sorted(
            all_files,
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )

    def scan_existing_files(self) -> None:
        """Scan for existing event files and set initial positions to end of file

        Call this before start_watching() to skip existing events and only
        receive new ones.
        """
        for path in self.get_all_event_files():
            try:
                self._file_positions[path] = path.stat().st_size
            except OSError:
                pass

    def read_new_events(self) -> list[tuple[str, EventBase]]:
        """Read new events since last call (incremental reading)

        Returns:
            List of (entity_id, event) tuples
        """
        results = []
        for event_file in self.get_all_event_files():
            entity_id = self._extract_entity_id(event_file)
            if not entity_id:
                continue

            last_pos = self._file_positions.get(event_file, 0)
            try:
                with event_file.open("r") as f:
                    f.seek(last_pos)
                    # Use readline() instead of iterator to allow tell()
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event_dict = json.loads(line)
                            event = EventBase.from_dict(event_dict)
                            results.append((entity_id, event))
                        except json.JSONDecodeError:
                            pass
                    self._file_positions[event_file] = f.tell()
            except OSError:
                pass
        return results

    def start_watching(
        self,
        on_event: Optional[EntityEventCallback] = None,
        on_deleted: Optional[EntityDeletedCallback] = None,
    ) -> None:
        """Start watching for file changes using ipcom

        Args:
            on_event: Callback called with (entity_id, event) for each new event
            on_deleted: Callback called with (entity_id,) when event files are deleted
        """
        from watchdog.events import FileSystemEventHandler

        from experimaestro.ipc import ipcom

        if on_event:
            self._event_callbacks.append(on_event)
        if on_deleted:
            self._deleted_callbacks.append(on_deleted)

        if self._watches:
            return  # Already watching

        # Create event handler
        reader = self

        class EventFileHandler(FileSystemEventHandler):
            def _is_event_file(self, path: Path) -> bool:
                """Check if path is an event file (experiment or job)"""
                # Experiment events: events-{count}.jsonl
                # Job events: event-{job_id}-{count}.jsonl
                return path.suffix == ".jsonl" and path.name.startswith(
                    ("events-", "event-")
                )

            def on_modified(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if self._is_event_file(path):
                    logger.debug("Detected modification of event file: %s", path)
                    reader._process_file_change(path)

            def on_created(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if self._is_event_file(path):
                    logger.debug("Detected creation of event file: %s", path)
                    reader._file_positions[path] = 0
                    reader._process_file_change(path)

            def on_deleted(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if self._is_event_file(path):
                    logger.debug("Detected deletion of event file: %s", path)
                    entity_id = reader._extract_entity_id(path)
                    dir_config = reader._find_dir_config(path)
                    reader._file_positions.pop(path, None)
                    if entity_id:
                        reader._handle_deletion(entity_id, path, dir_config)

        self._handler = EventFileHandler()
        ipc = ipcom()

        # Register watches for each directory
        for dir_config in self.directories:
            dir_config.path.mkdir(parents=True, exist_ok=True)
            watch = ipc.fswatch(self._handler, dir_config.path, recursive=True)
            self._watches.append(watch)
            logger.debug("Started watching %s", dir_config.path)

    def stop_watching(self) -> None:
        """Stop watching for file changes"""
        from experimaestro.ipc import ipcom

        ipc = ipcom()
        for watch in self._watches:
            ipc.fsunwatch(watch)
        self._watches.clear()
        self._handler = None
        logger.debug("Stopped watching all directories")

    def _extract_file_number(self, path: Path) -> int | None:
        """Extract the file number from event file name.

        E.g., "events-2.jsonl" -> 2, "events-10.jsonl" -> 10
        Returns None if the file name doesn't match the expected pattern.
        """
        import re

        name = path.name
        # Match events-{number}.jsonl pattern
        match = re.match(r"events-(\d+)\.jsonl$", name)
        if match:
            return int(match.group(1))
        return None

    def _process_file_change(self, path: Path) -> None:
        """Process a changed event file and notify callbacks (or buffer)

        IMPORTANT: When processing a file like events-N.jsonl, we first ensure
        all earlier files (events-{min_count}.jsonl through events-(N-1).jsonl)
        have been fully read. This guarantees event ordering even when file
        system notifications arrive out of order.

        Files numbered below the entity's events_count (from status.json) are
        skipped, as they have already been processed in a previous run.
        """
        entity_id = self._extract_entity_id(path)
        if not entity_id:
            return

        entity_dir = path.parent
        file_number = self._extract_file_number(path)

        # Get the minimum event file number to process
        min_count = self._get_entity_events_count(entity_dir)

        # Skip files below the events_count threshold
        if file_number is not None and file_number < min_count:
            logger.debug(
                "Skipping %s (file_number=%d < events_count=%d)",
                path,
                file_number,
                min_count,
            )
            return

        # Process all earlier files first (from min_count onwards) to maintain ordering
        if file_number is not None and file_number > min_count:
            for earlier_num in range(min_count, file_number):
                earlier_path = entity_dir / f"events-{earlier_num}.jsonl"
                if earlier_path.exists():
                    self._process_single_file(earlier_path, entity_id)

        self._process_single_file(path, entity_id)

    def _process_single_file(self, path: Path, entity_id: str) -> None:
        """Process a single event file and notify callbacks (or buffer)"""
        last_pos = self._file_positions.get(path, 0)
        try:
            with path.open("r") as f:
                f.seek(last_pos)
                # Use readline() instead of iterator to allow tell()
                while True:
                    line = f.readline()
                    if not line:
                        break

                    # Skip incomplete lines (writer may be mid-write)
                    if not line.endswith("\n"):
                        break

                    line = line.strip()
                    if not line:
                        # Update position for empty lines
                        last_pos = f.tell()
                        continue
                    try:
                        event_dict = json.loads(line)
                        event = EventBase.from_dict(event_dict)
                        if self._buffering:
                            # Queue event for later
                            self._event_buffer.append((entity_id, event))
                        else:
                            # Forward immediately
                            for callback in self._event_callbacks:
                                try:
                                    callback(entity_id, event)
                                except Exception:
                                    logger.exception("Error in event callback")
                    except json.JSONDecodeError:
                        pass

                    # Update position after each complete line
                    last_pos = f.tell()

                self._file_positions[path] = last_pos
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Failed to read event file %s: %s", path, e)

    def _handle_deletion(
        self,
        entity_id: str,
        _deleted_path: Path | None = None,
        dir_config: WatchedDirectory | None = None,
    ) -> None:
        """Handle entity deletion - read from permanent storage if available

        When event files are deleted from the temporary (.events/) directory,
        this method attempts to read any remaining events from permanent storage
        (if configured via permanent_storage_resolver).

        Args:
            entity_id: The entity identifier (job_id or experiment_id)
            deleted_path: The path of the deleted file (optional)
            dir_config: The WatchedDirectory config for this path (optional)
        """
        # Try to read remaining events from permanent storage
        if dir_config and dir_config.permanent_storage_resolver:
            permanent_dir = dir_config.permanent_storage_resolver(entity_id)

            # Read any events from permanent storage that we haven't processed
            events = self._read_events_from_permanent(permanent_dir, entity_id)
            for event in events:
                if self._buffering:
                    self._event_buffer.append((entity_id, event))
                else:
                    for callback in self._event_callbacks:
                        try:
                            callback(entity_id, event)
                        except Exception:
                            logger.exception("Error in event callback")

        # Notify deletion callbacks
        if self._buffering:
            self._deleted_buffer.append(entity_id)
        else:
            for callback in self._deleted_callbacks:
                try:
                    callback(entity_id)
                except Exception:
                    logger.exception("Error in deleted callback")

    def _read_events_from_permanent(
        self, permanent_dir: Path, entity_id: str
    ) -> list[EventBase]:
        """Read events from permanent storage directory

        Args:
            permanent_dir: Path to permanent storage directory
            entity_id: Entity identifier (for logging)

        Returns:
            List of events read from permanent storage
        """
        events = []
        if not permanent_dir.exists():
            return events

        # Read all event files in permanent storage
        event_files = sorted(permanent_dir.glob("event-*.jsonl"))
        for event_file in event_files:
            try:
                with event_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event_dict = json.loads(line)
                            event = EventBase.from_dict(event_dict)
                            events.append(event)
                        except json.JSONDecodeError:
                            pass
            except OSError as e:
                logger.warning(
                    "Failed to read permanent event file %s: %s", event_file, e
                )

        if events:
            logger.debug(
                "Read %d events from permanent storage for entity %s",
                len(events),
                entity_id,
            )
        return events

    def clear_callbacks(self) -> None:
        """Clear all registered callbacks"""
        self._event_callbacks.clear()
        self._deleted_callbacks.clear()

    def start_buffering(self) -> None:
        """Start buffering events instead of forwarding to callbacks

        Call this before scan_existing_files() to ensure events arriving
        during initialization are queued and not lost.
        """
        self._buffering = True
        self._event_buffer.clear()
        self._deleted_buffer.clear()

    def flush_buffer(self) -> None:
        """Stop buffering and forward all buffered events to callbacks

        Call this after initial state loading is complete.
        """
        self._buffering = False

        # Forward buffered events
        for entity_id, event in self._event_buffer:
            for callback in self._event_callbacks:
                try:
                    callback(entity_id, event)
                except Exception:
                    logger.exception("Error in event callback")

        # Forward buffered deletions
        for entity_id in self._deleted_buffer:
            for callback in self._deleted_callbacks:
                try:
                    callback(entity_id)
                except Exception:
                    logger.exception("Error in deleted callback")

        self._event_buffer.clear()
        self._deleted_buffer.clear()

    def read_events_since_count(
        self, entity_id: str, start_count: int, base_dir: Optional[Path] = None
    ) -> list[EventBase]:
        """Read events for an entity starting from a specific file count

        Args:
            entity_id: Entity identifier (job_id or experiment_id)
            start_count: File count to start reading from (events-{count}.jsonl)
            base_dir: Optional base directory to search in (defaults to first directory)

        Returns:
            List of events from files starting at start_count
        """
        events = []

        # Determine which directory to search
        if base_dir is None and self.directories:
            base_dir = self.directories[0].path

        if base_dir is None:
            return events

        entity_events_dir = base_dir / entity_id
        count = start_count
        while True:
            event_path = entity_events_dir / f"events-{count}.jsonl"
            if not event_path.exists():
                break

            try:
                with event_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event_dict = json.loads(line)
                                event = EventBase.from_dict(event_dict)
                                events.append(event)
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass
            count += 1
        return events


class JobProgressReader:
    """Convenience reader for job progress events

    Reads progress events from a job's event files.
    Used for monitoring job progress via CLI and scheduler.
    """

    def __init__(self, job_path: Path):
        """Initialize job progress reader

        Args:
            job_path: Path to the job directory (workspace/jobs/task_id/job_id)
        """
        self.job_path = job_path
        # Extract task_id and job_id from path
        self.job_id = job_path.name
        self.task_id = job_path.parent.name
        # Progress events are stored in workspace/.events/jobs/{task_id}/
        self._events_dir = (
            job_path.parent.parent.parent / ".events" / "jobs" / self.task_id
        )

    def get_event_files(self) -> list[Path]:
        """Get all event files for this job"""
        if not self._events_dir.exists():
            return []
        return sorted(self._events_dir.glob("events-*.jsonl"))

    def read_progress_events(self) -> list[JobProgressEvent]:
        """Read all progress events from event files

        Returns:
            List of JobProgressEvent objects in chronological order
        """
        events = []
        for event_file in self.get_event_files():
            try:
                with event_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("type") == "job_progress":
                                events.append(
                                    JobProgressEvent(
                                        job_id=data.get("job_id", ""),
                                        level=data.get("level", 0),
                                        progress=data.get("progress", 0.0),
                                        desc=data.get("desc"),
                                        timestamp=data.get("timestamp", 0.0),
                                    )
                                )
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass
        return events

    def get_current_progress(self) -> dict[int, JobProgressEvent]:
        """Get current progress state per level

        Returns:
            Dict mapping level to the latest JobProgressEvent for that level
        """
        progress: dict[int, JobProgressEvent] = {}
        for event in self.read_progress_events():
            progress[event.level] = event
        return progress

    def is_done(self) -> bool:
        """Check if job is complete (JobStateChangedEvent with done/error state)

        Returns:
            True if job state is "done" or "error" in event files
        """
        for event_file in self.get_event_files():
            try:
                with event_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # Check for job state changed event with final state
                            if data.get("type") == "job_state_changed":
                                state = data.get("state", "")
                                if state in ("done", "error"):
                                    return True
                            # Also check for old EOJ marker for backward compatibility
                            if (
                                data.get("type") == "job_progress"
                                and data.get("level") == -1
                            ):
                                return True
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass
        return False


__all__ = [
    # Hardlink utilities
    "supports_hardlinks",
    "safe_link_or_copy",
    # Event classes
    "EventBase",
    "JobSubmittedEvent",
    "JobStateChangedEvent",
    "JobProgressEvent",
    "ServiceAddedEvent",
    "ServiceStateChangedEvent",
    "RunCompletedEvent",
    "EVENT_TYPES",
    # Event writer classes
    "EventWriter",
    "JobEventWriter",
    "ExperimentEventWriter",
    # Event reader class
    "EventReader",
    "WatchedDirectory",
    "PermanentStorageResolver",
    "job_entity_id_extractor",
    "JobProgressReader",
    # Callback types
    "EntityEventCallback",
    "EntityDeletedCallback",
]
