"""Filesystem-based state tracking for experiments

This module provides event and status file handling for tracking experiment state
without using a database. It replaces the SQLite/peewee-based state tracking.

Key components:
- Event dataclasses: Serializable events for JSONL event files
- EventWriter/EventReader: Base classes for event I/O
- JobEventWriter: Job-specific event handling
- ExperimentEventWriter/ExperimentEventReader: Experiment-specific event handling

File structure:
- workspace/.events/experiments/{experiment-id}-{count}.jsonl  (flat)
- workspace/.events/jobs/{hash8}-{job-id}-{count}.jsonl  (flat, hash8 = SHA-256(task-id)[:8])
- workspace/experiments/{experiment-id}/{run-id}/status.json
- workspace/jobs/{task-id}/{job-id}/.experimaestro/information.json
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from experimaestro.filewatcher import DirectoryWatch, FileWatcherService


if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseExperiment
    from experimaestro.scheduler.state_provider import MockJob

logger = logging.getLogger("xpm.state_status")

# Status file version
STATUS_VERSION = 1


def task_id_hash(task_id: str) -> str:
    """Return the first 8 characters of the SHA-256 hash of a task_id.

    Used to create flat event file names: {hash8}-{job_id}-{count}.jsonl
    """
    return hashlib.sha256(task_id.encode()).hexdigest()[:8]


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
    """Base class for job-related events (have job_id and task_id)"""

    job_id: str = ""
    task_id: str = ""


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

    run_id: str = ""
    tags: list[JobTag] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    submitted_time: Optional[str] = None  # ISO format timestamp


@dataclass
class JobStateChangedEvent(JobEventBase):
    """Event: Job execution state changed (from job process events / disk)

    Fired when a job's execution state changes (scheduled, running, done, error, etc.)
    Timestamps are stored as ISO format strings for JSON serialization.
    Only written to job event files (.events/jobs/), never to experiment event files.
    """

    state: str = ""
    failure_reason: Optional[str] = None
    started_time: Optional[str] = None  # ISO format timestamp
    ended_time: Optional[str] = None  # ISO format timestamp
    exit_code: Optional[int] = None
    retry_count: int = 0
    progress: list[ProgressLevel] = field(default_factory=list)


@dataclass
class ExperimentJobStateEvent(ExperimentEventBase, JobEventBase):
    """Event: Scheduler lifecycle state changed for a job in an experiment

    Fired by the scheduler when a job's scheduler_state changes
    (UNSCHEDULED → WAITING → READY → SCHEDULED → RUNNING → DONE → ERROR).
    This is the experiment/scheduler view of the job lifecycle, distinct from
    the execution state events (JobStateChangedEvent) written by the job process.

    Supersedes JobSubmittedEvent: the initial event for a job includes tags,
    depends_on, and submitted_time fields.

    Used by TUI and WebUI to track scheduler-level lifecycle transitions.
    """

    scheduler_state: str = ""
    failure_reason: Optional[str] = None
    run_id: str = ""
    tags: list[JobTag] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    submitted_time: Optional[str] = None  # ISO format timestamp


@dataclass
class JobProgressEvent(JobEventBase):
    """Event: Job progress update

    Written by the running job process to report progress.
    """

    level: int = 0
    progress: float = 0.0
    desc: Optional[str] = None


@dataclass
class CarbonMetricsEvent(JobEventBase):
    """Event: Carbon metrics update during job execution.

    Written periodically by the running job process to report environmental
    impact metrics (CO2 emissions, energy consumption).
    """

    co2_kg: float = 0.0
    """Cumulative CO2 equivalent emissions in kilograms."""

    energy_kwh: float = 0.0
    """Cumulative energy consumed in kilowatt-hours."""

    cpu_power_w: float = 0.0
    """Average CPU power consumption in watts."""

    gpu_power_w: float = 0.0
    """Average GPU power consumption in watts."""

    ram_power_w: float = 0.0
    """Average RAM power consumption in watts."""

    duration_s: float = 0.0
    """Duration of tracking in seconds."""

    region: str = ""
    """Region/country code used for carbon intensity."""

    is_final: bool = False
    """True if this is the final measurement (on job completion)."""

    written: bool = False
    """True if the carbon record was successfully written to CarbonStorage."""

    run_group_id: str = ""
    """Run group ID linking metrics from the same retry sequence."""


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


@dataclass
class WarningEvent(ExperimentEventBase):
    """Event: Generic warning with user actions

    This event is emitted when something needs user attention/action.
    The TUI can display a dialog with the description and action buttons.
    When user clicks an action, it calls state_provider.execute_warning_action(key, action_key).
    """

    run_id: str = ""
    warning_key: str = ""  # Unique key for this warning (e.g., "stale_locks_gpu_token")
    description: str = ""  # Human-readable description for the UI
    actions: dict[str, str] = field(default_factory=dict)  # action_key -> button_label
    context: dict[str, Any] = field(default_factory=dict)  # Additional data for UI
    severity: str = "warning"  # "info", "warning", "error"


@dataclass
class ErrorEvent(ExperimentEventBase):
    """Event: Error occurred during warning action execution

    This event is emitted when execute_warning_action() fails.
    """

    run_id: str = ""
    warning_key: str = ""  # Key of the warning that caused the error
    action_key: str = ""  # Action that failed
    error_message: str = ""  # Error description


# =============================================================================
# Event Writer Classes
# =============================================================================


class EventWriter(ABC):
    """Base class for writing events to JSONL files

    Events are written to {events_dir}/events-{count}.jsonl
    Uses line buffering so each event is flushed immediately after write.

    Supports automatic file rotation when files exceed MAX_EVENT_FILE_SIZE.
    When rotating, the status file is updated with the new events_count.

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
        self._lock = threading.RLock()

    @property
    @abstractmethod
    def events_dir(self) -> Path:
        """Get the directory where events are written"""
        ...

    @property
    def status_path(self) -> Path | None:
        """Get the path to the status file, or None if not applicable

        Subclasses should override this to return the path to their status file.
        """
        return None

    def _write_events_count_to_status(self) -> None:
        """Write current events_count to status file

        Called when the first event file is opened and on rotation.
        This ensures the status file always reflects the current event file number.
        """
        status_path = self.status_path
        if status_path is None or not status_path.exists():
            return

        try:
            with status_path.open("r") as f:
                status = json.load(f)
            status["events_count"] = self._count
            with status_path.open("w") as f:
                json.dump(status, f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to update status with events_count: %s", e)

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
        """Write an event to the current event file (thread-safe)

        If permanent storage is configured and hardlinks are supported,
        creates a hardlink immediately when the file is first opened.
        """
        with self._lock:
            if self._file is None:
                self.events_dir.mkdir(parents=True, exist_ok=True)
                # Use line buffering (buffering=1) so each line is flushed
                self._file = self._get_event_file_path().open("a", buffering=1)

                # Write events_count to status file when opening first event file
                self._write_events_count_to_status()

                # Create hardlink to permanent storage immediately if supported
                if self._check_hardlink_support() and self._permanent_dir:
                    temp_path = self._get_event_file_path()
                    perm_path = self._get_permanent_event_file_path()
                    try:
                        perm_path.parent.mkdir(parents=True, exist_ok=True)
                        if not perm_path.exists():
                            os.link(temp_path, perm_path)
                            self._hardlink_created_for_count = self._count
                            logger.debug(
                                "Created hardlink %s -> %s", temp_path, perm_path
                            )
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
        """Update status file with new events_count before rotation

        This is called BEFORE the new event file is created, ensuring the status
        file reflects the new count before any events are written to the new file.

        Note: We update self._count first so any property that delegates to
        self._count (e.g., experiment.events_count) returns the new value
        when the status is written.

        Args:
            new_count: The new event file count after rotation
        """
        # Update count first (properties may delegate to self._count)
        self._count = new_count
        self._write_events_count_to_status()

    def flush(self) -> None:
        """Flush the current event file to disk (thread-safe)"""
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the current event file (thread-safe)"""
        with self._lock:
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

    Events are stored in: {workspace}/.events/jobs/{hash8}-{job_id}-{count}.jsonl
    Permanent storage: {job_path}/.experimaestro/events/event-{count}.jsonl

    Use JobEventWriter.get() to obtain a cached instance — this ensures that
    all threads (progress, carbon reporting) share the same writer and see
    consistent file rotation state.
    """

    # Class-level cache: (workspace_path, job_id) -> writer instance
    _cache: dict[tuple[str, str], "JobEventWriter"] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def get(
        cls,
        workspace_path: Path,
        task_id: str,
        job_id: str,
        initial_count: int = 0,
        job_path: Path | None = None,
    ) -> "JobEventWriter":
        """Get or create a cached JobEventWriter for the given job.

        Returns the same instance for the same (workspace_path, job_id),
        ensuring all threads share rotation state.
        """
        key = (str(workspace_path), job_id)
        with cls._cache_lock:
            writer = cls._cache.get(key)
            if writer is None:
                writer = cls(workspace_path, task_id, job_id, initial_count, job_path)
                cls._cache[key] = writer
            return writer

    @classmethod
    def clear_cache(cls, workspace_path: Path | None = None, job_id: str | None = None):
        """Remove a writer from the cache (e.g., on job completion).

        If both workspace_path and job_id are given, removes that specific entry.
        If neither is given, clears the entire cache.
        """
        with cls._cache_lock:
            if workspace_path is not None and job_id is not None:
                cls._cache.pop((str(workspace_path), job_id), None)
            elif workspace_path is None and job_id is None:
                cls._cache.clear()

    def __init__(
        self,
        workspace_path: Path,
        task_id: str,
        job_id: str,
        initial_count: int = 0,
        job_path: Path | None = None,
    ):
        """Initialize job event writer

        Prefer using JobEventWriter.get() to obtain a cached instance.

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
        self._events_dir = workspace_path / ".events" / "jobs"

        # Maintain a MockJob loaded from status.json so that on rotation
        # we can write back a fully up-to-date state (progress, carbon, etc.)
        self._mock_job = self._create_mock_job()

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    @property
    def status_path(self) -> Path | None:
        """Get the path to the job's status file"""
        if self.job_path is None:
            return None
        return self.job_path / ".experimaestro" / "status.json"

    def _create_mock_job(self) -> "MockJob":
        """Load or create a MockJob from the job's status.json.

        The mock job mirrors the on-disk state and is kept up-to-date
        by applying every event that flows through this writer.
        """
        from experimaestro.scheduler.state_provider import MockJob

        status_path = self.status_path
        if status_path is not None and status_path.exists():
            try:
                return MockJob.from_disk(self.job_path, self.task_id, self.job_id)
            except Exception:
                logger.debug("Failed to load MockJob from disk, creating empty")

        return MockJob(
            identifier=self.job_id,
            task_id=self.task_id,
            path=self.job_path,
            state="unscheduled",
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )

    def write_event(self, event: EventBase) -> None:
        """Write event and apply it to the in-memory mock job.

        Ensures task_id and job_id are set on JobEventBase events
        so that readers can resolve the real task_id from events.
        """
        if isinstance(event, JobEventBase):
            if not event.task_id:
                event.task_id = self.task_id
            if not event.job_id:
                event.job_id = self.job_id
        self._mock_job.apply_event(event)
        super().write_event(event)

    def _write_events_count_to_status(self) -> None:
        """Write the full job state (from mock job) to status.json on rotation.

        Instead of just patching events_count into the existing file,
        we write the mock job's complete state_dict(). This ensures that
        progress, carbon metrics, and any other event-derived state is
        persisted so that monitors skipping earlier event files still see
        the correct state.
        """
        status_path = self.status_path
        if status_path is None or not status_path.exists():
            return

        try:
            state = self._mock_job.state_dict()
            state["events_count"] = self._count
            with status_path.open("w") as f:
                json.dump(state, f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to write job state to status: %s", e)

    def _get_event_file_path(self) -> Path:
        """Get the path for job event file with hash8 prefix in filename"""
        h = task_id_hash(self.task_id)
        return self.events_dir / f"{h}-{self.job_id}-{self._count}.jsonl"

    def _get_temp_event_file_path(self, count: int) -> Path:
        """Get path for temporary event file at a specific count"""
        h = task_id_hash(self.task_id)
        return self.events_dir / f"{h}-{self.job_id}-{count}.jsonl"


class ExperimentEventWriter(EventWriter):
    """Writes events to experiment event files

    Events are stored in: {workspace}/.events/experiments/{experiment_id}-{count}.jsonl
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
        self._events_dir = workspace_path / ".events" / "experiments"

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def run_dir(self) -> Path | None:
        return self.experiment.run_dir

    @property
    def status_path(self) -> Path | None:
        """Get the path to the experiment's status file"""
        run_dir = self.experiment.run_dir
        if run_dir is None:
            return None
        return run_dir / "status.json"

    def _get_event_file_path(self) -> Path:
        """Flat format: {events_dir}/{experiment_id}-{count}.jsonl"""
        return self.events_dir / f"{self.experiment_id}-{self._count}.jsonl"

    def _get_temp_event_file_path(self, count: int) -> Path:
        """Flat format: {events_dir}/{experiment_id}-{count}.jsonl"""
        return self.events_dir / f"{self.experiment_id}-{count}.jsonl"

    def cleanup(self) -> None:
        """Delete all flat event files for this experiment"""
        self.close()
        for i in range(self._count + 1):
            path = self.events_dir / f"{self.experiment_id}-{i}.jsonl"
            if path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete event file %s: %s", path, e)

    def _write_events_count_to_status(self) -> None:
        """Write the full experiment state to status.json on rotation.

        Instead of just patching events_count, write the experiment's complete
        state_dict(). This ensures services, job counts, and other event-derived
        state are persisted so monitors loading from status.json see current state.
        """
        status_path = self.status_path
        if status_path is None or not status_path.exists():
            return

        try:
            state = self.experiment.state_dict()
            state["events_count"] = self._count
            state["last_updated"] = datetime.now().isoformat()
            with status_path.open("w") as f:
                json.dump(state, f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to write experiment state to status: %s", e)

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
        experiments/{experiment_id}/current -> run_dir

        This is placed in the main experiments directory (not .events/) to
        avoid the file watcher following the symlink into job permanent
        storage directories.
        """
        run_dir = self.experiment.run_dir
        if run_dir is None:
            return

        # Ensure the events base directory exists (flat: no per-experiment subdir)
        self._events_dir.mkdir(parents=True, exist_ok=True)

        # Handle legacy: remove old symlinks from .experimaestro and .events
        for events_base in [".experimaestro", ".events"]:
            experiments_dir = self.workspace_path / events_base / "experiments"
            # Old format: symlink was the experiment_id directory itself
            old_symlink = experiments_dir / self.experiment_id
            if old_symlink.is_symlink():
                old_symlink.unlink()
            # Previous format: current symlink inside .events experiment dir
            old_current = experiments_dir / self.experiment_id / "current"
            if old_current.is_symlink():
                old_current.unlink()

        # Create symlink in the main experiment directory
        exp_dir = self.workspace_path / "experiments" / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        symlink = exp_dir / "current"

        # Compute relative path from symlink location to run_dir
        try:
            rel_path = os.path.relpath(run_dir, exp_dir)
        except ValueError:
            # On Windows, relpath fails for paths on different drives
            rel_path = str(run_dir)

        # Remove existing symlink if present
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()

        symlink.symlink_to(rel_path)


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


# Regex for new flat format: {hash8}-{job_id}-{count}.jsonl
# hash8 = 8 hex chars, job_id = typically 64 hex chars, count = digits
# Uses non-greedy match + last dash before digits to handle job_ids with dashes
_JOB_EVENT_FLAT_RE = re.compile(r"([0-9a-f]{8})-(.+)-(\d+)\.jsonl$")

# Regex for old nested format: event-{job_id}-{count}.jsonl
_JOB_EVENT_OLD_RE = re.compile(r"event-(.+)-(\d+)\.jsonl$")


def job_entity_id_extractor(path: Path) -> str | None:
    """For jobs: entity_id is "{hash8}:{job_id}" extracted from path

    New flat format: {base_dir}/{hash8}-{job_id}-{count}.jsonl
    Old nested format: {base_dir}/{task_id}/event-{job_id}-{count}.jsonl

    Returns: "{hash8}:{job_id}"
    """
    name = path.name

    # Try new flat format first
    m = _JOB_EVENT_FLAT_RE.match(name)
    if m:
        return f"{m.group(1)}:{m.group(2)}"

    # Backwards compat: old nested format
    m = _JOB_EVENT_OLD_RE.match(name)
    if m:
        parent_task_id = path.parent.name
        h = task_id_hash(parent_task_id)
        return f"{h}:{m.group(1)}"

    return None


# Regex for flat experiment event format: {experiment_id}-{count}.jsonl
# experiment_id can contain any characters except the trailing -{digits}.jsonl
_EXPERIMENT_EVENT_FLAT_RE = re.compile(r"(.+)-(\d+)\.jsonl$")


def experiment_entity_id_extractor(path: Path) -> str | None:
    """Extract experiment_id from flat event file path.

    New flat format: {base_dir}/{experiment_id}-{count}.jsonl
    Old subdir format: {base_dir}/{experiment_id}/events-{count}.jsonl
    """
    name = path.name

    # Old subdir format: events-{count}.jsonl inside a subdirectory
    if name.startswith("events-"):
        return path.parent.name

    # New flat format: {experiment_id}-{count}.jsonl
    m = _EXPERIMENT_EVENT_FLAT_RE.match(name)
    if m:
        return m.group(1)

    return None


# Resolver type: given an entity_id, returns the permanent storage path
PermanentStorageResolver = Callable[[str], Path]

# Resolver type: given an entity_id, returns the starting events_count
EventsCountResolver = Callable[[str], int]


@dataclass
class WatchedDirectory:
    """Configuration for a directory to watch for events

    Attributes:
        path: Temporary events directory (.events/...)
        entity_id_extractor: Function to extract entity ID from event file path
        glob_pattern: Pattern for matching event files
        permanent_storage_resolver: Optional function that returns permanent storage
            path for an entity. Used for hardlink archiving and deletion recovery.
        events_count_resolver: Optional function that returns the starting
            events_count for an entity. When None, defaults to 0 (process all files).
            Replaces reading events_count from status.json in the entity directory.
        on_created: Callback when a new entity is discovered. Receives (entity_id, events)
            where events is the list of historical events read during catch-up. Returns
            True to follow this entity's events, False to ignore. If None, all entities
            are followed.
        on_event: Callback for each event (entity_id, event).
        on_deleted: Callback when entity's event files are deleted.
    """

    path: Path
    entity_id_extractor: EntityIdExtractor = field(
        default_factory=lambda: default_entity_id_extractor
    )
    glob_pattern: str = "*/events-*.jsonl"
    # For archiving and deletion handling:
    permanent_storage_resolver: PermanentStorageResolver | None = None
    # For resolving starting events_count per entity:
    events_count_resolver: EventsCountResolver | None = None
    # Callbacks for entity lifecycle
    on_created: Callable[[str, list[EventBase]], bool] | None = None
    on_event: Callable[[str, EventBase], None] | None = None
    on_deleted: Callable[[str], None] | None = None


class EventReader:
    """Generic reader for events from JSONL files

    Watches multiple directories with configurable entity ID extraction.

    Supports:
    - One-shot reading: read_events_since_count()
    - Incremental reading: read_new_events() - tracks file positions
    - File watching: start_watching(), stop_watching() - uses watchdog
    - Entity tracking: on_created callback determines which entities to follow
    """

    def __init__(self, *directories: WatchedDirectory, max_open_files: int = 128):
        """Initialize event reader

        Args:
            directories: Directories to watch with their configurations
            max_open_files: Maximum open FDs for file tailing
        """
        self.directories = list(directories)
        self._max_open_files = max_open_files
        # Initial file positions (staging area, transferred to watches on start)
        self._file_positions: dict[Path, int] = {}
        # File watchers (with tailing support)
        self._file_watcher: DirectoryWatch | None = None
        # Set of entity IDs being followed (entity_id -> dir_config)
        self._followed_entities: dict[str, WatchedDirectory] = {}
        # Per-entity tracking: entity_id -> path of the file currently being tailed
        self._current_file: dict[str, Path] = {}
        # All watches (for finding the right one per path)
        self._all_watchers: list[DirectoryWatch] = []

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

    def _get_entity_events_count(
        self, entity_id: str, dir_config: WatchedDirectory
    ) -> int:
        """Get starting events_count for an entity via the resolver callback.

        Returns the events_count from the resolver, or 0 if no resolver is set.
        """
        if dir_config.events_count_resolver is not None:
            try:
                return dir_config.events_count_resolver(entity_id)
            except Exception:
                logger.exception(
                    "Error resolving events_count for entity %s", entity_id
                )
                return 0
        return 0

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

    def _find_watcher(self, path: Path) -> DirectoryWatch | None:
        """Find the DirectoryWatch that covers the given path"""
        for watch in self._all_watchers:
            try:
                path.relative_to(watch._path)
                return watch
            except ValueError:
                continue
        return None

    @staticmethod
    def _is_event_file(path: Path) -> bool:
        """Check if path is an event file (experiment or job)"""
        name = path.name
        if not name.endswith(".jsonl") or name.startswith("."):
            return False
        # Old experiment events: events-{count}.jsonl (in subdirectory)
        if name.startswith("events-"):
            return True
        # New flat job events: {hash8}-{job_id}-{count}.jsonl
        if _JOB_EVENT_FLAT_RE.match(name):
            return True
        # Old job events: event-{job_id}-{count}.jsonl
        if name.startswith("event-"):
            return True
        # Flat experiment events: {experiment_id}-{count}.jsonl
        if _EXPERIMENT_EVENT_FLAT_RE.match(name):
            return True
        return False

    def _on_file_deleted(self, path: Path) -> None:
        """Handle file deletion event from FileWatcher"""
        logger.debug("Detected deletion of event file: %s", path)
        entity_id = self._extract_entity_id(path)
        dir_config = self._find_dir_config(path)
        # Remove from tailed pool
        watch = self._find_watcher(path)
        if watch:
            watch.remove_tail(path)
        self._file_positions.pop(path, None)
        if entity_id:
            self._handle_deletion(entity_id, path, dir_config)

    def _on_file_created(self, path: Path) -> None:
        """Handle file creation - ensure position is initialized before processing"""
        watch = self._find_watcher(path)
        if watch and watch._tailed_pool is not None:
            if watch.get_tail_position(path) == 0:
                watch.set_tail_position(path, 0)
        elif path not in self._file_positions:
            self._file_positions[path] = 0

    def _discover_entities(self) -> dict[str, tuple[WatchedDirectory, list[Path]]]:
        """Discover all entities from existing event files.

        Returns:
            Dict mapping entity_id to (dir_config, list of event files)
        """
        entities: dict[str, tuple[WatchedDirectory, list[Path]]] = {}

        for dir_config in self.directories:
            if not dir_config.path.exists():
                continue

            for event_file in dir_config.path.glob(dir_config.glob_pattern):
                entity_id = dir_config.entity_id_extractor(event_file)
                if entity_id:
                    if entity_id not in entities:
                        entities[entity_id] = (dir_config, [])
                    entities[entity_id][1].append(event_file)

        # Sort event files by name (which includes the count)
        for entity_id, (dir_config, files) in entities.items():
            files.sort(key=lambda p: p.name)

        return entities

    def _register_entity(
        self,
        entity_id: str,
        dir_config: WatchedDirectory,
        events: list[EventBase] | None = None,
    ) -> bool:
        """Register an entity to be followed.

        Calls on_created callback if present, passing historical events.
        Returns True if entity should be followed.

        Args:
            entity_id: Entity identifier
            dir_config: Directory configuration for this entity
            events: Historical events collected during catch-up (passed to on_created)

        Returns:
            True if entity is now being followed, False if rejected
        """
        if entity_id in self._followed_entities:
            return True  # Already following

        # Call on_created callback if present
        if dir_config.on_created is not None:
            if not dir_config.on_created(entity_id, events or []):
                return False  # Callback rejected this entity

        self._followed_entities[entity_id] = dir_config
        return True

    def start_watching(self) -> None:
        """Start watching for file changes using FileWatcher.

        Discovers existing entities, calls on_created for each, and
        replays all historical events for entities that should be followed.

        For each entity, only the latest event file is tracked (tailed).
        Earlier files are processed during catch-up but not watched.
        """
        if self._file_watcher is not None:
            return  # Already watching

        # Discover existing entities
        entities = self._discover_entities()

        # Files to track: only the latest file per entity
        files_to_track: list[Path] = []

        # Register entities and collect events for followed ones
        for entity_id, (dir_config, event_files) in entities.items():
            if not event_files:
                if not self._register_entity(entity_id, dir_config):
                    continue
                continue

            # Get starting events_count via resolver
            min_count = self._get_entity_events_count(entity_id, dir_config)

            # Filter files to those at or above min_count
            relevant_files = []
            for ef in event_files:
                fn = self._extract_file_number(ef)
                if fn is not None and fn >= min_count:
                    relevant_files.append(ef)

            # Collect all historical events
            collected_events: list[EventBase] = []
            if relevant_files:
                for event_file in relevant_files:
                    collected_events.extend(self._read_events_from_file(event_file))

            if not self._register_entity(entity_id, dir_config, collected_events):
                continue

            # When on_created is None, deliver events via on_event
            if dir_config.on_created is None and dir_config.on_event:
                for event in collected_events:
                    try:
                        dir_config.on_event(entity_id, event)
                    except Exception:
                        logger.exception("Error in on_event callback")

            # Track only the latest relevant file (skip files below events_count
            # to avoid reading old events when the job appends to them)
            if relevant_files:
                latest = relevant_files[-1]
                self._current_file[entity_id] = latest
                files_to_track.append(latest)
            # else: no current_file — _process_file_change will do
            # catch-up with events_count filtering when a new file appears

        # Create a DirectoryWatch for each directory (via FileWatcherService)
        svc = FileWatcherService.instance()

        # Use the first directory to create the primary watch
        first_dir = self.directories[0]
        first_dir.path.mkdir(parents=True, exist_ok=True)
        self._file_watcher = svc.watch_directory(
            first_dir.path,
            recursive=True,
            on_change=self._process_file_change,
            on_deleted=self._on_file_deleted,
            file_filter=self._is_event_file,
            on_created=self._on_file_created,
            enable_tailing=True,
            max_open_files=self._max_open_files,
        )
        self._all_watchers = [self._file_watcher]
        logger.debug("Started watching %s", first_dir.path)

        # Watch additional directories
        self._extra_watchers: list[DirectoryWatch] = []
        for dir_config in self.directories[1:]:
            dir_config.path.mkdir(parents=True, exist_ok=True)
            extra = svc.watch_directory(
                dir_config.path,
                recursive=True,
                on_change=self._process_file_change,
                on_deleted=self._on_file_deleted,
                file_filter=self._is_event_file,
                on_created=self._on_file_created,
                enable_tailing=True,
                max_open_files=self._max_open_files,
            )
            self._extra_watchers.append(extra)
            self._all_watchers.append(extra)
            logger.debug("Started watching %s", dir_config.path)

        # Transfer staged file positions to watch tail pools
        for path, pos in self._file_positions.items():
            watch = self._find_watcher(path)
            if watch:
                watch.set_tail_position(path, pos)
        self._file_positions.clear()

        # Only track the latest file per entity (not all files)
        for path in files_to_track:
            watch = self._find_watcher(path)
            if watch:
                watch.add_file(path)
            else:
                self._file_watcher.add_file(path)

    def stop_watching(self) -> None:
        """Stop watching for file changes (closes tailed file pools)"""
        if self._file_watcher:
            self._file_watcher.close()
            self._file_watcher = None
        if hasattr(self, "_extra_watchers"):
            for w in self._extra_watchers:
                w.close()
            self._extra_watchers.clear()
        self._all_watchers.clear()
        logger.debug("Stopped watching all directories")

    def follow(
        self,
        entity_id: str,
        dir_config: WatchedDirectory,
    ) -> list[EventBase]:
        """Start following an entity (e.g., when scheduler submits a new job).

        Returns all existing events for bulk consolidation by the caller,
        then starts tailing the latest event file for new events.

        If the entity is already being followed, returns immediately.

        Note: This bypasses the on_created callback - use this for explicit
        registration (e.g., scheduler submitting a job) rather than discovery.

        Args:
            entity_id: Entity identifier to follow
            dir_config: Directory configuration for this entity

        Returns:
            List of existing events for the caller to consolidate
        """
        if entity_id in self._followed_entities:
            return []

        # Register directly without calling on_created (explicit follow)
        self._followed_entities[entity_id] = dir_config

        collected: list[EventBase] = []
        latest_file: Path | None = None

        # Process existing event files for this entity
        if dir_config.path.exists():
            event_files = sorted(
                dir_config.path.glob(dir_config.glob_pattern),
                key=lambda p: p.name,
            )
            for event_file in event_files:
                file_entity_id = dir_config.entity_id_extractor(event_file)
                if file_entity_id == entity_id:
                    collected.extend(self._read_events_from_file(event_file))
                    latest_file = event_file

        # Track only the latest file for future changes
        if latest_file is not None:
            self._current_file[entity_id] = latest_file
            watch = self._find_watcher(latest_file)
            if watch:
                watch.add_file(latest_file)
            elif self._file_watcher:
                self._file_watcher.add_file(latest_file)

        return collected

    def ensure_file_polled(self, path: Path) -> None:
        """Ensure a file is being polled by the file watcher.

        Adds the file to the watcher's polling list if not already tracked.
        This is needed on shared/network filesystems where watchdog doesn't
        detect remote file creation (e.g., NFS, GPFS, Lustre).

        Args:
            path: Path to the event file to ensure is polled
        """
        watch = self._find_watcher(path)
        if watch:
            watch.add_file(path)
        elif self._file_watcher:
            self._file_watcher.add_file(path)

    def _read_events_from_file(self, path: Path) -> list[EventBase]:
        """Read all events from a file without calling callbacks.

        Updates file position tracking so subsequent reads don't re-read.

        Returns:
            List of parsed events
        """
        events: list[EventBase] = []
        try:
            with path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event_dict = json.loads(line)
                        events.append(EventBase.from_dict(event_dict))
                    except json.JSONDecodeError:
                        pass
                # Mark file as fully read
                end_pos = f.tell()
                watch = self._find_watcher(path)
                if watch and watch._tailed_pool is not None:
                    watch.set_tail_position(path, end_pos)
                else:
                    self._file_positions[path] = end_pos
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Failed to read events from %s: %s", path, e)
        return events

    def _extract_file_number(self, path: Path) -> int | None:
        """Extract the file number from event file name.

        E.g., "events-2.jsonl" -> 2, "abcd1234-abc123...-10.jsonl" -> 10,
        "event-abc123-10.jsonl" -> 10 (old format)
        Returns None if the file name doesn't match the expected pattern.
        """
        name = path.name
        # Match any format ending in -{number}.jsonl
        match = re.match(r".*-(\d+)\.jsonl$", name)
        if match:
            return int(match.group(1))
        return None

    def _extract_file_prefix(self, path: Path) -> str | None:
        """Extract the filename prefix before the count number.

        E.g., "events-2.jsonl" -> "events-",
        "abcd1234-abc123...-10.jsonl" -> "abcd1234-abc123...-",
        "event-abc123-10.jsonl" -> "event-abc123-" (old format)
        Returns None if the file name doesn't match.
        """
        name = path.name
        match = re.match(r"(.*-)\d+\.jsonl$", name)
        if match:
            return match.group(1)
        return None

    def _process_file_change(self, path: Path) -> None:
        """Process a changed event file and notify callbacks (or buffer)

        Follows one file per entity and switches on rotation:
        - If the notified file has a higher number than the current file,
          drain the current file first, then switch to the new one.
        - If the notified file has a lower number, ignore it (old file).
        - On first encounter, catch up from events_count via the resolver.
        """
        entity_id = self._extract_entity_id(path)
        if not entity_id:
            logger.debug("Could not extract entity_id from %s", path)
            return

        dir_config = self._find_dir_config(path)
        if not dir_config:
            logger.debug("No dir_config found for %s", path)
            return

        logger.debug("Processing file change for entity %s: %s", entity_id, path)

        # Check if this entity is being followed
        is_new = entity_id not in self._followed_entities

        file_number = self._extract_file_number(path)
        current = self._current_file.get(entity_id)
        current_number = self._extract_file_number(current) if current else None

        if is_new or current is None:
            # First file for this entity: catch up from events_count
            min_count = self._get_entity_events_count(entity_id, dir_config)

            # Skip files below the events_count threshold
            if file_number is not None and file_number < min_count:
                return

            # Collect catch-up events from earlier files
            collected_events: list[EventBase] = []
            entity_dir = path.parent
            prefix = self._extract_file_prefix(path)

            if file_number is not None and prefix:
                for earlier_num in range(min_count, file_number):
                    earlier_path = entity_dir / f"{prefix}{earlier_num}.jsonl"
                    if earlier_path.exists():
                        collected_events.extend(
                            self._read_events_from_file(earlier_path)
                        )

            # Also read the notified file itself for catch-up
            collected_events.extend(self._read_events_from_file(path))

            if is_new:
                if not self._register_entity(entity_id, dir_config, collected_events):
                    return  # Entity rejected by on_created

                # When on_created is None, events were not delivered to any
                # callback — fall through to deliver via on_event
                if dir_config.on_created is not None:
                    # on_created handled the events
                    self._current_file[entity_id] = path
                    return

            # Deliver events via on_event (either already-followed entity
            # with no current file, or new entity with no on_created)
            if dir_config.on_event and collected_events:
                for event in collected_events:
                    try:
                        dir_config.on_event(entity_id, event)
                    except Exception:
                        logger.exception("Error in on_event callback")

            # Set current file to the notified file
            self._current_file[entity_id] = path
            return  # Already read this file during catch-up
        elif file_number is not None and current_number is not None:
            if file_number > current_number:
                # Rotation: drain current file first, then switch
                self._process_single_file(current, entity_id, dir_config)
                self._current_file[entity_id] = path
            elif file_number < current_number:
                return  # Old file, ignore

        # Process the current/new file
        self._process_single_file(path, entity_id, dir_config)

    def _process_single_file(
        self, path: Path, entity_id: str, dir_config: WatchedDirectory
    ) -> None:
        """Process a single event file and notify callbacks"""
        watch = self._find_watcher(path)
        if watch and watch._tailed_pool is not None:
            # Use tailed file pool for efficient reading
            lines = watch.read_new_lines(path)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    event_dict = json.loads(line)
                    event = EventBase.from_dict(event_dict)
                    if dir_config.on_event:
                        try:
                            dir_config.on_event(entity_id, event)
                        except Exception:
                            logger.exception("Error in on_event callback")
                except json.JSONDecodeError:
                    pass
        else:
            # Fallback: open/seek/read/close (for standalone read_new_events)
            last_pos = self._file_positions.get(path, 0)
            try:
                with path.open("r") as f:
                    f.seek(last_pos)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            break
                        line = line.strip()
                        if not line:
                            last_pos = f.tell()
                            continue
                        try:
                            event_dict = json.loads(line)
                            event = EventBase.from_dict(event_dict)
                            if dir_config.on_event:
                                try:
                                    dir_config.on_event(entity_id, event)
                                except Exception:
                                    logger.exception("Error in on_event callback")
                        except json.JSONDecodeError:
                            pass
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
        # Only process if entity was being followed
        if entity_id not in self._followed_entities:
            return

        # Use stored dir_config if not provided
        if dir_config is None:
            dir_config = self._followed_entities.get(entity_id)

        # Try to read remaining events from permanent storage
        if dir_config and dir_config.permanent_storage_resolver:
            permanent_dir = dir_config.permanent_storage_resolver(entity_id)
            min_count = self._get_entity_events_count(entity_id, dir_config)

            # Read only events from files >= events_count (older ones are
            # already consolidated into status.json)
            events = self._read_events_from_permanent(
                permanent_dir, entity_id, min_count=min_count
            )
            for event in events:
                if dir_config.on_event:
                    try:
                        dir_config.on_event(entity_id, event)
                    except Exception:
                        logger.exception("Error in on_event callback")

        # Notify deletion callback
        if dir_config and dir_config.on_deleted:
            try:
                dir_config.on_deleted(entity_id)
            except Exception:
                logger.exception("Error in on_deleted callback")

        # Remove from followed entities and current file tracking
        self._followed_entities.pop(entity_id, None)
        self._current_file.pop(entity_id, None)

    def _read_events_from_permanent(
        self, permanent_dir: Path, entity_id: str, min_count: int = 0
    ) -> list[EventBase]:
        """Read events from permanent storage directory

        Args:
            permanent_dir: Path to permanent storage directory
            entity_id: Entity identifier (for logging)
            min_count: Only read files with count >= this value (files below
                are already consolidated into status.json)

        Returns:
            List of events read from permanent storage
        """
        events = []
        if not permanent_dir.exists():
            return events

        # Read event files in permanent storage, skipping consolidated ones
        event_files = sorted(permanent_dir.glob("event-*.jsonl"))
        if min_count > 0:
            event_files = [
                ef
                for ef in event_files
                if (fn := self._extract_file_number(ef)) is not None and fn >= min_count
            ]
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

    def read_events_since_count(
        self, entity_id: str, start_count: int, base_dir: Optional[Path] = None
    ) -> list[EventBase]:
        """Read events for an entity starting from a specific file count

        Tries flat format first ({entity_id}-{count}.jsonl), then falls back
        to old subdir format ({entity_id}/events-{count}.jsonl).

        Args:
            entity_id: Entity identifier (job_id or experiment_id)
            start_count: File count to start reading from
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

        count = start_count
        while True:
            # Try flat format first: {base_dir}/{entity_id}-{count}.jsonl
            event_path = base_dir / f"{entity_id}-{count}.jsonl"
            if not event_path.exists():
                # Fallback: old subdir format {base_dir}/{entity_id}/events-{count}.jsonl
                event_path = base_dir / entity_id / f"events-{count}.jsonl"
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
        # Progress events are stored in workspace/.events/jobs/
        self._events_dir = job_path.parent.parent.parent / ".events" / "jobs"

    def get_event_files(self) -> list[Path]:
        """Get all event files for this job"""
        if not self._events_dir.exists():
            return []
        h = task_id_hash(self.task_id)
        return sorted(self._events_dir.glob(f"{h}-{self.job_id}-*.jsonl"))

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
    "experiment_entity_id_extractor",
    "task_id_hash",
    "_JOB_EVENT_FLAT_RE",
    "_EXPERIMENT_EVENT_FLAT_RE",
    "JobProgressReader",
    # Callback types
    "EntityEventCallback",
    "EntityDeletedCallback",
]
