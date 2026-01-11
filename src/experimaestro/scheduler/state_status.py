"""Filesystem-based state tracking for experiments

This module provides event and status file handling for tracking experiment state
without using a database. It replaces the SQLite/peewee-based state tracking.

Key components:
- Event dataclasses: Serializable events for JSONL event files
- EventWriter/EventReader: Base classes for event I/O
- JobEventWriter: Job-specific event handling
- ExperimentEventWriter/ExperimentEventReader: Experiment-specific event handling
- StatusData: Dataclass representing status.json content
- StatusFile: Class for locked read/write of status files

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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import fasteners

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseJob, BaseService

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
    """

    state: str = ""
    failure_reason: Optional[str] = None
    submitted_time: Optional[float] = None
    started_time: Optional[float] = None
    ended_time: Optional[float] = None
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
    state: str = "STOPPED"
    service_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStateChangedEvent(ServiceEventBase):
    """Event: Service state changed (STOPPED, STARTING, RUNNING, STOPPING)"""

    run_id: str = ""
    state: str = ""


# =============================================================================
# Status File Data Structure
# =============================================================================


@dataclass
class StatusData:
    """Complete status data stored in status.json

    Stores structured objects (MockJob, MockService) for easy access.
    Uses state_dict() for JSON serialization and from_state_dict() for loading.
    """

    version: int = STATUS_VERSION
    experiment_id: str = ""
    run_id: str = ""
    events_count: int = 0
    hostname: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: str = "active"
    jobs: dict[str, "BaseJob"] = field(default_factory=dict)
    tags: dict[str, dict[str, str]] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    services: dict[str, "BaseService"] = field(default_factory=dict)
    last_updated: str = ""
    # Workspace path needed for creating MockJob instances
    workspace_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def empty(cls) -> "StatusData":
        """Create empty status data"""
        return cls()

    def state_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "version": self.version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "events_count": self.events_count,
            "hostname": self.hostname,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "jobs": {k: v.state_dict() for k, v in self.jobs.items()},
            "tags": self.tags,
            "dependencies": self.dependencies,
            "services": {k: v.state_dict() for k, v in self.services.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_state_dict(cls, d: dict, workspace_path: Path) -> "StatusData":
        """Create StatusData from dictionary

        Args:
            d: Dictionary from state_dict() or JSON file
            workspace_path: Workspace path for computing job paths
        """
        from experimaestro.scheduler.state_provider import MockJob, MockService

        # Parse jobs
        jobs_dict = d.get("jobs", {})
        jobs = {
            k: MockJob.from_state_dict(v, workspace_path) for k, v in jobs_dict.items()
        }

        # Parse services
        services_dict = d.get("services", {})
        services = {k: MockService.from_state_dict(v) for k, v in services_dict.items()}

        return cls(
            version=d.get("version", STATUS_VERSION),
            experiment_id=d.get("experiment_id", ""),
            run_id=d.get("run_id", ""),
            events_count=d.get("events_count", 0),
            hostname=d.get("hostname"),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            status=d.get("status", "active"),
            jobs=jobs,
            tags=d.get("tags", {}),
            dependencies=d.get("dependencies", {}),
            services=services,
            last_updated=d.get("last_updated", ""),
            workspace_path=workspace_path,
        )

    def apply_event(self, event: EventBase) -> None:
        """Apply an event to update the status data"""
        from experimaestro.scheduler.state_provider import MockJob, MockService
        from experimaestro.scheduler.transient import TransientMode

        if isinstance(event, JobSubmittedEvent):
            # Create MockJob for the new job
            job_path = (
                self.workspace_path / "jobs" / event.task_id / event.job_id
                if self.workspace_path
                else Path(f"/jobs/{event.task_id}/{event.job_id}")
            )
            self.jobs[event.job_id] = MockJob(
                identifier=event.job_id,
                task_id=event.task_id,
                path=job_path,
                state="unscheduled",
                submittime=event.timestamp,
                starttime=None,
                endtime=None,
                progress=[],
                updated_at="",
                transient=TransientMode(event.transient),
            )
            if event.tags:
                self.tags[event.job_id] = event.tags
            if event.depends_on:
                self.dependencies[event.job_id] = event.depends_on

        elif isinstance(event, JobStateChangedEvent):
            if event.job_id in self.jobs:
                job = self.jobs[event.job_id]
                # Update job state - MockJob stores state as JobState enum
                from experimaestro.scheduler.interfaces import STATE_NAME_TO_JOBSTATE

                job.state = STATE_NAME_TO_JOBSTATE.get(event.state, job.state)
                if event.failure_reason:
                    from experimaestro.scheduler.interfaces import JobFailureStatus

                    try:
                        job.failure_reason = JobFailureStatus[event.failure_reason]
                    except KeyError:
                        pass
                if event.submitted_time is not None:
                    job.submittime = event.submitted_time
                if event.started_time is not None:
                    job.starttime = event.started_time
                if event.ended_time is not None:
                    job.endtime = event.ended_time
                if event.exit_code is not None:
                    job.exit_code = event.exit_code
                if event.retry_count:
                    job.retry_count = event.retry_count
                if event.progress:
                    from experimaestro.notifications import (
                        get_progress_information_from_dict,
                    )

                    job.progress = get_progress_information_from_dict(event.progress)

        elif isinstance(event, JobProgressEvent):
            if event.job_id in self.jobs:
                from experimaestro.notifications import LevelInformation

                job = self.jobs[event.job_id]
                level = event.level
                # Truncate to level + 1 entries
                job.progress = job.progress[: (level + 1)]
                # Extend if needed
                while len(job.progress) <= level:
                    job.progress.append(LevelInformation(len(job.progress), None, 0.0))
                # Update the level's progress and description
                if event.desc:
                    job.progress[-1].desc = event.desc
                job.progress[-1].progress = event.progress

        elif isinstance(event, ServiceAddedEvent):
            self.services[event.service_id] = MockService(
                service_id=event.service_id,
                description_text=event.description,
                service_config_data=event.service_config,
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                state=event.state,
            )

        elif isinstance(event, RunCompletedEvent):
            self.status = event.status
            self.ended_at = event.ended_at


# =============================================================================
# Status File Handler
# =============================================================================


class StatusFile:
    """Base class for status file handling with locking

    Provides atomic read/write operations with inter-process locking.
    Works with plain dicts - the unified format from status_dict().

    Subclasses:
    - ExperimentStatusFile: For experiment status.json files
    - JobStatusFile: For job information.json files
    """

    def __init__(self, directory: Path, filename: str):
        """Initialize status file handler

        Args:
            directory: Directory containing the status file
            filename: Name of the status file
        """
        self.directory = directory
        self.path = directory / filename
        self._lock_path = directory / f".{filename}.lock"

    def read_locked(self) -> tuple[dict, fasteners.InterProcessLock]:
        """Acquire lock and read status. Caller MUST release lock when done."""
        lock = fasteners.InterProcessLock(str(self._lock_path))
        lock.acquire()
        try:
            return self._read(), lock
        except Exception:
            lock.release()
            raise

    def write_locked(self, data: dict, lock: fasteners.InterProcessLock) -> None:
        """Write status and release lock"""
        try:
            self._write(data)
        finally:
            lock.release()

    def read(self) -> dict:
        """Read status file (brief lock for consistency)"""
        if not self.path.exists():
            return {}
        lock = fasteners.InterProcessLock(str(self._lock_path))
        with lock:
            return self._read()

    def write(self, data: dict) -> None:
        """Write status file (with lock)"""
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = fasteners.InterProcessLock(str(self._lock_path))
        with lock:
            self._write(data)

    def _read(self) -> dict:
        """Internal read (no locking)"""
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", self.path, e)
            return {}

    def _write(self, data: dict) -> None:
        """Internal write - atomic via temp + rename"""
        data["last_updated"] = datetime.now().isoformat()
        temp_path = self.path.with_suffix(".json.tmp")
        with temp_path.open("w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.path)


class ExperimentStatusFile(StatusFile):
    """Handles experiment status.json files with StatusData conversion

    Extends StatusFile to provide StatusData-specific read/write methods
    for backward compatibility with existing code.
    """

    def __init__(self, run_dir: Path, workspace_path: Optional[Path] = None):
        super().__init__(run_dir, "status.json")
        self.run_dir = run_dir
        self.workspace_path = workspace_path or run_dir.parent.parent.parent

    def read_status_data(self) -> "StatusData":
        """Read and convert to StatusData object"""
        d = self.read()
        if not d:
            return StatusData.empty()
        return StatusData.from_state_dict(d, self.workspace_path)

    def write_status_data(self, data: "StatusData") -> None:
        """Write StatusData object"""
        self.write(data.state_dict())

    # Backward compatible methods that return StatusData
    def read_locked_status_data(
        self,
    ) -> tuple["StatusData", fasteners.InterProcessLock]:
        """Acquire lock and read status as StatusData"""
        d, lock = self.read_locked()
        if not d:
            return StatusData.empty(), lock
        return StatusData.from_state_dict(d, self.workspace_path), lock

    def write_locked_status_data(
        self, data: "StatusData", lock: fasteners.InterProcessLock
    ) -> None:
        """Write StatusData and release lock"""
        self.write_locked(data.state_dict(), lock)


class JobStatusFile(StatusFile):
    """Handles job information.json files

    Provides access to job status stored in the job's .experimaestro directory.
    """

    def __init__(self, job_path: Path):
        xpm_dir = job_path / ".experimaestro"
        super().__init__(xpm_dir, "information.json")
        self.job_path = job_path


# =============================================================================
# Event Writer Classes
# =============================================================================


class EventWriter(ABC):
    """Base class for writing events to JSONL files

    Events are written to {events_dir}/events-{count}.jsonl
    Uses line buffering so each event is flushed immediately after write.

    Supports proactive hardlinking: when a permanent_dir is set and hardlinks
    are supported, a hardlink is created to permanent storage immediately when
    the event file is opened. This ensures events are written to both locations
    simultaneously and no data is lost if the process crashes.
    """

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
        workspace_path: Path,
        experiment_id: str,
        initial_count: int = 0,
        run_dir: Path | None = None,
    ):
        """Initialize experiment event writer

        Args:
            workspace_path: Path to workspace directory
            experiment_id: Experiment identifier
            initial_count: Starting event file count for rotation
            run_dir: Optional run directory path for permanent storage
        """
        # Permanent storage: run_dir/events/
        permanent_dir = run_dir / "events" if run_dir else None
        super().__init__(initial_count, permanent_dir)
        self.workspace_path = workspace_path
        self.experiment_id = experiment_id
        self.run_dir = run_dir
        self._events_dir = workspace_path / ".events" / "experiments" / experiment_id

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    def init_status(
        self,
        run_dir: Path,
        run_id: str,
        hostname: str,
    ) -> StatusData:
        """Initialize status.json for a new run

        Args:
            run_dir: Path to the run directory
            run_id: Run identifier
            hostname: Hostname where experiment is running

        Returns:
            Initialized StatusData
        """
        data = StatusData(
            version=STATUS_VERSION,
            experiment_id=self.experiment_id,
            run_id=run_id,
            events_count=0,
            hostname=hostname,
            started_at=datetime.now().isoformat(),
            status="active",
        )
        status_file = ExperimentStatusFile(run_dir, self.workspace_path)
        status_file.write_status_data(data)
        return data

    def create_symlink(self, run_dir: Path) -> None:
        """Create/update symlink to current run directory

        The symlink is created at:
        .events/experiments/{experiment_id}/current -> run_dir

        Args:
            run_dir: Path to the current run directory
        """
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
        status_file_class: Optional StatusFile subclass for reading entity status
            when event files are deleted.
    """

    path: Path
    entity_id_extractor: EntityIdExtractor = field(
        default_factory=lambda: default_entity_id_extractor
    )
    glob_pattern: str = "*/events-*.jsonl"
    # NEW fields for archiving and deletion handling:
    permanent_storage_resolver: PermanentStorageResolver | None = None
    status_file_class: type[StatusFile] | None = None


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

    def _process_file_change(self, path: Path) -> None:
        """Process a changed event file and notify callbacks (or buffer)"""
        entity_id = self._extract_entity_id(path)
        if not entity_id:
            return

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
        deleted_path: Path | None = None,
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
    # Status classes
    "StatusData",
    "StatusFile",
    "ExperimentStatusFile",
    "JobStatusFile",
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
