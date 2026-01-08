"""Filesystem-based state tracking for experiments

This module provides event and status file handling for tracking experiment state
without using a database. It replaces the SQLite/peewee-based state tracking.

Key components:
- Event dataclasses: Serializable events for JSONL event files
- StatusData: Dataclass representing status.json content
- StatusFile: Class for locked read/write of status files
- Helper functions for experiment.py to use

File structure:
- workspace/.experimaestro/experiments/events-{count}@{experiment-id}.jsonl
- workspace/.experimaestro/experiments/{experiment-id} (symlink to current run)
- workspace/experiments/{experiment-id}/{run-id}/status.json
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import fasteners

if TYPE_CHECKING:
    from experimaestro.scheduler.interfaces import BaseJob, BaseService

logger = logging.getLogger("xpm.state_status")

# Status file version
STATUS_VERSION = 1


# =============================================================================
# Event Dataclasses (for JSONL event files)
# =============================================================================


@dataclass
class EventBase:
    """Base class for all events written to JSONL files"""

    type: str
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize event to JSON string"""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "EventBase":
        """Deserialize event from dictionary"""
        event_type = d.get("type")
        event_class = EVENT_TYPES.get(event_type, EventBase)
        # Filter to only known fields for the event class
        valid_fields = {f for f in event_class.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return event_class(**filtered)


@dataclass
class JobSubmittedEvent(EventBase):
    """Event: Job was submitted to the scheduler"""

    type: str = "job_submitted"
    job_id: str = ""
    task_id: str = ""
    transient: int = 0
    tags: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class JobStateChangedEvent(EventBase):
    """Event: Job state changed"""

    type: str = "job_state_changed"
    job_id: str = ""
    state: str = ""
    failure_reason: Optional[str] = None
    submitted_time: Optional[float] = None
    started_time: Optional[float] = None
    ended_time: Optional[float] = None
    exit_code: Optional[int] = None
    retry_count: int = 0
    progress: list[dict] = field(default_factory=list)


@dataclass
class ServiceAddedEvent(EventBase):
    """Event: Service was added to the experiment"""

    type: str = "service_added"
    service_id: str = ""
    description: str = ""
    state: str = "STOPPED"
    state_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunCompletedEvent(EventBase):
    """Event: Experiment run completed"""

    type: str = "run_completed"
    status: str = "completed"
    ended_at: str = ""


# Registry for event deserialization
EVENT_TYPES: dict[str, type] = {
    "job_submitted": JobSubmittedEvent,
    "job_state_changed": JobStateChangedEvent,
    "service_added": ServiceAddedEvent,
    "run_completed": RunCompletedEvent,
}


# =============================================================================
# Status File Data Structure
# =============================================================================


@dataclass
class StatusData:
    """Complete status data stored in status.json

    Stores structured objects (MockJob, MockService) for easy access.
    Uses db_state_dict() for JSON serialization and from_db_state_dict() for loading.
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

    def db_state_dict(self) -> dict:
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
            "jobs": {k: v.db_state_dict() for k, v in self.jobs.items()},
            "tags": self.tags,
            "dependencies": self.dependencies,
            "services": {k: v.db_state_dict() for k, v in self.services.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_db_state_dict(cls, d: dict, workspace_path: Path) -> "StatusData":
        """Create StatusData from dictionary

        Args:
            d: Dictionary from db_state_dict() or JSON file
            workspace_path: Workspace path for computing job paths
        """
        from experimaestro.scheduler.state_provider import MockJob, MockService

        # Parse jobs
        jobs_dict = d.get("jobs", {})
        jobs = {
            k: MockJob.from_db_state_dict(v, workspace_path)
            for k, v in jobs_dict.items()
        }

        # Parse services
        services_dict = d.get("services", {})
        services = {
            k: MockService.from_db_state_dict(v) for k, v in services_dict.items()
        }

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

        elif isinstance(event, ServiceAddedEvent):
            self.services[event.service_id] = MockService(
                service_id=event.service_id,
                description_text=event.description,
                state_dict_data=event.state_dict,
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
    """Handles reading/writing/locking of status.json files

    Uses file locking to coordinate between the running experiment (writer)
    and monitoring tools (readers).
    """

    def __init__(self, run_dir: Path, workspace_path: Optional[Path] = None):
        self.run_dir = run_dir
        self.path = run_dir / "status.json"
        self._lock_path = run_dir / ".status.json.lock"
        self.workspace_path = workspace_path

    def read_locked(self) -> tuple["StatusData", fasteners.InterProcessLock]:
        """Acquire lock and read status

        Returns:
            Tuple of (StatusData, lock). Caller MUST release lock when done.
        """
        lock = fasteners.InterProcessLock(str(self._lock_path))
        lock.acquire()
        try:
            data = self._read()
            return data, lock
        except Exception:
            lock.release()
            raise

    def write_locked(self, data: StatusData, lock: fasteners.InterProcessLock) -> None:
        """Write status and release lock

        Args:
            data: Status data to write
            lock: Lock acquired from read_locked()
        """
        try:
            self._write(data)
        finally:
            lock.release()

    def read(self) -> StatusData:
        """Read status file (brief lock for consistency)"""
        if not self.path.exists():
            return StatusData.empty()
        lock = fasteners.InterProcessLock(str(self._lock_path))
        with lock:
            return self._read()

    def write(self, data: StatusData) -> None:
        """Write status file (with lock)"""
        lock = fasteners.InterProcessLock(str(self._lock_path))
        with lock:
            self._write(data)

    def _read(self) -> StatusData:
        """Internal read (no locking)"""
        if not self.path.exists():
            return StatusData.empty()
        try:
            with self.path.open("r") as f:
                data = json.load(f)
                if self.workspace_path is not None:
                    return StatusData.from_db_state_dict(data, self.workspace_path)
                # Fallback for when workspace_path is not provided
                return StatusData.from_db_state_dict(
                    data, self.run_dir.parent.parent.parent
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read status file %s: %s", self.path, e)
            return StatusData.empty()

    def _write(self, data: StatusData) -> None:
        """Internal write (no locking)

        Uses atomic write (write to temp file then rename) to prevent
        partial writes from corrupting the file.
        """
        data.last_updated = datetime.now().isoformat()
        temp_path = self.path.with_suffix(".json.tmp")
        with temp_path.open("w") as f:
            json.dump(data.db_state_dict(), f, indent=2)
        temp_path.replace(self.path)


# =============================================================================
# Event File Writer
# =============================================================================


class EventFileWriter:
    """Writes events to JSONL file

    Events are written to workspace/.experimaestro/experiments/events-{count}@{exp_id}.jsonl
    Flushes to disk every 5 seconds to allow file watchers to detect changes.
    """

    FLUSH_INTERVAL = 5.0  # seconds

    def __init__(
        self, experiments_dir: Path, experiment_id: str, initial_count: int = 0
    ):
        import time

        self.experiments_dir = experiments_dir
        self.experiment_id = experiment_id
        self._count = initial_count
        self._file = None
        self._last_flush = time.time()

    def _get_event_file_path(self) -> Path:
        return self.experiments_dir / f"events-{self._count}@{self.experiment_id}.jsonl"

    def write_event(self, event: EventBase) -> None:
        """Write an event to the current event file"""
        import os
        import time

        if self._file is None:
            self.experiments_dir.mkdir(parents=True, exist_ok=True)
            self._file = self._get_event_file_path().open("a")

        self._file.write(event.to_json() + "\n")

        # Flush periodically to allow file watchers to detect changes
        now = time.time()
        if now - self._last_flush >= self.FLUSH_INTERVAL:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._last_flush = now

    def close(self) -> None:
        """Close the current event file"""
        import os

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
        """Delete all event files for this experiment (called when experiment finishes)"""
        self.close()
        for i in range(self._count + 1):
            path = self.experiments_dir / f"events-{i}@{self.experiment_id}.jsonl"
            if path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete event file %s: %s", path, e)


# =============================================================================
# Helper Functions for experiment.py
# =============================================================================


def init_status(
    run_dir: Path,
    experiment_id: str,
    run_id: str,
    hostname: str,
) -> StatusData:
    """Initialize status.json for a new run

    Args:
        run_dir: Path to the run directory
        experiment_id: Experiment identifier
        run_id: Run identifier
        hostname: Hostname where experiment is running

    Returns:
        Initialized StatusData
    """
    data = StatusData(
        version=STATUS_VERSION,
        experiment_id=experiment_id,
        run_id=run_id,
        events_count=0,
        hostname=hostname,
        started_at=datetime.now().isoformat(),
        status="active",
    )
    status_file = StatusFile(run_dir)
    status_file.write(data)
    return data


def create_experiment_symlink(
    experiments_dir: Path, experiment_id: str, run_dir: Path
) -> None:
    """Create/update symlink from experiment_id to current run directory

    Args:
        experiments_dir: Path to workspace/.experimaestro/experiments/
        experiment_id: Experiment identifier
        run_dir: Path to the current run directory
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)
    symlink = experiments_dir / experiment_id

    # Compute relative path from symlink location to run_dir
    try:
        rel_path = os.path.relpath(run_dir, experiments_dir)
    except ValueError:
        # On Windows, relpath fails for paths on different drives
        rel_path = str(run_dir)

    # Remove existing symlink if present
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()

    symlink.symlink_to(rel_path)


def read_events_since(
    experiments_dir: Path, experiment_id: str, since_count: int = 0
) -> list[EventBase]:
    """Read events from JSONL files starting from a given count

    Args:
        experiments_dir: Path to workspace/.experimaestro/experiments/
        experiment_id: Experiment identifier
        since_count: Start reading from this event file count

    Returns:
        List of events
    """
    events = []
    count = since_count
    while True:
        event_path = experiments_dir / f"events-{count}@{experiment_id}.jsonl"
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


__all__ = [
    # Event classes
    "EventBase",
    "JobSubmittedEvent",
    "JobStateChangedEvent",
    "ServiceAddedEvent",
    "RunCompletedEvent",
    "EVENT_TYPES",
    # Status classes
    "StatusData",
    "StatusFile",
    # Event writer
    "EventFileWriter",
    # Helper functions
    "init_status",
    "create_experiment_symlink",
    "read_events_since",
]
