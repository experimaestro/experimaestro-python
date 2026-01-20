"""Base interfaces for job and experiment data

This module defines abstract interfaces that represent job and experiment information.
These interfaces provide a common API between live jobs/experiments and those
loaded from the database.

- JobState: Base class for job states with singleton instances
- JobFailureStatus: Enum for failure reasons
- BaseJob: Interface defining job attributes and metadata operations
- BaseExperiment: Interface defining experiment attributes

The existing Job and experiment classes should provide these same attributes
to enable unified access in the TUI and other monitoring tools.
"""

import enum
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.scheduler.transient import TransientMode
    from experimaestro.scheduler.state_provider import CarbonMetricsData
    from experimaestro.scheduler.state_status import EventBase, JobStateChangedEvent
    from experimaestro.scheduler.experiment import experiment
    from experimaestro.carbon.base import CarbonImpactData

logger = logging.getLogger("xpm.interfaces")


@dataclass
class ExperimentJobInformation:
    """Lightweight job information for experiment state serialization

    This class contains the minimal job metadata stored in status.json and jobs.jsonl.
    Full job state (progress, state changes, etc.) comes from events.jsonl replay
    or from the state provider.
    """

    job_id: str
    task_id: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON"""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentJobInformation":
        """Create from dictionary"""
        return cls(
            job_id=d["job_id"],
            task_id=d["task_id"],
            tags=d.get("tags", {}),
            timestamp=d.get("timestamp"),
        )


def serialize_timestamp(ts: Optional[Union[float, datetime, str]]) -> Optional[str]:
    """Serialize timestamp to ISO format string for DB/network storage

    Handles:
    - None: returns None
    - float/int: Unix timestamp, converts to ISO format
    - datetime: converts to ISO format
    - str: returns as-is (already serialized)
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts  # Already serialized
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)


def deserialize_timestamp(ts: Optional[Union[float, str]]) -> Optional[float]:
    """Deserialize timestamp from ISO format string to Unix timestamp

    Handles:
    - None: returns None
    - float/int: returns as-is (already a Unix timestamp)
    - str: parses ISO format and converts to Unix timestamp
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return None
    return None


def deserialize_to_datetime(
    ts: Optional[Union[float, int, str, datetime]],
) -> Optional[datetime]:
    """Deserialize timestamp to datetime object (backward-compatible)

    Handles:
    - None: returns None
    - datetime: returns as-is
    - float/int: Unix timestamp, converts to datetime
    - str: parses ISO format to datetime
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


# =============================================================================
# Job State Classes
# =============================================================================


class JobState:
    """Base class for job states

    Job states are represented as instances of JobState subclasses.
    Singleton instances are available as class attributes (e.g., JobState.DONE)
    for backward compatibility.
    """

    name: str  # Readable name
    value: int  # Numeric value for ordering comparisons
    failure_reason: Optional["JobFailureStatus"] = (
        None  # Failure reason (for error states)
    )

    # Singleton instances (assigned after subclass definitions)
    UNSCHEDULED: "JobStateUnscheduled"
    WAITING: "JobStateWaiting"
    READY: "JobStateReady"
    SCHEDULED: "JobStateScheduled"
    RUNNING: "JobStateRunning"
    DONE: "JobStateDone"
    ERROR: "JobStateError"

    def notstarted(self):
        """Returns True if the job hasn't started yet"""
        return self.value <= 2  # READY

    def running(self):
        """Returns True if the job is currently running or scheduled"""
        return self.value == 4 or self.value == 3  # RUNNING or SCHEDULED

    def finished(self):
        """Returns True if the job has finished (success or error)"""
        return self.value >= 5  # DONE or ERROR

    def is_error(self) -> bool:
        return False

    def __eq__(self, other):
        """Compare job states by their numeric value"""
        if isinstance(other, JobState):
            return self.value == other.value
        return False

    def __hash__(self):
        """Allow JobState instances to be used as dict keys"""
        return hash(self.value)

    def __repr__(self):
        """String representation of the job state"""
        return f"{self.__class__.__name__}()"

    @staticmethod
    def from_path(basepath: Path, scriptname: str) -> "JobState":
        """Read job state from .done or .failed files

        Args:
            basepath: The job directory path
            scriptname: The script name (used for file naming)

        Returns:
            JobState.DONE if .done exists, JobStateError with details if .failed exists,
            or None if neither exists.
        """
        donepath = basepath / f"{scriptname}.done"
        failedpath = basepath / f"{scriptname}.failed"

        if donepath.is_file():
            return JobState.DONE

        if failedpath.is_file():
            content = failedpath.read_text().strip()

            # Try JSON first
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # New format: failure_status field
                    failure_status_str = data.get("failure_status")
                    if failure_status_str:
                        try:
                            failure_status = JobFailureStatus[
                                failure_status_str.upper()
                            ]
                            return JobStateError(failure_status)
                        except KeyError:
                            pass
                    # Legacy format: reason field
                    reason = data.get("reason")
                    if reason:
                        try:
                            failure_status = JobFailureStatus[reason.upper()]
                            return JobStateError(failure_status)
                        except KeyError:
                            pass
                    return JobStateError(JobFailureStatus.FAILED)
            except json.JSONDecodeError:
                pass

            # Fall back to legacy integer format
            try:
                code = int(content)
                if code == 0:
                    return JobState.DONE
                return JobStateError(JobFailureStatus.FAILED)
            except ValueError:
                logger.warning(
                    "Could not parse failed file %s: %s", failedpath, content
                )
                return JobStateError(JobFailureStatus.FAILED)

        return None


class JobStateUnscheduled(JobState):
    """Job is not yet scheduled"""

    name = "unscheduled"
    value = 0


class JobStateWaiting(JobState):
    """Job is waiting for dependencies to be done"""

    name = "waiting"
    value = 1


class JobStateReady(JobState):
    """Job is ready to run"""

    name = "ready"
    value = 2


class JobStateScheduled(JobState):
    """Job is scheduled (e.g., in SLURM queue)"""

    name = "scheduled"
    value = 3


class JobStateRunning(JobState):
    """Job is currently running"""

    name = "running"
    value = 4


class JobStateDone(JobState):
    """Job has completed successfully"""

    name = "done"
    value = 5


class JobFailureStatus(enum.Enum):
    """Reasons for job failure"""

    #: Job dependency failed
    DEPENDENCY = 0

    #: Job failed
    FAILED = 1

    #: Memory
    MEMORY = 2

    #: Timeout (can retry for resumable tasks)
    TIMEOUT = 3


class ExperimentStatus(enum.Enum):
    """Status of an experiment run"""

    #: Experiment is currently running
    RUNNING = "running"

    #: Experiment completed successfully
    DONE = "done"

    #: Experiment failed
    FAILED = "failed"


class JobStateError(JobState):
    """Job has failed

    This state carries information about the failure reason via JobFailureStatus enum.
    """

    name = "error"
    value = 6

    def __init__(self, failure_reason: Optional[JobFailureStatus] = None):
        """Create an error state, optionally with failure details

        Args:
            failure_reason: Optional reason for the failure (JobFailureStatus enum value)
        """
        self.failure_reason = failure_reason

    def __repr__(self):
        if self.failure_reason:
            return f"JobStateError(failure_reason={self.failure_reason})"
        return "JobStateError()"

    def __eq__(self, other):
        """Error states are equal if they have the same value

        Note: We intentionally ignore failure_reason in equality comparison
        to maintain backward compatibility with code that does:
        if job.state == JobState.ERROR: ...
        """
        if isinstance(other, JobState):
            return self.value == other.value
        return False

    def is_error(self):
        return True


# NOTE: Consider removing these singleton instances in a future refactor
# Create singleton instances for backward compatibility
# These can be used in comparisons: if state == JobState.DONE: ...
JobState.UNSCHEDULED = JobStateUnscheduled()
JobState.WAITING = JobStateWaiting()
JobState.READY = JobStateReady()
JobState.SCHEDULED = JobStateScheduled()
JobState.RUNNING = JobStateRunning()
JobState.DONE = JobStateDone()
JobState.ERROR = JobStateError()  # default error without failure details


# Mapping from state name string to JobState singleton
STATE_NAME_TO_JOBSTATE = {
    "unscheduled": JobState.UNSCHEDULED,
    "waiting": JobState.WAITING,
    "ready": JobState.READY,
    "scheduled": JobState.SCHEDULED,
    "running": JobState.RUNNING,
    "done": JobState.DONE,
    "error": JobState.ERROR,
}


# =============================================================================
# Base Job Interface
# =============================================================================


class BaseJob:
    """Base interface for job information and metadata operations

    This class defines the interface for job data and provides methods for
    reading/writing job metadata files. Both live Job instances and
    database-loaded MockJob instances should provide these attributes.

    Attributes:
        identifier: Unique identifier for the job (hash)
        task_id: Task class identifier (string)
        path: Path to job directory
        state: Current job state (JobState object or compatible)
        submittime: When job was submitted (datetime or None)
        starttime: When job started running (datetime or None)
        endtime: When job finished (datetime or None)
        progress: List of progress updates
        exit_code: Process exit code (optional)
        retry_count: Number of retries
        transient: Transient mode (NONE, TRANSIENT, or REMOVE)
    """

    identifier: str
    task_id: str
    path: Path
    _state: JobState
    submittime: Optional[datetime]
    starttime: Optional[datetime]
    endtime: Optional[datetime]
    progress: List[Dict]
    exit_code: Optional[int]
    retry_count: int
    transient: "TransientMode"
    carbon_metrics: Optional["CarbonMetricsData"]
    events_count: Optional[int]  # None = all events processed
    experiments: List["experiment"]  # Experiments this job belongs to

    #: The process
    _process: Optional["Process"]

    #: The process definition from JSON
    _process_dict: Optional["Process"]

    def __init__(self):
        super().__init__()
        self._state = JobState.UNSCHEDULED
        self.submittime: datetime | None = None
        self.starttime: datetime | None = None
        self.endtime: datetime | None = None
        self._process = None
        self._process_dict = None
        self.events_count: int | None = None
        self.experiments: List = []

    @property
    def state(self) -> JobState:
        """Access to job state"""
        return self._state

    @state.setter
    def state(self, new_state: JobState):
        """Set state via set_state() to ensure proper handling"""
        self.set_state(new_state)

    def set_state(
        self,
        new_state: JobState,
        *,
        loading: bool = False,
    ):
        """Set job state and update timestamps

        Args:
            new_state: The new job state
            loading: If True, timestamps are not modified (loading from disk)

        Timestamp rules (when loading=False):
        - WAITING: sets submittime, clears starttime and endtime
        - RUNNING: sets starttime
        - DONE/ERROR: sets endtime
        """
        old_state = self._state
        if old_state == new_state:
            return

        if not loading:
            ts = datetime.now()

            # Transitioning to WAITING clears later timestamps (for restarts)
            if new_state == JobState.WAITING:
                self.submittime = ts
                self.starttime = None
                self.endtime = None

            # Set starttime when entering RUNNING
            elif new_state == JobState.RUNNING:
                self.starttime = ts

            # Set endtime when finishing (DONE or ERROR)
            elif new_state.finished():
                self.endtime = ts

        self._state = new_state

    @property
    def locator(self) -> str:
        """Full task locator (identifier): {task_id}/{identifier}"""
        return f"{self.task_id}/{self.identifier}"

    @property
    def full_id(self) -> str:
        """Full job identifier: {task_id}:{identifier}

        This is used as a unique key for jobs across the entire workspace,
        combining the task identifier and job hash.
        """
        return f"{self.task_id}:{self.identifier}"

    @staticmethod
    def make_full_id(task_id: str, job_id: str) -> str:
        """Create full job identifier from components.

        This is a static helper for computing full_id when you don't have
        a job instance yet (e.g., for cache lookups).

        Args:
            task_id: Task identifier
            job_id: Job identifier (hash)

        Returns:
            Full job identifier in format "{task_id}:{job_id}"
        """
        return f"{task_id}:{job_id}"

    @staticmethod
    def parse_full_id(full_id: str) -> tuple[str, str]:
        """Parse full job identifier into components.

        Args:
            full_id: Full job identifier in format "{task_id}:{job_id}"

        Returns:
            Tuple of (task_id, job_id)
        """
        parts = full_id.split(":", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", full_id

    # -------------------------------------------------------------------------
    # Static path computation (for use without a job instance)
    # -------------------------------------------------------------------------

    @staticmethod
    def get_scriptname(task_id: str) -> str:
        """Extract script name from task_id (last component after '.')"""
        return task_id.rsplit(".", 1)[-1]

    @staticmethod
    def get_xpm_dir(job_path: Path) -> Path:
        """Get .experimaestro directory path for a job path"""
        return job_path / ".experimaestro"

    @staticmethod
    def get_status_path(job_path: Path) -> Path:
        """Get status file path for a job path"""
        return job_path / ".experimaestro" / "status.json"

    @staticmethod
    def get_pidfile(job_path: Path, scriptname: str) -> Path:
        """Get PID file path"""
        return job_path / f"{scriptname}.pid"

    @staticmethod
    def get_donefile(job_path: Path, scriptname: str) -> Path:
        """Get done marker file path"""
        return job_path / f"{scriptname}.done"

    @staticmethod
    def get_failedfile(job_path: Path, scriptname: str) -> Path:
        """Get failed marker file path"""
        return job_path / f"{scriptname}.failed"

    # -------------------------------------------------------------------------
    # Instance properties (using static methods for consistency)
    # -------------------------------------------------------------------------

    @property
    def scriptname(self) -> str:
        """The script name derived from task_id"""
        return BaseJob.get_scriptname(self.task_id)

    @property
    def xpm_dir(self) -> Path:
        """Path to the .experimaestro directory within job path"""
        return BaseJob.get_xpm_dir(self.path)

    @property
    def status_path(self) -> Path:
        """Path to the job status file"""
        return BaseJob.get_status_path(self.path)

    @property
    def pidfile(self) -> Path:
        """Path to the .pid file"""
        return BaseJob.get_pidfile(self.path, self.scriptname)

    @property
    def donefile(self) -> Path:
        """Path to the .done file"""
        return BaseJob.get_donefile(self.path, self.scriptname)

    @property
    def failedfile(self) -> Path:
        """Path to the .failed file"""
        return BaseJob.get_failedfile(self.path, self.scriptname)

    @property
    def lockpath(self) -> Path:
        """Path to the .lock file for job locking"""
        return self.xpm_dir / f"{self.scriptname}.lock"

    # -------------------------------------------------------------------------
    # State I/O (unified state_dict pattern)
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Get job state as dictionary (single source of truth)

        This is the canonical representation of job state used for both
        serialization to status files and network communication.

        Returns:
            Dictionary with all job state fields
        """
        failure_reason = None
        if self.state and self.state.failure_reason is not None:
            failure_reason = self.state.failure_reason.name

        result = {
            "job_id": self.identifier,
            "task_id": self.task_id,
            "path": str(self.path) if self.path else None,
            "state": self.state.name if self.state else None,
            "failure_reason": failure_reason,
            "submitted_time": serialize_timestamp(self.submittime),
            "started_time": serialize_timestamp(self.starttime),
            "ended_time": serialize_timestamp(self.endtime),
            "exit_code": self.exit_code,
            "retry_count": self.retry_count,
            "progress": [
                p.to_dict() if hasattr(p, "to_dict") else p
                for p in (self.progress or [])
            ],
            "process": self.process_state_dict(),
        }
        # Include carbon_metrics if available
        carbon_metrics = getattr(self, "carbon_metrics", None)
        if carbon_metrics is not None:
            # CarbonMetricsData is a dataclass, convert to dict
            result["carbon_metrics"] = {
                "co2_kg": carbon_metrics.co2_kg,
                "energy_kwh": carbon_metrics.energy_kwh,
                "cpu_power_w": carbon_metrics.cpu_power_w,
                "gpu_power_w": carbon_metrics.gpu_power_w,
                "ram_power_w": carbon_metrics.ram_power_w,
                "duration_s": carbon_metrics.duration_s,
                "region": carbon_metrics.region,
                "is_final": carbon_metrics.is_final,
                "written": carbon_metrics.written,
            }
        # Include events_count only if not None (None = all events processed)
        if self.events_count is not None:
            result["events_count"] = self.events_count
        return result

    def process_state_dict(self) -> dict | None:
        """Get process state as dictionary. Override in subclasses."""
        return None

    def state_changed_event(self) -> "JobStateChangedEvent":
        """Create a JobStateChangedEvent from the current job state.

        This centralizes event creation logic by using state_dict() to avoid
        duplication across the codebase (e.g., in StateListener, finalize_status).

        Returns:
            JobStateChangedEvent with current job state and metadata
        """
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        state_data = self.state_dict()

        # Extract progress from state_dict
        progress = state_data.get("progress", [])

        return JobStateChangedEvent(
            job_id=state_data["job_id"],
            state=state_data["state"],
            failure_reason=state_data.get("failure_reason"),
            submitted_time=state_data.get("submitted_time"),
            started_time=state_data.get("started_time"),
            ended_time=state_data.get("ended_time"),
            exit_code=state_data.get("exit_code"),
            retry_count=state_data.get("retry_count", 0),
            progress=progress,
        )

    def write_status(self) -> None:
        """Write job state to status.json file.

        This is a sync method for writing the canonical state_dict() to disk.
        Call this while holding the job lock to ensure atomic status updates.
        """
        import json

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.state_dict()))

    def apply_event(self, event: "EventBase") -> None:
        """Apply a job event to update this job's state.

        Handles CarbonMetricsEvent, JobStateChangedEvent, and JobProgressEvent.
        """
        from experimaestro.scheduler.state_status import (
            JobStateChangedEvent,
            JobProgressEvent,
            CarbonMetricsEvent,
        )
        from experimaestro.scheduler.state_provider import CarbonMetricsData
        from experimaestro.notifications import LevelInformation

        if isinstance(event, CarbonMetricsEvent):
            self.carbon_metrics = CarbonMetricsData(
                co2_kg=event.co2_kg,
                energy_kwh=event.energy_kwh,
                cpu_power_w=event.cpu_power_w,
                gpu_power_w=event.gpu_power_w,
                ram_power_w=event.ram_power_w,
                duration_s=event.duration_s,
                region=event.region,
                is_final=event.is_final,
                written=event.written,
            )
            logger.debug(
                "Applied carbon metrics to job %s: %.4f kg CO2",
                self.identifier,
                event.co2_kg,
            )

        elif isinstance(event, JobStateChangedEvent):
            new_state = STATE_NAME_TO_JOBSTATE.get(event.state)
            if new_state is not None:
                self.set_state(new_state, loading=True)
            if event.failure_reason:
                try:
                    self._state.failure_reason = JobFailureStatus[event.failure_reason]
                except KeyError:
                    pass
            # Convert ISO string timestamps to datetime
            if event.submitted_time is not None:
                self.submittime = deserialize_to_datetime(event.submitted_time)
            if event.started_time is not None:
                self.starttime = deserialize_to_datetime(event.started_time)
            if event.ended_time is not None:
                self.endtime = deserialize_to_datetime(event.ended_time)
            if event.exit_code is not None:
                self.exit_code = event.exit_code
            if event.retry_count:
                self.retry_count = event.retry_count
            logger.debug(
                "Applied state change to job %s: %s", self.identifier, self.state
            )

        elif isinstance(event, JobProgressEvent):
            level = event.level
            # Truncate to level + 1 entries
            self.progress = self.progress[: (level + 1)]
            # Extend if needed
            while len(self.progress) <= level:
                self.progress.append(LevelInformation(len(self.progress), None, 0.0))
            # Update the level's progress and description
            if event.desc:
                self.progress[-1].desc = event.desc
            self.progress[-1].progress = event.progress
            logger.debug(
                "Applied progress to job %s level %d: %.2f",
                self.identifier,
                level,
                event.progress,
            )

    def _cleanup_event_files(self) -> None:
        """Clean up job event files, applying pending events first.

        1. If events_count is set, reads and applies events from that count onwards
        2. Removes event files at .events/jobs/{task_id}/event-{job_id}-*.jsonl
        3. Sets events_count to None (all events processed)

        Called when a job is about to restart to ensure clean state while
        preserving carbon metrics and other event-based data.
        """
        import json
        from experimaestro.scheduler.state_status import EventBase

        # Get paths for event files
        # job.path is workspace/jobs/task_id/job_id
        workspace_path = self.path.parent.parent.parent
        events_dir = workspace_path / ".events" / "jobs" / self.task_id

        # Apply pending events if events_count is set
        if self.events_count is not None and events_dir.exists():
            try:
                # Job events are stored as event-{job_id}-{count}.jsonl
                # Read events from events_count onwards
                count = self.events_count
                while True:
                    event_file = events_dir / f"event-{self.identifier}-{count}.jsonl"
                    if not event_file.exists():
                        break

                    # Read events from this file
                    with event_file.open("r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    event_dict = json.loads(line)
                                    event = EventBase.from_dict(event_dict)
                                    self.apply_event(event)
                                    logger.debug(
                                        "Applied event %s to job %s",
                                        type(event).__name__,
                                        self.identifier,
                                    )
                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.warning(
                                        "Failed to parse event from %s: %s",
                                        event_file,
                                        e,
                                    )
                    count += 1
            except Exception as e:
                logger.warning(
                    "Failed to apply pending events for job %s: %s",
                    self.identifier,
                    e,
                )

        # Mark all events as processed
        self.events_count = None

        # Find and delete old event files for this job
        if events_dir.exists():
            pattern = f"event-{self.identifier}-*.jsonl"
            for event_file in events_dir.glob(pattern):
                try:
                    event_file.unlink()
                    logger.debug("Removed old job event file: %s", event_file)
                except OSError as e:
                    logger.warning(
                        "Failed to remove job event file %s: %s", event_file, e
                    )

    async def finalize_status(
        self,
        callback: "Callable[[BaseJob], None] | None" = None,
        cleanup_events: bool = False,
    ) -> bool:
        """Finalize job status: load from disk, apply callback, cleanup, and write.

        This method:
        1. Acquires the job lock
        2. Loads state from disk (to get carbon info, etc.)
        3. Calls the callback to modify job state (e.g., set state, increment retry_count)
        4. Optionally cleans up event files (for non-done jobs being restarted)
        5. Writes status if different from disk

        Args:
            callback: Optional function to modify job state after loading from disk.
                      Called with the job as argument.
            cleanup_events: If True, cleanup event files (for job restart scenarios).

        Returns:
            True if the status was written, False if unchanged.
        """
        import json
        from filelock import AsyncFileLock

        async with AsyncFileLock(self.lockpath):
            # Load state from disk (gets carbon info, timestamps, etc.)
            self.load_from_disk()
            logger.debug(
                "finalization: reading dict for %s: %s", self, self.state_dict()
            )

            # Apply callback to modify job state
            if callback is not None:
                callback(self)

            # Cleanup event files if requested (for restart scenarios)
            if cleanup_events:
                self._cleanup_event_files()

                # Write carbon record if we have metrics but not yet written
                if self.carbon_metrics is not None and not self.carbon_metrics.written:
                    try:
                        from experimaestro.carbon.storage import (
                            CarbonRecord,
                            CarbonStorage,
                        )

                        workspace_path = self.path.parent.parent.parent
                        record = CarbonRecord(
                            job_id=self.identifier,
                            task_id=self.task_id,
                            started_at=serialize_timestamp(self.starttime) or "",
                            ended_at=serialize_timestamp(self.endtime) or "",
                            co2_kg=self.carbon_metrics.co2_kg,
                            energy_kwh=self.carbon_metrics.energy_kwh,
                            cpu_power_w=self.carbon_metrics.cpu_power_w,
                            gpu_power_w=self.carbon_metrics.gpu_power_w,
                            ram_power_w=self.carbon_metrics.ram_power_w,
                            duration_s=self.carbon_metrics.duration_s,
                            region=self.carbon_metrics.region,
                        )
                        storage = CarbonStorage(workspace_path)
                        storage.write_record(record)
                        self.carbon_metrics.written = True
                        logger.debug(
                            "Wrote carbon record for job %s from events",
                            self.identifier,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to write carbon record for job %s: %s",
                            self.identifier,
                            e,
                        )

            # Read current state from disk for comparison
            current_dict = None
            if self.status_path.exists():
                try:
                    current_dict = json.loads(self.status_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

            # Get new state and create event
            new_dict = self.state_dict()
            event = self.state_changed_event()

            # Compare and write only if different
            status_written = False
            if current_dict != new_dict:
                self.status_path.parent.mkdir(parents=True, exist_ok=True)
                self.status_path.write_text(json.dumps(new_dict))
                status_written = True

            # After consolidating status.json, write job state event to all experiments
            # This ensures experiment events are immediately available when processing
            # (solves: "job is over but experiment events not yet available")
            if self.experiments:
                for xp in self.experiments:
                    if xp._event_writer is not None:
                        try:
                            xp._event_writer.write_event(event)
                            logger.debug(
                                "Wrote job state event for %s to experiment %s",
                                self.identifier[:8],
                                xp.name,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to write job state event to experiment %s: %s",
                                xp.name,
                                e,
                            )

            return status_written

    def load_from_disk(self):
        """Load job state from disk, prioritizing marker files over status.json

        This method is resilient - it returns False on any error instead of raising.

        Priority for state:
        1. Check .done/.failed marker files (most reliable, written atomically by job)
        2. Fall back to status.json for state info
        3. Check PID file for running state
        4. Default to current state

        Additional fields are loaded from status.json:
        - timestamps (submittime, starttime, endtime)
        - exit_code, retry_count
        - progress, carbon_metrics
        """
        # Load from status.json
        status_dict = None
        if self.status_path.exists():
            try:
                with self.status_path.open() as f:
                    status_dict = json.load(f)
                    self._load_from_status_dict(status_dict)
            except Exception as e:
                logger.debug("Failed to load status.json: %s", e)

        # If no state from any source, use directory mtime as fallback for timestamps
        if status_dict is None:
            self._load_fallback_timestamps()

        # Marker files (.done/.failed) overrides stored state
        if marker_state := JobState.from_path(self.path, self.scriptname):
            # Use marker file's mtime as endtime
            marker_file = (
                self.donefile if marker_state == JobState.DONE else self.failedfile
            )
            try:
                self.endtime = datetime.fromtimestamp(marker_file.stat().st_mtime)
            except OSError:
                pass

            # Load starttime from params.json if not already set
            if self.starttime is None:
                self._load_starttime_from_params()

            self.set_state(marker_state, loading=True)
            self._process_dict = None

        # Check PID file for running state
        elif self.pidfile.exists():
            try:
                self._process_dict = json.loads(self.pidfile.read_text())
                pid = self._process_dict.get("pid")
                if pid is not None:
                    pid = int(pid)
                    # Check if the process is still running
                    try:
                        import psutil

                        proc = psutil.Process(pid)
                        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                            self.set_state(JobState.RUNNING, loading=True)
                    except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass

    def _load_from_status_dict(self, status_dict: Dict[str, Any]) -> None:
        """Load fields from status.json dictionary

        Timestamps are loaded and preserved when calling set_state() with the
        appropriate timestamp parameter based on the target state.
        """
        # Load timestamps
        self.submittime = (
            deserialize_to_datetime(status_dict["submitted_time"])
            if status_dict.get("submitted_time")
            else None
        )
        self.starttime = (
            deserialize_to_datetime(status_dict["started_time"])
            if status_dict.get("started_time")
            else None
        )
        self.endtime = (
            deserialize_to_datetime(status_dict["ended_time"])
            if status_dict.get("ended_time")
            else None
        )

        # Always override fields from status_dict (None if not existent)
        self.exit_code = status_dict.get("exit_code")
        self.retry_count = status_dict.get("retry_count", 0)
        self.events_count = status_dict.get(
            "events_count"
        )  # None = all events processed

        # Determine state from status dict
        state_str = status_dict.get("state", "").lower()
        new_state = STATE_NAME_TO_JOBSTATE.get(state_str)

        # Handle error state with failure reason
        if state_str in ("error", "failed"):
            failure_reason = status_dict.get("failure_reason")
            if failure_reason:
                try:
                    new_state = JobStateError(JobFailureStatus[failure_reason])
                except (KeyError, ValueError):
                    new_state = JobStateError(JobFailureStatus.FAILED)
            else:
                new_state = JobStateError(JobFailureStatus.FAILED)

        # Set state without modifying timestamps (loading from disk)
        if new_state is not None:
            self.set_state(new_state, loading=True)

        # Load process information
        self._process_dict = status_dict.get("process", None)

        # Load progress
        progress_list = status_dict.get("progress", [])
        self._set_progress(progress_list)

        # Load carbon metrics
        carbon_dict = status_dict.get("carbon_metrics")
        if carbon_dict is None:
            self.carbon_metrics = None
        else:
            from experimaestro.carbon.utils import to_float
            from experimaestro.scheduler.state_provider import CarbonMetricsData

            self.carbon_metrics = CarbonMetricsData(
                co2_kg=to_float(carbon_dict.get("co2_kg", 0.0)),
                energy_kwh=to_float(carbon_dict.get("energy_kwh", 0.0)),
                cpu_power_w=to_float(carbon_dict.get("cpu_power_w", 0.0)),
                gpu_power_w=to_float(carbon_dict.get("gpu_power_w", 0.0)),
                ram_power_w=to_float(carbon_dict.get("ram_power_w", 0.0)),
                duration_s=to_float(carbon_dict.get("duration_s", 0.0)),
                region=carbon_dict.get("region", ""),
                is_final=carbon_dict.get("is_final", False),
                written=carbon_dict.get("written", False),
            )

    def _load_fallback_timestamps(self) -> None:
        """Load timestamps from directory mtime as fallback"""
        try:
            mtime = datetime.fromtimestamp(self.path.stat().st_mtime)
            if self.submittime is None:
                self.submittime = mtime
            if self.starttime is None:
                self.starttime = mtime
            if (
                self.state is not None
                and self.state.finished()
                and self.endtime is None
            ):
                self.endtime = mtime
        except OSError:
            pass

    def _load_starttime_from_params(self) -> None:
        """Load starttime from params.json mtime as fallback"""
        params_path = self.path / "params.json"
        try:
            if params_path.exists():
                self.starttime = datetime.fromtimestamp(params_path.stat().st_mtime)
        except OSError:
            pass

    def _clear_transient_fields(self) -> None:
        """Clear transient fields when job will run. Override in subclasses."""
        # Handle different attribute names (_progress vs progress)
        if hasattr(self, "_progress"):
            self._progress = []
        elif hasattr(self, "progress") and not isinstance(
            getattr(type(self), "progress", None), property
        ):
            self.progress = []
        # Clear carbon metrics
        if hasattr(self, "carbon_metrics"):
            self.carbon_metrics = None

    def _set_progress(self, progress_list: List) -> None:
        """Set progress from a list. Override in subclasses if needed."""
        # Handle different attribute names (_progress vs progress)
        if hasattr(self, "_progress"):
            self._progress = progress_list
        elif hasattr(self, "progress") and not isinstance(
            getattr(type(self), "progress", None), property
        ):
            self.progress = progress_list

    async def aio_process(self) -> Optional["Process"]:
        """Returns the process if there is one"""
        if self._process:
            return self._process

        if self.pidpath.is_file():
            # Get from pidpath file
            from experimaestro.connectors import Process

            pinfo = json.loads(self.pidpath.read_text())
            process = Process.fromDefinition(self.launcher.connector, pinfo)
            if process is None:
                return None

            if await process.aio_isrunning():
                self._process = process
                return self._process

            return None

        return None


# =============================================================================
# Base Experiment Interface
# =============================================================================


class BaseExperiment:
    """Base interface for experiment information

    This class defines the interface for experiment data. Both live experiment
    instances and MockExperiment instances should provide these attributes.

    Core attributes:
        workdir: Path to run directory (experiments/{exp-id}/{run-id}/)
        run_id: Run identifier

    State tracking (replaces StatusData):
        jobs: Dict mapping job_id to BaseJob
        services: Dict mapping service_id to BaseService
        tags: Dict mapping job_id to tag dict
        dependencies: Dict mapping job_id to list of dependency job_ids
        events_count: Number of events processed
        hostname: Hostname where experiment runs
        started_at: Start datetime
        ended_at: End datetime (None if running)
    """

    # Status file version
    STATUS_VERSION = 1

    workdir: Path
    run_id: str

    @property
    def experiment_id(self) -> str:
        """Experiment identifier derived from workdir structure"""
        # workdir is experiments/{exp-id}/{run-id}, so parent.name is exp-id
        return self.workdir.parent.name

    @property
    def run_dir(self) -> Path:
        """Path to run directory (same as workdir)"""
        return self.workdir

    @property
    def status(self) -> "ExperimentStatus":
        """Experiment status - override in subclasses"""
        raise NotImplementedError

    # State tracking properties (abstract - must be implemented by subclasses)

    @property
    def jobs(self) -> Dict[str, "BaseJob"]:
        """Jobs in this experiment"""
        raise NotImplementedError

    @property
    def services(self) -> Dict[str, "BaseService"]:
        """Services in this experiment"""
        raise NotImplementedError

    @property
    def tags(self) -> Dict[str, Dict[str, str]]:
        """Tags for jobs"""
        raise NotImplementedError

    @property
    def dependencies(self) -> Dict[str, List[str]]:
        """Job dependencies"""
        raise NotImplementedError

    @property
    def events_count(self) -> int:
        """Number of events processed"""
        raise NotImplementedError

    @property
    def hostname(self) -> Optional[str]:
        """Hostname where experiment runs"""
        raise NotImplementedError

    @property
    def started_at(self) -> Optional[datetime]:
        """Start datetime"""
        raise NotImplementedError

    @property
    def ended_at(self) -> Optional[datetime]:
        """End datetime (None if running)"""
        raise NotImplementedError

    @property
    def carbon_impact(self) -> Optional["CarbonImpactData"]:
        """Carbon impact metrics for this experiment (sum and latest aggregations)"""
        return None  # Default: no carbon metrics

    # Run tags - concrete implementation at base level (set for efficient lookup)
    _run_tags: set[str]

    @property
    def run_tags(self) -> list[str]:
        """Tags assigned to this run (as sorted list for JSON serialization)"""
        return sorted(self._run_tags)

    # Computed properties

    @property
    def total_jobs(self) -> int:
        """Total number of jobs"""
        return len(self.jobs)

    @property
    def finished_jobs(self) -> int:
        """Number of finished jobs"""
        return sum(1 for j in self.jobs.values() if j.state == JobState.DONE)

    @property
    def failed_jobs(self) -> int:
        """Number of failed jobs"""
        return sum(1 for j in self.jobs.values() if j.state.is_error())

    def get_services(self) -> List["BaseService"]:
        """Get services for this experiment as a list"""
        return list(self.services.values())

    @staticmethod
    def get_status_path(run_dir: Path) -> Path:
        """Get status file path for a run directory"""
        return run_dir / "status.json"

    def state_dict(self) -> Dict[str, Any]:
        """Get experiment state as dictionary (single source of truth)

        This is the canonical representation of experiment state used for both
        serialization to status files and network communication.

        Note: Jobs are not included here - they are stored in jobs.jsonl.
        """
        try:
            status_value = self.status.value
        except NotImplementedError:
            status_value = None

        result = {
            "version": self.STATUS_VERSION,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": status_value,
            "hostname": self.hostname,
            "started_at": serialize_timestamp(self.started_at),
            "ended_at": serialize_timestamp(self.ended_at),
            "finished_jobs": self.finished_jobs,
            "failed_jobs": self.failed_jobs,
            "services": {k: v.full_state_dict() for k, v in self.services.items()},
            "run_tags": self.run_tags,
        }
        # Include carbon_impact if available
        if self.carbon_impact:
            result["carbon_impact"] = self.carbon_impact.to_dict()
        return result

    def write_status(self) -> None:
        """Write status.json to disk (calls state_dict internally)

        Uses file locking to ensure atomic writes across processes.
        """
        import filelock

        run_dir = self.run_dir
        if run_dir is None:
            return

        status_path = run_dir / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = status_path.parent / f".{status_path.name}.lock"

        data = self.state_dict()
        data["last_updated"] = datetime.now().isoformat()

        with filelock.FileLock(lock_path):
            temp_path = status_path.with_suffix(".json.tmp")
            with temp_path.open("w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(status_path)

    def apply_event(self, event: "EventBase", merge_mode: bool = False) -> None:
        """Apply an event to update experiment state.

        This is a stub method - concrete implementations (MockExperiment)
        should override this to handle specific event types.

        Args:
            event: Event to apply
            merge_mode: If True, don't overwrite fields that are already set
                       (except for timestamps which should use the latest value)
        """
        # Default implementation does nothing
        # MockExperiment overrides this with actual event handling
        pass

    def _raise_orphaned_events_warning(
        self, events_dir: Path, event_files: list[Path]
    ) -> None:
        """Raise OrphanedEventsError for user confirmation.

        Args:
            events_dir: Directory containing orphaned events
            event_files: List of event file paths

        Raises:
            OrphanedEventsError: Always raises to request user confirmation
        """
        from experimaestro.scheduler.state_status import EventReader, WatchedDirectory
        from experimaestro.locking import OrphanedEventsError

        # Count total events
        event_count = 0
        for event_file in event_files:
            with open(event_file) as f:
                event_count += sum(1 for _ in f)

        description = (
            f"Found {event_count} orphaned events for experiment '{self.experiment_id}' (run: {self.run_id}).\n\n"
            f"The experiment status file exists but doesn't track which events have been processed.\n"
            f"This can happen if:\n"
            f"  • The experiment finished before event tracking was implemented\n"
            f"  • Files were manually modified or copied\n\n"
            f"Action options:\n"
            f"  • Merge Events: Replay all {event_count} events in merge mode (won't overwrite existing fields)\n"
            f"  • Skip & Archive: Archive events without replaying (safe, preserves current state)\n"
            f"  • Cancel: Leave events as-is for manual review"
        )

        # Create callbacks
        def merge_events():
            """Replay all events in merge mode"""
            reader = EventReader(WatchedDirectory(path=events_dir))
            events = reader.read_events_since_count(self.experiment_id, 0)
            for event in events:
                self.apply_event(event, merge_mode=True)
            logger.info(
                "Merged %d orphaned events into experiment %s",
                event_count,
                self.experiment_id,
            )

        def skip_events():
            """Just mark as processed without applying"""
            logger.info(
                "Skipping %d orphaned events for experiment %s",
                event_count,
                self.experiment_id,
            )
            # Do nothing - events will be archived below

        warning_key = f"orphaned_events_{self.experiment_id}_{self.run_id}"
        context = {
            "title": "Orphaned Events Detected",
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "event_count": event_count,
            "event_files": [str(f) for f in event_files],
            "events_dir": str(events_dir),
        }

        actions = {
            "merge": "Merge Events",
            "skip": "Skip & Archive",
            "cancel": "Cancel",
        }

        callbacks = {
            "merge": merge_events,
            "skip": skip_events,
            "cancel": lambda: None,  # Raising the exception will stop processing
        }

        raise OrphanedEventsError(
            warning_key=warning_key,
            description=description,
            actions=actions,
            context=context,
            callbacks=callbacks,
        )

    def _cleanup_experiment_event_files(self) -> None:
        """Clean up experiment event files, applying pending events first.

        1. If events_count is set, reads and applies events from that count onwards
        2. If events_count is None but events exist, raises OrphanedEventsError for user confirmation
        3. Archives event files from .events/experiments/{exp_id}/ to permanent storage
        4. Sets events_count to None (all events processed)

        Called when consolidating experiment state after completion or when cleaning up
        orphaned events.

        Raises:
            OrphanedEventsError: When status.json has no events_count but orphaned events exist
        """
        from experimaestro.scheduler.state_status import EventReader, WatchedDirectory

        # Get workspace path
        # workdir is {workspace}/experiments/{exp-id}/{run-id}, so parent.parent.parent is workspace
        workspace_path = self.workdir.parent.parent.parent
        events_dir = workspace_path / ".events" / "experiments" / self.experiment_id

        # Apply pending events if events_count is set
        try:
            current_events_count = self.events_count
        except NotImplementedError:
            current_events_count = None

        # Check for orphaned events (events exist but no events_count)
        if current_events_count is None and events_dir.exists():
            event_files = list(events_dir.glob("events-*.jsonl"))
            if event_files:
                # Edge case: status.json exists without events_count, but events remain
                # Ask user for confirmation before merging
                self._raise_orphaned_events_warning(events_dir, event_files)

        if current_events_count is not None and events_dir.exists():
            try:
                reader = EventReader(WatchedDirectory(path=events_dir))
                # Event files are at events_dir/events-{count}.jsonl (no subdirectory)
                # so we pass "." as entity_id to read directly from base_dir
                events = reader.read_events_since_count(".", current_events_count)
                for event in events:
                    self.apply_event(event)
                    logger.debug(
                        "Applied event %s to experiment %s",
                        type(event).__name__,
                        self.experiment_id,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to apply pending events for experiment %s: %s",
                    self.experiment_id,
                    e,
                )

        # Mark all events as processed (set events_count to None)
        # For MockExperiment, this sets _events_count
        if hasattr(self, "_events_count"):
            self._events_count = None

        # Archive event files to permanent storage
        if events_dir.exists():
            event_files = list(events_dir.glob("events-*.jsonl"))
            if event_files:
                perm_events_dir = self.run_dir / "events"
                perm_events_dir.mkdir(parents=True, exist_ok=True)

                for event_file in event_files:
                    try:
                        # Rename from "events-N.jsonl" to "event-N.jsonl" (singular)
                        perm_filename = event_file.name.replace("events-", "event-", 1)
                        perm_file = perm_events_dir / perm_filename
                        # Try hardlink first, copy if that fails
                        try:
                            perm_file.hardlink_to(event_file)
                        except (OSError, NotImplementedError):
                            import shutil

                            shutil.copy2(event_file, perm_file)

                        # Delete temp event file
                        event_file.unlink()
                        logger.debug("Archived experiment event file: %s", event_file)
                    except OSError as e:
                        logger.warning(
                            "Failed to archive experiment event file %s: %s",
                            event_file,
                            e,
                        )

    async def finalize_status(
        self,
        callback: "Callable[[BaseExperiment], None] | None" = None,
        cleanup_events: bool = False,
    ) -> bool:
        """Finalize experiment status: load from disk, apply callback, cleanup, and write.

        This method:
        1. Acquires the experiment lock
        2. Loads state from disk (to get latest event-based updates)
        3. Calls the callback to modify experiment state (e.g., set status, mark completed)
        4. Optionally cleans up event files (consolidates orphaned events)
        5. Writes status if different from disk

        Args:
            callback: Optional function to modify experiment state after loading from disk.
                      Called with the experiment as argument.
            cleanup_events: If True, cleanup event files (consolidates orphaned events).

        Returns:
            True if the status was written, False if unchanged.
        """
        import json
        from filelock import AsyncFileLock

        # Get experiment lock path (experiments/{exp-id}/lock)
        experiment_base = self.workdir.parent
        lock_path = experiment_base / "lock"

        async with AsyncFileLock(lock_path):
            # Get status path
            status_path = self.run_dir / "status.json"

            # Load current state from disk for comparison
            current_dict = None
            if status_path.exists():
                try:
                    current_dict = json.loads(status_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

            # Apply callback to modify experiment state
            if callback is not None:
                callback(self)

            # Cleanup event files if requested
            if cleanup_events:
                self._cleanup_experiment_event_files()

            # Get new state
            new_dict = self.state_dict()

            # Compare and write only if different
            status_written = False
            if current_dict != new_dict:
                status_path.parent.mkdir(parents=True, exist_ok=True)
                # Write without events_count field when cleanup_events=True
                if cleanup_events and "events_count" in new_dict:
                    new_dict = {
                        k: v for k, v in new_dict.items() if k != "events_count"
                    }
                status_path.write_text(json.dumps(new_dict, indent=2))
                status_written = True

            return status_written


class BaseService(ABC):
    """Base interface for service information

    This class defines the interface for service data. Both live Service instances
    and MockService instances should provide these attributes and methods.

    Attributes:
        id: Unique identifier for the service
        experiment_id: Experiment this service belongs to
        run_id: Run ID (timestamp format YYYYMMDD_HHMMSS)
        state: Current service state (ServiceState enum or compatible)
    """

    id: str

    @property
    @abstractmethod
    def experiment_id(self) -> str:
        """Experiment this service belongs to"""
        ...

    @property
    @abstractmethod
    def run_id(self) -> str:
        """Run ID (timestamp format YYYYMMDD_HHMMSS)"""
        ...

    @property
    @abstractmethod
    def state(self):
        """Current service state"""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the service"""
        ...

    def state_dict(self) -> dict:
        """Return service state for serialization/recreation"""
        return {}

    def full_state_dict(self) -> Dict[str, Any]:
        """Get service state as dictionary for JSON serialization.

        This method properly serializes Path objects and other non-JSON types.
        """
        return {
            "service_id": self.id,
            "description": self.description(),
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "state_dict": self.state_dict(),
        }

    def to_service(self) -> "BaseService":
        """Convert to a live Service instance.

        For live Service instances, returns self.
        For MockService instances, tries to recreate the service from config.

        Returns:
            A live Service instance, or self if conversion is not possible
        """
        return self  # Default: return self (for live services)

    @property
    def sync_status(self) -> Optional[str]:
        """Return sync status for display.

        Override in subclasses that manage file synchronization (e.g., SSHLocalService).

        Returns:
            Sync status string (e.g., "⟳ Syncing", "✓ 30s") or None if not applicable
        """
        return None

    @property
    def error(self) -> Optional[str]:
        """Return error message if service failed to start.

        Override in subclasses to provide error information.

        Returns:
            Error message string or None if no error
        """
        return None
