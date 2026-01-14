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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from experimaestro.scheduler.transient import TransientMode

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

    def notstarted(self):
        """Returns True if the job hasn't started yet"""
        return self.value <= 2  # READY

    def running(self):
        """Returns True if the job is currently running or scheduled"""
        return self.value == 4 or self.value == 3  # RUNNING or SCHEDULED

    def finished(self):
        """Returns True if the job has finished (success or error)"""
        return self.value >= 5  # DONE or ERROR

    def is_error(self):
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
    state: JobState
    submittime: Optional[datetime]
    starttime: Optional[datetime]
    endtime: Optional[datetime]
    progress: List[Dict]
    exit_code: Optional[int]
    retry_count: int
    transient: "TransientMode"

    @property
    def locator(self) -> str:
        """Full task locator (identifier): {task_id}/{identifier}"""
        return f"{self.task_id}/{self.identifier}"

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
        if (
            self.state
            and self.state.is_error()
            and hasattr(self.state, "failure_reason")
        ):
            fr = self.state.failure_reason
            if fr is not None:
                failure_reason = fr.name

        return {
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

    def process_state_dict(self) -> dict | None:
        """Get process state as dictionary. Override in subclasses."""
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

        return {
            "version": self.STATUS_VERSION,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": status_value,
            "events_count": self.events_count,
            "hostname": self.hostname,
            "started_at": serialize_timestamp(self.started_at),
            "ended_at": serialize_timestamp(self.ended_at),
            "finished_jobs": self.finished_jobs,
            "failed_jobs": self.failed_jobs,
            "services": {k: v.full_state_dict() for k, v in self.services.items()},
        }

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


class BaseService:
    """Base interface for service information

    This class defines the interface for service data. Both live Service instances
    and MockService instances should provide these attributes and methods.

    Attributes:
        id: Unique identifier for the service
        state: Current service state (ServiceState enum or compatible)
    """

    id: str

    @property
    def state(self):
        """Current service state"""
        raise NotImplementedError

    def description(self) -> str:
        """Human-readable description of the service"""
        raise NotImplementedError

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
