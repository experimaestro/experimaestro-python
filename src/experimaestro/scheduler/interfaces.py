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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("xpm.interfaces")


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


# FIXME: Get rid of those
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
        locator: Full task locator (identifier)
        path: Path to job directory
        state: Current job state (JobState object or compatible)
        submittime: When job was submitted (Unix timestamp or None)
        starttime: When job started running (Unix timestamp or None)
        endtime: When job finished (Unix timestamp or None)
        progress: List of progress updates
        tags: Dictionary of tag key-value pairs
        exit_code: Process exit code (optional)
        retry_count: Number of retries
    """

    identifier: str
    task_id: str
    locator: str
    path: Path
    state: JobState
    submittime: Optional[float]
    starttime: Optional[float]
    endtime: Optional[float]
    progress: List[Dict]
    tags: Dict[str, str]
    exit_code: Optional[int]
    retry_count: int

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
    def get_metadata_path(job_path: Path) -> Path:
        """Get metadata file path for a job path"""
        return job_path / ".experimaestro" / "information.json"

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
    def metadata_path(self) -> Path:
        """Path to the job metadata file"""
        return BaseJob.get_metadata_path(self.path)

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
    # Metadata I/O
    # -------------------------------------------------------------------------

    def write_metadata(self, **extra_fields) -> None:
        """Write or update job metadata in .experimaestro/information.json file

        Automatically extracts metadata from job attributes (identifier, state,
        submittime, starttime, endtime, retry_count) and writes to the metadata file.

        Performs atomic write using temp file + rename. If metadata exists,
        new fields are merged with existing ones. Updates last_updated timestamp.

        Args:
            **extra_fields: Optional extra fields (e.g., launcher, launcher_job_id, exit_code)
        """
        # Ensure .experimaestro directory exists
        self.xpm_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.metadata_path

        # Read existing metadata
        existing = {}
        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    existing = json.load(f)
            except Exception as e:
                logger.warning(
                    "Failed to read existing metadata from %s: %s", metadata_path, e
                )

        # Build metadata from job attributes
        fields = {
            "job_id": self.identifier,
            "task_id": self.task_id,
            "state": self.state.name if self.state else None,
        }

        # Add timing information if available
        if self.submittime is not None:
            fields["submitted_time"] = self.submittime
        if self.starttime is not None:
            fields["started_time"] = self.starttime
        if self.endtime is not None:
            fields["ended_time"] = self.endtime

        # Add exit code if available
        if self.exit_code is not None:
            fields["exit_code"] = self.exit_code

        # Add retry count
        if hasattr(self, "retry_count"):
            fields["retry_count"] = self.retry_count

        # Merge with extra fields (for launcher info, exit_code, etc.)
        fields.update(extra_fields)

        # Merge with existing and update timestamp
        existing.update(fields)
        existing["last_updated"] = datetime.now().timestamp()

        # Atomic write
        temp_path = metadata_path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w") as f:
                json.dump(existing, f, indent=2)
            temp_path.replace(metadata_path)
            logger.debug("Wrote metadata to %s: %s", metadata_path, list(fields.keys()))
        except Exception as e:
            logger.error("Failed to write metadata to %s: %s", metadata_path, e)
            if temp_path.exists():
                temp_path.unlink()
            raise

    def read_metadata(self) -> Optional[dict]:
        """Read job metadata from .experimaestro/information.json file

        Returns:
            Dictionary of metadata fields, or None if file doesn't exist
        """
        metadata_path = self.metadata_path
        if not metadata_path.exists():
            return None

        try:
            with metadata_path.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read metadata from %s: %s", metadata_path, e)
            return None


# =============================================================================
# Base Experiment Interface
# =============================================================================


class BaseExperiment:
    """Base interface for experiment information

    This class defines the interface for experiment data. Both live experiment
    instances and database-loaded MockExperiment instances should provide these attributes.

    Attributes:
        workdir: Path to experiment directory
        current_run_id: Current/latest run ID (or None)
    """

    workdir: Path
    current_run_id: Optional[str]

    @property
    def experiment_id(self) -> str:
        """Experiment identifier derived from workdir name"""
        return self.workdir.name
