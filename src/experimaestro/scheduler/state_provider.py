"""State provider interfaces for accessing experiment and job information

This module provides the abstract StateProvider interface and related data classes.
The concrete implementations are in db_state_provider.py (DbStateProvider) and
remote/client.py (SSHStateProviderClient).

Key features:
- StateProvider ABC: Abstract base class for all state providers
- StateEvent classes: Typed dataclass events for state changes
- Mock classes: Concrete implementations for database-loaded state objects
- StateListener: Type alias for listener callbacks
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple

from experimaestro.scheduler.interfaces import (
    BaseJob,
    BaseExperiment,
    BaseService,
    ExperimentRun,
    JobState,
    JobFailureStatus,
    STATE_NAME_TO_JOBSTATE,
)
from experimaestro.scheduler.transient import TransientMode
from experimaestro.notifications import (
    ProgressInformation,
    get_progress_information_from_dict,
)

logger = logging.getLogger("xpm.state")


# =============================================================================
# State Event Classes
# =============================================================================


@dataclass
class StateEvent:
    """Base class for state change events

    Each subclass represents a specific type of state change.
    """

    pass


@dataclass
class ExperimentUpdatedEvent(StateEvent):
    """Event fired when an experiment is created or updated"""

    experiment_id: str
    experiment: Optional["BaseExperiment"] = None


@dataclass
class RunUpdatedEvent(StateEvent):
    """Event fired when an experiment run is created or updated"""

    experiment_id: str
    run_id: str
    run: Optional["ExperimentRun"] = None


@dataclass
class JobUpdatedEvent(StateEvent):
    """Event fired when a job is created or updated"""

    experiment_id: str
    run_id: str
    job_id: str
    job: Optional["BaseJob"] = None


@dataclass
class JobExperimentUpdatedEvent(StateEvent):
    """Event fired when a job is added to an experiment/run

    This event signals that a job has been associated with an experiment.
    UIs can use this to update their job lists for the affected experiment.
    """

    experiment_id: str
    run_id: str
    job_id: str
    tags: Dict[str, str]  # Tags for this job in this experiment/run
    depends_on: List[str]  # List of job IDs this job depends on


@dataclass
class ServiceUpdatedEvent(StateEvent):
    """Event fired when a service is added or updated"""

    experiment_id: str
    run_id: str
    service_id: str
    service: Optional["BaseService"] = None


# Type alias for listener callbacks
StateListener = Callable[[StateEvent], None]


# =============================================================================
# State Provider ABC
# =============================================================================


class StateProvider(ABC):
    """Abstract base class for state providers

    Defines the interface that all state providers must implement.
    This enables both local (DbStateProvider), remote (SSHStateProviderClient),
    and live (Scheduler) providers to be used interchangeably.

    Concrete implementations:
    - Scheduler: Live in-memory state from running experiments
    - OfflineStateProvider: Base for cached/persistent state (in db_state_provider.py)
      - DbStateProvider: SQLite database-backed state
      - SSHStateProviderClient: Remote SSH-based state

    State listener management is provided by the base class with default implementations.
    """

    def __init__(self) -> None:
        """Initialize state listener management"""
        self._state_listeners: Set[StateListener] = set()
        self._state_listener_lock = threading.Lock()

    def add_listener(self, listener: StateListener) -> None:
        """Register a listener for state change events

        Args:
            listener: Callback function that receives StateEvent objects
        """
        with self._state_listener_lock:
            self._state_listeners.add(listener)

    def remove_listener(self, listener: StateListener) -> None:
        """Unregister a listener

        Args:
            listener: Previously registered callback function
        """
        with self._state_listener_lock:
            self._state_listeners.discard(listener)

    def _notify_state_listeners(self, event: StateEvent) -> None:
        """Notify all state listeners of an event

        Args:
            event: State change event to broadcast
        """
        with self._state_listener_lock:
            listeners = list(self._state_listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.exception("Error in state listener: %s", e)

    @abstractmethod
    def get_experiments(self, since: Optional[datetime] = None) -> List[BaseExperiment]:
        """Get list of all experiments"""
        ...

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[BaseExperiment]:
        """Get a specific experiment by ID"""
        ...

    @abstractmethod
    def get_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment

        Returns:
            List of ExperimentRun dataclass instances with job statistics
        """
        ...

    @abstractmethod
    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current run ID for an experiment"""
        ...

    @abstractmethod
    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[BaseJob]:
        """Query jobs with optional filters"""
        ...

    @abstractmethod
    def get_job(
        self, job_id: str, experiment_id: str, run_id: Optional[str] = None
    ) -> Optional[BaseJob]:
        """Get a specific job"""
        ...

    @abstractmethod
    def get_all_jobs(
        self,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[BaseJob]:
        """Get all jobs across all experiments"""
        ...

    @abstractmethod
    def get_tags_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Get tags map for jobs in an experiment/run

        Tags are stored per (job_id, experiment_id, run_id) in JobTagModel.
        This method returns a map from job_id to {tag_key: tag_value}.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            Dictionary mapping job identifiers to their tags dict
        """
        ...

    @abstractmethod
    def get_dependencies_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Get dependencies map for jobs in an experiment/run

        Dependencies are stored per (job_id, experiment_id, run_id) in JobDependenciesModel.
        This method returns a map from job_id to list of job_ids it depends on.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            Dictionary mapping job identifiers to list of job IDs they depend on
        """
        ...

    @abstractmethod
    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[BaseService]:
        """Get services for an experiment"""
        ...

    # add_listener and remove_listener are implemented in base class

    @abstractmethod
    def kill_job(self, job: BaseJob, perform: bool = False) -> bool:
        """Kill a running job"""
        ...

    @abstractmethod
    def clean_job(self, job: BaseJob, perform: bool = False) -> bool:
        """Clean a finished job"""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the state provider and release resources"""
        ...

    # Optional methods with default implementations

    def sync_path(self, path: str) -> Optional[Path]:
        """Sync a specific path from remote (remote providers only)

        Returns None for local providers or if sync fails.
        """
        return None

    def get_orphan_jobs(self) -> List[BaseJob]:
        """Get orphan jobs (jobs not associated with any experiment run)"""
        return []

    def delete_job_safely(self, job: BaseJob, perform: bool = True) -> Tuple[bool, str]:
        """Safely delete a job and its data"""
        return False, "Not implemented"

    def delete_experiment(
        self, experiment_id: str, perform: bool = True
    ) -> Tuple[bool, str]:
        """Delete an experiment and all its data"""
        return False, "Not implemented"

    def cleanup_orphan_partials(self, perform: bool = False) -> List[str]:
        """Clean up orphan partial directories"""
        return []

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last sync time (for incremental updates)"""
        return None

    @property
    def read_only(self) -> bool:
        """Whether this provider is read-only"""
        return True

    @property
    def is_remote(self) -> bool:
        """Whether this is a remote provider (e.g., SSH)

        Remote providers use periodic refresh instead of push notifications
        and support sync_path for on-demand file synchronization.
        """
        return False


# =============================================================================
# Mock Classes for Database-Loaded State
# =============================================================================


class MockJob(BaseJob):
    """Concrete implementation of BaseJob for database-loaded jobs

    This class is used when loading job information from the database,
    as opposed to live Job instances which are created during experiment runs.
    """

    def __init__(
        self,
        identifier: str,
        task_id: str,
        path: Path,
        state: str,  # State name string from DB
        submittime: Optional[float],
        starttime: Optional[float],
        endtime: Optional[float],
        progress: ProgressInformation,
        updated_at: str,
        exit_code: Optional[int] = None,
        retry_count: int = 0,
        failure_reason: Optional[JobFailureStatus] = None,
        transient: TransientMode = TransientMode.NONE,
    ):
        self.identifier = identifier
        self.task_id = task_id
        self.path = path
        # Convert state name to JobState instance
        self.state = STATE_NAME_TO_JOBSTATE.get(state, JobState.UNSCHEDULED)
        self.submittime = submittime
        self.starttime = starttime
        self.endtime = endtime
        self.progress = progress
        self.updated_at = updated_at
        self.exit_code = exit_code
        self.retry_count = retry_count
        self.failure_reason = failure_reason
        self.transient = transient

    @classmethod
    def from_disk(cls, path: Path) -> Optional["MockJob"]:
        """Create a MockJob by reading metadata from disk

        Args:
            path: Path to the job directory

        Returns:
            MockJob instance if metadata exists, None otherwise
        """
        metadata_path = path / ".xpm_metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with metadata_path.open("r") as f:
                metadata = json.load(f)

            return cls(
                identifier=metadata.get("job_id", path.name),
                task_id=metadata.get(
                    "task_id", path.parent.name if path.parent else "unknown"
                ),
                path=path,
                state=metadata.get("state", "unscheduled"),
                submittime=metadata.get("submitted_time"),
                starttime=metadata.get("started_time"),
                endtime=metadata.get("ended_time"),
                progress=[],  # Progress not stored in metadata
                updated_at=str(metadata.get("last_updated", "")),
                exit_code=metadata.get("exit_code"),
                retry_count=metadata.get("retry_count", 0),
            )
        except Exception as e:
            logger.warning("Failed to read job metadata from %s: %s", path, e)
            return None

    def getprocess(self):
        """Get process handle for running job

        This method is used for compatibility with filter expressions and
        for killing running jobs.

        Returns:
            Process instance or None if process info not available
        """
        from experimaestro.connectors import Process
        from experimaestro.connectors.local import LocalConnector

        # Get script name from task_id (last component after the last dot)
        scriptname = self.task_id.rsplit(".", 1)[-1]
        pid_file = self.path / f"{scriptname}.pid"

        if not pid_file.exists():
            return None

        try:
            connector = LocalConnector.instance()
            pinfo = json.loads(pid_file.read_text())
            return Process.fromDefinition(connector, pinfo)
        except Exception as e:
            logger.warning("Could not get process for job at %s: %s", self.path, e)
            return None

    @classmethod
    def from_db_state_dict(cls, d: Dict, workspace_path: Path) -> "MockJob":
        """Create MockJob from serialized dictionary

        Args:
            d: Dictionary from db_state_dict()
            workspace_path: Workspace path to compute job path if not provided

        Returns:
            MockJob instance
        """
        task_id = d["task_id"]
        identifier = d["identifier"]

        # Use path from dict if it's already a Path, otherwise compute it
        path = d.get("path")
        if path is None:
            path = workspace_path / "jobs" / task_id / identifier
        elif isinstance(path, str):
            path = Path(path)

        failure_reason = None
        if d.get("failure_reason"):
            failure_reason = JobFailureStatus[d["failure_reason"]]

        # Convert progress dicts to LevelInformation objects
        progress_list = get_progress_information_from_dict(d.get("progress", []))

        return cls(
            identifier=identifier,
            task_id=task_id,
            path=path,
            state=d["state"],
            submittime=d.get("submittime"),
            starttime=d.get("starttime"),
            endtime=d.get("endtime"),
            progress=progress_list,
            updated_at=d.get("updated_at", ""),
            exit_code=d.get("exit_code"),
            retry_count=d.get("retry_count", 0),
            failure_reason=failure_reason,
        )


class MockExperiment(BaseExperiment):
    """Concrete implementation of BaseExperiment for database-loaded experiments

    This class is used when loading experiment information from the database,
    as opposed to live experiment instances which are created during runs.
    """

    def __init__(
        self,
        workdir: Path,
        current_run_id: Optional[str],
        total_jobs: int,
        finished_jobs: int,
        failed_jobs: int,
        updated_at: str,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        hostname: Optional[str] = None,
        *,
        experiment_id: Optional[str] = None,
    ):
        self.workdir = workdir
        self.current_run_id = current_run_id
        self.total_jobs = total_jobs
        self.finished_jobs = finished_jobs
        self.failed_jobs = failed_jobs
        self.updated_at = updated_at
        self.started_at = started_at
        self.ended_at = ended_at
        self.hostname = hostname
        self._experiment_id = experiment_id

    @property
    def experiment_id(self) -> str:
        """Experiment identifier (explicitly set or derived from workdir structure)"""
        if self._experiment_id:
            return self._experiment_id
        # In new layout, workdir is experiments/{exp-id}/{run-id}
        # So parent.name gives experiment_id
        return self.workdir.parent.name

    def db_state_dict(self) -> Dict:
        """Serialize experiment to dictionary for DB/network storage

        Overrides BaseExperiment.db_state_dict() to include all MockExperiment fields.
        """
        return {
            "experiment_id": self.experiment_id,
            "workdir": str(self.workdir) if self.workdir else None,
            "current_run_id": self.current_run_id,
            "total_jobs": self.total_jobs,
            "finished_jobs": self.finished_jobs,
            "failed_jobs": self.failed_jobs,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "hostname": self.hostname,
        }

    @classmethod
    def from_db_state_dict(cls, d: Dict, workspace_path: Path) -> "MockExperiment":
        """Create MockExperiment from serialized dictionary

        Args:
            d: Dictionary from db_state_dict()
            workspace_path: Workspace path to compute experiment path if not provided

        Returns:
            MockExperiment instance
        """
        experiment_id = d["experiment_id"]
        current_run_id = d.get("current_run_id")

        # Use workdir from dict if provided, otherwise compute it
        workdir = d.get("workdir")
        if workdir is None:
            # New layout: experiments/{experiment_id}/{run_id}/
            if current_run_id:
                workdir = (
                    workspace_path / "experiments" / experiment_id / current_run_id
                )
            else:
                workdir = workspace_path / "experiments" / experiment_id
        elif isinstance(workdir, str):
            workdir = Path(workdir)

        return cls(
            workdir=workdir,
            current_run_id=d.get("current_run_id"),
            total_jobs=d.get("total_jobs", 0),
            finished_jobs=d.get("finished_jobs", 0),
            failed_jobs=d.get("failed_jobs", 0),
            updated_at=d.get("updated_at", ""),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            hostname=d.get("hostname"),
            experiment_id=experiment_id,
        )


class MockService(BaseService):
    """Mock service object for remote monitoring

    This class provides a service-like interface for services loaded from
    the remote server. It mimics the Service class interface sufficiently
    for display in the TUI ServicesList widget.
    """

    def __init__(
        self,
        service_id: str,
        description_text: str,
        state_dict_data: dict,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        url: Optional[str] = None,
        state: str = "STOPPED",
    ):
        self.id = service_id
        self._description = description_text
        self._state_name = state
        self._state_dict_data = state_dict_data
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.url = url

    @property
    def state(self):
        """Return state as a ServiceState-like object with a name attribute"""
        from experimaestro.scheduler.services import ServiceState

        # Convert state name to ServiceState enum
        try:
            return ServiceState[self._state_name]
        except KeyError:
            # Return a mock object with name attribute for unknown states
            class MockState:
                def __init__(self, name):
                    self.name = name

            return MockState(self._state_name)

    def description(self) -> str:
        """Return service description"""
        return self._description

    def state_dict(self) -> dict:
        """Return state dictionary for service recreation"""
        return self._state_dict_data

    @classmethod
    def from_db_state_dict(cls, d: Dict) -> "MockService":
        """Create MockService from serialized dictionary

        Args:
            d: Dictionary from db_state_dict()

        Returns:
            MockService instance
        """
        return cls(
            service_id=d["service_id"],
            description_text=d.get("description", ""),
            state_dict_data=d.get("state_dict", {}),
            experiment_id=d.get("experiment_id"),
            run_id=d.get("run_id"),
            url=d.get("url"),
            state=d.get("state", "STOPPED"),
        )


__all__ = [
    # Events
    "StateEvent",
    "ExperimentUpdatedEvent",
    "RunUpdatedEvent",
    "JobUpdatedEvent",
    "ServiceUpdatedEvent",
    "StateListener",
    # ABC
    "StateProvider",
    # Mock classes
    "MockJob",
    "MockExperiment",
    "MockService",
]
