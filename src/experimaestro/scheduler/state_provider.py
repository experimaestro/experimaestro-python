"""State provider interfaces for accessing experiment and job information

This module provides the abstract StateProvider interface and related data classes.
The concrete implementations are in workspace_state_provider.py (WorkspaceStateProvider)
and remote/client.py (SSHStateProviderClient).

Key features:
- StateProvider ABC: Abstract base class for all state providers
- OfflineStateProvider: Base class for cached/persistent state providers
- Mock classes: Concrete implementations for database-loaded state objects
- StateListener: Type alias for listener callbacks

Note: Event classes are defined in state_status.py (EventBase and subclasses).
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
    ExperimentJobInformation,
    ExperimentStatus,
    JobState,
    JobFailureStatus,
    STATE_NAME_TO_JOBSTATE,
    deserialize_to_datetime,
)
from experimaestro.scheduler.transient import TransientMode
from experimaestro.notifications import (
    ProgressInformation,
    get_progress_information_from_dict,
)
from experimaestro.scheduler.state_status import EventBase

logger = logging.getLogger("xpm.state")


# =============================================================================
# Process Information
# =============================================================================


@dataclass
class ProcessInfo:
    """Information about a running or completed process"""

    pid: int
    """Process ID"""

    type: str
    """Process type (e.g., 'local', 'slurm', 'oar')"""

    running: bool = False
    """Whether the process is currently running"""

    cpu_percent: Optional[float] = None
    """CPU usage percentage (if available)"""

    memory_mb: Optional[float] = None
    """Memory usage in MB (if available)"""

    num_threads: Optional[int] = None
    """Number of threads (if available)"""


# Type alias for listener callbacks (uses EventBase from state_status)
StateListener = Callable[[EventBase], None]


# =============================================================================
# State Provider ABC
# =============================================================================


class StateProvider(ABC):
    """Abstract base class for state providers

    Defines the interface that all state providers must implement.
    This enables both local (WorkspaceStateProvider), remote (SSHStateProviderClient),
    and live (Scheduler) providers to be used interchangeably.

    Concrete implementations:

    - Scheduler: Live in-memory state from running experiments
    - OfflineStateProvider: Base for cached/persistent state (WorkspaceStateProvider,
      SSHStateProviderClient)

    State listener management is provided by the base class with default implementations.
    """

    #: Whether this provider is connected to a live scheduler
    is_live: bool = False

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

    def _notify_state_listeners(self, event: EventBase) -> None:
        """Notify all state listeners of an event

        Args:
            event: State change event to broadcast
        """
        with self._state_listener_lock:
            listeners = list(self._state_listeners)

        logger.debug(
            "Notifying %d listeners of %s", len(listeners), type(event).__name__
        )
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.exception("Error in state listener: %s", e)

    def service_state_changed(self, service) -> None:
        """Called when a service's state changes - emit event to listeners

        StateProvider registers itself as a listener on services it returns,
        so this method is called when those services' states change.
        """
        from experimaestro.scheduler.state_status import ServiceStateChangedEvent

        experiment_id = getattr(service, "_experiment_id", "") or ""
        run_id = getattr(service, "_run_id", "") or ""
        state_name = service.state.name if hasattr(service.state, "name") else "UNKNOWN"

        logger.debug(
            "Service %s state changed to %s (experiment=%s)",
            service.id,
            state_name,
            experiment_id,
        )

        event = ServiceStateChangedEvent(
            experiment_id=experiment_id,
            run_id=run_id,
            service_id=service.id,
            state=state_name,
        )
        self._notify_state_listeners(event)

    @abstractmethod
    def get_experiments(self, since: Optional[datetime] = None) -> List[BaseExperiment]:
        """Get list of all experiments"""
        ...

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[BaseExperiment]:
        """Get a specific experiment by ID"""
        ...

    @abstractmethod
    def get_experiment_runs(self, experiment_id: str) -> List[BaseExperiment]:
        """Get all runs for an experiment

        Returns:
            List of BaseExperiment instances (MockExperiment for past runs,
            or live experiment for the current run in Scheduler)
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
    def get_job(self, task_id: str, job_id: str) -> Optional[BaseJob]:
        """Get a job directly by task_id and job_id

        Jobs are stored independently in workspace/jobs/task_id/job_id/,
        so they can be retrieved without knowing which experiment they belong to.

        Args:
            task_id: The task identifier
            job_id: The job identifier (hash)

        Returns:
            The job if found, None otherwise
        """
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

    def get_stray_jobs(self) -> List[BaseJob]:
        """Get stray jobs (running jobs not associated with any active experiment)

        Stray jobs are a subset of orphan jobs - they are orphan jobs that are
        currently running or scheduled. These represent jobs where the experimental
        plan changed but the job process is still running.

        Returns:
            List of running/scheduled jobs not in any active experiment
        """
        # Default implementation: filter orphan jobs to running ones
        return [j for j in self.get_orphan_jobs() if j.state and j.state.running()]

    def delete_job_safely(self, job: BaseJob, perform: bool = True) -> Tuple[bool, str]:
        """Safely delete a job and its data

        Only deletes jobs that are finished (not running). Uses clean_job
        for the actual deletion.

        Args:
            job: The job to delete
            perform: If True, actually perform deletion; if False, just check

        Returns:
            Tuple of (success, message)
        """
        # Check if job is running - cannot delete running jobs
        if job.state and job.state.running():
            return False, f"Cannot delete running job {job.identifier}"

        # Use clean_job for the actual deletion
        if self.clean_job(job, perform=perform):
            if perform:
                return True, f"Deleted job {job.identifier}"
            else:
                return True, f"Job {job.identifier} can be deleted"
        else:
            return False, f"Failed to delete job {job.identifier}"

    def delete_experiment(
        self, experiment_id: str, delete_jobs: bool = False, perform: bool = True
    ) -> Tuple[bool, str]:
        """Delete an experiment and optionally its job data

        Args:
            experiment_id: Experiment identifier to delete
            delete_jobs: If True, also delete job directories (default: False)
            perform: If True, actually perform deletion; if False, just check

        Returns:
            Tuple of (success, message)
        """
        return False, "Not implemented"

    def cleanup_orphan_partials(self, perform: bool = False) -> List[str]:
        """Clean up orphan partial directories"""
        return []

    def get_process_info(self, job: BaseJob) -> Optional[ProcessInfo]:
        """Get process information for a job

        Returns a ProcessInfo dataclass or None if not available.
        """
        return None

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
# Offline State Provider (with service caching)
# =============================================================================


class OfflineStateProvider(StateProvider):
    """State provider for offline/cached state access

    Provides state listener management, job/experiment/service caching shared by
    WorkspaceStateProvider and SSHStateProviderClient.

    This is an intermediate class between StateProvider (the ABC) and concrete
    implementations that need state listener support and caching.

    Caching strategy:
    - Jobs and experiments are cached by their identifiers
    - Events update cached objects in-place to maintain consistency
    - get_jobs/get_experiments return cached objects when available
    """

    def __init__(self):
        """Initialize offline state provider with caches and listener management"""
        super().__init__()  # Initialize state listener management
        self._init_service_cache()
        self._init_job_cache()
        self._init_experiment_cache()

    # =========================================================================
    # Job caching methods
    # =========================================================================

    def _init_job_cache(self) -> None:
        """Initialize job cache - call from subclass __init__"""
        # Job cache: full_id (task_id:job_id) -> MockJob
        self._job_cache: Dict[str, "MockJob"] = {}
        self._job_cache_lock = threading.Lock()

    def _clear_job_cache(self) -> None:
        """Clear the job cache"""
        with self._job_cache_lock:
            self._job_cache.clear()

    def _get_cached_job(self, full_id: str) -> Optional["MockJob"]:
        """Get a job from cache by full_id"""
        with self._job_cache_lock:
            return self._job_cache.get(full_id)

    def _cache_job(self, full_id: str, job: "MockJob") -> None:
        """Add a job to the cache"""
        with self._job_cache_lock:
            self._job_cache[full_id] = job

    def _get_or_load_job(self, full_id: str, *args, **kwargs) -> "MockJob":
        """Get job from cache or load/create it

        Checks cache first. If found, calls _on_cached_job_found (for updates).
        If not found, calls _create_job to create it and caches the result.

        Args:
            full_id: Job full identifier (task_id:job_id)
            *args, **kwargs: Passed to _create_job and _on_cached_job_found
        """
        with self._job_cache_lock:
            cached = self._job_cache.get(full_id)
            if cached is not None:
                self._on_cached_job_found(cached, *args, **kwargs)
                return cached

            job = self._create_job(full_id, *args, **kwargs)
            self._job_cache[full_id] = job
            return job

    def _on_cached_job_found(self, job: "MockJob", *args, **kwargs) -> None:
        """Called when a cached job is found - override to update it"""
        pass

    @abstractmethod
    def _create_job(self, full_id: str, *args, **kwargs) -> "MockJob":
        """Create a job instance - subclasses implement this"""
        ...

    # =========================================================================
    # Experiment caching methods
    # =========================================================================

    def _init_experiment_cache(self) -> None:
        """Initialize experiment cache - call from subclass __init__"""
        # Experiment cache: (experiment_id, run_id) -> MockExperiment
        self._experiment_cache: Dict[tuple[str, str], "MockExperiment"] = {}
        self._experiment_cache_lock = threading.Lock()

    def _clear_experiment_cache_all(self) -> None:
        """Clear the entire experiment cache"""
        with self._experiment_cache_lock:
            self._experiment_cache.clear()

    def _clear_experiment_cache(self, experiment_id: str) -> None:
        """Clear cached experiments for a specific experiment ID

        Removes all cache entries where the experiment_id matches,
        regardless of run_id.
        """
        with self._experiment_cache_lock:
            keys_to_remove = [
                k for k in self._experiment_cache if k[0] == experiment_id
            ]
            for key in keys_to_remove:
                del self._experiment_cache[key]

    def _get_cached_experiment(
        self, experiment_id: str, run_id: str
    ) -> Optional["MockExperiment"]:
        """Get an experiment from cache"""
        with self._experiment_cache_lock:
            return self._experiment_cache.get((experiment_id, run_id))

    def _cache_experiment(
        self, experiment_id: str, run_id: str, exp: "MockExperiment"
    ) -> None:
        """Add an experiment to the cache"""
        with self._experiment_cache_lock:
            self._experiment_cache[(experiment_id, run_id)] = exp

    def _get_or_load_experiment(
        self, experiment_id: str, run_id: str, *args, **kwargs
    ) -> "MockExperiment":
        """Get experiment from cache or load/create it

        Checks cache first. If found, calls _on_cached_experiment_found (for updates).
        If not found, calls _create_experiment to create it and caches the result.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            *args, **kwargs: Passed to _create_experiment and _on_cached_experiment_found
        """
        cache_key = (experiment_id, run_id)

        with self._experiment_cache_lock:
            cached = self._experiment_cache.get(cache_key)
            if cached is not None:
                self._on_cached_experiment_found(cached, *args, **kwargs)
                return cached

            exp = self._create_experiment(experiment_id, run_id, *args, **kwargs)
            self._experiment_cache[cache_key] = exp
            return exp

    def _on_cached_experiment_found(
        self, exp: "MockExperiment", *args, **kwargs
    ) -> None:
        """Called when a cached experiment is found - override to update it"""
        pass

    @abstractmethod
    def _create_experiment(
        self, experiment_id: str, run_id: str, *args, **kwargs
    ) -> "MockExperiment":
        """Create an experiment instance - subclasses implement this"""
        ...

    # =========================================================================
    # Event handling methods
    # =========================================================================

    def apply_event(self, event: "EventBase") -> None:
        """Apply an event to cached jobs and experiments

        This method is called when events are received (from event files
        or via notifications). It updates the cached objects in-place
        to maintain state consistency.

        Handles:
        - JobStateChangedEvent: Updates job state in cache
        - JobProgressEvent: Updates job progress in cache
        - JobSubmittedEvent: Adds new job to cache
        - ExperimentUpdatedEvent: Invalidates experiment cache

        Subclasses may override this for additional logic (e.g., setting
        from_experiment flag in WorkspaceStateProvider).
        """
        from experimaestro.scheduler.state_status import (
            JobStateChangedEvent,
            JobProgressEvent,
            JobSubmittedEvent,
            ExperimentUpdatedEvent,
        )
        from experimaestro.scheduler.interfaces import BaseJob

        # Handle job state/progress events
        if isinstance(event, (JobStateChangedEvent, JobProgressEvent)):
            job_id = event.job_id
            with self._job_cache_lock:
                # Find job in cache by job_id
                for full_id, job in self._job_cache.items():
                    if job.identifier == job_id:
                        job.apply_event(event)
                        logger.debug(
                            "Applied %s to cached job %s",
                            type(event).__name__,
                            full_id,
                        )
                        break

        # Handle job submitted event - add to cache if we have task_id
        elif isinstance(event, JobSubmittedEvent):
            if event.task_id:
                full_id = BaseJob.make_full_id(event.task_id, event.job_id)
                with self._job_cache_lock:
                    if full_id not in self._job_cache:
                        # Create a minimal MockJob for tracking
                        job = MockJob(
                            identifier=event.job_id,
                            task_id=event.task_id,
                            path=None,
                            state="scheduled",
                            submittime=None,
                            starttime=None,
                            endtime=None,
                            progress=[],
                            updated_at="",
                        )
                        self._job_cache[full_id] = job
                        logger.debug("Added job %s to cache from event", full_id)

        # Handle experiment events - invalidate cache to force refresh
        elif isinstance(event, ExperimentUpdatedEvent):
            with self._experiment_cache_lock:
                keys_to_remove = [
                    k for k in self._experiment_cache if k[0] == event.experiment_id
                ]
                for key in keys_to_remove:
                    del self._experiment_cache[key]

    def _apply_event_to_cache(self, event: "EventBase") -> None:
        """Apply an event to cached jobs/experiments

        Convenience method that delegates to apply_event.
        Used by subclasses when handling notifications.
        """
        self.apply_event(event)

    # =========================================================================
    # Service caching methods
    # =========================================================================

    def _init_service_cache(self) -> None:
        """Initialize service cache - call from subclass __init__"""
        self._service_cache: Dict[tuple[str, str], Dict[str, "BaseService"]] = {}
        self._service_cache_lock = threading.Lock()

    def _clear_service_cache(self) -> None:
        """Clear the service cache"""
        with self._service_cache_lock:
            self._service_cache.clear()

    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List["BaseService"]:
        """Get services for an experiment

        Uses caching to preserve service instances (and their URLs) across calls.
        Subclasses can override _get_live_services() for live service support
        and must implement _fetch_services_from_storage() for persistent storage.

        If experiment_id is None, returns services from all experiments.
        """
        # Handle "get all services" case
        if experiment_id is None:
            all_services = []
            for exp in self.get_experiments():
                exp_services = self.get_services(exp.experiment_id)
                all_services.extend(exp_services)
            return all_services

        # Resolve run_id if needed
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return []

        cache_key = (experiment_id, run_id)

        with self._service_cache_lock:
            # Try to get live services (scheduler, etc.) - may return None
            live_services = self._get_live_services(experiment_id, run_id)
            if live_services is not None:
                # Cache and return live services
                self._service_cache[cache_key] = {s.id: s for s in live_services}
                return live_services

            # Check cache
            cached = self._service_cache.get(cache_key)
            if cached is not None:
                return list(cached.values())

            # Fetch from persistent storage (filesystem or remote)
            services = self._fetch_services_from_storage(experiment_id, run_id)
            self._service_cache[cache_key] = {s.id: s for s in services}
            return services

    def _get_live_services(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> Optional[List["BaseService"]]:
        """Get live services if available (e.g., from scheduler).

        Returns None if no live services are available (default).
        Subclasses may override to check for live services.
        """
        return None

    @abstractmethod
    def _fetch_services_from_storage(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> List["BaseService"]:
        """Fetch services from persistent storage (filesystem or remote).

        Called when no live services and cache is empty.
        """
        ...

    # State listener methods (add_listener, remove_listener, _notify_state_listeners)
    # are inherited from StateProvider base class


# =============================================================================
# Mock Classes for Database-Loaded State
# =============================================================================


@dataclass
class CarbonMetricsData:
    """Carbon metrics data for a job."""

    co2_kg: float = 0.0
    energy_kwh: float = 0.0
    cpu_power_w: float = 0.0
    gpu_power_w: float = 0.0
    ram_power_w: float = 0.0
    duration_s: float = 0.0
    region: str = ""
    is_final: bool = False
    written: bool = False
    """True if the carbon record was successfully written to CarbonStorage."""


# Re-export aggregate data classes from carbon module
from experimaestro.carbon.base import CarbonAggregateData, CarbonImpactData  # noqa: E402


class MockJob(BaseJob):
    """Concrete implementation of BaseJob for database-loaded jobs

    This class is used when loading job information from the database,
    as opposed to live Job instances which are created during experiment runs.

    State resolution:
    - If _has_event_state is True, return _state (from job events)
    - Else if _experiment_status is set, return _experiment_status (from experiment)
    - Else return UNSCHEDULED
    """

    def __init__(
        self,
        identifier: str,
        task_id: str,
        path: Path,
        state: str,  # State name string from DB
        submittime: Optional[datetime],
        starttime: Optional[datetime],
        endtime: Optional[datetime],
        progress: ProgressInformation,
        updated_at: str,
        exit_code: Optional[int] = None,
        retry_count: int = 0,
        failure_reason: Optional[JobFailureStatus] = None,
        transient: TransientMode = TransientMode.NONE,
        process: dict | None = None,
        carbon_metrics: CarbonMetricsData | None = None,
    ):
        super().__init__()
        self.identifier = identifier
        self.task_id = task_id
        self.path = path

        # Experiment-provided status (fallback when no job events)
        self._experiment_status: JobState | None = None
        # Track if state was set from job events (takes priority)
        self._has_event_state: bool = False

        # Convert state name to JobState instance
        initial_state = STATE_NAME_TO_JOBSTATE.get(state, JobState.UNSCHEDULED)
        if initial_state != JobState.UNSCHEDULED:
            # State was explicitly provided (from disk), mark as having state
            self._state = initial_state
            self._has_event_state = True
        # Set failure_reason on the state object
        if failure_reason is not None:
            self._state.failure_reason = failure_reason
        self.submittime = submittime
        self.starttime = starttime
        self.endtime = endtime
        self.progress = progress
        self.updated_at = updated_at
        self.exit_code = exit_code
        self.retry_count = retry_count
        self.transient = transient
        self._process_dict = process
        self.carbon_metrics = carbon_metrics

    @property
    def state(self) -> JobState:
        """Get job state with fallback chain.

        Priority:
        1. _state if set from job events (_has_event_state=True)
        2. _experiment_status if set by experiment
        3. UNSCHEDULED as default
        """
        if self._has_event_state:
            return self._state
        if self._experiment_status is not None:
            return self._experiment_status
        return JobState.UNSCHEDULED

    @state.setter
    def state(self, new_state: JobState):
        """Set state and mark as having event state."""
        self._has_event_state = True
        self.set_state(new_state)

    def set_state(
        self,
        new_state: JobState,
        *,
        loading: bool = False,
    ):
        """Override to track when state is set from disk loading.

        Any non-UNSCHEDULED state indicates a real state that should be used.
        """
        if new_state != JobState.UNSCHEDULED:
            self._has_event_state = True
        super().set_state(new_state, loading=loading)

    def apply_event(self, event: "EventBase") -> None:
        """Apply a job event to update this job's state.

        Overrides BaseJob.apply_event to handle from_experiment flag:
        - from_experiment=False (job events): high priority, sets _has_event_state
        - from_experiment=True (experiment events): low priority, only sets _experiment_status
        """
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        if isinstance(event, JobStateChangedEvent):
            if event.from_experiment:
                # Event from experiment events - only update _experiment_status
                # This is lower priority than job events
                job_state = STATE_NAME_TO_JOBSTATE.get(
                    event.state, JobState.UNSCHEDULED
                )
                self._experiment_status = job_state
                # Don't call parent - we only want to update _experiment_status
                return
            else:
                # Event from job events - high priority
                self._has_event_state = True

        # Call parent implementation for non-experiment events
        super().apply_event(event)

    def process_state_dict(self) -> dict | None:
        """Get process state as dictionary."""
        return self._process_dict

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
    def from_state_dict(cls, d: Dict, workspace_path: Path) -> "MockJob":
        """Create MockJob from state dictionary

        Args:
            d: Dictionary from state_dict()
            workspace_path: Workspace path to compute job path if not provided

        Returns:
            MockJob instance
        """
        task_id = d["task_id"]
        identifier = d["job_id"]

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
            submittime=deserialize_to_datetime(d.get("submitted_time")),
            starttime=deserialize_to_datetime(d.get("started_time")),
            endtime=deserialize_to_datetime(d.get("ended_time")),
            progress=progress_list,
            updated_at=d.get("updated_at", ""),
            exit_code=d.get("exit_code"),
            retry_count=d.get("retry_count", 0),
            failure_reason=failure_reason,
            process=d.get("process"),
        )

    @classmethod
    def from_disk(
        cls,
        job_path: Path,
        task_id: str,
        job_id: str,
        workspace_path: Path | None = None,  # noqa: ARG003 - kept for API compatibility
    ) -> "MockJob":
        """Create MockJob from job directory on disk

        Creates a minimal instance and calls load_from_disk() to populate it.

        Args:
            job_path: Path to the job directory
            task_id: Task identifier
            job_id: Job identifier
            workspace_path: Workspace path (unused, kept for compatibility)

        Returns:
            MockJob instance with state loaded from disk
        """
        # Create minimal instance with required fields
        job = cls(
            identifier=job_id,
            task_id=task_id,
            path=job_path,
            state="unscheduled",
            submittime=None,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
        )

        # Load state from disk
        job.load_from_disk()

        return job


class MockExperiment(BaseExperiment):
    """Concrete implementation of BaseExperiment for loaded experiments

    This class is used when loading experiment information from disk,
    as opposed to live experiment instances which are created during runs.

    It stores all experiment state including jobs, services, tags,
    dependencies, and event tracking (replaces StatusData).

    Job state tracking:
    - _job_states: Dict[str, JobState] tracks the latest known state of each job
      from experiment events (JobStateChangedEvent). This is used to provide
      _experiment_status to MockJob instances when they are loaded.
    """

    def __init__(
        self,
        workdir: Path,
        run_id: str,
        *,
        status: ExperimentStatus = ExperimentStatus.RUNNING,
        events_count: int = 0,
        hostname: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        job_infos: Optional[Dict[str, "ExperimentJobInformation"]] = None,
        services: Optional[Dict[str, "MockService"]] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        experiment_id_override: Optional[str] = None,
        finished_jobs: int = 0,
        failed_jobs: int = 0,
        run_tags: Optional[set[str]] = None,
        carbon_impact: Optional["CarbonImpactData"] = None,
        job_states: Optional[Dict[str, JobState]] = None,
    ):
        self.workdir = workdir
        self.run_id = run_id
        self._status = status
        self._events_count = events_count
        self._hostname = hostname
        self._started_at = started_at
        self._ended_at = ended_at
        self._job_infos = job_infos or {}
        self._services = services or {}
        self._dependencies = dependencies or {}
        self._experiment_id_override = experiment_id_override
        self._finished_jobs = finished_jobs
        self._failed_jobs = failed_jobs
        self._run_tags = run_tags or set()
        self._carbon_impact = carbon_impact
        # Track job states from experiment events
        self._job_states: Dict[str, JobState] = job_states or {}

    @property
    def experiment_id(self) -> str:
        """Return experiment_id (overriding base class if needed for v1 layout)"""
        if self._experiment_id_override:
            return self._experiment_id_override
        return super().experiment_id

    # Implement abstract properties from BaseExperiment

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    @property
    def job_infos(self) -> Dict[str, "ExperimentJobInformation"]:
        """Lightweight job info from jobs.jsonl (job_id, task_id, tags, timestamp)"""
        return self._job_infos

    @property
    def services(self) -> Dict[str, "BaseService"]:
        return self._services

    @property
    def tags(self) -> Dict[str, Dict[str, str]]:
        """Build tags dict from job_infos"""
        return {
            job_id: job_info.tags
            for job_id, job_info in self._job_infos.items()
            if job_info.tags
        }

    @property
    def dependencies(self) -> Dict[str, List[str]]:
        return self._dependencies

    @property
    def events_count(self) -> int:
        return self._events_count

    @property
    def hostname(self) -> Optional[str]:
        return self._hostname

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    @property
    def ended_at(self) -> Optional[datetime]:
        return self._ended_at

    @property
    def total_jobs(self) -> int:
        return len(self._job_infos)

    @property
    def finished_jobs(self) -> int:
        return self._finished_jobs

    @property
    def carbon_impact(self) -> Optional["CarbonImpactData"]:
        """Carbon impact metrics for this experiment"""
        return self._carbon_impact

    @property
    def failed_jobs(self) -> int:
        return self._failed_jobs

    def get_job_state(self, job_id: str) -> JobState | None:
        """Get the tracked state of a job from experiment events.

        Args:
            job_id: The job identifier

        Returns:
            JobState if tracked, None otherwise
        """
        return self._job_states.get(job_id)

    # state_dict() is inherited from BaseExperiment

    @classmethod
    def from_disk(
        cls, run_dir: Path, workspace_path: Path
    ) -> Optional["MockExperiment"]:
        """Load MockExperiment from status.json and jobs.jsonl on disk

        Args:
            run_dir: Path to the run directory containing status.json
            workspace_path: Workspace path for resolving relative paths

        Returns:
            MockExperiment instance or None if status.json doesn't exist
        """
        import filelock

        status_path = run_dir / "status.json"
        if not status_path.exists():
            return None

        lock_path = status_path.parent / f".{status_path.name}.lock"
        with filelock.FileLock(lock_path):
            try:
                with status_path.open("r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", status_path, e)
                return None

        # Create experiment from status.json
        exp = cls.from_state_dict(data, workspace_path)

        # Load jobs from jobs.jsonl
        jobs_jsonl_path = run_dir / "jobs.jsonl"
        if jobs_jsonl_path.exists():
            try:
                with jobs_jsonl_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            job_info = ExperimentJobInformation.from_dict(record)
                            exp._job_infos[job_info.job_id] = job_info
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError as e:
                logger.warning("Failed to read %s: %s", jobs_jsonl_path, e)

        return exp

    @classmethod
    def from_state_dict(cls, d: Dict, workspace_path: Path) -> "MockExperiment":
        """Create MockExperiment from state dictionary

        Args:
            d: Dictionary from state_dict()
            workspace_path: Workspace path to compute experiment path if not provided

        Returns:
            MockExperiment instance
        """
        experiment_id = d.get("experiment_id", "")
        run_id = d.get("run_id", "")

        # Use workdir from dict if provided, otherwise compute it
        workdir = d.get("workdir")
        if workdir is None:
            # New layout: experiments/{experiment_id}/{run_id}/
            workdir = workspace_path / "experiments" / experiment_id / run_id
        elif isinstance(workdir, str):
            workdir = Path(workdir)

        # Parse status from string to enum
        status_str = d.get("status", "running")
        try:
            status = ExperimentStatus(status_str)
        except ValueError:
            # Handle legacy status values
            if status_str in ("active", "running"):
                status = ExperimentStatus.RUNNING
            elif status_str in ("completed", "done"):
                status = ExperimentStatus.DONE
            elif status_str == "failed":
                status = ExperimentStatus.FAILED
            else:
                status = ExperimentStatus.RUNNING

        # Parse services from dict (can be list or dict)
        services_data = d.get("services", {})
        if isinstance(services_data, list):
            services = {
                s.get("service_id", ""): MockService.from_full_state_dict(s)
                for s in services_data
            }
        else:
            services = {
                k: MockService.from_full_state_dict(v) for k, v in services_data.items()
            }

        return cls(
            workdir=workdir,
            run_id=run_id,
            status=status,
            events_count=d.get("events_count", 0),
            hostname=d.get("hostname"),
            started_at=deserialize_to_datetime(d.get("started_at")),
            ended_at=deserialize_to_datetime(d.get("ended_at")),
            services=services,
            dependencies=d.get("dependencies", {}),
            finished_jobs=d.get("finished_jobs", 0),
            failed_jobs=d.get("failed_jobs", 0),
            run_tags=set(d.get("run_tags", [])),
            carbon_impact=CarbonImpactData.from_dict(d.get("carbon_impact")),
        )

    def apply_event(self, event: "EventBase") -> None:
        """Apply an event to update experiment state

        Args:
            event: Event to apply
        """
        from experimaestro.scheduler.state_status import (
            JobSubmittedEvent,
            JobStateChangedEvent,
            ServiceAddedEvent,
            ServiceStateChangedEvent,
            RunCompletedEvent,
        )

        if isinstance(event, JobSubmittedEvent):
            # Add lightweight job info (tags are stored in ExperimentJobInformation)
            self._job_infos[event.job_id] = ExperimentJobInformation(
                job_id=event.job_id,
                task_id=event.task_id,
                tags=event.tags or {},
                timestamp=event.timestamp,
            )
            if event.depends_on:
                self._dependencies[event.job_id] = event.depends_on

        elif isinstance(event, ServiceAddedEvent):
            self._services[event.service_id] = MockService(
                service_id=event.service_id,
                description_text=event.description,
                state_dict_data=event.state_dict,
                service_class=event.service_class,
                experiment_id=self.experiment_id,
                run_id=self.run_id,
            )

        elif isinstance(event, ServiceStateChangedEvent):
            # Update service state when it changes
            if event.service_id in self._services:
                self._services[event.service_id]._state_name = event.state

        elif isinstance(event, JobStateChangedEvent):
            # Track individual job states
            job_state = STATE_NAME_TO_JOBSTATE.get(event.state, JobState.UNSCHEDULED)
            self._job_states[event.job_id] = job_state

            # Update finished/failed counters when jobs complete
            if event.state == "done":
                self._finished_jobs += 1
            elif event.state == "error":
                self._failed_jobs += 1

        elif isinstance(event, RunCompletedEvent):
            # Map status string to ExperimentStatus
            if event.status in ("completed", "done"):
                self._status = ExperimentStatus.DONE
            elif event.status == "failed":
                self._status = ExperimentStatus.FAILED
            else:
                self._status = ExperimentStatus.RUNNING
            self._ended_at = event.ended_at


class MockService(BaseService):
    """Mock service object for offline/monitor mode.

    This class provides a service-like interface for services loaded from
    persistent storage. It mimics the Service class interface sufficiently
    for display in the TUI ServicesList widget.

    Live service recreation happens lazily via to_service() when needed.
    Subclasses (e.g., SSHMockService) can override to_service() for
    specialized behavior like path translation and sync management.
    """

    def __init__(
        self,
        service_id: str,
        description_text: str,
        state_dict_data: dict,
        service_class: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        url: Optional[str] = None,
        state: Optional[str] = None,
    ):
        self.id = service_id
        self._description = description_text
        # Default to STOPPED state (services start stopped)
        self._state_name = state or "STOPPED"
        self._state_dict_data = state_dict_data
        self._service_class = service_class
        self._experiment_id = experiment_id or ""
        self._run_id = run_id or ""
        self._url = url
        self._live_service: Optional["BaseService"] = None  # Cached live service
        self._error: Optional[str] = None  # Error message if start failed

    @property
    def experiment_id(self) -> str:
        """Return the experiment ID this service belongs to"""
        return self._experiment_id

    @property
    def run_id(self) -> str:
        """Return the run ID (timestamp format YYYYMMDD_HHMMSS)"""
        return self._run_id

    @property
    def url(self) -> Optional[str]:
        """Return service URL.

        Delegates to live service if available (URL is set when service is started).
        """
        if self._live_service is not None:
            return getattr(self._live_service, "url", None)
        return self._url

    @property
    def state(self):
        """Return state as a ServiceState-like object with a name attribute

        If a live service has been created via to_service(), delegates to its state.
        """
        # If we have a live service, return its state (keeps in sync when started)
        if self._live_service is not None:
            return self._live_service.state

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

    @property
    def sync_status(self) -> Optional[str]:
        """Return sync status for display.

        Delegates to live service if available.
        """
        if self._live_service is not None:
            return self._live_service.sync_status
        return None

    @property
    def error(self) -> Optional[str]:
        """Return error message if service failed to start.

        Delegates to live service if available.
        """
        if self._live_service is not None:
            return self._live_service.error
        return self._error

    def set_error(self, error: Optional[str]) -> None:
        """Set error message and update state to ERROR."""
        self._error = error
        if error:
            self._state_name = "ERROR"

    def set_starting(self) -> None:
        """Set state to STARTING and clear any previous error."""
        self._state_name = "STARTING"
        self._error = None

    def state_dict(self) -> dict:
        """Return service state for recreation"""
        return self._state_dict_data

    def full_state_dict(self) -> dict:
        """Get full state as dictionary for JSON serialization.

        Overrides BaseService.full_state_dict() to preserve the original
        service class name instead of using MockService's class name.
        """
        return {
            "service_id": self.id,
            "description": self._description,
            "class": self._service_class,
            "state_dict": self._state_dict_data,
        }

    @property
    def service_class(self) -> Optional[str]:
        """Return service class name"""
        return self._service_class

    @classmethod
    def from_full_state_dict(cls, d: Dict) -> "MockService":
        """Create MockService from full state dictionary

        Args:
            d: Dictionary from full_state_dict()

        Returns:
            MockService instance (state is always MOCK, not from dict)
        """
        return cls(
            service_id=d["service_id"],
            description_text=d.get("description", ""),
            state_dict_data=d.get("state_dict", {}),
            service_class=d.get("class"),
            experiment_id=d.get("experiment_id"),
            run_id=d.get("run_id"),
            url=d.get("url"),
        )

    def to_service(self) -> "BaseService":
        """Try to recreate a live Service instance from this mock.

        Attempts to recreate the service using the stored configuration.
        The result is cached so subsequent calls return the same instance.
        If recreation fails, sets error and returns self.

        Returns:
            A live Service instance or self if recreation is not possible
        """
        # Return cached live service if available
        if self._live_service is not None:
            return self._live_service

        from experimaestro.scheduler.services import Service

        if not self._service_class:
            self.set_error("No service class stored")
            return self

        try:
            self._live_service = Service.from_state_dict(
                self._service_class, self._state_dict_data
            )
            return self._live_service
        except Exception as e:
            self.set_error(str(e))
            return self


__all__ = [
    # Data classes
    "ProcessInfo",
    "CarbonMetricsData",
    "CarbonAggregateData",  # Re-exported from carbon.base
    "CarbonImpactData",  # Re-exported from carbon.base
    # Listener type alias
    "StateListener",
    # ABC
    "StateProvider",
    "OfflineStateProvider",
    # Mock classes
    "MockJob",
    "MockExperiment",
    "MockService",
]
