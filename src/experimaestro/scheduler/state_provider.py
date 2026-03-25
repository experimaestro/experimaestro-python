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
from functools import cached_property
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from experimaestro.core.serialization import ExperimentInfo
    from experimaestro.scheduler.state_status import WarningEvent

from experimaestro.scheduler.interfaces import (
    BaseAction,
    BaseJob,
    BaseExperiment,
    BaseService,
    ExperimentJobInformation,
    ExperimentStatus,
    JobState,
    JobStateError,
    JobFailureStatus,
    STATE_NAME_TO_JOBSTATE,
    deserialize_to_datetime,
)
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
        """Initialize state listener management and warning action cache"""
        self._state_listeners: Set[StateListener] = set()
        self._state_listener_lock = threading.Lock()
        # Cache for warning action callbacks: warning_key -> {action_key -> callback}
        self._warning_actions: Dict[str, Dict[str, Callable[[], None]]] = {}
        # Cache for warning metadata: warning_key -> WarningEvent
        self._warning_metadata: Dict[str, "WarningEvent"] = {}
        self._warning_actions_lock = threading.Lock()

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

    def register_warning_actions(
        self,
        warning_key: str,
        actions: Dict[str, Callable[[], None]],
        warning_event: Optional["WarningEvent"] = None,
    ) -> None:
        """Register action callbacks for a warning

        Args:
            warning_key: Unique identifier for the warning
            actions: Dict mapping action_key to callback function
            warning_event: Optional WarningEvent with full metadata for display
        """
        with self._warning_actions_lock:
            self._warning_actions[warning_key] = actions
            if warning_event is not None:
                self._warning_metadata[warning_key] = warning_event
            logger.debug(
                "Registered %d actions for warning '%s'", len(actions), warning_key
            )

    def execute_warning_action(
        self,
        warning_key: str,
        action_key: str,
        experiment_id: str = "",
        run_id: str = "",
    ) -> None:
        """Execute a warning action and emit error event if it fails

        Args:
            warning_key: The warning identifier
            action_key: The action to execute
            experiment_id: Experiment ID for error events
            run_id: Run ID for error events

        Raises:
            KeyError: If warning_key or action_key not found
        """
        from experimaestro.scheduler.state_status import ErrorEvent

        with self._warning_actions_lock:
            if warning_key not in self._warning_actions:
                error_msg = f"Warning '{warning_key}' not found"
                logger.error(error_msg)
                self._notify_state_listeners(
                    ErrorEvent(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        warning_key=warning_key,
                        action_key=action_key,
                        error_message=error_msg,
                    )
                )
                raise KeyError(error_msg)

            actions = self._warning_actions[warning_key]
            if action_key not in actions:
                error_msg = (
                    f"Action '{action_key}' not found for warning '{warning_key}'"
                )
                logger.error(error_msg)
                self._notify_state_listeners(
                    ErrorEvent(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        warning_key=warning_key,
                        action_key=action_key,
                        error_message=error_msg,
                    )
                )
                raise KeyError(error_msg)

            callback = actions[action_key]

        # Execute callback outside the lock
        try:
            logger.info(
                "Executing action '%s' for warning '%s'", action_key, warning_key
            )
            callback()
            logger.info("Action '%s' completed successfully", action_key)

            # Remove warning from metadata and actions (it's been resolved)
            with self._warning_actions_lock:
                self._warning_actions.pop(warning_key, None)
                self._warning_metadata.pop(warning_key, None)
        except Exception as e:
            error_msg = f"Action '{action_key}' failed: {e}"
            logger.error(error_msg, exc_info=True)
            self._notify_state_listeners(
                ErrorEvent(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    warning_key=warning_key,
                    action_key=action_key,
                    error_message=error_msg,
                )
            )
            raise

    def get_unresolved_warnings(self) -> List["WarningEvent"]:
        """Get all unresolved warnings

        Returns:
            List of WarningEvent objects with metadata for all pending warnings
        """
        with self._warning_actions_lock:
            return list(self._warning_metadata.values())

    @staticmethod
    def get_resolved_state(
        job: BaseJob, experiment: BaseExperiment | None
    ) -> tuple[JobState, JobState | None]:
        """Resolve display state from experiment and execution states.

        Returns (resolved_state, scheduler_state_or_none).
        - When there's no conflict: (resolved_state, None)
        - When there's a conflict: (exec_state, exp_state) — caller shows both icons

        Uses JobState.resolve() for dispatch to subclass-specific logic.
        """
        exec_state = job.state
        exp_state = experiment.get_job_state(job.identifier) if experiment else None

        if exp_state is None:
            return exec_state, None

        resolved = exp_state.resolve(exec_state)
        if resolved is not None:
            return resolved, None

        # Conflict: resolve() returned None
        return exec_state, exp_state

    def load_xp_info(
        self, experiment_id: str, run_id: str | None = None
    ) -> "ExperimentInfo":
        """Load all serialized objects from a past experiment run.

        Returns an ExperimentInfo with .jobs and .actions dictionaries.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current/latest run)

        Returns:
            ExperimentInfo with jobs and actions

        Raises:
            FileNotFoundError: If objects.jsonl/configs.json doesn't exist
            NotImplementedError: If the provider doesn't support this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support load_xp_info"
        )

    def load_configs(
        self, experiment_id: str, run_id: str | None = None
    ) -> dict[str, Any]:
        """Load all job configs from a past experiment run.

        .. deprecated::
            Use :meth:`load_xp_info` instead.
        """
        import warnings

        warnings.warn(
            "load_configs() is deprecated, use load_xp_info() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_xp_info(experiment_id, run_id).jobs

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
    def get_experiment_job_info(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, ExperimentJobInformation]:
        """Get experiment-level job info (submittime, transient) for jobs

        Returns:
            Dictionary mapping job_id to ExperimentJobInformation
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
        # Only allow deletion of finished or unscheduled jobs
        if job.state and not (job.state.finished() or job.state.is_unscheduled()):
            return (
                False,
                f"Cannot delete job {job.identifier} (state: {job.state.name})",
            )

        # Use clean_job for the actual deletion
        try:
            if self.clean_job(job, perform=perform):
                if perform:
                    return True, f"Deleted job {job.identifier}"
                else:
                    return True, f"Job {job.identifier} can be deleted"
            else:
                state_name = job.state.name if job.state else "unknown"
                return (
                    False,
                    f"Failed to delete job {job.identifier} (state: {state_name})",
                )
        except Exception as e:
            return False, f"Failed to delete job {job.identifier}: {e}"

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

    def get_display_path(self, job: BaseJob) -> str:
        """Get the path to display/copy for a job

        For local providers, returns the local path.
        For remote providers, returns the remote path.

        Args:
            job: Job to get display path for

        Returns:
            Path string suitable for display and copying
        """
        return str(job.path) if job.path else ""


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
        # Job cache: keyed by "{hash8}:{job_id}" where hash8 = task_id_hash(task_id)
        # This matches the entity_id format from job_entity_id_extractor / EventReader
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
        - CarbonMetricsEvent: Updates job carbon metrics in cache
        - JobSubmittedEvent: Adds new job to cache
        - ExperimentUpdatedEvent: Invalidates experiment cache

        Subclasses may override this for additional logic.
        """
        from experimaestro.scheduler.state_status import (
            CarbonMetricsEvent,
            ExperimentJobStateEvent,
            JobStateChangedEvent,
            JobProgressEvent,
            JobSubmittedEvent,
            ExperimentUpdatedEvent,
        )

        # Handle job state/progress/carbon events
        if isinstance(
            event, (JobStateChangedEvent, JobProgressEvent, CarbonMetricsEvent)
        ):
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

        # Handle ExperimentJobStateEvent with scheduler_state="submitted"
        # (supersedes JobSubmittedEvent)
        elif isinstance(event, ExperimentJobStateEvent):
            if event.scheduler_state == "submitted" and event.task_id:
                from experimaestro.scheduler.state_status import task_id_hash

                cache_key = f"{task_id_hash(event.task_id)}:{event.job_id}"
                with self._job_cache_lock:
                    if cache_key not in self._job_cache:
                        job = MockJob(
                            identifier=event.job_id,
                            task_id=event.task_id,
                            path=None,
                            state="scheduled",
                            starttime=None,
                            endtime=None,
                            progress=[],
                            updated_at="",
                        )
                        self._job_cache[cache_key] = job
                        logger.debug("Added job %s to cache from event", cache_key)

        # Legacy: handle old JobSubmittedEvent from event files
        elif isinstance(event, JobSubmittedEvent):
            if event.task_id:
                from experimaestro.scheduler.state_status import task_id_hash

                cache_key = f"{task_id_hash(event.task_id)}:{event.job_id}"
                with self._job_cache_lock:
                    if cache_key not in self._job_cache:
                        job = MockJob(
                            identifier=event.job_id,
                            task_id=event.task_id,
                            path=None,
                            state="scheduled",
                            starttime=None,
                            endtime=None,
                            progress=[],
                            updated_at="",
                        )
                        self._job_cache[cache_key] = job
                        logger.debug("Added job %s to cache from event", cache_key)

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
        starttime: Optional[datetime],
        endtime: Optional[datetime],
        progress: ProgressInformation,
        updated_at: str,
        exit_code: Optional[int] = None,
        retry_count: int = 0,
        failure_reason: Optional[JobFailureStatus] = None,
        process: dict | None = None,
        carbon_metrics: CarbonMetricsData | None = None,
        run_group_id: str | None = None,
        previous_carbon_metrics: CarbonMetricsData | None = None,
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
        if failure_reason is not None:
            # Create a new JobStateError with the specific failure reason
            # (don't mutate the singleton)
            initial_state = JobStateError(failure_reason)
        if not initial_state.is_unscheduled():
            # State was explicitly provided (from disk), mark as having state
            self._state = initial_state
            self._has_event_state = True
        self.starttime = starttime
        self.endtime = endtime
        self.progress = progress
        self.updated_at = updated_at
        self.exit_code = exit_code
        self.retry_count = retry_count
        self._process_dict = process
        self._process = None  # Cached Process handle (avoids repeated fromDefinition)
        self.carbon_metrics = carbon_metrics
        self.run_group_id = run_group_id
        self._previous_carbon_metrics = previous_carbon_metrics

    @property
    def state(self) -> JobState:
        """Get job state with fallback chain.

        Priority:
        1. _state if set from job events (_has_event_state=True)
        2. _experiment_status if set by experiment events
        3. _state from status.json (may be stale but better than hardcoded default)
        """
        if self._has_event_state:
            return self._state
        if self._experiment_status is not None:
            return self._experiment_status
        return self._state

    @state.setter
    def state(self, new_state: JobState):
        """Set state and mark as having event state."""
        self._has_event_state = True
        self.set_state(new_state)

    @cached_property
    def cache_key(self) -> str:
        """Cache key: {hash8_task_id}:{job_id}

        Matches the entity_id format from job_entity_id_extractor / EventReader.
        """
        from experimaestro.scheduler.state_status import task_id_hash

        return f"{task_id_hash(self.task_id)}:{self.identifier}"

    @property
    def scheduler_state(self) -> JobState:
        """Scheduler lifecycle state for offline jobs.

        Returns _experiment_status (from experiment events) if available,
        otherwise falls back to execution state.
        """
        if self._experiment_status is not None:
            return self._experiment_status
        return self.state

    def set_state(
        self,
        new_state: JobState,
        *,
        loading: bool = False,
    ):
        """Override to track when state comes from a reliable source.

        _has_event_state is NOT set here — it is set explicitly by:
        - apply_event() for job events (highest priority)
        - load_from_disk() for marker files and PID file (reliable on-disk state)

        State loaded from status.json (via _load_from_status_dict) is NOT
        considered reliable since it may be stale from a previous run.
        This ensures experiment events can override stale status.json state.
        """
        super().set_state(new_state, loading=loading)

    def load_from_disk(self):
        """Override to skip expensive Process.fromDefinition() calls.

        In the monitoring context (TUI/web), we trust marker files and
        status.json for state. We still read PID file content for metadata
        but don't check if the process is actually alive — cleanup handles
        stale PID files instead.

        Marker files (.done/.failed) and PID files are reliable indicators
        of actual job state. status.json may be stale from a previous run.
        """
        # Load from status.json (same as parent)
        status_dict = None
        if self.status_path.exists():
            try:
                with self.status_path.open() as f:
                    status_dict = json.load(f)
                    self._load_from_status_dict(status_dict)
            except Exception as e:
                logger.debug("Failed to load status.json: %s", e)

        if status_dict is None:
            self._load_fallback_timestamps()

        # Marker files (.done/.failed) override stored state (same as parent)
        if marker_state := JobState.from_path(self.path, self.scriptname):
            marker_file = (
                self.donefile if marker_state == JobState.DONE else self.failedfile
            )
            try:
                self.endtime = datetime.fromtimestamp(marker_file.stat().st_mtime)
            except OSError:
                pass

            if self.starttime is None:
                self._load_starttime_from_params()

            self.set_state(marker_state, loading=True)
            self._process_dict = None

        # PID file exists — job is SCHEDULED or RUNNING.
        # Trust status.json for the specific state; only infer RUNNING
        # as fallback when status.json didn't provide a state.
        elif self.pidfile.exists():
            try:
                self._process_dict = json.loads(self.pidfile.read_text())
                if self._state.is_unscheduled():
                    self.set_state(JobState.RUNNING, loading=True)
            except Exception:
                pass

        # Set _has_event_state for reliable on-disk sources
        if self.donefile.exists() or self.failedfile.exists() or self.pidfile.exists():
            self._has_event_state = True

    def apply_event(self, event: "EventBase") -> None:
        """Apply a job event to update this job's state.

        All events here come from job event files (.events/jobs/).
        """
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        if isinstance(event, JobStateChangedEvent):
            self._has_event_state = True

        super().apply_event(event)

    def process_state_dict(self) -> dict | None:
        """Get process state as dictionary."""
        return self._process_dict

    def getprocess(self):
        """Get process handle for running job

        This method is used for compatibility with filter expressions and
        for killing running jobs. The result is cached to avoid repeated
        Process.fromDefinition() calls (which can be expensive for SLURM).

        Returns:
            Process instance or None if process info not available
        """
        if self._process is not None:
            return self._process

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
            self._process = Process.fromDefinition(connector, pinfo)
            return self._process
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

        # Restore carbon metrics if present
        carbon_metrics = None
        carbon_dict = d.get("carbon_metrics")
        if carbon_dict:
            carbon_metrics = CarbonMetricsData(**carbon_dict)

        # Restore previous carbon metrics if present
        previous_carbon_metrics = None
        prev_carbon_dict = d.get("previous_carbon_metrics")
        if prev_carbon_dict:
            previous_carbon_metrics = CarbonMetricsData(**prev_carbon_dict)

        job = cls(
            identifier=identifier,
            task_id=task_id,
            path=path,
            state=d["state"],
            starttime=deserialize_to_datetime(d.get("started_time")),
            endtime=deserialize_to_datetime(d.get("ended_time")),
            progress=progress_list,
            updated_at=d.get("updated_at", ""),
            exit_code=d.get("exit_code"),
            retry_count=d.get("retry_count", 0),
            failure_reason=failure_reason,
            process=d.get("process"),
            carbon_metrics=carbon_metrics,
            run_group_id=d.get("run_group_id"),
            previous_carbon_metrics=previous_carbon_metrics,
        )
        job.resumable = d.get("resumable", False)
        return job

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
      from experiment events (ExperimentJobStateEvent). This is used to provide
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
        total_jobs: int = 0,
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
        self._actions: Dict[str, "BaseAction"] = {}
        self._dependencies = dependencies or {}
        self._experiment_id_override = experiment_id_override
        self._finished_jobs = finished_jobs
        self._failed_jobs = failed_jobs
        self._run_tags = run_tags or set()
        self._carbon_impact = carbon_impact
        # Track job states from experiment events
        self._job_states: Dict[str, JobState] = job_states or {}
        # total_jobs from status.json (0 means use len(_job_infos) fallback)
        self._total_jobs = total_jobs
        # Lazy loading support for jobs.jsonl
        self._jobs_jsonl_path: Optional[Path] = None
        self._jobs_jsonl_loaded: bool = True

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

    @status.setter
    def status(self, value: ExperimentStatus) -> None:
        self._status = value

    @property
    def job_infos(self) -> Dict[str, "ExperimentJobInformation"]:
        """Lightweight job info from jobs.jsonl (job_id, task_id, tags, timestamp)

        Lazy-loaded: when created from from_disk() with status.json, the
        jobs.jsonl is not read until this property is first accessed.
        """
        if not self._jobs_jsonl_loaded:
            self._load_jobs_jsonl()
        return self._job_infos

    @property
    def services(self) -> Dict[str, "BaseService"]:
        return self._services

    @property
    def actions(self) -> Dict[str, "BaseAction"]:
        return self._actions

    @property
    def tags(self) -> Dict[str, Dict[str, str]]:
        """Build tags dict from job_infos"""
        return {
            job_id: job_info.tags
            for job_id, job_info in self.job_infos.items()
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
        if self._total_jobs > 0:
            return self._total_jobs
        return max(len(self.job_infos), len(self._job_states))

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

    def _serialize_job_states(self) -> dict[str, str]:
        result = {}
        for job_id, state in self._job_states.items():
            if state.is_error() and state.failure_reason is not None:
                result[job_id] = f"error:{state.failure_reason.name}"
            else:
                result[job_id] = state.name
        return result

    def get_job_state(self, job_id: str) -> JobState | None:
        """Get the tracked state of a job from experiment events.

        Args:
            job_id: The job identifier

        Returns:
            JobState if tracked, None otherwise
        """
        return self._job_states.get(job_id)

    # state_dict() is inherited from BaseExperiment

    def _load_jobs_jsonl(self) -> None:
        """Load job infos from jobs.jsonl (deferred loading)"""
        self._jobs_jsonl_loaded = True
        if self._jobs_jsonl_path is None:
            return
        jobs_jsonl_path = self._jobs_jsonl_path
        if not jobs_jsonl_path.exists():
            return
        try:
            with jobs_jsonl_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        job_info = ExperimentJobInformation.from_dict(record)
                        self._job_infos[job_info.job_id] = job_info
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as e:
            logger.warning("Failed to read %s: %s", jobs_jsonl_path, e)

    @classmethod
    def from_disk(
        cls, run_dir: Path, workspace_path: Path
    ) -> Optional["MockExperiment"]:
        """Load MockExperiment from status.json and jobs.jsonl on disk

        If status.json doesn't exist, attempts to recover experiment state from:
        - Event files in .events/experiments/{experiment_id}/
        - Jobs directory (jobs/ symlinks)
        - Disk state (job status files)

        Args:
            run_dir: Path to the run directory containing status.json
            workspace_path: Workspace path for resolving relative paths

        Returns:
            MockExperiment instance or None if recovery fails
        """
        from experimaestro.locking import create_file_lock

        status_path = run_dir / "status.json"

        # Try to load from status.json first
        if status_path.exists():
            lock_path = status_path.parent / f".{status_path.name}.lock"
            with create_file_lock(lock_path):
                try:
                    with status_path.open("r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to read %s: %s", status_path, e)
                    return None

            # Create experiment from status.json
            exp = cls.from_state_dict(data, workspace_path)

            # Defer jobs.jsonl loading until job_infos is accessed
            jobs_jsonl_path = run_dir / "jobs.jsonl"
            exp._jobs_jsonl_path = jobs_jsonl_path
            exp._jobs_jsonl_loaded = False

            return exp

        # status.json doesn't exist - try to recover from disk
        logger.info(
            "status.json not found at %s, attempting recovery from disk", status_path
        )

        # Get experiment_id and run_id from path
        run_id = run_dir.name
        experiment_id = run_dir.parent.name

        # Create minimal experiment instance
        exp = cls(
            workdir=run_dir,
            run_id=run_id,
            status=ExperimentStatus.RUNNING,
            started_at=None,
            ended_at=None,
            services={},
            experiment_id_override=experiment_id,
        )

        # Try to recover jobs from jobs directory
        jobs_dir = run_dir / "jobs"
        if jobs_dir.exists():
            logger.debug("Recovering jobs from %s", jobs_dir)
            for task_dir in jobs_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name

                for job_link in task_dir.iterdir():
                    if not job_link.is_symlink():
                        continue

                    job_id = job_link.name
                    try:
                        # Resolve symlink to get actual job path
                        job_path = job_link.resolve()

                        # Load job from disk (validates it exists and is loadable)
                        MockJob.from_disk(job_path, task_id, job_id, workspace_path)

                        # Add job info to experiment
                        job_info = ExperimentJobInformation(
                            task_id=task_id,
                            job_id=job_id,
                            tags={},
                        )
                        exp._job_infos[job_id] = job_info
                        logger.debug("Recovered job %s/%s", task_id, job_id)
                    except Exception as e:
                        logger.warning(
                            "Failed to recover job %s/%s: %s", task_id, job_id, e
                        )

        # Assume experiment is done since we're recovering from a crash
        # The events will update the status if needed
        if exp._job_infos:
            exp.status = ExperimentStatus.DONE

        # Set total_jobs from recovered job count
        exp._total_jobs = len(exp._job_infos)

        logger.info(
            "Recovered experiment %s/%s with %d job(s)",
            experiment_id,
            run_id,
            len(exp._job_infos),
        )

        # Write status.json so future loads don't need expensive recovery
        try:
            exp.write_status()
            logger.info(
                "Wrote status.json for recovered experiment %s/%s",
                experiment_id,
                run_id,
            )
        except OSError as e:
            logger.warning(
                "Failed to write status.json for %s/%s: %s", experiment_id, run_id, e
            )

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

        # Parse job_infos (included in RPC responses, not in status.json)
        job_infos_data = d.get("job_infos", {})
        job_infos = {
            job_id: ExperimentJobInformation.from_dict(info)
            if isinstance(info, dict)
            else info
            for job_id, info in job_infos_data.items()
        } or None

        # Parse job_states from status.json and compute counters
        job_states_raw = d.get("job_states", {})
        job_states = {}
        for job_id, state_str in job_states_raw.items():
            if state_str.startswith("error:"):
                reason_name = state_str[6:]
                try:
                    job_states[job_id] = JobStateError(JobFailureStatus[reason_name])
                except KeyError:
                    job_states[job_id] = JobStateError()
            else:
                job_states[job_id] = STATE_NAME_TO_JOBSTATE.get(
                    state_str, JobState.UNSCHEDULED
                )
        if job_states:
            finished_jobs = sum(1 for s in job_states.values() if s == JobState.DONE)
            failed_jobs = sum(1 for s in job_states.values() if s.is_error())
        else:
            # Backward compat: old status.json without job_states
            finished_jobs = d.get("finished_jobs", 0)
            failed_jobs = d.get("failed_jobs", 0)

        exp = cls(
            workdir=workdir,
            run_id=run_id,
            status=status,
            events_count=d.get("events_count", 0),
            hostname=d.get("hostname"),
            started_at=deserialize_to_datetime(d.get("started_at")),
            ended_at=deserialize_to_datetime(d.get("ended_at")),
            job_infos=job_infos,
            services=services,
            dependencies=d.get("dependencies", {}),
            finished_jobs=finished_jobs,
            failed_jobs=failed_jobs,
            run_tags=set(d.get("run_tags", [])),
            carbon_impact=CarbonImpactData.from_dict(d.get("carbon_impact")),
            total_jobs=d.get("total_jobs", 0),
            job_states=job_states,
        )

        # Parse actions from status.json
        actions_data = d.get("actions", {})
        for action_id, action_dict in actions_data.items():
            exp._actions[action_id] = BaseAction.from_dict(action_dict)

        return exp

    def _update_job_state(
        self, job_id: str, job_state: JobState, merge_mode: bool
    ) -> None:
        """Update _job_states and counters for a job state change"""
        if merge_mode and job_id in self._job_states:
            return

        previous_state = self._job_states.get(job_id)
        self._job_states[job_id] = job_state

        if not merge_mode and previous_state != job_state:
            if previous_state is not None:
                if previous_state == JobState.DONE:
                    self._finished_jobs = max(0, self._finished_jobs - 1)
                elif previous_state.is_error():
                    self._failed_jobs = max(0, self._failed_jobs - 1)

            if job_state == JobState.DONE:
                self._finished_jobs += 1
            elif job_state.is_error():
                self._failed_jobs += 1

    def apply_event(self, event: "EventBase", merge_mode: bool = False) -> None:
        """Apply an event to update experiment state

        Args:
            event: Event to apply
            merge_mode: If True, don't overwrite fields that are already set
                       (except for timestamps which should use the latest value)
        """
        from experimaestro.scheduler.state_status import (
            ActionAddedEvent,
            ExperimentJobStateEvent,
            JobSubmittedEvent,
            ServiceAddedEvent,
            ServiceStateChangedEvent,
            RunCompletedEvent,
        )

        if isinstance(event, JobSubmittedEvent):
            # Legacy event type — still supported for reading old event files
            if merge_mode and event.job_id in self._job_infos:
                return  # Already exists, skip

            self._job_infos[event.job_id] = ExperimentJobInformation(
                job_id=event.job_id,
                task_id=event.task_id,
                tags=event.tags or {},
                timestamp=event.timestamp,
            )
            if event.depends_on:
                self._dependencies[event.job_id] = event.depends_on

        elif isinstance(event, ServiceAddedEvent):
            # In merge mode, only add if not already present
            if merge_mode and event.service_id in self._services:
                return  # Already exists, skip

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

        elif isinstance(event, ActionAddedEvent):
            if not (merge_mode and event.action_id in self._actions):
                self._actions[event.action_id] = BaseAction(
                    action_id=event.action_id,
                    description=event.description,
                    action_class=event.action_class,
                )

        elif isinstance(event, ExperimentJobStateEvent):
            # Track scheduler lifecycle state from the scheduler
            job_state = STATE_NAME_TO_JOBSTATE.get(
                event.scheduler_state, JobState.UNSCHEDULED
            )
            if event.failure_reason:
                try:
                    job_state = JobStateError(JobFailureStatus[event.failure_reason])
                except (KeyError, ValueError):
                    pass
            self._update_job_state(event.job_id, job_state, merge_mode)

            # Populate job info from ExperimentJobStateEvent (supersedes
            # JobSubmittedEvent) — only when tags/depends are present
            if event.tags or event.depends_on:
                if not (merge_mode and event.job_id in self._job_infos):
                    self._job_infos[event.job_id] = ExperimentJobInformation(
                        job_id=event.job_id,
                        task_id=event.task_id,
                        tags={t.key: t.value for t in event.tags} if event.tags else {},
                        timestamp=event.timestamp,
                    )
                    if event.depends_on:
                        self._dependencies[event.job_id] = event.depends_on

        elif isinstance(event, RunCompletedEvent):
            # Map status string to ExperimentStatus
            new_status = None
            if event.status in ("completed", "done"):
                new_status = ExperimentStatus.DONE
            elif event.status == "failed":
                new_status = ExperimentStatus.FAILED
            else:
                new_status = ExperimentStatus.RUNNING

            # In merge mode, only update if current status is less final
            if merge_mode:
                # Don't overwrite DONE or FAILED status with RUNNING
                if self._status in (ExperimentStatus.DONE, ExperimentStatus.FAILED):
                    if new_status == ExperimentStatus.RUNNING:
                        return  # Keep the more final status

            self._status = new_status

            # For timestamps, always use the latest (even in merge mode)
            ended_at = deserialize_to_datetime(event.ended_at)
            if ended_at is not None and (
                self._ended_at is None or ended_at > self._ended_at
            ):
                self._ended_at = ended_at


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
    def log_directory(self) -> Optional[Path]:
        if self._live_service is not None:
            return self._live_service.log_directory
        return None

    @property
    def stdout(self) -> Optional[Path]:
        if self._live_service is not None:
            return self._live_service.stdout
        return None

    @property
    def stderr(self) -> Optional[Path]:
        if self._live_service is not None:
            return self._live_service.stderr
        return None

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
