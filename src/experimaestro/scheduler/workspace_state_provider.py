"""Filesystem-based state provider implementation

This module provides the concrete implementation of StateProvider that
uses the filesystem for persistent state storage, replacing the SQLite/peewee
based DbStateProvider.

Classes:
- WorkspaceStateProvider: Filesystem-backed state provider (read-only for monitoring)
"""

import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from experimaestro.scheduler.interfaces import (
    BaseService,
    ExperimentRun,
    JobState,
    STATE_NAME_TO_JOBSTATE,
)
from experimaestro.scheduler.state_provider import (
    StateProvider,
    MockJob,
    MockExperiment,
    ProcessInfo,
)
from experimaestro.scheduler.state_status import (
    StatusFile,
    StatusData,
    EventBase,
    JobSubmittedEvent,
    JobStateChangedEvent,
    JobProgressEvent,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    ServiceAddedEvent,
    EventReader,
    WatchedDirectory,
    job_entity_id_extractor,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("xpm.workspace_state")


class WorkspaceStateProvider(StateProvider):
    """Filesystem-based state provider for monitoring experiments

    This provider reads experiment state from status.json and events JSONL files.
    It is read-only and used by TUI/web monitors to observe running and past experiments.

    Singleton per workspace path - use get_instance() to obtain instances.
    """

    _instances: Dict[Path, "WorkspaceStateProvider"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        workspace_path: Path,
        standalone: bool = False,
    ) -> "WorkspaceStateProvider":
        """Get or create singleton instance for workspace

        Args:
            workspace_path: Path to workspace directory
            standalone: If True, start event file watcher automatically

        Returns:
            WorkspaceStateProvider instance
        """
        workspace_path = Path(workspace_path).resolve()

        with cls._lock:
            instance = cls._instances.get(workspace_path)
            if instance is None:
                instance = cls(workspace_path, standalone=standalone)
                cls._instances[workspace_path] = instance
            return instance

    def __init__(self, workspace_path: Path, standalone: bool = False):
        """Initialize workspace state provider

        Args:
            workspace_path: Path to workspace directory
            standalone: If True, start event file watcher automatically
        """
        super().__init__()
        self.workspace_path = Path(workspace_path).resolve()
        self._experiments_dir = self.workspace_path / ".experimaestro" / "experiments"
        self._standalone = standalone

        # Status cache: (experiment_id, run_id) -> StatusData
        # Only caches active experiments (those with event files)
        self._status_cache: Dict[tuple[str, str], StatusData] = {}
        self._status_cache_lock = threading.Lock()

        # Service cache
        self._service_cache: Dict[tuple[str, str], Dict[str, BaseService]] = {}
        self._service_cache_lock = threading.Lock()

        # Event reader (with built-in watching capability)
        self._event_reader: Optional[EventReader] = None
        self._jobs_dir = self.workspace_path / ".experimaestro" / "jobs"
        if standalone:
            self._start_watcher()

    def _start_watcher(self) -> None:
        """Start the event file watcher for experiments and jobs"""
        if self._event_reader is None:
            self._event_reader = EventReader(
                [
                    WatchedDirectory(path=self._experiments_dir),
                    WatchedDirectory(
                        path=self._jobs_dir,
                        glob_pattern="*/event-*-*.jsonl",
                        entity_id_extractor=job_entity_id_extractor,
                    ),
                ]
            )
            # Start buffering before watching to avoid race condition:
            # events arriving before scan_existing_files completes should be
            # queued and processed after initial state is loaded
            self._event_reader.start_buffering()
            self._event_reader.start_watching(
                on_event=self._on_event,
                on_deleted=self._on_deleted,
            )
            self._event_reader.scan_existing_files()
            # Now flush any events that arrived during initialization
            self._event_reader.flush_buffer()

    def _stop_watcher(self) -> None:
        """Stop the event file watcher"""
        if self._event_reader is not None:
            self._event_reader.stop_watching()
            self._event_reader = None

    def _on_event(self, entity_id: str, event: EventBase) -> None:
        """Unified callback for events from file watcher

        Detects whether the event is from experiments or jobs based on event type.
        """
        # Experiment events: JobSubmittedEvent, ServiceAddedEvent
        # Job events: JobStateChangedEvent, JobProgressEvent (can appear in both)
        if isinstance(event, JobSubmittedEvent):
            # This is an experiment event (job submitted to experiment)
            experiment_id = entity_id
            self._apply_event_to_cache(experiment_id, event)
            # Forward the event directly - it already has all needed info
            self._notify_state_listeners(event)
        elif isinstance(event, ServiceAddedEvent):
            # This is an experiment event - forward directly
            experiment_id = entity_id
            self._apply_event_to_cache(experiment_id, event)
            self._notify_state_listeners(event)
        elif isinstance(event, JobStateChangedEvent):
            # Could be experiment event or job event
            # Try as experiment first
            if (self._experiments_dir / entity_id).exists():
                experiment_id = entity_id
                self._apply_event_to_cache(experiment_id, event)
            # Forward the event directly
            self._notify_state_listeners(event)
        elif isinstance(event, JobProgressEvent):
            # Job event (progress from job process) - forward directly
            self._notify_state_listeners(event)

    def _on_deleted(self, entity_id: str) -> None:
        """Unified callback when event files are deleted"""
        # Check if this is an experiment directory
        if (self._experiments_dir / entity_id).exists() or self.get_current_run(
            entity_id
        ):
            # Experiment event files deleted (experiment finalized)
            experiment_id = entity_id
            self._clear_status_cache(experiment_id)
            run_id = self.get_current_run(experiment_id) or ""
            self._notify_state_listeners(
                ExperimentUpdatedEvent(experiment_id=experiment_id)
            )
            self._notify_state_listeners(
                RunUpdatedEvent(experiment_id=experiment_id, run_id=run_id)
            )
        # Job deletion doesn't need special handling

    @property
    def read_only(self) -> bool:
        """This provider is always read-only"""
        return True

    @property
    def is_remote(self) -> bool:
        """This is a local provider"""
        return False

    # =========================================================================
    # Status cache methods
    # =========================================================================

    def _get_cached_status(
        self, experiment_id: str, run_id: str, run_dir: Path
    ) -> StatusData:
        """Get status from cache or load from disk

        For active experiments (with event files), maintains an in-memory cache
        that is updated when events arrive via the file watcher.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            run_dir: Path to run directory

        Returns:
            StatusData with all events applied (contains MockJob and MockService objects)
        """
        cache_key = (experiment_id, run_id)

        with self._status_cache_lock:
            # Check cache first
            if cache_key in self._status_cache:
                return self._status_cache[cache_key]

            # Load from disk - pass workspace_path so StatusData creates MockJob/MockService
            status_file = StatusFile(run_dir, workspace_path=self.workspace_path)
            status = status_file.read()

            # Set workspace_path for event application
            status.workspace_path = self.workspace_path

            # Apply pending events from event files
            # Events are in experiments/{experiment_id}/events-{count}.jsonl
            reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
            events = reader.read_events_since_count(experiment_id, status.events_count)
            for event in events:
                status.apply_event(event)

            # Only cache if experiment is active (has event files)
            # This avoids caching finished experiments that won't change
            if self._has_event_files(experiment_id):
                self._status_cache[cache_key] = status

            return status

    def _has_event_files(self, experiment_id: str) -> bool:
        """Check if experiment has any event files (is active)"""
        # Format: experiments/{experiment_id}/events-*.jsonl
        exp_events_dir = self._experiments_dir / experiment_id
        return exp_events_dir.is_dir() and any(exp_events_dir.glob("events-*.jsonl"))

    def _apply_event_to_cache(self, experiment_id: str, event: EventBase) -> None:
        """Apply an event to the cached status (called by EventFileWatcher)"""
        run_id = self.get_current_run(experiment_id)
        if run_id is None:
            return

        cache_key = (experiment_id, run_id)

        with self._status_cache_lock:
            if cache_key in self._status_cache:
                self._status_cache[cache_key].apply_event(event)

    def _clear_status_cache(self, experiment_id: str) -> None:
        """Clear cached status for an experiment (called when experiment finishes)"""
        with self._status_cache_lock:
            # Remove all cache entries for this experiment
            keys_to_remove = [k for k in self._status_cache if k[0] == experiment_id]
            for key in keys_to_remove:
                del self._status_cache[key]

    # =========================================================================
    # Experiment methods
    # =========================================================================

    def get_experiments(self, since: Optional[datetime] = None) -> List[MockExperiment]:
        """Get list of all experiments (v2 and v1 layouts)"""
        experiments = []
        seen_ids = set()

        # v2 layout: experiments/{exp-id}/{run-id}/
        experiments_base = self.workspace_path / "experiments"
        if experiments_base.exists():
            for exp_dir in experiments_base.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_id = exp_dir.name
                seen_ids.add(experiment_id)
                experiment = self._load_experiment(experiment_id)
                if experiment is not None:
                    # Filter by since if provided
                    if since is not None and experiment.updated_at:
                        try:
                            updated = datetime.fromisoformat(experiment.updated_at)
                            if updated < since:
                                continue
                        except ValueError:
                            pass
                    experiments.append(experiment)

        # v1 layout: xp/{exp-id}/ (with jobs/, jobs.bak/)
        old_xp_dir = self.workspace_path / "xp"
        if old_xp_dir.exists():
            for exp_dir in old_xp_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_id = exp_dir.name
                if experiment_id in seen_ids:
                    continue  # Already loaded from v2

                experiment = self._load_v1_experiment(experiment_id, exp_dir)
                if experiment is not None:
                    if since is not None and experiment.updated_at:
                        try:
                            updated = datetime.fromisoformat(experiment.updated_at)
                            if updated < since:
                                continue
                        except ValueError:
                            pass
                    experiments.append(experiment)

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[MockExperiment]:
        """Get a specific experiment by ID (v2 or v1 layout)"""
        # Try v2 layout first
        experiment = self._load_experiment(experiment_id)
        if experiment is not None:
            return experiment

        # Try v1 layout
        old_exp_dir = self.workspace_path / "xp" / experiment_id
        if old_exp_dir.exists():
            return self._load_v1_experiment(experiment_id, old_exp_dir)

        return None

    def _load_experiment(self, experiment_id: str) -> Optional[MockExperiment]:
        """Load experiment from filesystem"""
        exp_dir = self.workspace_path / "experiments" / experiment_id
        if not exp_dir.exists():
            return None

        # Find current run (latest by directory name or from symlink)
        current_run_id = self.get_current_run(experiment_id)
        if current_run_id is None:
            # No runs yet, return empty experiment
            return MockExperiment(
                workdir=exp_dir,
                current_run_id=None,
                total_jobs=0,
                finished_jobs=0,
                failed_jobs=0,
                updated_at="",
                experiment_id=experiment_id,
            )

        run_dir = exp_dir / current_run_id
        if not run_dir.exists():
            return None

        # Get status from cache or load from disk
        status = self._get_cached_status(experiment_id, current_run_id, run_dir)

        # Count jobs by state (jobs are MockJob objects with JobState enum)
        total_jobs = len(status.jobs)
        finished_jobs = 0
        failed_jobs = 0
        for job in status.jobs.values():
            if job.state == JobState.DONE:
                finished_jobs += 1
            elif job.state == JobState.ERROR:
                failed_jobs += 1

        return MockExperiment(
            workdir=run_dir,
            current_run_id=current_run_id,
            total_jobs=total_jobs,
            finished_jobs=finished_jobs,
            failed_jobs=failed_jobs,
            updated_at=status.last_updated,
            started_at=_parse_timestamp(status.started_at),
            ended_at=_parse_timestamp(status.ended_at),
            hostname=status.hostname,
            experiment_id=experiment_id,
        )

    def _load_v1_experiment(
        self, experiment_id: str, exp_dir: Path
    ) -> Optional[MockExperiment]:
        """Load experiment from v1 layout (xp/{exp-id}/ with jobs/, jobs.bak/)"""
        from experimaestro.scheduler.interfaces import JobState as JobStateClass

        # Count jobs from jobs/ and jobs.bak/ directories
        jobs_dir = exp_dir / "jobs"
        jobs_bak_dir = exp_dir / "jobs.bak"

        total_jobs = 0
        finished_jobs = 0
        failed_jobs = 0
        seen_jobs = set()

        for jdir in [jobs_dir, jobs_bak_dir]:
            if not jdir.exists():
                continue

            for job_link in jdir.glob("*/*"):
                # Job key is task_id/job_id
                key = str(job_link.relative_to(jdir))
                if key in seen_jobs:
                    continue
                seen_jobs.add(key)

                # Resolve symlink to check if job exists
                try:
                    job_path = job_link.resolve()
                    if not job_path.is_dir():
                        continue
                except OSError:
                    # Broken symlink - skip
                    continue

                total_jobs += 1

                # Read state from .done/.failed files
                task_id = job_link.parent.name
                scriptname = task_id.rsplit(".", 1)[-1]
                state = JobStateClass.from_path(job_path, scriptname)

                if state is not None:
                    if state == JobState.DONE:
                        finished_jobs += 1
                    elif state.is_error():
                        failed_jobs += 1

        # Get modification time for updated_at
        try:
            mtime = exp_dir.stat().st_mtime
            updated_at = datetime.fromtimestamp(mtime).isoformat()
        except OSError:
            updated_at = ""

        return MockExperiment(
            workdir=exp_dir,
            current_run_id="v1",  # Mark as v1 experiment
            total_jobs=total_jobs,
            finished_jobs=finished_jobs,
            failed_jobs=failed_jobs,
            updated_at=updated_at,
            experiment_id=experiment_id,
        )

    def _get_v1_jobs(self, experiment_id: str) -> List[MockJob]:
        """Get jobs from v1 experiment layout

        v1 layout: xp/{exp-id}/jobs/{task_id}/{job_hash} -> symlink to jobs/{task_id}/{job_hash}
        """
        from experimaestro.scheduler.interfaces import JobState as JobStateClass

        exp_dir = self.workspace_path / "xp" / experiment_id
        if not exp_dir.exists():
            return []

        jobs = []
        jobs_dir = exp_dir / "jobs"
        jobs_bak_dir = exp_dir / "jobs.bak"

        for jdir in [jobs_dir, jobs_bak_dir]:
            if not jdir.exists():
                continue

            for job_link in jdir.glob("*/*"):
                # Resolve symlinks to get actual job path
                try:
                    job_path = job_link.resolve()
                    if not job_path.is_dir():
                        continue
                except OSError:
                    # Broken symlink
                    continue

                task_id = job_link.parent.name
                job_id = job_link.name

                # Try to load job from metadata (v2 style)
                job = MockJob.from_disk(job_path)
                if job is None:
                    # v1 jobs: read state from .done/.failed files
                    # Script name is the last component of task_id
                    scriptname = task_id.rsplit(".", 1)[-1]
                    state = JobStateClass.from_path(job_path, scriptname)
                    if state is None:
                        state = JobState.UNSCHEDULED

                    # Get modification time for timestamps
                    try:
                        mtime = job_path.stat().st_mtime
                    except OSError:
                        mtime = None

                    job = MockJob(
                        identifier=job_id,
                        task_id=task_id,
                        path=job_path,
                        state=state.name,  # Convert JobState to string name
                        submittime=mtime,
                        starttime=mtime,
                        endtime=mtime if state.finished() else None,
                        progress=[],
                        updated_at="",
                    )
                jobs.append(job)

        return jobs

    def get_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment"""
        runs = []
        exp_dir = self.workspace_path / "experiments" / experiment_id

        # Check for v1 layout first (xp/{exp-id}/ without separate runs)
        v1_exp_dir = self.workspace_path / "xp" / experiment_id
        if v1_exp_dir.exists() and not exp_dir.exists():
            # v1 experiment: return single synthetic run
            exp = self._load_v1_experiment(experiment_id, v1_exp_dir)
            if exp:
                # Get modification time for started_at
                try:
                    mtime = v1_exp_dir.stat().st_mtime
                except OSError:
                    mtime = None

                runs.append(
                    ExperimentRun(
                        run_id="v1",
                        experiment_id=experiment_id,
                        started_at=mtime,
                        ended_at=mtime,
                        status="completed",
                        hostname=None,
                        total_jobs=exp.total_jobs,
                        finished_jobs=exp.finished_jobs,
                        failed_jobs=exp.failed_jobs,
                    )
                )
            return runs

        if not exp_dir.exists():
            return runs

        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            run_id = run_dir.name
            status_file = StatusFile(run_dir, workspace_path=self.workspace_path)
            status = status_file.read()

            # Count jobs by state (jobs are MockJob objects with JobState enum)
            total_jobs = len(status.jobs)
            finished_jobs = 0
            failed_jobs = 0
            for job in status.jobs.values():
                if job.state == JobState.DONE:
                    finished_jobs += 1
                elif job.state == JobState.ERROR:
                    failed_jobs += 1

            runs.append(
                ExperimentRun(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    started_at=status.started_at,
                    ended_at=status.ended_at,
                    status=status.status,
                    hostname=status.hostname,
                    total_jobs=total_jobs,
                    finished_jobs=finished_jobs,
                    failed_jobs=failed_jobs,
                )
            )

        return runs

    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current run ID for an experiment"""
        # Check new symlink location: .experimaestro/experiments/{experiment_id}/current
        exp_events_dir = self._experiments_dir / experiment_id
        symlink = exp_events_dir / "current"
        if symlink.is_symlink():
            try:
                target = symlink.resolve()
                return target.name
            except OSError:
                pass

        # Check legacy symlink location: .experimaestro/experiments/{experiment_id}
        legacy_symlink = self._experiments_dir / experiment_id
        if legacy_symlink.is_symlink():
            try:
                target = legacy_symlink.resolve()
                return target.name
            except OSError:
                pass

        # Fall back to finding latest run directory
        exp_dir = self.workspace_path / "experiments" / experiment_id
        if not exp_dir.exists():
            return None

        runs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
            reverse=True,
        )
        return runs[0].name if runs else None

    # =========================================================================
    # Job methods
    # =========================================================================

    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[MockJob]:
        """Query jobs with optional filters"""
        if experiment_id is None:
            return self.get_all_jobs(state=state, tags=tags, since=since)

        if run_id is None:
            run_id = self.get_current_run(experiment_id)

        # Check for v1 experiment
        if run_id == "v1" or run_id is None:
            v1_exp_dir = self.workspace_path / "xp" / experiment_id
            if v1_exp_dir.exists():
                return self._get_v1_jobs(experiment_id)
            if run_id is None:
                return []

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return []

        # Get status from cache or load from disk (contains MockJob objects)
        status = self._get_cached_status(experiment_id, run_id, run_dir)

        # Filter MockJob objects
        jobs = []
        for job_id, job in status.jobs.items():
            # Apply filters
            if task_id and job.task_id != task_id:
                continue
            # state filter expects string, job.state is JobState enum
            if state:
                state_enum = STATE_NAME_TO_JOBSTATE.get(state)
                if state_enum and job.state != state_enum:
                    continue
            if tags:
                job_tags = status.tags.get(job_id, {})
                if not all(job_tags.get(k) == v for k, v in tags.items()):
                    continue

            jobs.append(job)

        return jobs

    def get_job(
        self, job_id: str, experiment_id: str, run_id: Optional[str] = None
    ) -> Optional[MockJob]:
        """Get a specific job"""
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return None

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return None

        # Get status from cache or load from disk (contains MockJob objects)
        status = self._get_cached_status(experiment_id, run_id, run_dir)

        # Return MockJob directly
        return status.jobs.get(job_id)

    def get_all_jobs(
        self,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[MockJob]:
        """Get all jobs across all experiments"""
        all_jobs = []
        experiments_base = self.workspace_path / "experiments"
        if not experiments_base.exists():
            return all_jobs

        for exp_dir in experiments_base.iterdir():
            if not exp_dir.is_dir():
                continue
            experiment_id = exp_dir.name
            jobs = self.get_jobs(
                experiment_id=experiment_id, state=state, tags=tags, since=since
            )
            all_jobs.extend(jobs)

        return all_jobs

    # =========================================================================
    # Tags and dependencies
    # =========================================================================

    def get_tags_map(
        self, experiment_id: str, run_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get tags map for jobs in an experiment/run"""
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return {}

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return {}

        # Get status from cache or load from disk
        status = self._get_cached_status(experiment_id, run_id, run_dir)

        return status.tags

    def get_dependencies_map(
        self, experiment_id: str, run_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get dependencies map for jobs in an experiment/run"""
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return {}

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return {}

        # Get status from cache or load from disk
        status = self._get_cached_status(experiment_id, run_id, run_dir)

        return status.dependencies

    # =========================================================================
    # Services
    # =========================================================================

    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[BaseService]:
        """Get services for an experiment

        Tries to recreate real Service objects from state_dict, falls back to
        MockService if recreation fails.

        If experiment_id is None, returns services from all experiments.
        """
        if experiment_id is None:
            # Return services from all experiments
            all_services = []
            for exp in self.get_experiments():
                exp_services = self.get_services(exp.experiment_id)
                all_services.extend(exp_services)
            return all_services

        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return []

        cache_key = (experiment_id, run_id)

        with self._service_cache_lock:
            # Check cache
            cached = self._service_cache.get(cache_key)
            if cached is not None:
                return list(cached.values())

            # Fetch and try to recreate services
            services = self._fetch_services_from_storage(experiment_id, run_id)
            # Store experiment_id on services for global view
            for s in services:
                s._experiment_id = experiment_id
                s._run_id = run_id
            self._service_cache[cache_key] = {s.id: s for s in services}
            return services

    def _fetch_services_from_storage(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> List[BaseService]:
        """Fetch services from status.json and try to recreate real Service objects"""
        from experimaestro.scheduler.services import Service

        if experiment_id is None or run_id is None:
            return []

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return []

        # Get status from cache or load from disk
        status = self._get_cached_status(experiment_id, run_id, run_dir)

        services = []
        for service_id, mock_service in status.services.items():
            # Try to recreate service from state_dict
            state_dict = mock_service.state_dict()
            if state_dict and "__class__" in state_dict:
                try:
                    service = Service.from_state_dict(state_dict)
                    # Store experiment info on the service
                    service._experiment_id = experiment_id
                    service._run_id = run_id
                    # Register as listener to emit events when state changes
                    service.add_listener(self)
                    services.append(service)
                    logger.debug("Recreated service %s from state_dict", service_id)
                except Exception as e:
                    # Failed to recreate - use MockService with error description
                    from experimaestro.scheduler.state_provider import MockService

                    service = MockService(
                        service_id=service_id,
                        description_text=f"error: {e}",
                        state_dict_data={},
                        experiment_id=experiment_id,
                        run_id=run_id,
                    )
                    services.append(service)
                    logger.warning(
                        "Failed to recreate service %s from state_dict: %s",
                        service_id,
                        e,
                    )
                    if isinstance(e, ModuleNotFoundError):
                        logger.warning(
                            "Missing module for service recreation. Python Path: %s",
                            sys.path,
                        )
            else:
                # No valid state_dict - use MockService with error
                from experimaestro.scheduler.state_provider import MockService

                service = MockService(
                    service_id=service_id,
                    description_text="error: no state_dict",
                    state_dict_data={},
                    experiment_id=experiment_id,
                    run_id=run_id,
                )
                services.append(service)
                logger.debug("Service %s has no state_dict for recreation", service_id)

        return services

    # =========================================================================
    # Job operations
    # =========================================================================

    def kill_job(self, job: MockJob, perform: bool = False) -> bool:
        """Kill a running job"""
        if not perform:
            return job.state.running()

        process = job.getprocess()
        if process is None:
            return False

        try:
            process.kill()
            return True
        except Exception as e:
            logger.warning("Failed to kill job %s: %s", job.identifier, e)
            return False

    def clean_job(self, job: MockJob, perform: bool = False) -> bool:
        """Clean a finished job"""
        if not job.state.finished():
            return False

        if not perform:
            return True

        try:
            import shutil

            if job.path.exists():
                shutil.rmtree(job.path)
            return True
        except Exception as e:
            logger.warning("Failed to clean job %s: %s", job.identifier, e)
            return False

    # =========================================================================
    # Orphan job detection
    # =========================================================================

    def get_orphan_jobs(self) -> List[MockJob]:
        """Get orphan jobs (jobs not associated with any experiment run)

        Scans workspace/jobs/ for all job directories and compares against
        jobs referenced by experiments (both v1 and v2 layouts).

        Returns:
            List of MockJob objects for jobs that exist on disk but are not
            referenced by any experiment.
        """
        jobs_base = self.workspace_path / "jobs"
        if not jobs_base.exists():
            return []

        # Collect all job paths referenced by experiments
        referenced_jobs = self._collect_referenced_job_paths()

        # Scan workspace/jobs/ for all job directories
        orphan_jobs = []
        for task_dir in jobs_base.iterdir():
            if not task_dir.is_dir():
                continue

            task_id = task_dir.name

            for job_dir in task_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name

                # Resolve to canonical path for comparison
                try:
                    job_path = job_dir.resolve()
                except OSError:
                    continue

                # Check if this job is referenced by any experiment
                if job_path not in referenced_jobs:
                    # This is an orphan job
                    job = MockJob.from_disk(job_path)
                    if job is None:
                        # No metadata file - create minimal MockJob from filesystem
                        job = self._create_mock_job_from_path(job_path, task_id, job_id)
                    orphan_jobs.append(job)

        return orphan_jobs

    def get_stray_jobs(self) -> list[MockJob]:
        """Get stray jobs (running jobs not in the latest run of any experiment)

        A stray job is a running job that was submitted by a previous run of an
        experiment, but the experiment has since been relaunched with different
        parameters (i.e., a new run was started).

        This differs from orphan jobs which considers ALL runs. Stray jobs only
        look at the LATEST run of each experiment.

        Returns:
            List of MockJob objects for running jobs not in any current experiment
        """
        jobs_base = self.workspace_path / "jobs"
        if not jobs_base.exists():
            return []

        # Collect job paths from LATEST runs only
        latest_run_jobs = self._collect_latest_run_job_paths()

        # Scan workspace/jobs/ for all running job directories
        stray_jobs = []
        for task_dir in jobs_base.iterdir():
            if not task_dir.is_dir():
                continue

            task_id = task_dir.name

            for job_dir in task_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name

                # Resolve to canonical path for comparison
                try:
                    job_path = job_dir.resolve()
                except OSError:
                    continue

                # Check if this job is in any latest run
                if job_path not in latest_run_jobs:
                    # Always verify running state from PID file (don't trust metadata)
                    scriptname = task_id.rsplit(".", 1)[-1]
                    actual_state = self._check_running_from_pid(job_path, scriptname)

                    # Only include if the job is actually running
                    if actual_state == JobState.RUNNING:
                        # Create MockJob for the running job
                        job = MockJob.from_disk(job_path)
                        if job is None:
                            job = self._create_mock_job_from_path(
                                job_path, task_id, job_id
                            )
                        else:
                            # Update state to verified running state
                            job.state = JobState.RUNNING
                        stray_jobs.append(job)

        return stray_jobs

    def _collect_latest_run_job_paths(self) -> set[Path]:
        """Collect job paths from the latest run of each experiment only

        Returns:
            Set of resolved job paths that are in the latest run of any experiment
        """
        referenced = set()

        # v2 layout: experiments/{exp-id}/{run-id}/status.json
        experiments_base = self.workspace_path / "experiments"
        if experiments_base.exists():
            for exp_dir in experiments_base.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_id = exp_dir.name

                # Get the latest run for this experiment
                latest_run_id = self.get_current_run(experiment_id)
                if latest_run_id is None:
                    continue

                run_dir = exp_dir / latest_run_id
                if not run_dir.is_dir():
                    continue

                # Read status.json for this run
                status_file = StatusFile(run_dir, workspace_path=self.workspace_path)
                status = status_file.read()

                # Also apply pending events
                # Events are in experiments/{experiment_id}/events-{count}.jsonl
                reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
                events = reader.read_events_since_count(
                    experiment_id, status.events_count
                )
                for event in events:
                    status.apply_event(event)

                # Add all job paths from this run
                for job in status.jobs.values():
                    if job.path:
                        try:
                            referenced.add(job.path.resolve())
                        except OSError:
                            pass

        # v1 layout: only most recent jobs/ (not jobs.bak/)
        old_xp_dir = self.workspace_path / "xp"
        if old_xp_dir.exists():
            for exp_dir in old_xp_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                # Only check current jobs/ (not jobs.bak/)
                jobs_dir = exp_dir / "jobs"
                if not jobs_dir.exists():
                    continue

                for job_link in jobs_dir.glob("*/*"):
                    try:
                        job_path = job_link.resolve()
                        referenced.add(job_path)
                    except OSError:
                        pass

        return referenced

    def _collect_referenced_job_paths(self) -> set[Path]:
        """Collect all job paths referenced by experiments (v1 and v2 layouts)

        Returns:
            Set of resolved job paths that are referenced by at least one experiment
        """
        referenced = set()

        # v2 layout: experiments/{exp-id}/{run-id}/status.json
        experiments_base = self.workspace_path / "experiments"
        if experiments_base.exists():
            for exp_dir in experiments_base.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_id = exp_dir.name

                for run_dir in exp_dir.iterdir():
                    if not run_dir.is_dir() or run_dir.name.startswith("."):
                        continue

                    # Read status.json for this run
                    status_file = StatusFile(
                        run_dir, workspace_path=self.workspace_path
                    )
                    status = status_file.read()

                    # Also apply pending events
                    # Events are in experiments/{experiment_id}/events-{count}.jsonl
                    reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
                    events = reader.read_events_since_count(
                        experiment_id, status.events_count
                    )
                    for event in events:
                        status.apply_event(event)

                    # Add all job paths from this run
                    for job in status.jobs.values():
                        if job.path:
                            try:
                                referenced.add(job.path.resolve())
                            except OSError:
                                pass

        # v1 layout: xp/{exp-id}/jobs/{task_id}/{job_hash} -> symlinks
        old_xp_dir = self.workspace_path / "xp"
        if old_xp_dir.exists():
            for exp_dir in old_xp_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                # Check jobs/ and jobs.bak/ directories
                for jdir_name in ["jobs", "jobs.bak"]:
                    jobs_dir = exp_dir / jdir_name
                    if not jobs_dir.exists():
                        continue

                    for job_link in jobs_dir.glob("*/*"):
                        try:
                            job_path = job_link.resolve()
                            referenced.add(job_path)
                        except OSError:
                            # Broken symlink - skip
                            pass

        return referenced

    def _create_mock_job_from_path(
        self, job_path: Path, task_id: str, job_id: str
    ) -> MockJob:
        """Create a MockJob from a job directory path (when no metadata exists)"""
        from experimaestro.scheduler.interfaces import JobState as JobStateClass

        # Try to determine state from marker files
        scriptname = task_id.rsplit(".", 1)[-1]
        state = JobStateClass.from_path(job_path, scriptname)

        # If no done/failed marker, check if job is running via PID file
        if state is None:
            state = self._check_running_from_pid(job_path, scriptname)

        if state is None:
            state = JobState.UNSCHEDULED

        # Get modification time for timestamps
        try:
            mtime = job_path.stat().st_mtime
        except OSError:
            mtime = None

        return MockJob(
            identifier=job_id,
            task_id=task_id,
            path=job_path,
            state=state.name,
            submittime=mtime,
            starttime=mtime,
            endtime=mtime if state.finished() else None,
            progress=[],
            updated_at="",
        )

    def _check_running_from_pid(
        self, job_path: Path, scriptname: str
    ) -> Optional[JobState]:
        """Check if a job is running by reading its PID file and checking the process

        Args:
            job_path: Path to the job directory
            scriptname: The script name (used for file naming)

        Returns:
            JobState.RUNNING if the process is still running, None otherwise
        """
        pid_file = job_path / f"{scriptname}.pid"
        if not pid_file.exists():
            return None

        try:
            pinfo = json.loads(pid_file.read_text())
            pid = pinfo.get("pid")
            if pid is None:
                return None

            # Ensure pid is an integer (JSON may store it as string)
            pid = int(pid)

            # Check if the process is still running
            try:
                import psutil

                proc = psutil.Process(pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    return JobState.RUNNING
            except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

        return None

    # =========================================================================
    # Process information
    # =========================================================================

    def get_process_info(self, job: MockJob) -> Optional[ProcessInfo]:
        """Get process information for a job

        Returns a ProcessInfo dataclass or None if not available.
        """
        if not job.path or not job.task_id:
            return None

        # Get script name from task_id
        scriptname = job.task_id.rsplit(".", 1)[-1]
        pid_file = job.path / f"{scriptname}.pid"

        if not pid_file.exists():
            return None

        try:
            pinfo = json.loads(pid_file.read_text())
            pid = pinfo.get("pid")
            proc_type = pinfo.get("type", "unknown")

            if pid is None:
                return None

            result = ProcessInfo(pid=pid, type=proc_type, running=False)

            # Try to get more info for running jobs
            if job.state and job.state.running():
                try:
                    import psutil

                    proc = psutil.Process(pid)
                    if proc.is_running():
                        result.running = True
                        # Get CPU and memory usage
                        result.cpu_percent = proc.cpu_percent(interval=0.1)
                        mem_info = proc.memory_info()
                        result.memory_mb = mem_info.rss / (1024 * 1024)
                        result.num_threads = proc.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                except ImportError:
                    pass  # psutil not available

            return result
        except (json.JSONDecodeError, OSError):
            return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the state provider and release resources"""
        self._stop_watcher()

        with self._lock:
            if self.workspace_path in self._instances:
                del self._instances[self.workspace_path]


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse ISO format timestamp to Unix timestamp"""
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


__all__ = [
    "WorkspaceStateProvider",
]
