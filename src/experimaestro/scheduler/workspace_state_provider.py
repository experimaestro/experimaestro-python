"""Filesystem-based state provider implementation

This module provides the concrete implementation of StateProvider that
uses the filesystem for persistent state storage, replacing the SQLite/peewee
based DbStateProvider.

Classes:
- WorkspaceStateProvider: Filesystem-backed state provider (read-only for monitoring)
- EventFileWatcher: Watches event files and notifies listeners
"""

import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from experimaestro.scheduler.interfaces import (
    BaseService,
    ExperimentRun,
    JobState,
    STATE_NAME_TO_JOBSTATE,
)
from experimaestro.scheduler.state_provider import (
    StateProvider,
    StateEvent,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    JobUpdatedEvent,
    JobExperimentUpdatedEvent,
    ServiceUpdatedEvent,
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
    ServiceAddedEvent,
    read_events_since,
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

        # Event file watcher
        self._event_watcher: Optional[EventFileWatcher] = None
        if standalone:
            self._start_watcher()

    def _start_watcher(self) -> None:
        """Start the event file watcher"""
        if self._event_watcher is None:
            self._event_watcher = EventFileWatcher(self)
            self._event_watcher.start()

    def _stop_watcher(self) -> None:
        """Stop the event file watcher"""
        if self._event_watcher is not None:
            self._event_watcher.stop()
            self._event_watcher = None

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
            events = read_events_since(
                self._experiments_dir, experiment_id, status.events_count
            )
            for event in events:
                status.apply_event(event)

            # Only cache if experiment is active (has event files)
            # This avoids caching finished experiments that won't change
            if self._has_event_files(experiment_id):
                self._status_cache[cache_key] = status

            return status

    def _has_event_files(self, experiment_id: str) -> bool:
        """Check if experiment has any event files (is active)"""
        pattern = f"events-*@{experiment_id}.jsonl"
        return any(self._experiments_dir.glob(pattern))

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
        # Check symlink first
        symlink = self._experiments_dir / experiment_id
        if symlink.is_symlink():
            try:
                target = symlink.resolve()
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
        """
        if experiment_id is None:
            return []

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
                    events = read_events_since(
                        self._experiments_dir, experiment_id, status.events_count
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


class EventFileWatcher(FileSystemEventHandler):
    """Watches event files for changes and notifies listeners

    Monitors workspace/.experimaestro/experiments/ for .jsonl event file changes.
    When events are detected, reads new lines and converts them to StateEvents.
    """

    def __init__(self, state_provider: WorkspaceStateProvider):
        super().__init__()
        self.state_provider = state_provider
        self._observer: Optional[Observer] = None
        self._file_positions: Dict[Path, int] = {}

    def start(self) -> None:
        """Start watching for file changes"""
        # Watch .experimaestro/experiments for event files only
        events_dir = self.state_provider._experiments_dir
        events_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting file watcher on %s", events_dir)

        # Use polling observer with 0.5s interval for reliable cross-platform support
        self._observer = Observer(timeout=0.5)
        self._observer.schedule(self, str(events_dir), recursive=False)
        self._observer.start()

        # Process any existing event files
        self._scan_existing_files()

    def stop(self) -> None:
        """Stop watching for file changes"""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def _scan_existing_files(self) -> None:
        """Scan for existing event files and set initial positions"""
        experiments_dir = self.state_provider._experiments_dir
        for path in experiments_dir.glob("events-*@*.jsonl"):
            # Just record the position at end of file
            try:
                self._file_positions[path] = path.stat().st_size
            except OSError:
                pass

    def on_any_event(self, event) -> None:
        """Log all file system events for debugging"""
        logger.debug("FSEvent: %s %s", event.event_type, event.src_path)

    def on_modified(self, event) -> None:
        """Handle file modification events"""
        if event.is_directory:
            return

        path = Path(event.src_path)
        logger.debug("File modified: %s", path)
        if path.suffix == ".jsonl" and path.name.startswith("events-"):
            self._process_event_file(path)

    def on_created(self, event) -> None:
        """Handle file creation events"""
        if event.is_directory:
            return

        path = Path(event.src_path)
        logger.debug("File created: %s", path)
        if path.suffix == ".jsonl" and path.name.startswith("events-"):
            self._file_positions[path] = 0
            self._process_event_file(path)

    def on_deleted(self, event) -> None:
        """Handle file deletion events - triggers when experiment finalizes"""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix == ".jsonl" and path.name.startswith("events-"):
            # Extract experiment_id from filename: events-{count}@{experiment_id}.jsonl
            filename = path.stem
            parts = filename.split("@", 1)
            if len(parts) == 2:
                experiment_id = parts[1]
                # Clean up position tracking
                self._file_positions.pop(path, None)
                # Clear status cache for this experiment (it's finalized)
                self.state_provider._clear_status_cache(experiment_id)
                # Notify that experiment was updated (finalized)
                run_id = self.state_provider.get_current_run(experiment_id) or ""
                self.state_provider._notify_state_listeners(
                    ExperimentUpdatedEvent(
                        experiment_id=experiment_id,
                        experiment=None,  # Will be loaded on demand
                    )
                )
                self.state_provider._notify_state_listeners(
                    RunUpdatedEvent(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        run=None,
                    )
                )

    def _process_event_file(self, path: Path) -> None:
        """Read new events from file and notify listeners

        No locking needed - event files are append-only.
        """
        last_pos = self._file_positions.get(path, 0)
        logger.debug("Processing event file %s from position %d", path, last_pos)

        # Extract experiment_id from filename: events-{count}@{experiment_id}.jsonl
        filename = path.stem
        parts = filename.split("@", 1)
        experiment_id = parts[1] if len(parts) == 2 else None

        try:
            with path.open("r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event_dict = json.loads(line)
                            event = EventBase.from_dict(event_dict)

                            # Apply event to cache
                            if experiment_id:
                                self.state_provider._apply_event_to_cache(
                                    experiment_id, event
                                )

                            # Notify listeners
                            state_event = self._to_state_event(event, path)
                            if state_event:
                                logger.debug(
                                    "Notifying listeners of %s",
                                    type(state_event).__name__,
                                )
                                self.state_provider._notify_state_listeners(state_event)
                        except json.JSONDecodeError:
                            pass

                self._file_positions[path] = f.tell()
        except FileNotFoundError:
            # File may have been deleted - ignore
            pass
        except OSError as e:
            logger.warning("Failed to read event file %s: %s", path, e)

    def _to_state_event(self, event: EventBase, path: Path) -> Optional[StateEvent]:
        """Convert filesystem event to StateEvent"""
        # Extract experiment_id from filename: events-{count}@{experiment_id}.jsonl
        filename = path.stem  # events-{count}@{experiment_id}
        parts = filename.split("@", 1)
        if len(parts) != 2:
            return None
        experiment_id = parts[1]

        # Get current run_id
        run_id = self.state_provider.get_current_run(experiment_id) or ""

        if isinstance(event, JobSubmittedEvent):
            return JobExperimentUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=event.job_id,
                tags=event.tags,
                depends_on=event.depends_on,
            )
        elif isinstance(event, JobStateChangedEvent):
            return JobUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=event.job_id,
                job=None,  # Will be loaded on demand
            )
        elif isinstance(event, ServiceAddedEvent):
            return ServiceUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                service_id=event.service_id,
                service=None,
            )

        return None


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse ISO format timestamp to Unix timestamp"""
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


# Alias for backwards compatibility
# (DbStateProvider was previously called WorkspaceStateProvider)
__all__ = [
    "WorkspaceStateProvider",
    "EventFileWatcher",
]
