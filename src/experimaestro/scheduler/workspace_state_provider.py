"""Filesystem-based state provider implementation

This module provides the concrete implementation of StateProvider that
uses the filesystem for persistent state storage, replacing the SQLite/peewee
based DbStateProvider.

Classes:
- WorkspaceStateProvider: Filesystem-backed state provider (read-only for monitoring)
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from experimaestro.scheduler.interfaces import (
    BaseExperiment,
    BaseService,
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
    EventBase,
    JobEventBase,
    ExperimentEventBase,
    JobSubmittedEvent,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
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

    On initialization, this provider performs crash recovery:
    - Consolidates orphaned event files from .events/ into status.json
    - Recovers job state from marker files (.done, .failed, .pid)
    """

    _instances: Dict[Path, "WorkspaceStateProvider"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        workspace_path: Path,
    ) -> "WorkspaceStateProvider":
        """Get or create singleton instance for workspace

        Args:
            workspace_path: Path to workspace directory

        Returns:
            WorkspaceStateProvider instance
        """
        workspace_path = Path(workspace_path).resolve()

        with cls._lock:
            instance = cls._instances.get(workspace_path)
            if instance is None:
                instance = cls(workspace_path)
                cls._instances[workspace_path] = instance
            return instance

    def __init__(self, workspace_path: Path):
        """Initialize workspace state provider

        Args:
            workspace_path: Path to workspace directory
        """
        super().__init__()
        self.workspace_path = Path(workspace_path).resolve()
        self._experiments_dir = self.workspace_path / ".events" / "experiments"

        # Experiment cache: (experiment_id, run_id) -> MockExperiment
        # Only caches active experiments (those with event files)
        self._experiment_cache: Dict[tuple[str, str], MockExperiment] = {}
        self._experiment_cache_lock = threading.Lock()

        # Job cache: job_id -> MockJob
        # Shared cache for all jobs, updated directly by job events
        self._job_cache: Dict[str, MockJob] = {}
        self._job_cache_lock = threading.Lock()

        # Service cache
        self._service_cache: Dict[tuple[str, str], Dict[str, BaseService]] = {}
        self._service_cache_lock = threading.Lock()

        # Event reader (with built-in watching capability)
        self._event_reader: Optional[EventReader] = None
        self._jobs_dir = self.workspace_path / ".events" / "jobs"

        # Perform crash recovery: consolidate orphaned events
        self._consolidate_orphaned_events()

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

    # =========================================================================
    # Crash recovery: Consolidate orphaned events
    # =========================================================================

    def _consolidate_orphaned_events(self) -> None:
        """Consolidate orphaned event files from crashed experiments/jobs

        This method is called during initialization to recover from crashes.
        It scans the .events/ directory for orphaned event files and consolidates
        them into status.json files.

        Crash scenarios handled:
        1. Experiment crashed after writing events but before finalizing status.json
        2. Job crashed before writing final status

        Logic:
        - If status.json has NO 'events_count' field → already consolidated, just cleanup
        - If status.json HAS 'events_count' field → events need to be processed
        """
        # Consolidate experiment events
        if self._experiments_dir.exists():
            for exp_events_dir in self._experiments_dir.iterdir():
                if exp_events_dir.is_dir():
                    experiment_id = exp_events_dir.name
                    self._consolidate_experiment_events(experiment_id, exp_events_dir)

    def _consolidate_experiment_events(
        self, experiment_id: str, exp_events_dir: Path
    ) -> None:
        """Consolidate orphaned event files for a specific experiment

        If there are event files in .events/experiments/{experiment_id}/ and
        status.json has an events_count field, apply the events to update status.json.

        If events_count is absent from status.json, all events have been processed
        and we just need to cleanup the orphaned event files.

        Args:
            experiment_id: The experiment identifier
            exp_events_dir: Path to .events/experiments/{experiment_id}/
        """
        # Find event files
        event_files = list(exp_events_dir.glob("events-*.jsonl"))
        if not event_files:
            return  # No orphaned events

        # Get current run_id from symlink
        symlink = exp_events_dir / "current"
        if not symlink.is_symlink():
            logger.debug(
                "No 'current' symlink for experiment %s, skipping consolidation",
                experiment_id,
            )
            return

        try:
            run_dir = symlink.resolve()
            run_id = run_dir.name
        except OSError:
            logger.warning(
                "Could not resolve 'current' symlink for experiment %s", experiment_id
            )
            return

        # Check if status.json exists and check for events_count
        status_path = run_dir / "status.json"
        exp_data = None
        events_count = None  # None means "already consolidated"

        if status_path.exists():
            try:
                with status_path.open("r") as f:
                    exp_data = json.load(f)
                # events_count absent means already consolidated
                events_count = exp_data.get("events_count")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to read status.json for %s: %s", experiment_id, e
                )

        # If events_count is None (absent), events are already consolidated
        # Just cleanup orphaned event files
        if events_count is None:
            logger.debug(
                "Experiment %s already consolidated (no events_count), cleaning up",
                experiment_id,
            )
            self._cleanup_event_files(exp_events_dir, event_files)
            return

        # We have unprocessed events - need to consolidate
        logger.info(
            "Consolidating orphaned events for experiment %s (events_count=%d)",
            experiment_id,
            events_count,
        )

        # Load or create experiment
        from experimaestro.scheduler.interfaces import ExperimentStatus

        if exp_data:
            exp = MockExperiment.from_state_dict(exp_data, self.workspace_path)
        else:
            exp = MockExperiment(
                workdir=run_dir,
                run_id=run_id,
                status=ExperimentStatus.RUNNING,
            )

        # Read and apply all events from events_count onwards
        reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
        events = reader.read_events_since_count(experiment_id, events_count)

        for event in events:
            exp.apply_event(event)
            logger.debug(
                "Applied event %s to experiment %s", type(event).__name__, experiment_id
            )

        # Check for RunCompletedEvent to determine final status
        from experimaestro.scheduler.state_status import RunCompletedEvent

        final_event = None
        for event in reversed(events):
            if isinstance(event, RunCompletedEvent):
                final_event = event
                break

        if final_event:
            # Experiment completed - write final status.json
            if final_event.status in ("completed", "done"):
                exp._status = ExperimentStatus.DONE
            elif final_event.status == "failed":
                exp._status = ExperimentStatus.FAILED
            if final_event.ended_at:
                from experimaestro.scheduler.interfaces import deserialize_to_datetime

                exp._ended_at = deserialize_to_datetime(final_event.ended_at)

        # Remove events_count from state_dict to mark as consolidated
        # (MockExperiment.state_dict() includes events_count, so we set it to 0
        # and then remove the key after getting state_dict)
        exp._events_count = 0

        # Write consolidated status.json (without events_count)
        self._write_experiment_status(exp, status_path, remove_events_count=True)

        # Archive event files to permanent storage
        perm_events_dir = run_dir / "events"
        self._archive_event_files(exp_events_dir, perm_events_dir, event_files)

        logger.info(
            "Consolidated experiment %s: %d events applied, status=%s",
            experiment_id,
            len(events),
            exp.status.value,
        )

    def _write_experiment_status(
        self, exp: MockExperiment, status_path: Path, remove_events_count: bool = False
    ) -> None:
        """Write experiment status.json atomically

        Args:
            exp: MockExperiment with current state
            status_path: Path to status.json
            remove_events_count: If True, remove events_count from the output
                (indicates all events have been processed)
        """
        import filelock
        import tempfile

        status_path.parent.mkdir(parents=True, exist_ok=True)

        # Use file lock for atomic writes
        lock_path = status_path.parent / f".{status_path.name}.lock"

        state = exp.state_dict()
        if remove_events_count:
            state.pop("events_count", None)

        with filelock.FileLock(lock_path):
            # Write to temp file first, then atomic rename
            fd, temp_path = tempfile.mkstemp(
                dir=status_path.parent, suffix=".tmp", prefix="status"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2)
                os.replace(temp_path, status_path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

    def _cleanup_event_files(self, events_dir: Path, event_files: List[Path]) -> None:
        """Remove temporary event files after consolidation

        Args:
            events_dir: Directory containing event files
            event_files: List of event files to remove
        """
        for event_file in event_files:
            try:
                event_file.unlink()
                logger.debug("Removed event file %s", event_file)
            except OSError as e:
                logger.warning("Failed to remove event file %s: %s", event_file, e)

        # Remove empty events directory
        try:
            # Only remove if empty (may still have 'current' symlink)
            remaining = list(events_dir.iterdir())
            if not remaining or (
                len(remaining) == 1 and remaining[0].name == "current"
            ):
                # Remove symlink if present
                current = events_dir / "current"
                if current.is_symlink():
                    current.unlink()
                events_dir.rmdir()
                logger.debug("Removed empty events directory %s", events_dir)
        except OSError:
            pass  # Directory not empty or other error

    def _archive_event_files(
        self, temp_dir: Path, perm_dir: Path, event_files: List[Path]
    ) -> None:
        """Archive event files to permanent storage and clean up temp files

        Args:
            temp_dir: Temporary events directory (.events/experiments/{id}/)
            perm_dir: Permanent events directory (run_dir/events/)
            event_files: List of temporary event files
        """
        import shutil

        perm_dir.mkdir(parents=True, exist_ok=True)

        for temp_file in event_files:
            # Convert filename: events-N.jsonl -> event-N.jsonl
            perm_name = temp_file.name.replace("events-", "event-")
            perm_path = perm_dir / perm_name

            if perm_path.exists():
                # Already archived (hardlink case) - just remove temp
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            else:
                # Move to permanent storage
                try:
                    shutil.move(str(temp_file), str(perm_path))
                    logger.debug("Archived %s -> %s", temp_file, perm_path)
                except OSError as e:
                    logger.warning("Failed to archive %s: %s", temp_file, e)

        # Clean up empty temp directory
        self._cleanup_event_files(temp_dir, [])

    def _on_event(self, entity_id: str, event: EventBase) -> None:
        """Unified callback for events from file watcher

        Uses event class hierarchy to determine how to handle events:
        - ExperimentEventBase: update experiment status cache
        - JobEventBase: update job cache directly
        """
        logger.debug("Received event for entity %s: %s", entity_id, event)

        # Handle experiment events (update experiment status cache)
        if isinstance(event, ExperimentEventBase):
            experiment_id = entity_id
            self._apply_event_to_cache(experiment_id, event)

        # Handle job events (update job cache directly)
        # Note: JobSubmittedEvent is both, but job is created when status loads
        if isinstance(event, JobEventBase) and not isinstance(event, JobSubmittedEvent):
            with self._job_cache_lock:
                job = self._job_cache.get(event.job_id)
                if job is not None:
                    job.apply_event(event)

        # Always forward to listeners
        self._notify_state_listeners(event)

    def _on_deleted(self, entity_id: str) -> None:
        """Unified callback when event files are deleted"""
        # Check if this is an experiment directory
        if (self._experiments_dir / entity_id).exists() or self.get_current_run(
            entity_id
        ):
            # Experiment event files deleted (experiment finalized)
            experiment_id = entity_id
            self._clear_experiment_cache(experiment_id)
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

    def _get_cached_experiment(
        self, experiment_id: str, run_id: str, run_dir: Path
    ) -> MockExperiment:
        """Get experiment from cache or load from disk

        For active experiments (with event files), maintains an in-memory cache
        that is updated when events arrive via the file watcher.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            run_dir: Path to run directory

        Returns:
            MockExperiment with all events applied
        """
        cache_key = (experiment_id, run_id)

        with self._experiment_cache_lock:
            # Check cache first
            if cache_key in self._experiment_cache:
                return self._experiment_cache[cache_key]

            # Load from disk using MockExperiment.from_disk
            exp = MockExperiment.from_disk(run_dir, self.workspace_path)
            if exp is None:
                # Create empty experiment if no status.json exists
                exp = MockExperiment(
                    workdir=run_dir,
                    run_id=run_id,
                )

            # Apply pending events from event files
            # Events are in experiments/{experiment_id}/events-{count}.jsonl
            reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
            events = reader.read_events_since_count(experiment_id, exp.events_count)
            for event in events:
                exp.apply_event(event)

            # Only cache if experiment is active (has event files)
            # This avoids caching finished experiments that won't change
            if self._has_event_files(experiment_id):
                self._experiment_cache[cache_key] = exp

            return exp

    def _has_event_files(self, experiment_id: str) -> bool:
        """Check if experiment has any event files (is active)"""
        # Format: experiments/{experiment_id}/events-*.jsonl
        exp_events_dir = self._experiments_dir / experiment_id
        return exp_events_dir.is_dir() and any(exp_events_dir.glob("events-*.jsonl"))

    def _apply_event_to_cache(self, experiment_id: str, event: EventBase) -> None:
        """Apply an event to the cached experiment (called by EventFileWatcher)"""
        run_id = self.get_current_run(experiment_id)
        if run_id is None:
            return

        cache_key = (experiment_id, run_id)

        with self._experiment_cache_lock:
            if cache_key in self._experiment_cache:
                self._experiment_cache[cache_key].apply_event(event)

    def _clear_experiment_cache(self, experiment_id: str) -> None:
        """Clear cached experiment for an experiment (called when experiment finishes)"""
        with self._experiment_cache_lock:
            # Remove all cache entries for this experiment
            keys_to_remove = [
                k for k in self._experiment_cache if k[0] == experiment_id
            ]
            for key in keys_to_remove:
                del self._experiment_cache[key]

    def _get_or_load_job(
        self, job_id: str, task_id: str, submit_time: float | datetime | None
    ) -> MockJob:
        """Get job from cache or load from disk and cache it.

        This ensures that job events (progress, state changes) can be applied
        to cached jobs, keeping them up to date between get_jobs() calls.

        Args:
            job_id: Job identifier
            task_id: Task identifier (for job path)
            submit_time: Submit timestamp (fallback if job directory doesn't exist)

        Returns:
            MockJob from cache or freshly loaded from disk
        """
        with self._job_cache_lock:
            if job_id in self._job_cache:
                return self._job_cache[job_id]

            # Load from disk
            job_path = self.workspace_path / "jobs" / task_id / job_id
            if job_path.exists():
                job = self._create_mock_job_from_path(job_path, task_id, job_id)
            else:
                # Job directory doesn't exist - create minimal MockJob
                # Convert float timestamp to datetime if needed
                submittime_dt = None
                if submit_time is not None:
                    if isinstance(submit_time, datetime):
                        submittime_dt = submit_time
                    else:
                        submittime_dt = datetime.fromtimestamp(submit_time)
                job = MockJob(
                    identifier=job_id,
                    task_id=task_id,
                    path=job_path,
                    state="unscheduled",
                    submittime=submittime_dt,
                    starttime=None,
                    endtime=None,
                    progress=[],
                    updated_at="",
                )

            self._job_cache[job_id] = job
            return job

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
                run_id="",
            )

        run_dir = exp_dir / current_run_id
        if not run_dir.exists():
            return None

        # Get experiment from cache or load from disk
        return self._get_cached_experiment(experiment_id, current_run_id, run_dir)

    def _load_v1_experiment(
        self, experiment_id: str, exp_dir: Path
    ) -> Optional[MockExperiment]:
        """Load experiment from v1 layout (xp/{exp-id}/ with jobs/, jobs.bak/)"""
        from experimaestro.scheduler.interfaces import (
            ExperimentJobInformation,
            ExperimentStatus,
        )

        # Build job_infos from jobs/ and jobs.bak/ directories
        jobs_dir = exp_dir / "jobs"
        jobs_bak_dir = exp_dir / "jobs.bak"

        job_infos: Dict[str, ExperimentJobInformation] = {}
        seen_jobs: set[str] = set()
        status = ExperimentStatus.DONE
        finished_count = 0
        failed_count = 0

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

                task_id = job_link.parent.name
                job_id = job_link.name

                # Create ExperimentJobInformation
                try:
                    mtime = job_path.stat().st_mtime
                except OSError:
                    mtime = None
                job_infos[job_id] = ExperimentJobInformation(
                    job_id=job_id,
                    task_id=task_id,
                    tags={},
                    timestamp=mtime,
                )

                # Check job state for experiment status and counting
                job = self._create_mock_job_from_path(job_path, task_id, job_id)
                if job.state.is_error():
                    status = ExperimentStatus.FAILED
                    failed_count += 1
                elif job.state.finished():
                    finished_count += 1
                else:
                    status = ExperimentStatus.RUNNING

        # Get modification time for started_at
        try:
            mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
        except OSError:
            mtime = None

        return MockExperiment(
            workdir=exp_dir,
            run_id="v1",  # Mark as v1 experiment
            status=status,
            job_infos=job_infos,
            started_at=mtime,
            experiment_id_override=experiment_id,
            finished_jobs=finished_count,
            failed_jobs=failed_count,
        )

    def _get_v1_jobs(self, experiment_id: str) -> List[MockJob]:
        """Get jobs from v1 experiment layout

        v1 layout: xp/{exp-id}/jobs/{task_id}/{job_hash} -> symlink to jobs/{task_id}/{job_hash}
        """

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

                # Create MockJob from filesystem state (done/failed files, etc.)
                job = self._create_mock_job_from_path(job_path, task_id, job_id)
                jobs.append(job)

        return jobs

    def get_experiment_runs(self, experiment_id: str) -> List[BaseExperiment]:
        """Get all runs for an experiment"""
        runs: List[BaseExperiment] = []
        exp_dir = self.workspace_path / "experiments" / experiment_id

        # Check for v1 layout first (xp/{exp-id}/ without separate runs)
        v1_exp_dir = self.workspace_path / "xp" / experiment_id
        if v1_exp_dir.exists() and not exp_dir.exists():
            # v1 experiment: return single synthetic run
            exp = self._load_v1_experiment(experiment_id, v1_exp_dir)
            if exp:
                runs.append(exp)
            return runs

        if not exp_dir.exists():
            return runs

        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            # Use MockExperiment.from_disk to load the experiment
            mock_exp = MockExperiment.from_disk(run_dir, self.workspace_path)
            if mock_exp is not None:
                runs.append(mock_exp)

        return runs

    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current run ID for an experiment"""
        # Check new symlink location: .events/experiments/{experiment_id}/current
        exp_events_dir = self._experiments_dir / experiment_id
        symlink = exp_events_dir / "current"
        if symlink.is_symlink():
            try:
                target = symlink.resolve()
                return target.name
            except OSError:
                pass

        # Check legacy symlink location (old .experimaestro path)
        legacy_experiments_dir = self.workspace_path / ".experimaestro" / "experiments"
        legacy_symlink = legacy_experiments_dir / experiment_id
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

        # Get experiment from cache or load from disk
        exp = self._get_cached_experiment(experiment_id, run_id, run_dir)

        # Load jobs using job_infos
        jobs = []
        for job_id, job_info in exp.job_infos.items():
            # Apply task_id filter early
            if task_id and job_info.task_id != task_id:
                continue

            # Apply tags filter early using job_info.tags
            if tags:
                if not all(job_info.tags.get(k) == v for k, v in tags.items()):
                    continue

            # Get job from cache or load from disk
            job = self._get_or_load_job(job_id, job_info.task_id, job_info.timestamp)

            # Apply state filter on loaded job
            if state:
                state_enum = STATE_NAME_TO_JOBSTATE.get(state)
                if state_enum and job.state != state_enum:
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

        # Get experiment from cache or load from disk
        exp = self._get_cached_experiment(experiment_id, run_id, run_dir)

        # Get job_info and load full job data
        job_info = exp.job_infos.get(job_id)
        if job_info is None:
            return None

        return self._get_or_load_job(job_id, job_info.task_id, job_info.timestamp)

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

        # Get experiment from cache or load from disk
        exp = self._get_cached_experiment(experiment_id, run_id, run_dir)

        return exp.tags

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

        # Get experiment from cache or load from disk
        exp = self._get_cached_experiment(experiment_id, run_id, run_dir)

        return exp.dependencies

    # =========================================================================
    # Services
    # =========================================================================

    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[BaseService]:
        """Get services for an experiment

        Tries to recreate real Service objects from service_config, falls back to
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

        # Get experiment from cache or load from disk
        exp = self._get_cached_experiment(experiment_id, run_id, run_dir)

        services = []
        for service_id, mock_service in exp.services.items():
            # Try to recreate service from state_dict
            service_class = mock_service.service_class
            state_dict = mock_service.state_dict()
            if service_class:
                try:
                    service = Service.from_state_dict(service_class, state_dict)
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
                # No service_class - use MockService with error
                from experimaestro.scheduler.state_provider import MockService

                service = MockService(
                    service_id=service_id,
                    description_text="error: no service_class",
                    state_dict_data={},
                    experiment_id=experiment_id,
                    run_id=run_id,
                )
                services.append(service)
                logger.debug(
                    "Service %s has no service_class for recreation", service_id
                )

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
                    # This is an orphan job - create MockJob from filesystem state
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
                        job = self._create_mock_job_from_path(job_path, task_id, job_id)
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

                # Load experiment from disk (with locking)
                exp = MockExperiment.from_disk(run_dir, self.workspace_path)
                if exp is None:
                    continue

                # Also apply pending events
                # Events are in experiments/{experiment_id}/events-{count}.jsonl
                reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
                events = reader.read_events_since_count(experiment_id, exp.events_count)
                for event in events:
                    exp.apply_event(event)

                # Add all job paths from this run
                for job_info in exp.job_infos.values():
                    job_path = (
                        self.workspace_path
                        / "jobs"
                        / job_info.task_id
                        / job_info.job_id
                    )
                    try:
                        referenced.add(job_path.resolve())
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

                    # Load experiment from disk (with locking)
                    exp = MockExperiment.from_disk(run_dir, self.workspace_path)
                    if exp is None:
                        continue

                    # Also apply pending events
                    # Events are in experiments/{experiment_id}/events-{count}.jsonl
                    reader = EventReader([WatchedDirectory(path=self._experiments_dir)])
                    events = reader.read_events_since_count(
                        experiment_id, exp.events_count
                    )
                    for event in events:
                        exp.apply_event(event)

                    # Add all job paths from this run
                    for job_info in exp.job_infos.values():
                        job_path = (
                            self.workspace_path
                            / "jobs"
                            / job_info.task_id
                            / job_info.job_id
                        )
                        try:
                            referenced.add(job_path.resolve())
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

        # Get modification time for timestamps (convert to datetime)
        try:
            mtime = datetime.fromtimestamp(job_path.stat().st_mtime)
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


__all__ = [
    "WorkspaceStateProvider",
]
