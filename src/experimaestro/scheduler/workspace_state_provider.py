"""Filesystem-based state provider implementation

This module provides the concrete implementation of OfflineStateProvider that
uses the filesystem for persistent state storage.

Classes:
- WorkspaceStateProvider: Filesystem-backed state provider (read-only for monitoring)
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from experimaestro.scheduler.interfaces import (
    BaseExperiment,
    BaseJob,
    BaseService,
    JobState,
    STATE_NAME_TO_JOBSTATE,
)
from experimaestro.scheduler.state_provider import (
    OfflineStateProvider,
    MockJob,
    MockExperiment,
    ProcessInfo,
    CarbonMetricsData,
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


class WorkspaceStateProvider(OfflineStateProvider):
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
        super().__init__()  # Initializes job/experiment/service caches
        self.workspace_path = Path(workspace_path).resolve()
        self._experiments_dir = self.workspace_path / ".events" / "experiments"

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
                WatchedDirectory(
                    path=self._experiments_dir,
                    on_created=self._on_experiment_created,
                    on_event=self._on_experiment_event,
                    on_deleted=self._on_experiment_deleted,
                ),
                WatchedDirectory(
                    path=self._jobs_dir,
                    glob_pattern="*/event-*-*.jsonl",
                    entity_id_extractor=job_entity_id_extractor,
                    on_created=self._on_job_created,
                    on_event=self._on_job_event,
                    on_deleted=self._on_job_deleted,
                ),
            )
            # Start buffering before watching to avoid race condition:
            # events arriving before entity discovery completes should be
            # queued and processed after initial state is loaded
            self._event_reader.start_buffering()
            self._event_reader.start_watching()
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

        Note: If OrphanedEventsError is raised (events without events_count), it's caught
        and handled by emitting a WarningEvent that the TUI/user can respond to.
        """
        from experimaestro.locking import OrphanedEventsError
        from experimaestro.scheduler.state_status import WarningEvent

        # Consolidate experiment events
        if self._experiments_dir.exists():
            for exp_events_dir in self._experiments_dir.iterdir():
                if exp_events_dir.is_dir():
                    experiment_id = exp_events_dir.name
                    try:
                        self._consolidate_experiment_events(
                            experiment_id, exp_events_dir
                        )
                    except OrphanedEventsError as e:
                        # Create warning event for TUI/listeners to handle
                        warning_event = WarningEvent(
                            experiment_id=e.context.get("experiment_id", ""),
                            run_id=e.context.get("run_id", ""),
                            warning_key=e.warning_key,
                            description=e.description,
                            actions=e.actions,
                            context=e.context,
                        )

                        # Register callbacks and metadata with state provider
                        self.register_warning_actions(
                            e.warning_key, e.callbacks, warning_event
                        )

                        # Emit warning event for TUI/listeners to handle
                        self._notify_state_listeners(warning_event)
                        logger.info(
                            "Orphaned events detected for experiment %s, awaiting user action",
                            experiment_id,
                        )

        # Consolidate job events
        # Note: We do this inline without using finalize_status() to avoid
        # async/lock complications during initialization
        if self._jobs_dir.exists():
            for task_dir in self._jobs_dir.iterdir():
                if task_dir.is_dir():
                    task_id = task_dir.name
                    # Group event files by job_id
                    job_event_files = list(task_dir.glob("event-*-*.jsonl"))

                    # Group files by job_id
                    job_files_map: dict[str, list[Path]] = {}
                    for event_file in job_event_files:
                        # Extract job_id from filename: event-{job_id}-{count}.jsonl
                        filename = event_file.name
                        parts = filename.split("-")
                        if len(parts) >= 3:
                            job_id = parts[1]
                            if job_id not in job_files_map:
                                job_files_map[job_id] = []
                            job_files_map[job_id].append(event_file)

                    # Consolidate events for each job
                    for job_id, files in job_files_map.items():
                        try:
                            self._consolidate_job_events(task_id, job_id, files)
                        except Exception as e:
                            logger.warning(
                                f"Failed to consolidate events for job {task_id}/{job_id}: {e}"
                            )

    def _consolidate_job_events(
        self, task_id: str, job_id: str, event_files: list[Path]
    ) -> None:
        """Consolidate orphaned job event files

        Uses the job's own _cleanup_event_files() method to apply events and cleanup.

        Args:
            task_id: Task identifier
            job_id: Job identifier
            event_files: List of event file paths for this job
        """
        import json

        job_path = self.workspace_path / "jobs" / task_id / job_id
        if not job_path.exists():
            # Job directory doesn't exist, just delete orphaned event files
            for event_file in event_files:
                try:
                    event_file.unlink()
                    logger.debug(f"Deleted orphaned event file: {event_file}")
                except OSError:
                    pass
            return

        # Load job from disk
        from experimaestro.scheduler.state_provider import MockJob

        job = MockJob.from_disk(
            job_path=job_path,
            task_id=task_id,
            job_id=job_id,
            workspace_path=self.workspace_path,
        )

        # Check if job has events_count (needs consolidation)
        if job.events_count is None:
            # Already consolidated, just delete event files
            for event_file in event_files:
                try:
                    event_file.unlink()
                    logger.debug(
                        f"Deleted already-consolidated event file: {event_file}"
                    )
                except OSError:
                    pass
            return

        # Use the job's own cleanup method to apply events and delete files
        # This ensures we use the same logic as normal job consolidation
        job._cleanup_event_files()

        # Write updated status.json without events_count
        status_dict = job.state_dict()
        status_dict.pop("events_count", None)

        status_path = job_path / ".experimaestro" / "status.json"
        try:
            status_path.write_text(json.dumps(status_dict, indent=2))
            logger.debug(
                f"Consolidated and updated status.json for job {task_id}/{job_id}"
            )
        except OSError as e:
            logger.warning(f"Failed to write status.json for {task_id}/{job_id}: {e}")

    def _consolidate_experiment_events(
        self, experiment_id: str, exp_events_dir: Path
    ) -> None:
        """Consolidate orphaned event files for a specific experiment

        If there are event files in .events/experiments/{experiment_id}/ and
        status.json has an events_count field, apply the events to update status.json.

        If events_count is absent from status.json, all events have been processed
        and we just need to cleanup the orphaned event files.

        IMPORTANT: Only consolidate if we can acquire the experiment lock.
        If the lock is held (experiment is running), skip consolidation.

        This method now uses BaseExperiment.finalize_status() for consolidation.

        Args:
            experiment_id: The experiment identifier
            exp_events_dir: Path to .events/experiments/{experiment_id}/
        """
        import filelock

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

        # If events_count is None (absent), this might be:
        # 1. Already consolidated - just cleanup
        # 2. Orphaned events that need user confirmation
        # We'll let finalize_status() decide by checking if files exist
        if events_count is None:
            logger.debug(
                "Experiment %s has no events_count, will check for orphaned events",
                experiment_id,
            )

        # Log what we're doing
        if events_count is not None:
            logger.info(
                "Consolidating orphaned events for experiment %s (events_count=%d)",
                experiment_id,
                events_count,
            )
        else:
            logger.info(
                "Checking experiment %s for orphaned events",
                experiment_id,
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

        # Try to acquire experiment lock with timeout
        # If locked (experiment is running), skip consolidation
        experiment_base = run_dir.parent
        lock_path = experiment_base / "lock"

        try:
            lock = filelock.FileLock(lock_path, timeout=0.1)
            lock.acquire()
        except filelock.Timeout:
            # Experiment is locked (running), skip consolidation
            logger.debug(
                "Experiment %s is locked (running), skipping consolidation",
                experiment_id,
            )
            return

        try:
            # If events_count is None, events are already consolidated
            # Just clean up orphaned event files without applying them
            if events_count is None:
                # Events already processed, just delete the orphaned files
                for event_file in event_files:
                    try:
                        event_file.unlink()
                        logger.debug(
                            "Deleted already-consolidated event file: %s", event_file
                        )
                    except OSError as e:
                        logger.warning(
                            "Failed to delete event file %s: %s", event_file, e
                        )
                logger.debug(
                    "Cleaned up %d already-consolidated event files for experiment %s",
                    len(event_files),
                    experiment_id,
                )
                return  # Done, no need to apply events or update status.json

            # Use the experiment's own cleanup method to apply events and archive files
            # This ensures we use the same logic as normal experiment consolidation
            try:
                exp._cleanup_experiment_event_files()
            except Exception as e:
                # Let OrphanedEventsError bubble up
                from experimaestro.locking import OrphanedEventsError

                if isinstance(e, OrphanedEventsError):
                    raise
                logger.warning(
                    "Failed to cleanup experiment event files for %s: %s",
                    experiment_id,
                    e,
                )
                return

            # Write status.json without events_count field
            status_dict = exp.state_dict()
            status_dict.pop("events_count", None)

            try:
                status_path.write_text(json.dumps(status_dict, indent=2))
                logger.info(
                    "Consolidated experiment %s, status=%s",
                    experiment_id,
                    exp.status.value,
                )
            except OSError as e:
                logger.warning(
                    "Failed to write status.json for experiment %s: %s",
                    experiment_id,
                    e,
                )
        finally:
            # Release the lock
            lock.release()

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

    # =========================================================================
    # Experiment event callbacks
    # =========================================================================

    def _on_experiment_created(self, experiment_id: str) -> bool:
        """Called when a new experiment is discovered. Pre-load into cache."""
        logger.debug("Discovered experiment: %s", experiment_id)

        # Get current run and pre-load experiment into cache
        run_id = self.get_current_run(experiment_id)
        if run_id is not None:
            run_dir = self.workspace_path / "experiments" / experiment_id / run_id
            if run_dir.exists():
                exp = MockExperiment.from_disk(run_dir, self.workspace_path)
                if exp is not None:
                    cache_key = (experiment_id, run_id)
                    with self._experiment_cache_lock:
                        self._experiment_cache[cache_key] = exp

        return True  # Follow all experiments

    def _on_experiment_event(self, experiment_id: str, event: EventBase) -> None:
        """Handle experiment event

        Experiment events include both ExperimentEventBase (experiment-level events)
        and JobEventBase (job events written to experiment's event file).
        """
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        logger.debug("Received experiment event for %s: %s", experiment_id, event)

        # Mark JobStateChangedEvent as coming from experiment events
        if isinstance(event, JobStateChangedEvent):
            event.from_experiment = True

        # Update experiment status cache for all experiment events
        # This includes JobStateChangedEvent which updates exp._job_states
        if isinstance(event, (ExperimentEventBase, JobEventBase)):
            self._handle_experiment_event(experiment_id, event)

        # Forward to listeners
        self._notify_state_listeners(event)

    def _on_experiment_deleted(self, experiment_id: str) -> None:
        """Handle experiment event files deletion (experiment finalized)"""
        self._clear_experiment_cache(experiment_id)
        run_id = self.get_current_run(experiment_id) or ""
        self._notify_state_listeners(
            ExperimentUpdatedEvent(experiment_id=experiment_id)
        )
        self._notify_state_listeners(
            RunUpdatedEvent(experiment_id=experiment_id, run_id=run_id)
        )

    # =========================================================================
    # Job event callbacks
    # =========================================================================

    def _on_job_created(self, entity_id: str) -> bool:
        """Called when a new job is discovered.

        Args:
            entity_id: Job entity ID in format "{task_id}:{job_id}" (same as full_id)
        """
        task_id, job_id = BaseJob.parse_full_id(entity_id)
        logger.debug("Discovered job: %s (task: %s)", job_id, task_id)
        return True  # Follow all jobs

    def _on_job_event(self, entity_id: str, event: EventBase) -> None:
        """Handle job event

        Args:
            entity_id: Job entity ID in format "{task_id}:{job_id}" (same as full_id)
            event: The job event
        """
        task_id, job_id = BaseJob.parse_full_id(entity_id)
        # entity_id is already in full_id format (task_id:job_id)
        full_id = entity_id
        logger.debug("Received job event for %s: %s", full_id, event)

        if isinstance(event, JobEventBase):
            with self._job_cache_lock:
                job = self._job_cache.get(full_id)

                if job is None:
                    # Job not in cache - create/load it
                    # This handles both JobSubmittedEvent and other events
                    # (e.g., JobStateChangedEvent arriving without prior JobSubmittedEvent)
                    job_path = self.workspace_path / "jobs" / task_id / job_id
                    submit_time = None

                    if isinstance(event, JobSubmittedEvent):
                        # Use task_id from event if available (more reliable)
                        if event.task_id:
                            task_id = event.task_id
                            full_id = BaseJob.make_full_id(task_id, job_id)
                            job_path = self.workspace_path / "jobs" / task_id / job_id
                        if event.timestamp:
                            submit_time = datetime.fromtimestamp(event.timestamp)

                    # Load from disk if exists, otherwise create minimal job
                    if job_path.exists():
                        job = self._mock_job_from_disk(job_path, task_id, job_id)
                    else:
                        job = MockJob(
                            identifier=job_id,
                            task_id=task_id,
                            path=job_path,
                            state="unscheduled",
                            submittime=submit_time,
                            starttime=None,
                            endtime=None,
                            progress=[],
                            updated_at="",
                        )
                    self._job_cache[full_id] = job

                # Apply the event to update job state
                job.apply_event(event)

        # Forward to listeners
        self._notify_state_listeners(event)

    def _on_job_deleted(self, entity_id: str) -> None:
        """Handle job event files deletion

        Args:
            entity_id: Job entity ID in format "{task_id}:{job_id}"
        """
        # Job deletion doesn't need special handling - job remains in cache
        # until explicitly removed or cache is cleared
        pass

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

    def _get_or_load_experiment(
        self, experiment_id: str, run_id: str, run_dir: Path
    ) -> MockExperiment:
        """Get experiment from cache or load from disk

        For active experiments (with event files), the cache is maintained by
        the EventReader callbacks (on_created and on_event).

        For non-active experiments (no event files), loads from disk directly.

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            run_dir: Path to run directory

        Returns:
            MockExperiment with current state
        """
        return super()._get_or_load_experiment(experiment_id, run_id, run_dir=run_dir)

    def _create_experiment(
        self, experiment_id: str, run_id: str, *, run_dir: Path
    ) -> MockExperiment:
        """Create an experiment by loading from disk or creating minimal instance"""
        exp = MockExperiment.from_disk(run_dir, self.workspace_path)
        if exp is None:
            exp = MockExperiment(
                workdir=run_dir,
                run_id=run_id,
            )
        return exp

    def _has_event_files(self, experiment_id: str) -> bool:
        """Check if experiment has any event files (is active)"""
        # Format: experiments/{experiment_id}/events-*.jsonl
        exp_events_dir = self._experiments_dir / experiment_id
        return exp_events_dir.is_dir() and any(exp_events_dir.glob("events-*.jsonl"))

    def _handle_experiment_event(self, experiment_id: str, event: EventBase) -> None:
        """Update cached experiment from an experiment event

        Called by EventReader when experiment events are received.
        For JobStateChangedEvent with from_experiment=True, also applies to
        cached job to set _experiment_status.
        """
        from experimaestro.scheduler.state_status import JobStateChangedEvent

        run_id = self.get_current_run(experiment_id)
        if run_id is None:
            return

        cache_key = (experiment_id, run_id)

        with self._experiment_cache_lock:
            if cache_key in self._experiment_cache:
                self._experiment_cache[cache_key].apply_event(event)

        # Also apply JobStateChangedEvent to job cache to update _experiment_status
        if isinstance(event, JobStateChangedEvent) and event.from_experiment:
            job_id = event.job_id
            # Find the job's task_id from the experiment's job_infos
            with self._experiment_cache_lock:
                exp = self._experiment_cache.get(cache_key)
                if exp is not None:
                    job_info = exp.job_infos.get(job_id)
                    if job_info is not None:
                        full_id = BaseJob.make_full_id(job_info.task_id, job_id)
                        with self._job_cache_lock:
                            job = self._job_cache.get(full_id)
                            if job is not None:
                                job.apply_event(event)

    def _load_carbon_metrics_for_job(self, job_id: str) -> CarbonMetricsData | None:
        """Load carbon metrics from carbon storage for a job.

        Args:
            job_id: Job identifier

        Returns:
            CarbonMetricsData or None if not available
        """
        try:
            from experimaestro.carbon.storage import CarbonStorage

            storage = CarbonStorage(self.workspace_path)
            record = storage.get_latest_job_record(job_id)
            if record:
                return CarbonMetricsData(
                    co2_kg=record.co2_kg,
                    energy_kwh=record.energy_kwh,
                    cpu_power_w=record.cpu_power_w,
                    gpu_power_w=record.gpu_power_w,
                    ram_power_w=record.ram_power_w,
                    duration_s=record.duration_s,
                    region=record.region,
                    is_final=True,  # Data from storage is always final
                )
        except Exception as e:
            logger.debug("Failed to load carbon metrics for job %s: %s", job_id, e)
        return None

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
        full_id = BaseJob.make_full_id(task_id, job_id)
        return super()._get_or_load_job(
            full_id, job_id=job_id, task_id=task_id, submit_time=submit_time
        )

    def _create_job(
        self,
        full_id: str,
        *,
        job_id: str,
        task_id: str,
        submit_time: float | datetime | None,
    ) -> MockJob:
        """Create a job by loading from disk or creating minimal instance"""
        job_path = self.workspace_path / "jobs" / task_id / job_id
        if job_path.exists():
            # _mock_job_from_disk handles carbon metrics loading
            return self._mock_job_from_disk(job_path, task_id, job_id)

        # Job directory doesn't exist - create minimal MockJob
        submittime_dt = None
        if submit_time is not None:
            if isinstance(submit_time, datetime):
                submittime_dt = submit_time
            else:
                submittime_dt = datetime.fromtimestamp(submit_time)

        return MockJob(
            identifier=job_id,
            task_id=task_id,
            path=job_path,
            state="unscheduled",
            submittime=submittime_dt,
            starttime=None,
            endtime=None,
            progress=[],
            updated_at="",
            carbon_metrics=self._load_carbon_metrics_for_job(job_id),
        )

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
        return self._get_or_load_experiment(experiment_id, current_run_id, run_dir)

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
                job = self._mock_job_from_disk(job_path, task_id, job_id)
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
                job = self._mock_job_from_disk(job_path, task_id, job_id)
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
        exp = self._get_or_load_experiment(experiment_id, run_id, run_dir)

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

            # Set experiment status from tracked job states
            exp_state = exp.get_job_state(job_id)
            if exp_state is not None:
                job._experiment_status = exp_state

            # Apply state filter on loaded job
            if state:
                state_enum = STATE_NAME_TO_JOBSTATE.get(state)
                if state_enum and job.state != state_enum:
                    continue

            jobs.append(job)

        return jobs

    def get_job(self, task_id: str, job_id: str) -> Optional[MockJob]:
        """Get a job directly by task_id and job_id"""
        job_path = self.workspace_path / "jobs" / task_id / job_id
        if not job_path.exists():
            return None

        return self._get_or_load_job(job_id, task_id, submit_time=None)

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
        exp = self._get_or_load_experiment(experiment_id, run_id, run_dir)

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
        exp = self._get_or_load_experiment(experiment_id, run_id, run_dir)

        return exp.dependencies

    # =========================================================================
    # Services
    # =========================================================================

    def _fetch_services_from_storage(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> List[BaseService]:
        """Fetch services from status.json

        Returns MockService objects directly from the cached experiment.
        These are updated when ServiceStateChangedEvent is applied.
        """
        if experiment_id is None or run_id is None:
            return []

        run_dir = self.workspace_path / "experiments" / experiment_id / run_id
        if not run_dir.exists():
            return []

        # Get experiment from cache or load from disk
        exp = self._get_or_load_experiment(experiment_id, run_id, run_dir)

        # Return MockService objects directly - they're updated by apply_event
        # experiment_id and run_id are already set when MockService is created
        return list(exp.services.values())

    # =========================================================================
    # Job operations
    # =========================================================================

    def kill_job(self, job: MockJob, perform: bool = False) -> bool:
        """Kill a running job.

        Raises:
            RuntimeError: If the job cannot be killed (no process or kill failed).
        """
        if not perform:
            return job.state.running()

        process = job.getprocess()
        if process is None:
            raise RuntimeError("No process found for job")

        try:
            process.kill()
            return True
        except Exception as e:
            logger.warning("Failed to kill job %s: %s", job.identifier, e)
            raise RuntimeError(f"Failed to kill process: {e}") from e

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

            # Clear job from cache
            with self._job_cache_lock:
                self._job_cache.pop(job.full_id, None)

            # Emit state change event (unscheduled = job removed)
            from experimaestro.scheduler.state_status import JobStateChangedEvent

            self._notify_state_listeners(
                JobStateChangedEvent(
                    job_id=job.identifier,
                    state="unscheduled",
                )
            )

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
                    job = self._mock_job_from_disk(job_path, task_id, job_id)
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
                        job = self._mock_job_from_disk(job_path, task_id, job_id)
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

                # Use cached experiment (EventReader keeps it up to date)
                exp = self._get_or_load_experiment(
                    experiment_id, latest_run_id, run_dir
                )

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
                current_run_id = self.get_current_run(experiment_id)

                for run_dir in exp_dir.iterdir():
                    if not run_dir.is_dir() or run_dir.name.startswith("."):
                        continue

                    run_id = run_dir.name

                    # Use cached experiment for current run (kept up to date by EventReader)
                    # For historical runs, load from disk (no pending events)
                    if run_id == current_run_id:
                        exp = self._get_or_load_experiment(
                            experiment_id, run_id, run_dir
                        )
                    else:
                        exp = MockExperiment.from_disk(run_dir, self.workspace_path)
                        if exp is None:
                            continue

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

    def _mock_job_from_disk(self, job_path: Path, task_id: str, job_id: str) -> MockJob:
        """Create a MockJob from disk and load carbon metrics from storage"""
        job = MockJob.from_disk(
            job_path=job_path,
            task_id=task_id,
            job_id=job_id,
            workspace_path=self.workspace_path,
        )

        # Load carbon metrics from carbon storage (may be more recent than status.json)
        carbon_metrics = self._load_carbon_metrics_for_job(job_id)
        if carbon_metrics is not None:
            job.carbon_metrics = carbon_metrics

        return job

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

            # Try to get more info for running jobs using the process abstraction
            if job.state and job.state.running():
                try:
                    from experimaestro.connectors import Process
                    from experimaestro.connectors.local import LocalConnector

                    connector = LocalConnector.instance()
                    proc = Process.fromDefinition(connector, pinfo)

                    # Check if process is running (sync via async)
                    import asyncio

                    try:
                        asyncio.get_running_loop()
                        # We're in an async context, can't use asyncio.run
                        # Just mark as running based on job state
                        result.running = True
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run
                        result.running = asyncio.run(proc.aio_isrunning())

                    # For local processes, try to get CPU/memory info via psutil
                    if proc_type == "local" and result.running:
                        try:
                            import psutil

                            local_pid = int(pid)
                            ps_proc = psutil.Process(local_pid)
                            if ps_proc.is_running():
                                result.cpu_percent = ps_proc.cpu_percent(interval=0.1)
                                mem_info = ps_proc.memory_info()
                                result.memory_mb = mem_info.rss / (1024 * 1024)
                                result.num_threads = ps_proc.num_threads()
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            ImportError,
                            ValueError,
                        ):
                            pass
                except Exception:
                    pass  # Process abstraction not available or failed

            return result
        except (json.JSONDecodeError, OSError):
            return None

    # =========================================================================
    # Experiment deletion
    # =========================================================================

    def delete_experiment(
        self, experiment_id: str, delete_jobs: bool = False, perform: bool = True
    ) -> tuple[bool, str]:
        """Delete an experiment and optionally its job data

        Args:
            experiment_id: Experiment identifier to delete
            delete_jobs: If True, also delete job directories (default: False)
            perform: If True, actually perform deletion; if False, just check

        Returns:
            Tuple of (success, message)
        """
        import shutil

        # Check for running jobs first
        jobs = self.get_jobs(experiment_id=experiment_id)
        running_jobs = [j for j in jobs if j.state and j.state.running()]
        if running_jobs:
            return False, f"Cannot delete: {len(running_jobs)} jobs are still running"

        # Find experiment directories (v2 layout)
        exp_dir = self.workspace_path / "experiments" / experiment_id
        events_dir = self._experiments_dir / experiment_id

        # Check for v1 layout
        v1_exp_dir = self.workspace_path / "xp" / experiment_id

        if not exp_dir.exists() and not v1_exp_dir.exists():
            return False, f"Experiment {experiment_id} not found"

        if not perform:
            return True, f"Experiment {experiment_id} can be deleted"

        errors = []

        # Delete job directories if requested
        if delete_jobs:
            for job in jobs:
                if job.path and job.path.exists():
                    try:
                        shutil.rmtree(job.path)
                    except OSError as e:
                        errors.append(f"Failed to delete job {job.identifier}: {e}")

        # Delete v2 experiment directory
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
            except OSError as e:
                errors.append(f"Failed to delete experiment dir: {e}")

        # Delete events directory
        if events_dir.exists():
            try:
                shutil.rmtree(events_dir)
            except OSError as e:
                errors.append(f"Failed to delete events dir: {e}")

        # Delete v1 experiment directory
        if v1_exp_dir.exists():
            try:
                shutil.rmtree(v1_exp_dir)
            except OSError as e:
                errors.append(f"Failed to delete v1 experiment dir: {e}")

        # Clear caches
        self._clear_experiment_cache(experiment_id)
        with self._job_cache_lock:
            for job in jobs:
                self._job_cache.pop(job.full_id, None)

        if errors:
            return False, f"Partial deletion: {'; '.join(errors)}"

        return True, f"Deleted experiment {experiment_id}"

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the state provider and release resources"""
        self._stop_watcher()

        # Clear caches (inherited from OfflineStateProvider)
        self._clear_job_cache()
        self._clear_experiment_cache_all()
        self._clear_service_cache()

        with self._lock:
            if self.workspace_path in self._instances:
                del self._instances[self.workspace_path]


__all__ = [
    "WorkspaceStateProvider",
]
