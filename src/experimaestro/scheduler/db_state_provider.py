"""Database-backed state provider implementation

This module provides the concrete implementation of StateProvider that
uses SQLite database for persistent state storage.

Classes:
- OfflineStateProvider: Intermediate class with state listener management
- DbStateProvider: SQLite-backed state provider (previously WorkspaceStateProvider)
- SchedulerListener: Adapter connecting scheduler events to DbStateProvider
"""

import json
import logging
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from experimaestro.utils.fswatcher import (
    FSEventsMarkerWorkaround,
    DB_CHANGE_MARKER,
)

from watchdog.events import FileSystemEventHandler
from watchdog.observers.api import ObservedWatch

from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    JobExperimentsModel,
    JobDependenciesModel,
    ServiceModel,
    PartialModel,
    JobPartialModel,
    WorkspaceSyncMetadata,
    ALL_MODELS,
    CURRENT_DB_VERSION,
)
from abc import abstractmethod

from experimaestro.scheduler.interfaces import (
    BaseService,
    ExperimentRun,
    JobFailureStatus,
)
from experimaestro.scheduler.state_provider import (
    StateProvider,
    StateEvent,
    StateListener,
    ExperimentUpdatedEvent,
    RunUpdatedEvent,
    JobUpdatedEvent,
    JobExperimentUpdatedEvent,
    ServiceUpdatedEvent,
    MockJob,
    MockExperiment,
    MockService,
)
from experimaestro.notifications import get_progress_information_from_dict
from experimaestro.scheduler.transient import TransientMode

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.scheduler.services import Service

logger = logging.getLogger("xpm.state")


def _with_db_context(func):
    """Decorator to wrap method in database bind context

    This ensures all database queries have the models bound to the database.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            with self.workspace_db.bind_ctx(ALL_MODELS):
                return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception("Error in %s with database context: %s", func.__name__, e)
            raise

    return wrapper


class OfflineStateProvider(StateProvider):
    """State provider for offline/cached state access

    Provides state listener management and service caching shared by
    DbStateProvider and SSHStateProviderClient.

    This is an intermediate class between StateProvider (the ABC) and concrete
    implementations that need state listener support and service caching.
    """

    def __init__(self):
        """Initialize offline state provider with service cache and listener management"""
        super().__init__()  # Initialize state listener management
        self._init_service_cache()

    # =========================================================================
    # Service caching methods
    # =========================================================================

    def _init_service_cache(self) -> None:
        """Initialize service cache - call from subclass __init__"""
        from typing import Tuple

        self._service_cache: Dict[Tuple[str, str], Dict[str, "BaseService"]] = {}
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
        """
        # Resolve run_id if needed
        if experiment_id is not None and run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return []

        cache_key = (experiment_id or "", run_id or "")

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

            # Fetch from persistent storage (DB or remote)
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
        """Fetch services from persistent storage (DB or remote).

        Called when no live services and cache is empty.
        """
        ...

    # State listener methods (add_listener, remove_listener, _notify_state_listeners)
    # are inherited from StateProvider base class


class _DatabaseChangeDetector:
    """Background thread that detects database changes and notifies listeners

    Uses a semaphore pattern so that the watchdog event handler never blocks.
    The watchdog just signals the semaphore, and this thread does the actual
    database queries and listener notifications.

    Thread safety:
    - Uses a lock to protect start/stop transitions
    - Once stop() is called, the stop event cannot be cleared by start()
    - Uses a Condition for atomic wait-and-clear of change notifications
    """

    def __init__(self, state_provider: "DbStateProvider"):
        self.state_provider = state_provider
        self._last_check_time: Optional[datetime] = None
        self._change_condition = threading.Condition()
        self._change_pending = False  # Protected by _change_condition
        self._thread: Optional[threading.Thread] = None
        self._debounce_seconds = 0.5  # Wait before processing to batch rapid changes
        self._state_lock = threading.Lock()  # Protects start/stop transitions
        self._stopped = False  # Once True, cannot be restarted

    def start(self) -> None:
        """Start the change detection thread"""
        with self._state_lock:
            # Once stopped, cannot restart
            if self._stopped:
                logger.debug("Cannot start change detector - already stopped")
                return

            if self._thread is not None and self._thread.is_alive():
                return  # Already running

            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="DBChangeDetector",
            )
            self._thread.start()
            logger.debug("Started database change detector thread")

    def stop(self) -> None:
        """Stop the change detection thread"""
        with self._state_lock:
            self._stopped = True  # Mark as permanently stopped

        # Wake up the thread so it can exit
        with self._change_condition:
            self._change_condition.notify_all()

        # Join outside the lock to avoid deadlock
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("Stopped database change detector thread")

    def signal_change(self) -> None:
        """Signal that a database change was detected (non-blocking)"""
        with self._change_condition:
            self._change_pending = True
            self._change_condition.notify()

    def _run(self) -> None:
        """Main loop: wait for changes and process them"""
        while not self._stopped:
            # Wait for a change signal and clear it atomically
            with self._change_condition:
                while not self._change_pending and not self._stopped:
                    self._change_condition.wait()

                if self._stopped:
                    break

                # Clear the pending flag atomically while holding the lock
                self._change_pending = False

            # Debounce - wait a bit for more changes to accumulate
            time.sleep(self._debounce_seconds)

            # Process all accumulated changes
            self._detect_and_notify_changes()

    def _detect_and_notify_changes(self) -> None:
        """Query the database to detect what changed and send events"""
        try:
            since = self._last_check_time
            self._last_check_time = datetime.now()

            # Query for changed experiments
            with self.state_provider.workspace_db.bind_ctx([ExperimentModel]):
                query = ExperimentModel.select()
                if since:
                    query = query.where(ExperimentModel.updated_at > since)

                for exp in query:
                    self.state_provider._notify_state_listeners(
                        ExperimentUpdatedEvent(
                            experiment_id=exp.experiment_id,
                        )
                    )

            # Query for changed jobs - need to join with JobExperimentsModel
            # to get experiment_id and run_id
            with self.state_provider.workspace_db.bind_ctx(
                [JobModel, JobExperimentsModel]
            ):
                query = JobModel.select()
                if since:
                    query = query.where(JobModel.updated_at > since)

                for job in query:
                    # Get experiment info from JobExperimentsModel
                    job_experiments = list(
                        JobExperimentsModel.select().where(
                            JobExperimentsModel.job_id == job.job_id
                        )
                    )

                    for job_exp in job_experiments:
                        self.state_provider._notify_state_listeners(
                            JobUpdatedEvent(
                                experiment_id=job_exp.experiment_id,
                                run_id=job_exp.run_id,
                                job_id=job.job_id,
                            )
                        )

        except Exception as e:
            logger.warning("Error detecting database changes: %s", e)


class _DatabaseFileHandler(FileSystemEventHandler):
    """Watchdog handler for SQLite database file changes

    Simply signals the change detector when database files are modified.
    Does not block - all processing happens in the detector thread.
    """

    def __init__(
        self,
        change_detector: _DatabaseChangeDetector,
        fsevents_workaround: Optional[FSEventsMarkerWorkaround] = None,
    ):
        super().__init__()
        self.change_detector = change_detector
        self.fsevents_workaround = fsevents_workaround

    def on_any_event(self, event) -> None:
        """Handle all file system events"""
        logger.debug(
            "File watcher received event: %s %s (is_dir=%s)",
            event.event_type,
            event.src_path,
            event.is_directory,
        )

        # Only handle modification-like events
        if event.event_type not in ("modified", "created", "moved"):
            return

        if event.is_directory:
            return

        # Only react to database files or the change marker
        path = Path(event.src_path)
        if path.name not in ("workspace.db", "workspace.db-wal", DB_CHANGE_MARKER):
            return

        logger.debug(
            "Database file changed: %s (event: %s) - signaling detector",
            path.name,
            event.event_type,
        )

        # If we detected a real DB file change (not marker), cancel the workaround
        # since FSEvents worked for this change
        if path.name != DB_CHANGE_MARKER and self.fsevents_workaround:
            self.fsevents_workaround.cancel()

        # Signal the detector thread (non-blocking)
        self.change_detector.signal_change()


class DbStateProvider(OfflineStateProvider):
    """Database-backed state provider (singleton per workspace path)

    Provides access to experiment and job state from a single workspace database.
    Supports both read-only (monitoring) and read-write (scheduler) modes.

    Previously named WorkspaceStateProvider.

    Only one DbStateProvider instance exists per workspace path. Subsequent
    requests for the same path return the existing instance.

    Thread safety:
    - Database connections are thread-local (managed by state_db module)
    - Singleton registry is protected by a lock
    - Each thread gets its own database connection

    Run tracking:
    - Each experiment can have multiple runs
    - Jobs/services are scoped to (experiment_id, run_id)
    - Tags are scoped to (job_id, experiment_id, run_id) - fixes GH #128
    """

    # Registry of state provider instances by absolute path
    _instances: Dict[Path, "DbStateProvider"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        workspace_path: Path,
        read_only: bool = False,
        sync_on_start: bool = False,
        sync_interval_minutes: int = 5,
    ) -> "DbStateProvider":
        """Get or create DbStateProvider instance for a workspace path

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)

        Returns:
            DbStateProvider instance (singleton per path)
        """
        # Normalize path
        if isinstance(workspace_path, Path):
            workspace_path = workspace_path.absolute()
        else:
            workspace_path = Path(workspace_path).absolute()

        # Check if instance already exists
        with cls._lock:
            if workspace_path in cls._instances:
                existing = cls._instances[workspace_path]
                # Fail if requesting different read_only mode than cached instance
                if existing.read_only != read_only:
                    raise RuntimeError(
                        f"DbStateProvider for {workspace_path} already exists "
                        f"with read_only={existing.read_only}, cannot open with "
                        f"read_only={read_only}. Close the existing instance first."
                    )
                return existing

            # Create new instance - register BEFORE __init__ to handle
            # nested get_instance calls during sync_on_start
            instance = object.__new__(cls)
            cls._instances[workspace_path] = instance

        # Initialize outside the lock to avoid deadlock during sync
        try:
            instance.__init__(
                workspace_path, read_only, sync_on_start, sync_interval_minutes
            )
        except Exception:
            # Remove from registry if initialization fails
            with cls._lock:
                cls._instances.pop(workspace_path, None)
            raise
        return instance

    def __init__(
        self,
        workspace_path: Path,
        read_only: bool = False,
        sync_on_start: bool = False,
        sync_interval_minutes: int = 5,
        standalone: bool = False,
    ):
        """Initialize database state provider (called by get_instance())

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)
            standalone: If True, behave as if no scheduler exists (for external
                monitoring). When standalone, file watcher auto-starts on
                add_listener.
        """
        # Initialize parent (service cache and state listeners)
        super().__init__()

        # Normalize path
        if isinstance(workspace_path, Path):
            workspace_path = workspace_path.absolute()
        else:
            workspace_path = Path(workspace_path).absolute()

        self.workspace_path = workspace_path
        self._read_only = read_only
        self.sync_interval_minutes = sync_interval_minutes
        self._standalone = standalone

        # Additional listeners for push notifications (legacy interface)
        self._listeners: Set[StateListener] = set()
        self._listeners_lock = threading.Lock()

        # File watcher for database changes (started when listeners are added)
        self._change_detector: Optional[_DatabaseChangeDetector] = None
        self._db_file_handler: Optional[_DatabaseFileHandler] = None
        self._db_file_watch: Optional[ObservedWatch] = None
        self._fsevents_workaround: Optional[FSEventsMarkerWorkaround] = None

        # Check and update workspace version
        from .workspace import WORKSPACE_VERSION

        version_file = self.workspace_path / ".__experimaestro__"

        if version_file.exists():
            # Read existing version
            content = version_file.read_text().strip()
            if content == "":
                # Empty file = v0
                workspace_version = 0
            else:
                try:
                    workspace_version = int(content)
                except ValueError:
                    raise RuntimeError(
                        f"Invalid workspace version file at {version_file}: expected integer, got '{content}'"
                    )

            # Check if workspace version is supported
            if workspace_version > WORKSPACE_VERSION:
                raise RuntimeError(
                    f"Workspace version {workspace_version} is not supported by "
                    f"this version of experimaestro (supports up to version "
                    f"{WORKSPACE_VERSION}). Please upgrade experimaestro."
                )
            if workspace_version < WORKSPACE_VERSION:
                raise RuntimeError(
                    f"Workspace version {workspace_version} is not supported by "
                    "this version of experimaestro (please upgrade the experimaestro "
                    "workspace)"
                )
        else:
            # New workspace - create the file
            workspace_version = WORKSPACE_VERSION

        # Write current version to file (update empty v0 workspaces)
        if not read_only and (
            not version_file.exists() or version_file.read_text().strip() == ""
        ):
            version_file.write_text(str(WORKSPACE_VERSION))

        # Initialize workspace database in hidden .experimaestro directory
        from .state_db import initialize_workspace_database

        experimaestro_dir = self.workspace_path / ".experimaestro"
        if not read_only:
            experimaestro_dir.mkdir(parents=True, exist_ok=True)

        db_path = experimaestro_dir / "workspace.db"
        self.workspace_db, needs_resync = initialize_workspace_database(
            db_path, read_only=read_only
        )
        self._db_dir = experimaestro_dir  # Store for file watcher

        # Initialize FSEvents workaround for macOS (only for write mode)
        # Use shorter interval for testing via environment variable
        import os

        marker_interval = float(os.environ.get("XPM_FSEVENTS_MARKER_INTERVAL", "1.0"))
        if not read_only:
            self._fsevents_workaround = FSEventsMarkerWorkaround(
                experimaestro_dir, debounce_seconds=marker_interval
            )

        # Sync from disk if needed due to schema version change
        if needs_resync and not read_only:
            logger.info(
                "Database schema version changed, triggering full resync from disk"
            )
            sync_on_start = True  # Force sync

        # Optionally sync from disk on start (only in write mode)
        # Syncing requires write access to update the database and sync timestamp
        if sync_on_start and not read_only:
            from .state_sync import sync_workspace_from_disk

            sync_workspace_from_disk(
                self.workspace_path,
                write_mode=True,
                force=needs_resync,  # Force full sync if schema changed
                sync_interval_minutes=sync_interval_minutes,
            )

            # Update db_version after successful sync
            if needs_resync:
                with self.workspace_db.bind_ctx([WorkspaceSyncMetadata]):
                    WorkspaceSyncMetadata.update(db_version=CURRENT_DB_VERSION).where(
                        WorkspaceSyncMetadata.id == "workspace"
                    ).execute()
                logger.info("Database schema updated to version %d", CURRENT_DB_VERSION)

        logger.info(
            "DbStateProvider initialized (read_only=%s, workspace=%s)",
            read_only,
            workspace_path,
        )

    @property
    def read_only(self) -> bool:
        """Whether this provider is read-only"""
        return self._read_only

    def add_listener(self, listener: StateListener) -> None:
        """Register a listener for state change events

        When not in a scheduler context (or standalone mode), automatically
        starts the file watcher to detect database changes from other processes.

        Args:
            listener: Callback function that receives StateEvent objects
        """
        # Call parent to register the listener
        super().add_listener(listener)

        # Auto-start file watcher if not in a scheduler context
        # or if standalone mode is enabled
        if self._db_file_watch is None:
            from experimaestro.scheduler.base import Scheduler

            if self._standalone or not Scheduler.has_instance():
                logger.info(
                    "Starting file watcher for cross-process monitoring "
                    "(standalone=%s, scheduler=%s)",
                    self._standalone,
                    Scheduler.has_instance(),
                )
                self._start_file_watcher()

    # Experiment management methods

    @_with_db_context
    def ensure_experiment(self, experiment_id: str):
        """Create or update experiment record

        Args:
            experiment_id: Unique identifier for the experiment
        """
        if self.read_only:
            raise RuntimeError("Cannot modify experiments in read-only mode")

        now = datetime.now()
        ExperimentModel.insert(
            experiment_id=experiment_id,
            created_at=now,
            updated_at=now,
        ).on_conflict(
            conflict_target=[ExperimentModel.experiment_id],
            update={
                ExperimentModel.updated_at: now,
            },
        ).execute()

        logger.debug("Ensured experiment: %s", experiment_id)

        # Notify listeners
        self._notify_listeners(
            ExperimentUpdatedEvent(
                experiment_id=experiment_id,
            )
        )

    @_with_db_context
    def create_run(self, experiment_id: str, run_id: Optional[str] = None) -> str:
        """Create a new run for an experiment

        Args:
            experiment_id: Experiment identifier
            run_id: Optional run ID (auto-generated from timestamp if not provided)

        Returns:
            The run_id that was created

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot create runs in read-only mode")

        # Auto-generate run_id from timestamp if not provided
        # Note: For filesystem collision detection, experiment.py generates run_id
        # This fallback uses simple timestamp format without microseconds
        if run_id is None:
            now = datetime.now()
            run_id = now.strftime("%Y%m%d_%H%M%S")

        # Capture hostname
        hostname = socket.gethostname()
        started_at = datetime.now()

        # Create run record with hostname
        ExperimentRunModel.insert(
            experiment_id=experiment_id,
            run_id=run_id,
            started_at=started_at,
            status="active",
            hostname=hostname,
        ).execute()

        # Update experiment's current_run_id and updated_at
        now = datetime.now()
        ExperimentModel.update(
            current_run_id=run_id,
            updated_at=now,
        ).where(ExperimentModel.experiment_id == experiment_id).execute()

        logger.info(
            "Created run %s for experiment %s on host %s",
            run_id,
            experiment_id,
            hostname,
        )

        # Notify listeners
        self._notify_listeners(
            RunUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
            )
        )

        return run_id

    @_with_db_context
    def get_current_run(self, experiment_id: str) -> Optional[str]:
        """Get the current/latest run_id for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            Current run_id or None if no runs exist
        """
        try:
            experiment = ExperimentModel.get(
                ExperimentModel.experiment_id == experiment_id
            )
            return experiment.current_run_id
        except ExperimentModel.DoesNotExist:
            return None

    @_with_db_context
    def get_experiments(self, since: Optional[datetime] = None) -> List[MockExperiment]:
        """Get list of all experiments

        Args:
            since: If provided, only return experiments updated after this timestamp

        Returns:
            List of MockExperiment objects with attributes:
            - workdir: Path to experiment directory
            - experiment_id: Unique identifier (property derived from workdir.name)
            - current_run_id: Current/latest run ID
            - total_jobs: Total number of jobs (for current run)
            - finished_jobs: Number of completed jobs (for current run)
            - failed_jobs: Number of failed jobs (for current run)
            - updated_at: When experiment was last modified
            - hostname: Host where the current run was launched
        """
        experiments = []

        query = ExperimentModel.select()
        if since is not None:
            query = query.where(ExperimentModel.updated_at > since)

        for exp_model in query:
            # Count jobs for current run
            total_jobs = 0
            finished_jobs = 0
            failed_jobs = 0

            started_at = None
            ended_at = None
            hostname = None

            if exp_model.current_run_id:
                # Query jobs via JobExperimentsModel join
                job_ids_in_run = JobExperimentsModel.select(
                    JobExperimentsModel.job_id
                ).where(
                    (JobExperimentsModel.experiment_id == exp_model.experiment_id)
                    & (JobExperimentsModel.run_id == exp_model.current_run_id)
                )
                total_jobs = (
                    JobModel.select().where(JobModel.job_id.in_(job_ids_in_run)).count()
                )
                finished_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.job_id.in_(job_ids_in_run))
                        & (JobModel.state == "done")
                    )
                    .count()
                )
                failed_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.job_id.in_(job_ids_in_run))
                        & (JobModel.state == "error")
                    )
                    .count()
                )

                # Get run timestamps and hostname
                try:
                    run_model = ExperimentRunModel.get(
                        (ExperimentRunModel.experiment_id == exp_model.experiment_id)
                        & (ExperimentRunModel.run_id == exp_model.current_run_id)
                    )
                    if run_model.started_at:
                        started_at = run_model.started_at.timestamp()
                    if run_model.ended_at:
                        ended_at = run_model.ended_at.timestamp()
                    hostname = run_model.hostname
                except ExperimentRunModel.DoesNotExist:
                    pass

            # Compute experiment workdir path (includes run_id)
            if exp_model.current_run_id:
                exp_path = (
                    self.workspace_path
                    / "experiments"
                    / exp_model.experiment_id
                    / exp_model.current_run_id
                )
            else:
                # Fallback for experiments without runs
                exp_path = self.workspace_path / "experiments" / exp_model.experiment_id

            experiments.append(
                MockExperiment(
                    workdir=exp_path,
                    current_run_id=exp_model.current_run_id,
                    total_jobs=total_jobs,
                    finished_jobs=finished_jobs,
                    failed_jobs=failed_jobs,
                    updated_at=exp_model.updated_at.isoformat(),
                    started_at=started_at,
                    ended_at=ended_at,
                    hostname=hostname,
                )
            )

        return experiments

    @_with_db_context
    def get_experiment(self, experiment_id: str) -> Optional[MockExperiment]:
        """Get a specific experiment by ID

        Args:
            experiment_id: Experiment identifier

        Returns:
            MockExperiment object or None if not found
        """
        try:
            exp_model = ExperimentModel.get(
                ExperimentModel.experiment_id == experiment_id
            )
        except ExperimentModel.DoesNotExist:
            return None

        # Count jobs for current run
        total_jobs = 0
        finished_jobs = 0
        failed_jobs = 0
        hostname = None

        if exp_model.current_run_id:
            # Query jobs via JobExperimentsModel
            job_ids_in_run = JobExperimentsModel.select(
                JobExperimentsModel.job_id
            ).where(
                (JobExperimentsModel.experiment_id == exp_model.experiment_id)
                & (JobExperimentsModel.run_id == exp_model.current_run_id)
            )
            total_jobs = (
                JobModel.select().where(JobModel.job_id.in_(job_ids_in_run)).count()
            )
            finished_jobs = (
                JobModel.select()
                .where(
                    (JobModel.job_id.in_(job_ids_in_run)) & (JobModel.state == "done")
                )
                .count()
            )
            failed_jobs = (
                JobModel.select()
                .where(
                    (JobModel.job_id.in_(job_ids_in_run)) & (JobModel.state == "error")
                )
                .count()
            )

            # Get hostname from run model
            try:
                run_model = ExperimentRunModel.get(
                    (ExperimentRunModel.experiment_id == exp_model.experiment_id)
                    & (ExperimentRunModel.run_id == exp_model.current_run_id)
                )
                hostname = run_model.hostname
            except ExperimentRunModel.DoesNotExist:
                pass

        # Compute experiment workdir path (includes run_id)
        if exp_model.current_run_id:
            exp_path = (
                self.workspace_path
                / "experiments"
                / exp_model.experiment_id
                / exp_model.current_run_id
            )
        else:
            # Fallback for experiments without runs
            exp_path = self.workspace_path / "experiments" / exp_model.experiment_id

        return MockExperiment(
            workdir=exp_path,
            current_run_id=exp_model.current_run_id,
            total_jobs=total_jobs,
            finished_jobs=finished_jobs,
            failed_jobs=failed_jobs,
            updated_at=exp_model.updated_at.isoformat(),
            hostname=hostname,
        )

    @_with_db_context
    def get_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of ExperimentRun dataclass instances with job statistics
        """
        runs = []
        for run_model in (
            ExperimentRunModel.select()
            .where(ExperimentRunModel.experiment_id == experiment_id)
            .order_by(ExperimentRunModel.started_at.desc())
        ):
            # Count jobs for this run via JobExperimentsModel
            job_ids_in_run = JobExperimentsModel.select(
                JobExperimentsModel.job_id
            ).where(
                (JobExperimentsModel.experiment_id == experiment_id)
                & (JobExperimentsModel.run_id == run_model.run_id)
            )
            total_jobs = (
                JobModel.select().where(JobModel.job_id.in_(job_ids_in_run)).count()
            )
            finished_jobs = (
                JobModel.select()
                .where(
                    (JobModel.job_id.in_(job_ids_in_run)) & (JobModel.state == "done")
                )
                .count()
            )
            failed_jobs = (
                JobModel.select()
                .where(
                    (JobModel.job_id.in_(job_ids_in_run)) & (JobModel.state == "error")
                )
                .count()
            )

            runs.append(
                ExperimentRun(
                    run_id=run_model.run_id,
                    experiment_id=run_model.experiment_id,
                    hostname=run_model.hostname,
                    started_at=(
                        run_model.started_at.timestamp()
                        if run_model.started_at
                        else None
                    ),
                    ended_at=(
                        run_model.ended_at.timestamp() if run_model.ended_at else None
                    ),
                    status=run_model.status,
                    total_jobs=total_jobs,
                    finished_jobs=finished_jobs,
                    failed_jobs=failed_jobs,
                )
            )
        return runs

    @_with_db_context
    def complete_run(self, experiment_id: str, run_id: str, status: str = "completed"):
        """Mark a run as completed

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            status: Final status (completed, failed, abandoned)

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot modify runs in read-only mode")

        ExperimentRunModel.update(ended_at=datetime.now(), status=status).where(
            (ExperimentRunModel.experiment_id == experiment_id)
            & (ExperimentRunModel.run_id == run_id)
        ).execute()

        logger.info("Marked run %s/%s as %s", experiment_id, run_id, status)

    # Job operations

    @_with_db_context
    def get_jobs(
        self,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[MockJob]:
        """Query jobs with optional filters

        Args:
            experiment_id: Filter by experiment (None = all experiments)
            run_id: Filter by run (None = current run if experiment_id provided)
            task_id: Filter by task class identifier
            state: Filter by job state
            tags: Filter by tags (all tags must match)
            since: If provided, only return jobs updated after this timestamp

        Returns:
            List of MockJob objects
        """
        # Build base query
        query = JobModel.select()

        # Apply since filter for incremental updates
        if since is not None:
            query = query.where(JobModel.updated_at > since)

        # Apply experiment filter via JobExperimentsModel
        if experiment_id is not None:
            # If experiment_id provided but not run_id, use current run
            if run_id is None:
                current_run = self.get_current_run(experiment_id)
                if current_run is None:
                    return []  # No runs exist for this experiment
                run_id = current_run

            # Filter by jobs in this experiment/run
            job_ids_in_run = JobExperimentsModel.select(
                JobExperimentsModel.job_id
            ).where(
                (JobExperimentsModel.experiment_id == experiment_id)
                & (JobExperimentsModel.run_id == run_id)
            )
            query = query.where(JobModel.job_id.in_(job_ids_in_run))

        # Apply task_id filter
        if task_id is not None:
            query = query.where(JobModel.task_id == task_id)

        # Apply state filter
        if state is not None:
            query = query.where(JobModel.state == state)

        # Apply tag filters
        if tags and experiment_id is not None and run_id is not None:
            for tag_key, tag_value in tags.items():
                # Join with JobTagModel for each tag filter
                query = query.join(
                    JobTagModel,
                    on=(
                        (JobTagModel.job_id == JobModel.job_id)
                        & (JobTagModel.experiment_id == experiment_id)
                        & (JobTagModel.run_id == run_id)
                        & (JobTagModel.tag_key == tag_key)
                        & (JobTagModel.tag_value == tag_value)
                    ),
                )

        # Execute query and convert to MockJob objects
        jobs = []
        for job_model in query:
            jobs.append(self._job_model_to_mock_job(job_model))

        return jobs

    @_with_db_context
    def get_job(
        self, job_id: str, experiment_id: str, run_id: Optional[str] = None
    ) -> Optional[MockJob]:
        """Get a specific job

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            MockJob object or None if not found
        """
        # Use current run if not specified
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return None

        # Check if job belongs to this experiment/run
        try:
            JobExperimentsModel.get(
                (JobExperimentsModel.job_id == job_id)
                & (JobExperimentsModel.experiment_id == experiment_id)
                & (JobExperimentsModel.run_id == run_id)
            )
        except JobExperimentsModel.DoesNotExist:
            return None

        try:
            job_model = JobModel.get(JobModel.job_id == job_id)
        except JobModel.DoesNotExist:
            return None

        return self._job_model_to_mock_job(job_model)

    @_with_db_context
    def update_job_submitted(self, job: "Job", experiment_id: str, run_id: str):
        """Record that a job has been submitted

        Args:
            job: Job instance
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update jobs in read-only mode")

        task_id = str(job.type.identifier)

        # Create or update job record
        now = datetime.now()
        JobModel.insert(
            job_id=job.identifier,
            task_id=task_id,
            state=job.state.name,
            submitted_time=job.submittime,
            transient=int(job.transient),
            updated_at=now,
        ).on_conflict(
            conflict_target=[JobModel.job_id, JobModel.task_id],
            update={
                JobModel.state: job.state.name,
                JobModel.submitted_time: job.submittime,
                JobModel.transient: int(job.transient),
                JobModel.updated_at: now,
                JobModel.failure_reason: None,  # Clear old failure reason on resubmit
            },
        ).execute()

        # Create or update job-experiment relationship
        JobExperimentsModel.insert(
            job_id=job.identifier,
            experiment_id=experiment_id,
            run_id=run_id,
        ).on_conflict_ignore().execute()

        # Update tags (run-scoped) - delete existing and insert new
        JobTagModel.delete().where(
            (JobTagModel.job_id == job.identifier)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ).execute()

        if job.tags:
            tag_records = [
                {
                    "job_id": job.identifier,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "tag_key": key,
                    "tag_value": value,
                }
                for key, value in job.tags.items()
            ]
            JobTagModel.insert_many(tag_records).execute()

        # Store dependencies (run-scoped) - delete existing and insert new
        from experimaestro.scheduler.jobs import JobDependency

        JobDependenciesModel.delete().where(
            (JobDependenciesModel.job_id == job.identifier)
            & (JobDependenciesModel.experiment_id == experiment_id)
            & (JobDependenciesModel.run_id == run_id)
        ).execute()

        depends_on_ids = []
        dependency_records = []
        for dep in job.dependencies:
            if isinstance(dep, JobDependency):
                dep_job = dep.origin
                dep_task_id = str(dep_job.type.identifier)
                depends_on_ids.append(dep_job.identifier)
                dependency_records.append(
                    {
                        "job_id": job.identifier,
                        "task_id": task_id,
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "depends_on_job_id": dep_job.identifier,
                        "depends_on_task_id": dep_task_id,
                    }
                )

        if dependency_records:
            JobDependenciesModel.insert_many(dependency_records).execute()

        # Notify that a job was added to this experiment (with its tags and dependencies)
        self._notify_listeners(
            JobExperimentUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=job.identifier,
                tags=job.tags or {},
                depends_on=depends_on_ids,
            )
        )

        # Register partials for all declared partial
        partial = job.type._partials
        for name, sp in partial.items():
            partial_id = job.config.__xpm__.get_partial_identifier(sp)
            partial_id_hex = partial_id.all.hex()

            # Register the partial directory
            self.register_partial(partial_id_hex, task_id, name)

            # Link job to partial
            self.register_job_partial(
                job.identifier, experiment_id, run_id, partial_id_hex
            )

        logger.debug(
            "Recorded job submission: %s (experiment=%s, run=%s)",
            job.identifier,
            experiment_id,
            run_id,
        )

        # Notify listeners
        self._notify_listeners(
            JobUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=job.identifier,
            )
        )

    @_with_db_context
    def update_job_state(self, job: "Job", experiment_id: str, run_id: str):
        """Update the state of a job

        Args:
            job: Job instance
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update jobs in read-only mode")

        # Build update dict with updated_at timestamp
        now = datetime.now()
        update_data = {
            JobModel.state: job.state.name,
            JobModel.updated_at: now,
        }

        # Add or clear failure reason based on state
        from experimaestro.scheduler.jobs import JobStateError

        if isinstance(job.state, JobStateError) and job.state.failure_reason:
            update_data[JobModel.failure_reason] = job.state.failure_reason.name
        else:
            # Clear failure reason when job is not in error state
            update_data[JobModel.failure_reason] = None

        # Add timing information
        if job.starttime:
            update_data[JobModel.started_time] = job.starttime
        if job.endtime:
            update_data[JobModel.ended_time] = job.endtime

        # Add progress information
        if job._progress:
            update_data[JobModel.progress] = json.dumps(
                [
                    {"level": p.level, "progress": p.progress, "desc": p.desc}
                    for p in job._progress
                ]
            )

        # Update the job record (job_id is unique in JobModel)
        JobModel.update(update_data).where(JobModel.job_id == job.identifier).execute()

        logger.debug(
            "Updated job state: %s -> %s (experiment=%s, run=%s)",
            job.identifier,
            job.state.name,
            experiment_id,
            run_id,
        )

        # Notify listeners
        self._notify_listeners(
            JobUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                job_id=job.identifier,
            )
        )

    @_with_db_context
    def delete_job(self, job_id: str):
        """Remove a job, its tags, dependencies, and partial references

        Args:
            job_id: Job identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot delete jobs in read-only mode")

        # Delete tags first (foreign key constraint)
        JobTagModel.delete().where(JobTagModel.job_id == job_id).execute()

        # Delete dependencies (both where this job is the source or target)
        JobDependenciesModel.delete().where(
            JobDependenciesModel.job_id == job_id
        ).execute()
        JobDependenciesModel.delete().where(
            JobDependenciesModel.depends_on_job_id == job_id
        ).execute()

        # Delete partial references
        JobPartialModel.delete().where(JobPartialModel.job_id == job_id).execute()

        # Delete job
        JobModel.delete().where(JobModel.job_id == job_id).execute()

        logger.debug("Deleted job %s", job_id)

    # CLI utility methods for job management

    @_with_db_context
    def get_all_jobs(
        self,
        state: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
    ) -> List[MockJob]:
        """Query all jobs across all experiments/runs

        This method is designed for CLI tools that need to list or manage jobs
        across the entire workspace, regardless of experiment or run.

        Args:
            state: Filter by job state (e.g., "done", "error", "running")
            tags: Filter by tags (all tags must match)
            since: If provided, only return jobs updated after this timestamp

        Returns:
            List of MockJob objects
        """
        # Build base query
        query = JobModel.select()

        # Apply since filter for incremental updates
        if since is not None:
            query = query.where(JobModel.updated_at > since)

        # Apply state filter
        if state is not None:
            query = query.where(JobModel.state == state)

        # Tag filters are not supported for get_all_jobs (requires experiment/run context)
        # Use get_jobs(experiment_id=...) for tag filtering

        # Execute query and convert to MockJob objects
        jobs = []
        for job_model in query:
            jobs.append(self._job_model_to_mock_job(job_model))

        return jobs

    @_with_db_context
    def get_tags_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Get tags map for jobs in an experiment/run

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            Dictionary mapping job identifiers to their tags dict
        """
        # Use current run if not specified
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return {}

        # Fetch all tags for this experiment/run
        tags_map: Dict[str, Dict[str, str]] = {}
        all_tags = JobTagModel.select().where(
            (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        )
        for tag in all_tags:
            if tag.job_id not in tags_map:
                tags_map[tag.job_id] = {}
            tags_map[tag.job_id][tag.tag_key] = tag.tag_value

        return tags_map

    @_with_db_context
    def get_dependencies_map(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Get dependencies map for jobs in an experiment/run

        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier (None = current run)

        Returns:
            Dictionary mapping job identifiers to list of job IDs they depend on
        """
        # Use current run if not specified
        if run_id is None:
            run_id = self.get_current_run(experiment_id)
            if run_id is None:
                return {}

        # Fetch all dependencies for this experiment/run
        deps_map: Dict[str, List[str]] = {}
        all_deps = JobDependenciesModel.select().where(
            (JobDependenciesModel.experiment_id == experiment_id)
            & (JobDependenciesModel.run_id == run_id)
        )
        for dep in all_deps:
            if dep.job_id not in deps_map:
                deps_map[dep.job_id] = []
            deps_map[dep.job_id].append(dep.depends_on_job_id)

        return deps_map

    def kill_job(self, job: MockJob, perform: bool = False) -> bool:
        """Kill a running job process

        This method finds the process associated with a running job and kills it.
        It also updates the job state in the database to ERROR.

        Args:
            job: MockJob instance to kill
            perform: If True, actually kill the process. If False, just check
                if the job can be killed (dry run).

        Returns:
            True if job was killed (or would be killed in dry run),
            False if job is not running or process not found
        """
        # Check if job is in a running state
        if not job.state.running():
            logger.debug("Job %s is not running (state=%s)", job.identifier, job.state)
            return False

        # Get process from job
        process = job.getprocess()
        if process is None:
            logger.warning("Could not get process for job %s", job.identifier)
            return False

        if perform:
            try:
                logger.info("Killing job %s (process: %s)", job.identifier, process)
                process.kill()

                # Update job state in database
                if not self.read_only:
                    self._update_job_state_to_error(job, "killed")
            except Exception as e:
                logger.error("Error killing job %s: %s", job.identifier, e)
                return False

        return True

    def _update_job_state_to_error(self, job: MockJob, reason: str):
        """Update job state to ERROR in database

        Args:
            job: MockJob instance
            reason: Failure reason
        """
        if self.read_only:
            return

        now = datetime.now()
        with self.workspace_db.bind_ctx([JobModel]):
            JobModel.update(
                state="error",
                failure_reason=reason,
                ended_time=now.timestamp(),
                updated_at=now,
            ).where(JobModel.job_id == job.identifier).execute()

        logger.debug(
            "Updated job %s state to error (reason=%s)", job.identifier, reason
        )

    def clean_job(self, job: MockJob, perform: bool = False) -> bool:
        """Clean a finished job (delete directory and DB entry)

        This method removes the job's working directory and its database entry.
        Only finished jobs (DONE or ERROR state) can be cleaned.

        Args:
            job: MockJob instance to clean
            perform: If True, actually delete the job. If False, just check
                if the job can be cleaned (dry run).

        Returns:
            True if job was cleaned (or would be cleaned in dry run),
            False if job is not finished or cannot be cleaned
        """
        from shutil import rmtree

        # Check if job is in a finished state
        if not job.state.finished():
            logger.debug(
                "Job %s is not finished (state=%s), cannot clean",
                job.identifier,
                job.state,
            )
            return False

        if perform:
            # Delete job directory
            if job.path.exists():
                logger.info("Cleaning job %s: removing %s", job.identifier, job.path)
                rmtree(job.path)
            else:
                logger.warning("Job directory does not exist: %s", job.path)

            # Delete from database
            if not self.read_only:
                self.delete_job(job.identifier)

        return True

    def kill_jobs(self, jobs: List[MockJob], perform: bool = False) -> int:
        """Kill multiple jobs

        Args:
            jobs: List of MockJob instances to kill
            perform: If True, actually kill the processes. If False, dry run.

        Returns:
            Number of jobs that were killed (or would be killed in dry run)
        """
        count = 0
        for job in jobs:
            if self.kill_job(job, perform=perform):
                count += 1
        return count

    def clean_jobs(self, jobs: List[MockJob], perform: bool = False) -> int:
        """Clean multiple finished jobs

        Args:
            jobs: List of MockJob instances to clean
            perform: If True, actually delete the jobs. If False, dry run.

        Returns:
            Number of jobs that were cleaned (or would be cleaned in dry run)
        """
        count = 0
        for job in jobs:
            if self.clean_job(job, perform=perform):
                count += 1
        return count

    def delete_job_safely(
        self, job: MockJob, cascade_orphans: bool = True
    ) -> tuple[bool, str]:
        """Delete a job with proper locking and orphan cleanup

        This method is designed for TUI/UI use. It acquires a lock on the job
        to prevent race conditions, then deletes the job directory and DB entry.

        Args:
            job: MockJob instance to delete
            cascade_orphans: If True, clean up orphan partials after deletion

        Returns:
            Tuple of (success: bool, message: str)
        """
        import fasteners
        from shutil import rmtree

        # Check if job is running
        if job.state.running():
            return False, "Cannot delete a running job"

        # Check if path exists
        if not job.path or not job.path.exists():
            # Just delete from database if path doesn't exist
            if not self.read_only:
                self.delete_job(job.identifier)
            if cascade_orphans:
                self.cleanup_orphan_partials(perform=True)
            return True, f"Job {job.identifier} deleted (directory already gone)"

        # Try to acquire job lock (non-blocking)
        # Lock file is typically {script_name}.lock, but we use .lock for general locking
        lock_path = job.path / ".lock"
        lock = fasteners.InterProcessLock(str(lock_path))

        if not lock.acquire(blocking=False):
            return False, "Job is currently locked (possibly running)"

        try:
            # Delete all files except the lock file
            for item in job.path.iterdir():
                if item.name != ".lock":
                    if item.is_dir():
                        rmtree(item)
                    else:
                        item.unlink()

            # Mark job as "phantom" in database (don't delete - keep as phantom)
            if not self.read_only:
                from datetime import datetime

                JobModel.update(
                    state="phantom",
                    updated_at=datetime.now(),
                ).where(JobModel.job_id == job.identifier).execute()

        finally:
            lock.release()
            # Now delete the lock file and directory
            try:
                lock_path.unlink(missing_ok=True)
                if job.path.exists() and not any(job.path.iterdir()):
                    job.path.rmdir()
            except Exception as e:
                logger.warning("Could not clean up lock file: %s", e)

        # Clean up orphan partials if requested
        if cascade_orphans:
            self.cleanup_orphan_partials(perform=True)

        return True, f"Job {job.identifier} deleted successfully"

    @_with_db_context
    def delete_experiment(
        self, experiment_id: str, delete_jobs: bool = False
    ) -> tuple[bool, str]:
        """Delete an experiment from the database

        Args:
            experiment_id: Experiment identifier
            delete_jobs: If True, also delete associated jobs (default: False)

        Returns:
            Tuple of (success: bool, message: str)
        """
        from shutil import rmtree

        if self.read_only:
            return False, "Cannot delete in read-only mode"

        # Get all jobs for this experiment
        jobs = self.get_jobs(experiment_id)
        running_jobs = [j for j in jobs if j.state.running()]

        if running_jobs:
            return (
                False,
                f"Cannot delete experiment with {len(running_jobs)} running job(s)",
            )

        # Delete jobs if requested
        if delete_jobs:
            for job in jobs:
                success, msg = self.delete_job_safely(job, cascade_orphans=False)
                if not success:
                    logger.warning("Failed to delete job %s: %s", job.identifier, msg)

        # Delete experiment runs
        ExperimentRunModel.delete().where(
            ExperimentRunModel.experiment_id == experiment_id
        ).execute()

        # Delete experiment
        ExperimentModel.delete().where(
            ExperimentModel.experiment_id == experiment_id
        ).execute()

        # Optionally delete experiment directory (includes all runs)
        exp_path = self.workspace_path / "experiments" / experiment_id
        if exp_path.exists():
            try:
                rmtree(exp_path)
            except Exception as e:
                logger.warning("Could not delete experiment directory: %s", e)

        # Clean up orphan partials
        self.cleanup_orphan_partials(perform=True)

        return True, f"Experiment {experiment_id} deleted successfully"

    @_with_db_context
    def get_orphan_jobs(self) -> List[MockJob]:
        """Find jobs that have no associated experiment in the database

        Returns:
            List of MockJob instances for orphan jobs
        """
        # Subquery to get all job IDs that are linked to experiments
        job_ids_with_experiments = JobExperimentsModel.select(
            JobExperimentsModel.job_id
        )

        # Query jobs where job_id is not in any experiment
        orphan_job_models = (
            JobModel.select()
            .where(JobModel.job_id.not_in(job_ids_with_experiments))
            .order_by(JobModel.updated_at.desc())
        )

        # Convert to MockJob instances
        orphan_jobs = []
        for job_model in orphan_job_models:
            orphan_jobs.append(self._job_model_to_mock_job(job_model))

        return orphan_jobs

    # Service operations

    @_with_db_context
    def register_service(
        self,
        service_id: str,
        experiment_id: str,
        run_id: str,
        description: str,
        state_dict: Optional[str] = None,
    ):
        """Register a service in the database

        Services are only added or removed, not updated. Runtime state
        is managed by the Service object itself.

        Args:
            service_id: Service identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            description: Human-readable description
            state_dict: JSON serialized state_dict for service recreation

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot register services in read-only mode")

        insert_data = {
            "service_id": service_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "description": description,
            "created_at": datetime.now(),
        }

        if state_dict is not None:
            insert_data["state_dict"] = state_dict

        # Use INSERT OR IGNORE - services are only added, not updated
        ServiceModel.insert(**insert_data).on_conflict_ignore().execute()

        logger.debug(
            "Registered service %s (experiment=%s, run=%s)",
            service_id,
            experiment_id,
            run_id,
        )

        # Notify listeners
        self._notify_listeners(
            ServiceUpdatedEvent(
                experiment_id=experiment_id,
                run_id=run_id,
                service_id=service_id,
            )
        )

    def _get_live_services(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> Optional[List["Service"]]:
        """Get live services from scheduler if available.

        Returns None if no live services (experiment not in scheduler).
        """
        if experiment_id is None:
            return None

        try:
            from experimaestro.scheduler.base import Scheduler

            if not Scheduler.has_instance():
                return None

            scheduler = Scheduler.instance()
            if experiment_id not in scheduler.experiments:
                logger.debug("Experiment %s not in scheduler", experiment_id)
                return None

            exp = scheduler.experiments[experiment_id]
            services = list(exp.services.values())
            logger.debug(
                "Returning %d live services for experiment %s",
                len(services),
                experiment_id,
            )
            return services

        except Exception as e:
            logger.warning("Could not get live services: %s", e)
            return None

    @_with_db_context
    def _fetch_services_from_storage(
        self, experiment_id: Optional[str], run_id: Optional[str]
    ) -> List["Service"]:
        """Fetch services from database.

        Called when no live services and cache is empty.
        """
        from experimaestro.scheduler.services import Service

        query = ServiceModel.select()

        if experiment_id is not None:
            query = query.where(
                (ServiceModel.experiment_id == experiment_id)
                & (ServiceModel.run_id == run_id)
            )

        services = []

        for service_model in query:
            service_id = service_model.service_id

            # Try to recreate service from state_dict
            state_dict_json = service_model.state_dict
            if state_dict_json and state_dict_json != "{}":
                try:
                    state_dict = json.loads(state_dict_json)
                    if "__class__" in state_dict:
                        service = Service.from_state_dict(state_dict)
                except Exception as e:
                    service = MockService(
                        service_id,
                        f"error: {e}",
                        {},
                        experiment_id=experiment_id,
                        run_id=run_id,
                    )

                    logger.warning(
                        "Failed to recreate service %s from state_dict: %s",
                        service_id,
                        e,
                    )
            else:
                # If we can't recreate, skip this service (it's not usable)
                logger.debug(
                    "Service %s has no state_dict for recreation, skipping",
                    service_id,
                )
                service = MockService(
                    service_id,
                    "error: no state_dict",
                    {},
                    experiment_id=experiment_id,
                    run_id=run_id,
                )

            # Add to services
            service.id = service_id
            services.append(service)
            continue

        return services

    def get_live_job_states(self, experiment_id: str) -> Dict[str, str]:
        """Get live job states from the scheduler if available

        This is useful for debugging to compare live state vs database state.

        Args:
            experiment_id: The experiment ID to get live jobs for

        Returns:
            Dict mapping job identifier to live state name, empty if scheduler
            not available or experiment not registered
        """
        try:
            from experimaestro.scheduler.base import Scheduler

            if not Scheduler.has_instance():
                logger.debug("No scheduler instance available for live states")
                return {}

            scheduler = Scheduler.instance()
            live_states = {}

            logger.debug(
                "get_live_job_states: looking for exp=%s, scheduler has %d jobs",
                experiment_id,
                len(scheduler.jobs),
            )

            for job_id, job in scheduler.jobs.items():
                # Filter by experiment if needed
                if hasattr(job, "experiment") and job.experiment is not None:
                    if hasattr(job.experiment, "name"):
                        job_exp_id = job.experiment.name
                        if job_exp_id == experiment_id:
                            live_states[job_id] = job.state.name
                        else:
                            logger.debug(
                                "Job %s exp_id=%s != requested %s",
                                job_id[:8],
                                job_exp_id,
                                experiment_id,
                            )
                else:
                    # Job not associated with experiment, include it anyway
                    live_states[job_id] = job.state.name
                    logger.debug(
                        "Job %s has no experiment, including anyway", job_id[:8]
                    )

            logger.debug("Returning %d live job states", len(live_states))
            return live_states

        except Exception as e:
            logger.debug("Could not get live job states: %s", e)
            return {}

    # Sync metadata methods

    @_with_db_context
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync

        Returns:
            datetime of last sync, or None if never synced
        """
        from peewee import OperationalError

        from .state_db import WorkspaceSyncMetadata

        try:
            metadata = WorkspaceSyncMetadata.get_or_none(
                WorkspaceSyncMetadata.id == "workspace"
            )
            if metadata and metadata.last_sync_time:
                return metadata.last_sync_time
        except OperationalError:
            # Table might not exist in older workspaces opened in read-only mode
            pass
        return None

    @_with_db_context
    def update_last_sync_time(self) -> None:
        """Update the last sync timestamp to now

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update sync time in read-only mode")

        from .state_db import WorkspaceSyncMetadata

        WorkspaceSyncMetadata.insert(
            id="workspace", last_sync_time=datetime.now()
        ).on_conflict(
            conflict_target=[WorkspaceSyncMetadata.id],
            update={WorkspaceSyncMetadata.last_sync_time: datetime.now()},
        ).execute()
        logger.debug("Updated last sync time")

    # Partial management methods

    @_with_db_context
    def register_partial(
        self, partial_id: str, task_id: str, partial_name: str
    ) -> None:
        """Register a partial directory (creates if not exists)

        Args:
            partial_id: Hex hash of the partial identifier
            task_id: Task class identifier
            partial_name: Name of the partial definition

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot register partials in read-only mode")

        PartialModel.insert(
            partial_id=partial_id,
            task_id=task_id,
            partial_name=partial_name,
            created_at=datetime.now(),
        ).on_conflict_ignore().execute()

        logger.debug(
            "Registered partial: %s (task=%s, subparams=%s)",
            partial_id,
            task_id,
            partial_name,
        )

    @_with_db_context
    def register_job_partial(
        self, job_id: str, experiment_id: str, run_id: str, partial_id: str
    ) -> None:
        """Link a job to a partial directory it uses

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            partial_id: Partial directory identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot register job partials in read-only mode")

        JobPartialModel.insert(
            job_id=job_id,
            experiment_id=experiment_id,
            run_id=run_id,
            partial_id=partial_id,
        ).on_conflict_ignore().execute()

        logger.debug(
            "Linked job %s to partial %s (experiment=%s, run=%s)",
            job_id,
            partial_id,
            experiment_id,
            run_id,
        )

    @_with_db_context
    def unregister_job_partials(
        self, job_id: str, experiment_id: str, run_id: str
    ) -> None:
        """Remove all partial links for a job

        Called when a job is deleted to clean up its partial references.

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot unregister job partials in read-only mode")

        JobPartialModel.delete().where(
            (JobPartialModel.job_id == job_id)
            & (JobPartialModel.experiment_id == experiment_id)
            & (JobPartialModel.run_id == run_id)
        ).execute()

        logger.debug(
            "Unregistered partials for job %s (experiment=%s, run=%s)",
            job_id,
            experiment_id,
            run_id,
        )

    @_with_db_context
    def get_orphan_partials(self) -> List[Dict]:
        """Find partial directories that are not referenced by any job

        Returns:
            List of dictionaries with partial_id, task_id, partial_name
        """
        # Find partials that have no job references
        # Using a subquery to find referenced partial_ids
        referenced_partials = JobPartialModel.select(JobPartialModel.partial_id)

        orphan_query = PartialModel.select().where(
            PartialModel.partial_id.not_in(referenced_partials)
        )

        orphans = []
        for partial in orphan_query:
            orphans.append(
                {
                    "partial_id": partial.partial_id,
                    "task_id": partial.task_id,
                    "partial_name": partial.partial_name,
                    "created_at": partial.created_at.isoformat(),
                }
            )

        return orphans

    def cleanup_orphan_partials(self, perform: bool = False) -> List[Path]:
        """Clean up orphan partial directories

        Finds partial directories not referenced by any job and removes them.

        Args:
            perform: If True, actually delete. If False, dry run (list only).

        Returns:
            List of paths that were deleted (or would be deleted in dry run)
        """
        from shutil import rmtree

        orphans = self.get_orphan_partials()
        deleted_paths = []

        for orphan in orphans:
            # Reconstruct path: WORKSPACE/partials/TASK_ID/SUBPARAM_NAME/PARTIAL_ID
            partial_path = (
                self.workspace_path
                / "partials"
                / orphan["task_id"]
                / orphan["partial_name"]
                / orphan["partial_id"]
            )

            if perform:
                # Delete directory if it exists
                if partial_path.exists():
                    logger.info("Cleaning orphan partial: %s", partial_path)
                    rmtree(partial_path)

                # Delete from database
                if not self.read_only:
                    with self.workspace_db.bind_ctx([PartialModel]):
                        PartialModel.delete().where(
                            PartialModel.partial_id == orphan["partial_id"]
                        ).execute()

            deleted_paths.append(partial_path)

        return deleted_paths

    # Utility methods

    def close(self):
        """Close the database connection and remove from registry

        This should be called when done with the workspace to free resources.
        """
        # Stop file watcher if running
        self._stop_file_watcher()

        # Stop FSEvents workaround if running
        if self._fsevents_workaround:
            self._fsevents_workaround.stop()

        # Close database connection
        if hasattr(self, "workspace_db") and self.workspace_db is not None:
            from .state_db import close_workspace_database

            close_workspace_database(self.workspace_db)
            self.workspace_db = None

        # Remove from registry
        with DbStateProvider._lock:
            if self.workspace_path in DbStateProvider._instances:
                del DbStateProvider._instances[self.workspace_path]

        logger.debug("DbStateProvider closed for %s", self.workspace_path)

    # Legacy listener methods (for push notifications with file watcher)
    # These are in addition to state listeners from OfflineStateProvider

    def _notify_listeners(self, event: StateEvent) -> None:
        """Notify all registered listeners of a state change

        This is called internally by state-modifying methods.
        Notifies both legacy listeners and state listeners.

        Args:
            event: State change event to broadcast
        """
        # Schedule FSEvents workaround marker touch (macOS only, rate-limited)
        if self._fsevents_workaround:
            self._fsevents_workaround.schedule_touch()

        # Notify legacy listeners
        with self._listeners_lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Listener %s raised exception: %s", listener, e)

        # Notify state listeners (from OfflineStateProvider)
        self._notify_state_listeners(event)

    def _start_file_watcher(self) -> None:
        """Start watching the database file for changes"""
        if self._db_file_watch is not None:
            logger.info("File watcher already running for %s", self._db_dir)
            return  # Already watching

        from experimaestro.ipc import ipcom

        # Create and start the change detector thread
        self._change_detector = _DatabaseChangeDetector(self)
        self._change_detector.start()

        # Create the file handler that signals the detector
        self._db_file_handler = _DatabaseFileHandler(
            self._change_detector,
            fsevents_workaround=self._fsevents_workaround,
        )
        self._db_file_watch = ipcom().fswatch(
            self._db_file_handler,
            self._db_dir,
            recursive=False,
        )
        logger.info("Started database file watcher for %s", self._db_dir)

    def _stop_file_watcher(self) -> None:
        """Stop watching the database file"""
        if self._db_file_watch is None:
            return  # Not watching

        from experimaestro.ipc import ipcom

        # Stop the file watcher first
        ipcom().fsunwatch(self._db_file_watch)
        self._db_file_watch = None
        self._db_file_handler = None

        # Stop the change detector thread
        if self._change_detector is not None:
            self._change_detector.stop()
            self._change_detector = None

        logger.debug("Stopped database file watcher for %s", self.workspace_path)

    # Helper methods

    def _job_model_to_mock_job(self, job_model: JobModel) -> MockJob:
        """Convert a JobModel to a MockJob object

        Args:
            job_model: JobModel instance

        Returns:
            MockJob object
        """
        # Parse progress JSON and convert to LevelInformation objects
        progress_dicts = json.loads(job_model.progress)
        progress_list = get_progress_information_from_dict(progress_dicts)

        # Compute job path from workspace_path, task_id, and job_id
        job_path = self.workspace_path / "jobs" / job_model.task_id / job_model.job_id

        # Convert failure_reason string to enum if present
        failure_reason = None
        if job_model.failure_reason:
            try:
                failure_reason = JobFailureStatus[job_model.failure_reason]
            except KeyError:
                pass  # Unknown failure reason, leave as None

        return MockJob(
            identifier=job_model.job_id,
            task_id=job_model.task_id,
            path=job_path,
            state=job_model.state,
            submittime=job_model.submitted_time,
            starttime=job_model.started_time,
            endtime=job_model.ended_time,
            progress=progress_list,
            updated_at=job_model.updated_at.isoformat(),
            failure_reason=failure_reason,
            transient=TransientMode(job_model.transient),
        )

    def _format_time(self, timestamp: Optional[float]) -> str:
        """Format timestamp for UI

        Args:
            timestamp: Unix timestamp or None

        Returns:
            ISO format datetime string or empty string
        """
        if not timestamp:
            return ""
        return datetime.fromtimestamp(timestamp).isoformat()


# Scheduler listener adapter
class SchedulerListener:
    """Adapter to connect scheduler events to DbStateProvider

    This class implements the scheduler listener interface and forwards
    events to the DbStateProvider. It tracks which experiment/run
    each job belongs to for proper database updates.
    """

    def __init__(self, state_provider: DbStateProvider):
        """Initialize listener

        Args:
            state_provider: DbStateProvider instance to update
        """
        self.state_provider = state_provider
        # Map job_id -> (experiment_id, run_id) for tracking
        self.job_experiments: Dict[str, tuple] = {}

        logger.info("SchedulerListener initialized")

    @_with_db_context
    def job_submitted(self, job: "Job", experiment_id: str, run_id: str):
        """Called when a job is submitted

        Args:
            job: The submitted job
            experiment_id: Experiment this job belongs to
            run_id: Run this job belongs to
        """
        # Track job's experiment/run
        self.job_experiments[job.identifier] = (experiment_id, run_id)

        # Update state provider
        try:
            self.state_provider.update_job_submitted(job, experiment_id, run_id)
        except Exception as e:
            logger.exception(
                "Error updating job submission for %s: %s", job.identifier, e
            )

    @_with_db_context
    def job_state(self, job: "Job"):
        """Called when a job's state changes

        Args:
            job: The job with updated state
        """
        # Look up job's experiment/run
        if job.identifier not in self.job_experiments:
            logger.warning(
                "State change for unknown job %s (not tracked by listener)",
                job.identifier,
            )
            return

        experiment_id, run_id = self.job_experiments[job.identifier]

        # Update state provider
        try:
            self.state_provider.update_job_state(job, experiment_id, run_id)
        except Exception as e:
            logger.exception("Error updating job state for %s: %s", job.identifier, e)

    @_with_db_context
    def service_add(self, service: "Service", experiment_id: str, run_id: str):
        """Called when a service is added

        Args:
            service: The added service
            experiment_id: Experiment identifier
            run_id: Run identifier
        """
        from experimaestro.scheduler.services import Service

        try:
            # Get state_dict for service recreation
            state_dict_json = None
            try:
                # _full_state_dict includes __class__ automatically
                state_dict = service._full_state_dict()
                # Serialize paths automatically
                serialized = Service.serialize_state_dict(state_dict)
                state_dict_json = json.dumps(serialized)
            except Exception as e:
                # Service cannot be serialized - store unserializable marker
                logger.warning(
                    "Could not get state_dict for service %s: %s", service.id, e
                )
                state_dict_json = json.dumps(
                    {
                        "__class__": f"{service.__class__.__module__}.{service.__class__.__name__}",
                        "__unserializable__": True,
                        "__reason__": f"Cannot serialize: {e}",
                    }
                )

            self.state_provider.register_service(
                service.id,
                experiment_id,
                run_id,
                service.description(),
                state_dict=state_dict_json,
            )
        except Exception as e:
            logger.exception("Error updating service %s: %s", service.id, e)


# Backward compatibility alias
WorkspaceStateProvider = DbStateProvider
