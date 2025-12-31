"""Unified workspace state provider for accessing experiment and job information

This module provides a single WorkspaceStateProvider class that accesses state
from the workspace-level database (.experimaestro/workspace.db). This replaces
the previous multi-provider architecture with a unified approach.

Key features:
- Single .experimaestro/workspace.db database shared across all experiments
- Support for multiple runs per experiment
- Run-scoped tags (fixes GH #128)
- Thread-safe database access via thread-local connections
- Real-time updates via scheduler listener interface
- Push notifications via listener callbacks (for reactive UI)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING

from watchdog.events import FileSystemEventHandler
from watchdog.observers.api import ObservedWatch

from experimaestro.scheduler.state_db import (
    ExperimentModel,
    ExperimentRunModel,
    JobModel,
    JobTagModel,
    ServiceModel,
    PartialModel,
    JobPartialModel,
    ALL_MODELS,
)
from experimaestro.scheduler.interfaces import (
    BaseJob,
    BaseExperiment,
    JobState,
    STATE_NAME_TO_JOBSTATE,
)

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.scheduler.services import Service

logger = logging.getLogger("xpm.state")


# Event types for state provider notifications
class StateEventType(Enum):
    """Types of state change events"""

    EXPERIMENT_UPDATED = auto()
    RUN_UPDATED = auto()
    JOB_UPDATED = auto()


@dataclass
class StateEvent:
    """Base class for state change events

    Attributes:
        event_type: Type of the event
        data: Event-specific data dictionary
    """

    event_type: StateEventType
    data: Dict


# Type alias for listener callbacks
StateListener = Callable[[StateEvent], None]


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

    def __init__(self, state_provider: "WorkspaceStateProvider"):
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
                    self.state_provider._notify_listeners(
                        StateEvent(
                            event_type=StateEventType.EXPERIMENT_UPDATED,
                            data={
                                "experiment_id": exp.experiment_id,
                            },
                        )
                    )

            # Query for changed jobs
            with self.state_provider.workspace_db.bind_ctx([JobModel]):
                query = JobModel.select()
                if since:
                    query = query.where(JobModel.updated_at > since)

                for job in query:
                    self.state_provider._notify_listeners(
                        StateEvent(
                            event_type=StateEventType.JOB_UPDATED,
                            data={
                                "jobId": job.job_id,
                                "experimentId": job.experiment_id,
                                "runId": job.run_id,
                                "status": job.state,
                            },
                        )
                    )

        except Exception as e:
            logger.warning("Error detecting database changes: %s", e)


class _DatabaseFileHandler(FileSystemEventHandler):
    """Watchdog handler for SQLite database file changes

    Simply signals the change detector when database files are modified.
    Does not block - all processing happens in the detector thread.
    """

    def __init__(self, change_detector: _DatabaseChangeDetector):
        super().__init__()
        self.change_detector = change_detector

    def on_any_event(self, event) -> None:
        """Handle all file system events"""
        # Only handle modification-like events
        if event.event_type not in ("modified", "created", "moved"):
            return

        if event.is_directory:
            return

        # Only react to database files
        path = Path(event.src_path)
        if path.name not in ("workspace.db", "workspace.db-wal"):
            return

        logger.debug(
            "Database file changed: %s (event: %s)", path.name, event.event_type
        )

        # Signal the detector thread (non-blocking)
        self.change_detector.signal_change()


class MockJob(BaseJob):
    """Concrete implementation of BaseJob for database-loaded jobs

    This class is used when loading job information from the database,
    as opposed to live Job instances which are created during experiment runs.
    """

    def __init__(
        self,
        identifier: str,
        task_id: str,
        locator: str,
        path: Path,
        state: str,  # State name string from DB
        submittime: Optional[float],
        starttime: Optional[float],
        endtime: Optional[float],
        progress: List[Dict],
        tags: Dict[str, str],
        experiment_id: str,
        run_id: str,
        updated_at: str,
        exit_code: Optional[int] = None,
        retry_count: int = 0,
    ):
        self.identifier = identifier
        self.task_id = task_id
        self.locator = locator
        self.path = path
        # Convert state name to JobState instance
        self.state = STATE_NAME_TO_JOBSTATE.get(state, JobState.UNSCHEDULED)
        self.submittime = submittime
        self.starttime = starttime
        self.endtime = endtime
        self.progress = progress
        self.tags = tags
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.updated_at = updated_at
        self.exit_code = exit_code
        self.retry_count = retry_count

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
            import json

            with metadata_path.open("r") as f:
                metadata = json.load(f)

            return cls(
                identifier=metadata.get("job_id", path.name),
                task_id=metadata.get(
                    "task_id", path.parent.name if path.parent else "unknown"
                ),
                locator=metadata.get("job_id", path.name),
                path=path,
                state=metadata.get("state", "unscheduled"),
                submittime=metadata.get("submitted_time"),
                starttime=metadata.get("started_time"),
                endtime=metadata.get("ended_time"),
                progress=[],  # Progress not stored in metadata
                tags={},  # Tags come from jobs.jsonl, not metadata
                experiment_id="",  # Not stored in job metadata
                run_id="",  # Not stored in job metadata
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
    ):
        self.workdir = workdir
        self.current_run_id = current_run_id
        self.total_jobs = total_jobs
        self.finished_jobs = finished_jobs
        self.failed_jobs = failed_jobs
        self.updated_at = updated_at
        self.started_at = started_at
        self.ended_at = ended_at

    @property
    def experiment_id(self) -> str:
        """Experiment identifier derived from workdir name"""
        return self.workdir.name


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


class WorkspaceStateProvider:
    """Unified state provider for workspace-level database (singleton per workspace path)

    Provides access to experiment and job state from a single workspace database.
    Supports both read-only (monitoring) and read-write (scheduler) modes.

    Only one WorkspaceStateProvider instance exists per workspace path. Subsequent
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
    _instances: Dict[Path, "WorkspaceStateProvider"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        workspace_path: Path,
        read_only: bool = False,
        sync_on_start: bool = False,
        sync_interval_minutes: int = 5,
    ) -> "WorkspaceStateProvider":
        """Get or create WorkspaceStateProvider instance for a workspace path

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)

        Returns:
            WorkspaceStateProvider instance (singleton per path)
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
                        f"WorkspaceStateProvider for {workspace_path} already exists "
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
    ):
        """Initialize workspace state provider (called by get_instance())

        Args:
            workspace_path: Root workspace directory
            read_only: If True, database is in read-only mode
            sync_on_start: If True, sync from disk on initialization
            sync_interval_minutes: Minimum interval between syncs (default: 5)
        """
        # Normalize path
        if isinstance(workspace_path, Path):
            workspace_path = workspace_path.absolute()
        else:
            workspace_path = Path(workspace_path).absolute()

        self.workspace_path = workspace_path
        self.read_only = read_only
        self.sync_interval_minutes = sync_interval_minutes

        # Listeners for push notifications
        self._listeners: Set[StateListener] = set()
        self._listeners_lock = threading.Lock()

        # File watcher for database changes (started when listeners are added)
        self._change_detector: Optional[_DatabaseChangeDetector] = None
        self._db_file_handler: Optional[_DatabaseFileHandler] = None
        self._db_file_watch: Optional[ObservedWatch] = None

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
                        f"Invalid workspace version file at {version_file}: "
                        f"expected integer, got '{content}'"
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
        self.workspace_db = initialize_workspace_database(db_path, read_only=read_only)
        self._db_dir = experimaestro_dir  # Store for file watcher

        # Optionally sync from disk on start (only in write mode)
        # Syncing requires write access to update the database and sync timestamp
        if sync_on_start and not read_only:
            from .state_sync import sync_workspace_from_disk

            sync_workspace_from_disk(
                self.workspace_path,
                write_mode=True,
                force=False,
                sync_interval_minutes=sync_interval_minutes,
            )

        logger.info(
            "WorkspaceStateProvider initialized (read_only=%s, workspace=%s)",
            read_only,
            workspace_path,
        )

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
        exp_path = str(self.workspace_path / "xp" / experiment_id)
        self._notify_listeners(
            StateEvent(
                event_type=StateEventType.EXPERIMENT_UPDATED,
                data={
                    "experiment_id": experiment_id,
                    "workdir_path": exp_path,
                    "updated_at": now.isoformat(),
                },
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
        if run_id is None:
            now = datetime.now()
            run_id = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"

        # Create run record
        ExperimentRunModel.insert(
            experiment_id=experiment_id,
            run_id=run_id,
            started_at=datetime.now(),
            status="active",
        ).execute()

        # Update experiment's current_run_id and updated_at
        now = datetime.now()
        ExperimentModel.update(
            current_run_id=run_id,
            updated_at=now,
        ).where(ExperimentModel.experiment_id == experiment_id).execute()

        logger.info("Created run %s for experiment %s", run_id, experiment_id)

        # Notify listeners
        self._notify_listeners(
            StateEvent(
                event_type=StateEventType.RUN_UPDATED,
                data={
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "status": "active",
                    "started_at": now.isoformat(),
                },
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

            if exp_model.current_run_id:
                total_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                    )
                    .count()
                )
                finished_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                        & (JobModel.state == "done")
                    )
                    .count()
                )
                failed_jobs = (
                    JobModel.select()
                    .where(
                        (JobModel.experiment_id == exp_model.experiment_id)
                        & (JobModel.run_id == exp_model.current_run_id)
                        & (JobModel.state == "error")
                    )
                    .count()
                )

                # Get run timestamps
                try:
                    run_model = ExperimentRunModel.get(
                        (ExperimentRunModel.experiment_id == exp_model.experiment_id)
                        & (ExperimentRunModel.run_id == exp_model.current_run_id)
                    )
                    if run_model.started_at:
                        started_at = run_model.started_at.timestamp()
                    if run_model.ended_at:
                        ended_at = run_model.ended_at.timestamp()
                except ExperimentRunModel.DoesNotExist:
                    pass

            # Compute experiment path from workspace_path and experiment_id
            exp_path = self.workspace_path / "xp" / exp_model.experiment_id

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

        if exp_model.current_run_id:
            total_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                )
                .count()
            )
            finished_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                    & (JobModel.state == "done")
                )
                .count()
            )
            failed_jobs = (
                JobModel.select()
                .where(
                    (JobModel.experiment_id == exp_model.experiment_id)
                    & (JobModel.run_id == exp_model.current_run_id)
                    & (JobModel.state == "error")
                )
                .count()
            )

        # Compute experiment path from workspace_path and experiment_id
        exp_path = self.workspace_path / "xp" / exp_model.experiment_id

        return MockExperiment(
            workdir=exp_path,
            current_run_id=exp_model.current_run_id,
            total_jobs=total_jobs,
            finished_jobs=finished_jobs,
            failed_jobs=failed_jobs,
            updated_at=exp_model.updated_at.isoformat(),
        )

    @_with_db_context
    def get_experiment_runs(self, experiment_id: str) -> List[Dict]:
        """Get all runs for an experiment

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of run dictionaries with keys:
            - experiment_id: Experiment ID
            - run_id: Run ID
            - started_at: When run started
            - ended_at: When run completed (None if active)
            - status: Run status (active, completed, failed, abandoned)
        """
        runs = []
        for run_model in (
            ExperimentRunModel.select()
            .where(ExperimentRunModel.experiment_id == experiment_id)
            .order_by(ExperimentRunModel.started_at.desc())
        ):
            runs.append(
                {
                    "experiment_id": run_model.experiment_id,
                    "run_id": run_model.run_id,
                    "started_at": run_model.started_at.isoformat(),
                    "ended_at": (
                        run_model.ended_at.isoformat() if run_model.ended_at else None
                    ),
                    "status": run_model.status,
                }
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

        # Apply experiment filter
        if experiment_id is not None:
            # If experiment_id provided but not run_id, use current run
            if run_id is None:
                current_run = self.get_current_run(experiment_id)
                if current_run is None:
                    return []  # No runs exist for this experiment
                run_id = current_run

            query = query.where(
                (JobModel.experiment_id == experiment_id) & (JobModel.run_id == run_id)
            )

        # Apply task_id filter
        if task_id is not None:
            query = query.where(JobModel.task_id == task_id)

        # Apply state filter
        if state is not None:
            query = query.where(JobModel.state == state)

        # Apply tag filters
        if tags:
            for tag_key, tag_value in tags.items():
                # Join with JobTagModel for each tag filter
                query = query.join(
                    JobTagModel,
                    on=(
                        (JobTagModel.job_id == JobModel.job_id)
                        & (JobTagModel.experiment_id == JobModel.experiment_id)
                        & (JobTagModel.run_id == JobModel.run_id)
                        & (JobTagModel.tag_key == tag_key)
                        & (JobTagModel.tag_value == tag_value)
                    ),
                )

        # Execute query and convert to dictionaries
        jobs = []
        for job_model in query:
            # Get tags for this job
            job_tags = self._get_job_tags(
                job_model.job_id, job_model.experiment_id, job_model.run_id
            )

            jobs.append(self._job_model_to_dict(job_model, job_tags))

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

        try:
            job_model = JobModel.get(
                (JobModel.job_id == job_id)
                & (JobModel.experiment_id == experiment_id)
                & (JobModel.run_id == run_id)
            )
        except JobModel.DoesNotExist:
            return None

        # Get tags for this job
        job_tags = self._get_job_tags(job_id, experiment_id, run_id)

        return self._job_model_to_dict(job_model, job_tags)

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
            experiment_id=experiment_id,
            run_id=run_id,
            task_id=task_id,
            locator=job.identifier,
            state=job.state.name,
            submitted_time=job.submittime,
            updated_at=now,
        ).on_conflict(
            conflict_target=[JobModel.job_id, JobModel.experiment_id, JobModel.run_id],
            update={
                JobModel.state: job.state.name,
                JobModel.submitted_time: job.submittime,
                JobModel.updated_at: now,
            },
        ).execute()

        # Update tags (run-scoped)
        self.update_job_tags(job.identifier, experiment_id, run_id, job.tags)

        # Register partials for all declared subparameters
        subparameters = job.type._subparameters
        for name, sp in subparameters.items():
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
        job_path = str(
            self.workspace_path / "jobs" / str(job.type.identifier) / job.identifier
        )
        self._notify_listeners(
            StateEvent(
                event_type=StateEventType.JOB_UPDATED,
                data={
                    "jobId": job.identifier,
                    "taskId": str(job.type.identifier),
                    "experimentId": experiment_id,
                    "runId": run_id,
                    "status": job.state.name,
                    "path": job_path,
                    "updatedAt": now.isoformat(),
                },
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

        # Add failure reason if available
        from experimaestro.scheduler.jobs import JobStateError

        if isinstance(job.state, JobStateError) and job.state.failure_reason:
            update_data[JobModel.failure_reason] = job.state.failure_reason.name

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

        # Update the job record
        JobModel.update(update_data).where(
            (JobModel.job_id == job.identifier)
            & (JobModel.experiment_id == experiment_id)
            & (JobModel.run_id == run_id)
        ).execute()

        logger.debug(
            "Updated job state: %s -> %s (experiment=%s, run=%s)",
            job.identifier,
            job.state.name,
            experiment_id,
            run_id,
        )

        # Notify listeners
        job_path = str(
            self.workspace_path / "jobs" / str(job.type.identifier) / job.identifier
        )
        self._notify_listeners(
            StateEvent(
                event_type=StateEventType.JOB_UPDATED,
                data={
                    "jobId": job.identifier,
                    "taskId": str(job.type.identifier),
                    "experimentId": experiment_id,
                    "runId": run_id,
                    "status": job.state.name,
                    "path": job_path,
                    "updatedAt": now.isoformat(),
                },
            )
        )

    @_with_db_context
    def update_job_tags(
        self, job_id: str, experiment_id: str, run_id: str, tags_dict: Dict[str, str]
    ):
        """Update tags for a job (run-scoped - fixes GH #128)

        Deletes existing tags for this (job_id, experiment_id, run_id) combination
        and inserts new tags. This ensures that the same job in different runs can
        have different tags.

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            tags_dict: Dictionary of tag key-value pairs

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update tags in read-only mode")

        # Delete existing tags for this job/experiment/run
        JobTagModel.delete().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ).execute()

        # Insert new tags
        if tags_dict:
            tag_records = [
                {
                    "job_id": job_id,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "tag_key": key,
                    "tag_value": value,
                }
                for key, value in tags_dict.items()
            ]
            JobTagModel.insert_many(tag_records).execute()

        logger.debug(
            "Updated tags for job %s (experiment=%s, run=%s): %s",
            job_id,
            experiment_id,
            run_id,
            tags_dict,
        )

    @_with_db_context
    def delete_job(self, job_id: str, experiment_id: str, run_id: str):
        """Remove a job, its tags, and partial references

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot delete jobs in read-only mode")

        # Delete tags first (foreign key constraint)
        JobTagModel.delete().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ).execute()

        # Delete partial references
        JobPartialModel.delete().where(
            (JobPartialModel.job_id == job_id)
            & (JobPartialModel.experiment_id == experiment_id)
            & (JobPartialModel.run_id == run_id)
        ).execute()

        # Delete job
        JobModel.delete().where(
            (JobModel.job_id == job_id)
            & (JobModel.experiment_id == experiment_id)
            & (JobModel.run_id == run_id)
        ).execute()

        logger.debug(
            "Deleted job %s (experiment=%s, run=%s)", job_id, experiment_id, run_id
        )

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

        # Apply tag filters
        if tags:
            for tag_key, tag_value in tags.items():
                query = query.join(
                    JobTagModel,
                    on=(
                        (JobTagModel.job_id == JobModel.job_id)
                        & (JobTagModel.experiment_id == JobModel.experiment_id)
                        & (JobTagModel.run_id == JobModel.run_id)
                        & (JobTagModel.tag_key == tag_key)
                        & (JobTagModel.tag_value == tag_value)
                    ),
                )

        # Execute query and convert to MockJob objects
        jobs = []
        for job_model in query:
            # Get tags for this job
            job_tags = self._get_job_tags(
                job_model.job_id, job_model.experiment_id, job_model.run_id
            )
            jobs.append(self._job_model_to_dict(job_model, job_tags))

        return jobs

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
            ).where(
                (JobModel.job_id == job.identifier)
                & (JobModel.experiment_id == job.experiment_id)
                & (JobModel.run_id == job.run_id)
            ).execute()

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
                self.delete_job(job.identifier, job.experiment_id, job.run_id)

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
                self.delete_job(job.identifier, job.experiment_id, job.run_id)
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
                ).where(
                    (JobModel.job_id == job.identifier)
                    & (JobModel.experiment_id == job.experiment_id)
                    & (JobModel.run_id == job.run_id)
                ).execute()

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

        # Optionally delete experiment directory
        exp_path = self.workspace_path / "xp" / experiment_id
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
        # Get all jobs
        all_jobs = self.get_all_jobs()

        # Get all experiment IDs
        experiments = self.get_experiments()
        experiment_ids = {exp.experiment_id for exp in experiments}

        # Find jobs with no matching experiment
        orphan_jobs = [
            job for job in all_jobs if job.experiment_id not in experiment_ids
        ]

        return orphan_jobs

    # Service operations

    @_with_db_context
    def update_service(
        self,
        service_id: str,
        experiment_id: str,
        run_id: str,
        description: str,
        state: str,
    ):
        """Update service information

        Args:
            service_id: Service identifier
            experiment_id: Experiment identifier
            run_id: Run identifier
            description: Human-readable description
            state: Service state

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot update services in read-only mode")

        ServiceModel.insert(
            service_id=service_id,
            experiment_id=experiment_id,
            run_id=run_id,
            description=description,
            state=state,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ).on_conflict(
            conflict_target=[
                ServiceModel.service_id,
                ServiceModel.experiment_id,
                ServiceModel.run_id,
            ],
            update={
                ServiceModel.description: description,
                ServiceModel.state: state,
                ServiceModel.updated_at: datetime.now(),
            },
        ).execute()

        logger.debug(
            "Updated service %s (experiment=%s, run=%s)",
            service_id,
            experiment_id,
            run_id,
        )

    @_with_db_context
    def get_services(
        self, experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[Dict]:
        """Get services, optionally filtered by experiment/run

        Args:
            experiment_id: Filter by experiment (None = all)
            run_id: Filter by run (None = current run if experiment_id provided)

        Returns:
            List of service dictionaries
        """
        query = ServiceModel.select()

        if experiment_id is not None:
            # Use current run if not specified
            if run_id is None:
                run_id = self.get_current_run(experiment_id)
                if run_id is None:
                    return []

            query = query.where(
                (ServiceModel.experiment_id == experiment_id)
                & (ServiceModel.run_id == run_id)
            )

        services = []
        for service_model in query:
            services.append(
                {
                    "service_id": service_model.service_id,
                    "experiment_id": service_model.experiment_id,
                    "run_id": service_model.run_id,
                    "description": service_model.description,
                    "state": service_model.state,
                    "created_at": service_model.created_at.isoformat(),
                    "updated_at": service_model.updated_at.isoformat(),
                }
            )
        return services

    # Sync metadata methods

    @_with_db_context
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync

        Returns:
            datetime of last sync, or None if never synced
        """
        from .state_db import WorkspaceSyncMetadata

        metadata = WorkspaceSyncMetadata.get_or_none(
            WorkspaceSyncMetadata.id == "workspace"
        )
        if metadata and metadata.last_sync_time:
            return metadata.last_sync_time
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
        self, partial_id: str, task_id: str, subparameters_name: str
    ) -> None:
        """Register a partial directory (creates if not exists)

        Args:
            partial_id: Hex hash of the partial identifier
            task_id: Task class identifier
            subparameters_name: Name of the subparameters definition

        Raises:
            RuntimeError: If in read-only mode
        """
        if self.read_only:
            raise RuntimeError("Cannot register partials in read-only mode")

        PartialModel.insert(
            partial_id=partial_id,
            task_id=task_id,
            subparameters_name=subparameters_name,
            created_at=datetime.now(),
        ).on_conflict_ignore().execute()

        logger.debug(
            "Registered partial: %s (task=%s, subparams=%s)",
            partial_id,
            task_id,
            subparameters_name,
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
            List of dictionaries with partial_id, task_id, subparameters_name
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
                    "subparameters_name": partial.subparameters_name,
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
                / orphan["subparameters_name"]
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

        # Close database connection
        if hasattr(self, "workspace_db") and self.workspace_db is not None:
            from .state_db import close_workspace_database

            close_workspace_database(self.workspace_db)
            self.workspace_db = None

        # Remove from registry
        with WorkspaceStateProvider._lock:
            if self.workspace_path in WorkspaceStateProvider._instances:
                del WorkspaceStateProvider._instances[self.workspace_path]

        logger.debug("WorkspaceStateProvider closed for %s", self.workspace_path)

    # Listener methods for push notifications

    def add_listener(self, listener: StateListener) -> None:
        """Register a listener for state change notifications

        Listeners are called synchronously when state changes occur.
        For UI applications, listeners should queue updates for their
        own event loop to avoid blocking database operations.

        When the first listener is added, starts watching the database
        file for changes to enable push notifications.

        Args:
            listener: Callback function that receives StateEvent objects
        """
        with self._listeners_lock:
            was_empty = len(self._listeners) == 0
            self._listeners.add(listener)

        # Start file watcher when first listener is added
        if was_empty:
            self._start_file_watcher()

        logger.info(
            "Added state listener: %s (total: %d)", listener, len(self._listeners)
        )

    def remove_listener(self, listener: StateListener) -> None:
        """Unregister a state change listener

        When the last listener is removed, stops watching the database file.

        Args:
            listener: Previously registered callback function
        """
        with self._listeners_lock:
            self._listeners.discard(listener)
            is_empty = len(self._listeners) == 0

        # Stop file watcher when last listener is removed
        if is_empty:
            self._stop_file_watcher()

        logger.debug("Removed state listener: %s", listener)

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
        self._db_file_handler = _DatabaseFileHandler(self._change_detector)
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

    def _notify_listeners(self, event: StateEvent) -> None:
        """Notify all registered listeners of a state change

        This is called internally by state-modifying methods.
        Listeners are called synchronously - they should be fast.

        Args:
            event: State change event to broadcast
        """
        with self._listeners_lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Listener %s raised exception: %s", listener, e)

    # Helper methods

    @_with_db_context
    def _get_job_tags(
        self, job_id: str, experiment_id: str, run_id: str
    ) -> Dict[str, str]:
        """Get tags for a job

        Args:
            job_id: Job identifier
            experiment_id: Experiment identifier
            run_id: Run identifier

        Returns:
            Dictionary of tag key-value pairs
        """
        tags = {}
        for tag_model in JobTagModel.select().where(
            (JobTagModel.job_id == job_id)
            & (JobTagModel.experiment_id == experiment_id)
            & (JobTagModel.run_id == run_id)
        ):
            tags[tag_model.tag_key] = tag_model.tag_value
        return tags

    def _job_model_to_dict(self, job_model: JobModel, tags: Dict[str, str]) -> MockJob:
        """Convert a JobModel to a MockJob object

        Args:
            job_model: JobModel instance
            tags: Dictionary of tags for this job

        Returns:
            MockJob object
        """
        # Parse progress JSON
        progress_list = json.loads(job_model.progress)

        # Compute job path from workspace_path, task_id, and job_id
        job_path = self.workspace_path / "jobs" / job_model.task_id / job_model.job_id

        return MockJob(
            identifier=job_model.job_id,
            task_id=job_model.task_id,
            locator=job_model.locator,
            path=job_path,
            state=job_model.state,
            submittime=job_model.submitted_time,
            starttime=job_model.started_time,
            endtime=job_model.ended_time,
            progress=progress_list,
            tags=tags,
            experiment_id=job_model.experiment_id,
            run_id=job_model.run_id,
            updated_at=job_model.updated_at.isoformat(),
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
    """Adapter to connect scheduler events to WorkspaceStateProvider

    This class implements the scheduler listener interface and forwards
    events to the WorkspaceStateProvider. It tracks which experiment/run
    each job belongs to for proper database updates.
    """

    def __init__(self, state_provider: WorkspaceStateProvider):
        """Initialize listener

        Args:
            state_provider: WorkspaceStateProvider instance to update
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
        try:
            self.state_provider.update_service(
                service.id,
                experiment_id,
                run_id,
                service.description(),
                service.state.name,
            )
        except Exception as e:
            logger.exception("Error updating service %s: %s", service.id, e)
