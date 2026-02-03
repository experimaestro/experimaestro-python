from collections import ChainMap
from enum import Enum
from functools import cached_property
import logging
from pathlib import Path
from typing import Optional
from experimaestro.settings import WorkspaceSettings, Settings

logger = logging.getLogger(__name__)

# Current workspace version
WORKSPACE_VERSION = 0

# Default service log retention period
DEFAULT_SERVICE_LOG_RETENTION_DAYS = 15


class RunMode(str, Enum):
    NORMAL = "normal"
    """Normal run"""

    GENERATE_ONLY = "generate"
    """Do not run, but generate the params.json file"""

    DRY_RUN = "dry-run"
    """Do not run"""


class Workspace:
    """Workspace environment for experiments

    This is a simple container for workspace settings, environment, and configuration.
    Multiple Workspace instances can exist for the same path - the singleton pattern
    is handled by WorkspaceStateProvider which manages the database per workspace path.
    """

    CURRENT = None
    settings: "Settings"
    workspace_settings: "WorkspaceSettings"

    def __init__(
        self,
        settings: "Settings",
        workspace_settings: "WorkspaceSettings",
        launcher=None,
        run_mode: RunMode = None,
    ):
        """Initialize workspace environment

        Args:
            settings: Global settings
            workspace_settings: Workspace-specific settings
            launcher: Default launcher for this workspace
            run_mode: Run mode for experiments in this workspace
        """
        self.settings = settings
        self.workspace_settings = workspace_settings

        path = self.workspace_settings.path
        self.notificationURL: Optional[str] = None
        if isinstance(path, Path):
            path = path.absolute()
        else:
            path = Path(path).absolute()
        self.path = path
        self.run_mode = run_mode
        self.python_path = []

        from ..launchers import Launcher

        self.launcher = launcher or Launcher.get(path)

        self.env = ChainMap({}, workspace_settings.env, settings.env)

        # Reference counting for nested context managers
        self._ref_count = 0

        # Scheduler run ID (set when first used)
        self._scheduler_run_id: Optional[str] = None

    def __enter__(self):
        # Increment reference count
        self._ref_count += 1

        # Only initialize on first entry
        if self._ref_count == 1:
            # Check if a different workspace is already active
            if Workspace.CURRENT is not None and Workspace.CURRENT.path != self.path:
                raise RuntimeError(
                    f"Cannot activate workspace at {self.path} - "
                    f"workspace at {Workspace.CURRENT.path} is already active. "
                    "Multiple workspaces are not yet supported."
                )

            self.old_workspace = Workspace.CURRENT
            Workspace.CURRENT = self

        return self

    def __exit__(self, *args):
        # Decrement reference count
        self._ref_count -= 1

        # Only cleanup on last exit
        if self._ref_count == 0:
            Workspace.CURRENT = self.old_workspace

    @cached_property
    def alt_workspaces(self):
        for ws_id in self.workspace_settings.alt_workspaces:
            yield self.settings.workspaces[ws_id]

    @property
    def alt_workdirs(self):
        yield from map(lambda ws: ws.path, self.workspace_settings.alt_workspaces)

    @property
    def connector(self):
        """Returns the default connector"""
        return self.launcher.connector

    @property
    def jobspath(self):
        """Folder for jobs"""
        return self.path / "jobs"

    @property
    def partialspath(self):
        """Folder for partial job directories (shared checkpoints, etc.)"""
        return self.path / "partials"

    @property
    def experimentspath(self):
        """Folder for experiments"""
        return self.path / "experiments"

    @property
    def configcachepath(self):
        """Folder for jobs"""
        return self.path / "config"

    def _ensure_scheduler_run_id(self) -> str:
        """Ensure scheduler run ID is set and return it

        Creates a timestamped folder under .scheduler/ for this run.
        If timestamp conflicts, adds suffix .1, .2, etc.

        Returns:
            The scheduler run ID (timestamp string)
        """
        if self._scheduler_run_id is not None:
            return self._scheduler_run_id

        from datetime import datetime

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.path / ".scheduler"
        base_path.mkdir(parents=True, exist_ok=True)

        # Check for conflicts and add suffix if needed
        run_path = base_path / timestamp
        if run_path.exists():
            ix = 1
            while True:
                run_path = base_path / f"{timestamp}.{ix}"
                if not run_path.exists():
                    break
                ix += 1
            self._scheduler_run_id = f"{timestamp}.{ix}"
        else:
            self._scheduler_run_id = timestamp

        # Create the run directory
        run_path.mkdir(parents=True, exist_ok=True)

        return self._scheduler_run_id

    @property
    def schedulerpath(self) -> Path:
        """Folder for scheduler metadata (base .scheduler directory)"""
        return self.path / ".scheduler"

    @property
    def scheduler_run_path(self) -> Path:
        """Folder for this run's scheduler data"""
        run_id = self._ensure_scheduler_run_id()
        return self.schedulerpath / run_id

    @property
    def scheduler_services_path(self) -> Path:
        """Folder for service logs"""
        return self.scheduler_run_path / "services"

    @property
    def scheduler_lock_path(self) -> Path:
        """Lock file for scheduler directory operations"""
        return self.scheduler_run_path / ".lock"

    @classmethod
    def set_launcher(cls, launcher) -> None:
        """Set the launcher for the current workspace

        Args:
            launcher: The launcher to use for task execution
        """
        if cls.CURRENT is None:
            raise RuntimeError("No active workspace - use within an experiment context")
        cls.CURRENT.launcher = launcher

    def cleanup_old_scheduler_runs(
        self,
        retention_days: int = DEFAULT_SERVICE_LOG_RETENTION_DAYS,
        force: bool = False,
    ) -> tuple[int, int]:
        """Clean up old scheduler run directories

        Args:
            retention_days: Delete run directories older than this many days
            force: Force cleanup even if recently run

        Returns:
            Tuple of (directories_deleted, errors_count)
        """
        import shutil
        import time
        from experimaestro.locking import create_file_lock

        scheduler_base = self.schedulerpath
        if not scheduler_base.exists():
            return 0, 0

        # Use lock file in the current run directory to prevent concurrent cleanup
        lock_file = self.scheduler_lock_path
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock = create_file_lock(lock_file, timeout=0)

        try:
            lock.acquire()
        except Exception:
            logger.debug("Cleanup skipped: another process holds the lock")
            return 0, 0

        try:
            # Check last cleanup time to avoid frequent runs
            last_cleanup_file = scheduler_base / ".last_cleanup"
            if not force and last_cleanup_file.exists():
                last_cleanup = last_cleanup_file.stat().st_mtime
                hours_since = (time.time() - last_cleanup) / 3600
                if hours_since < 24:  # Run at most once per day
                    logger.debug(f"Cleanup skipped: last run {hours_since:.1f}h ago")
                    return 0, 0

            # Find run directories older than retention period
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            deleted = 0
            errors = 0

            for run_dir in scheduler_base.iterdir():
                if not run_dir.is_dir():
                    continue
                if run_dir.name.startswith("."):
                    continue  # Skip .last_cleanup and other hidden files

                # Skip current run directory
                if self._scheduler_run_id and run_dir.name == self._scheduler_run_id:
                    continue

                # Check directory modification time
                mtime = run_dir.stat().st_mtime
                if mtime < cutoff_time:
                    try:
                        shutil.rmtree(run_dir)
                        logger.info(f"Deleted old scheduler run: {run_dir}")
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {run_dir}: {e}")
                        errors += 1

            # Update last cleanup timestamp
            last_cleanup_file.touch()

            return deleted, errors

        finally:
            lock.release()
