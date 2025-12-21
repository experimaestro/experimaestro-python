from collections import ChainMap
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Dict
import threading
from experimaestro.settings import WorkspaceSettings, Settings
from .state_db import initialize_workspace_database, close_workspace_database
from .state_provider import WorkspaceStateProvider


# Current workspace version
WORKSPACE_VERSION = 0


class RunMode(str, Enum):
    NORMAL = "normal"
    """Normal run"""

    GENERATE_ONLY = "generate"
    """Do not run, but generate the params.json file"""

    DRY_RUN = "dry-run"
    """Do not run"""


class Workspace:
    """An experimental workspace

    This workspace is created by an experiment object and is used by launchers
    to set up jobs.

    Only one Workspace instance exists per path - subsequent creations with the
    same path return the existing instance.
    """

    CURRENT = None
    settings: "Settings"
    worspace: "WorkspaceSettings"

    # Registry of workspace instances by absolute path
    _instances: Dict[Path, "Workspace"] = {}
    _lock = threading.Lock()

    """Creates a workspace for experiments"""

    def __new__(
        cls,
        settings: "Settings",
        workspace_settings: "WorkspaceSettings",
        launcher=None,
        run_mode: RunMode = None,
        sync_on_init: bool = False,
        sync_interval_minutes: int = 5,
    ):
        # Get absolute path for registry lookup
        path = workspace_settings.path
        if isinstance(path, Path):
            path = path.absolute()
        else:
            path = Path(path).absolute()

        # Check if instance already exists for this path
        with cls._lock:
            if path in cls._instances:
                existing = cls._instances[path]
                # Update configuration if needed
                existing.run_mode = run_mode
                existing.sync_on_init = sync_on_init
                existing.sync_interval_minutes = sync_interval_minutes
                return existing

            # Create new instance
            instance = super().__new__(cls)
            cls._instances[path] = instance
            # Mark as uninitialized
            instance._initialized = False
            return instance

    def __init__(
        self,
        settings: "Settings",
        workspace_settings: "WorkspaceSettings",
        launcher=None,
        run_mode: RunMode = None,
        sync_on_init: bool = False,
        sync_interval_minutes: int = 5,
    ):
        # Skip initialization if already initialized (returning existing instance)
        if self._initialized:
            return

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

        # Database sync configuration
        self.sync_on_init = sync_on_init
        self.sync_interval_minutes = sync_interval_minutes
        self.workspace_db = None

        # State provider for database access
        self._state_provider = WorkspaceStateProvider(self.path, read_only=False)

        # Reference counting for nested context managers
        self._ref_count = 0

        # Mark as initialized
        self._initialized = True

    def __enter__(self):
        # Increment reference count
        self._ref_count += 1

        # Only initialize on first entry
        if self._ref_count == 1:
            # Check and update workspace version
            version_file = self.path / ".__experimaestro__"

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
            if not version_file.exists() or version_file.read_text().strip() == "":
                version_file.write_text(str(WORKSPACE_VERSION))

            # Initialize workspace database
            self.workspace_db = initialize_workspace_database(self.workspace_db_path)

            # Optionally sync from disk if requested
            # Note: sync_workspace_from_disk will be implemented in Phase 5
            if self.sync_on_init:
                # TODO: Call sync_workspace_from_disk once implemented
                # from .state_sync import sync_workspace_from_disk
                # sync_workspace_from_disk(
                #     self,
                #     write_mode=True,
                #     force=False,
                #     sync_interval_minutes=self.sync_interval_minutes
                # )
                pass

            self.old_workspace = Workspace.CURRENT
            Workspace.CURRENT = self

        return self

    def __exit__(self, *args):
        # Decrement reference count
        self._ref_count -= 1

        # Only cleanup on last exit
        if self._ref_count == 0:
            # Close workspace database connection
            if self.workspace_db is not None:
                close_workspace_database(self.workspace_db)
                self.workspace_db = None

            # Remove from registry to allow fresh instance on next creation
            with Workspace._lock:
                if self.path in Workspace._instances:
                    del Workspace._instances[self.path]
                # Mark as uninitialized so it can be reinitialized
                self._initialized = False

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
    def experimentspath(self):
        """Folder for experiments"""
        return self.path / "xp"

    @property
    def configcachepath(self):
        """Folder for jobs"""
        return self.path / "config"

    @property
    def workspace_db_path(self):
        """Path to workspace-level database"""
        return self.path / "workspace.db"

    @property
    def state_provider(self):
        """Get the workspace state provider

        Returns the WorkspaceStateProvider instance for querying and updating
        the workspace database.
        """
        return self._state_provider
