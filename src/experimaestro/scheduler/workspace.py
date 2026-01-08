from collections import ChainMap
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional
from experimaestro.settings import WorkspaceSettings, Settings


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
