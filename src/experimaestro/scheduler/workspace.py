from collections import ChainMap
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional
from experimaestro.settings import WorkspaceSettings, Settings


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
    to set up jobs
    """

    CURRENT = None
    settings: "Settings"
    worspace: "WorkspaceSettings"

    """Creates a workspace for experiments"""

    def __init__(
        self,
        settings: "Settings",
        workspace_settings: "WorkspaceSettings",
        launcher=None,
        run_mode: RunMode = None,
    ):
        self.settings = settings
        self.workspace_settings = workspace_settings

        path = self.workspace_settings.path
        self.notificationURL: Optional[str] = None
        if isinstance(path, Path):
            path = path.absolute()
        self.path = path
        self.run_mode = run_mode
        self.python_path = []
        from ..launchers import Launcher

        self.launcher = launcher or Launcher.get(path)

        self.env = ChainMap({}, workspace_settings.env, settings.env)

    def __enter__(self):
        self.old_workspace = Workspace.CURRENT
        Workspace.CURRENT = self

    def __exit__(self, *args):
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
