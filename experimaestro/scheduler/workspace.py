from pathlib import Path
from typing import Optional

from experimaestro.scheduler.environment import Environment


class Workspace:
    """An experimental workspace

    This workspace is created by an experiment object and is used by launchers
    to set up jobs
    """

    CURRENT = None
    environment: Environment

    """Creates a workspace for experiments"""

    def __init__(self, environment: Environment, launcher=None):
        self.environment = environment
        path = environment.workdir
        self.notificationURL: Optional[str] = None
        if isinstance(path, Path):
            path = path.absolute()
        self.path = path
        from ..launchers import Launcher

        self.launcher = launcher or Launcher.get(path)

    def __enter__(self):
        self.old_workspace = Workspace.CURRENT
        Workspace.CURRENT = self

    def __exit__(self, *args):
        Workspace.CURRENT = self.old_workspace

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
