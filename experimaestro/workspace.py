from pathlib import Path


class Workspace:
    """A workspace"""

    CURRENT = None

    """Creates a workspace for experiments"""

    def __init__(self, path: Path):
        if isinstance(path, Path):
            path = path.absolute()
        self.path = path
        from .launchers import Launcher

        self.launcher = Launcher.get(path)

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
