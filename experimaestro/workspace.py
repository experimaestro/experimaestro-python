import os
from pathlib import Path
from .connectors import Launcher

class Experiment():
    """Represents an experiment"""


    CURRENT = None

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.old_experiment = Experiment.CURRENT
        Experiment.CURRENT = self


    def __exit__(self, *args):
        # Wait until all tasks are completed
        Experiment.CURRENT = self.old_experiment


class Workspace():
    """A workspace
    """
    CURRENT = None

    """Creates a workspace for experiments"""
    def __init__(self, path: Path):
        if isinstance(path, Path):
            path = path.absolute()
        self.path = path
        self.launcher = Launcher.get(path)


    def __enter__(self):
        self.old_workspace = Workspace.CURRENT
        Workspace.CURRENT = self


    def __exit__(self, *args):
        Workspace.CURRENT = self.old_workspace

    @property
    def jobspath(self):
        """Folder for jobs"""
        return self.path / "jobs"

class experiment:
    """Experiment environment"""
    def __init__(self, path: Path, name: str, *, port:int=None):
        
        self.workspace = Workspace(path)
        self.experiment = Experiment(name)
        self.port = port

    def __enter__(self):
        self.workspace.__enter__()
        self.experiment.__enter__()
        return self.workspace

    def __exit__(self, *args):
        self.experiment.__exit__()
        self.workspace.__exit__()
