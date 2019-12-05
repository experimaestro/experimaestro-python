from pathlib import Path

# Annotations
from .annotations import Type, Task, Argument, Typename, Array, Launcher, Workspace

# Deprecated Annotation
from .annotations import RegisterType, RegisterTask, TypeArgument

class experiment:
    """Experiment environment"""
    def __init__(self, path: Path, name: str, port:int=None):
        self.old_workspace = Workspace.CURRENT
        
        if isinstance(path, Path):
            path = path.absolute()
        self.path = path
        self.name = name
        self.port = port

    def __enter__(self):
        workspace = Workspace(self.path)
        Workspace.setcurrent(workspace)
        workspace.experiment(self.name)
        workspace.launcher = DirectLauncher(LocalConnector())
        if os.getenv("PYTHONPATH"):
            workspace.launcher.setenv("PYTHONPATH", os.getenv("PYTHONPATH"))
        return workspace

    def __exit__(self, *args):
        Workspace.setcurrent(self.old_workspace)

def set_launcher(launcher):
    Workspace.launcher = launcher


def parse_commandline():
    register.parse()

# Signal handling

import atexit
import signal

EXIT_MODE = False

def handleKill():
    EXIT_MODE = True
    logger.warn("Received SIGINT or SIGTERM")

signal.signal(signal.SIGINT, handleKill)
signal.signal(signal.SIGTERM, handleKill)
signal.signal(signal.SIGQUIT, handleKill)

