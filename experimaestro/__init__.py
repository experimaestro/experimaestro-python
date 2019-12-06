from pathlib import Path

# Annotations
from .annotations import Type, Task, Argument, PathArgument, ConstantArgument, Typename, Array
from .workspace import Workspace, experiment
from .connectors import Launcher

# Deprecated Annotation
from .annotations import RegisterType, RegisterTask, TypeArgument


def set_launcher(launcher):
    Workspace.CURRENT.launcher = launcher

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

