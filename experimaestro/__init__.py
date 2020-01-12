from pathlib import Path

# Annotations
from .annotations import config, task, argument, pathargument, ConstantArgument, Identifier, Array, tag
from .workspace import Workspace
from .scheduler import Scheduler, experiment
from .notifications import progress
from .register import parse_commandline
from .api import Any

def set_launcher(launcher):
    Workspace.CURRENT.launcher = launcher
