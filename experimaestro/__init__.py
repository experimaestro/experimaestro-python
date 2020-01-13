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

# Get version
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = None
