# flake8: noqa: F401

from pathlib import Path

# Annotations
from .annotations import (
    config,
    task,
    param,
    subparam,
    ConstantParam,
    constant,
    option,
    pathoption,
    cache,
    Identifier,
    Array,
    tag,
    tags,
    tagspath,
    STDOUT,
    STDERR,
    deprecate,
    # deprecated
    argument,
    initializer,
)
from .core.arguments import (
    # Types
    Param,
    SubParam,
    Option,
    Meta,
    Annotated,
    Constant,
    # Annotations helpers
    help,
    default,
)
from .generators import pathgenerator
from .core.objects import (
    Config,
    copyconfig,
    setmeta,
    Task,
    SerializedConfig,
    Serialized,
)
from .scheduler.environment import Environment
from .scheduler.workspace import Workspace
from .scheduler import Scheduler, experiment, FailedExperiment
from .notifications import progress, tqdm
from .core.types import Any
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation


def set_launcher(launcher):
    Workspace.CURRENT.launcher = launcher


# Get version
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "?"
