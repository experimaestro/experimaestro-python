# flake8: noqa: F401

from pathlib import Path

# Annotations
from .annotations import (
    config,
    task,
    param,
    ConstantParam,
    constant,
    option,
    pathoption,
    cache,
    Identifier,
    Array,
    TagDict,
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
    Option,
    Meta,
    DataPath,
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
    LightweightTask,
)
from .core.serializers import SerializationLWTask, PathSerializationLWTask
from .core.types import Any, SubmitHook
from .launchers import Launcher
from .scheduler.environment import Environment
from .scheduler.workspace import Workspace, RunMode
from .scheduler import Scheduler, experiment, FailedExperiment
from .notifications import progress, tqdm
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation
from .scheduler.base import Job


def set_launcher(launcher: Launcher):
    Workspace.CURRENT.launcher = launcher


# Get version
try:
    from .version import __version__, __version_tuple__
except:
    __version__ = "?"
    __version_tuple__ = (0, 0, 0, "", "")
