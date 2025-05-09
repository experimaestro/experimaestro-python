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
    # Method
    config_only,
)
from .core.serialization import load, save, state_dict, from_state_dict, from_task_dir
from .core.arguments import (
    # Types
    Param,
    Option,
    Meta,
    DataPath,
    Annotated,
    Constant,
    field,
    # Annotations helpers
    help,
    default,
)
from .generators import pathgenerator, PathGenerator
from .core.objects import (
    Config,
    copyconfig,
    setmeta,
    DependentMarker,
    Task,
    LightweightTask,
    ObjectStore,
)
from .core.context import SerializationContext
from .core.serializers import SerializationLWTask, PathSerializationLWTask
from .core.types import Any, SubmitHook
from .launchers import Launcher
from .scheduler import Scheduler, experiment, FailedExperiment
from .scheduler.workspace import Workspace, RunMode
from .scheduler.state import get_experiment
from .notifications import progress, tqdm
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation
from .scheduler.base import Job
from .launcherfinder.registry import LauncherRegistry


def set_launcher(launcher: Launcher):
    Workspace.CURRENT.launcher = launcher


# Get version
__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)
