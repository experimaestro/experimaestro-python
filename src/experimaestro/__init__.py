# flake8: noqa: F401

from pathlib import Path

# Annotations
from .annotations import (
    cache,
    Array,
    TagDict,
    tag,
    tags,
    tagspath,
    STDOUT,
    STDERR,
    deprecate,
    initializer,
    # Method
    config_only,
)
from .core.types import Identifier
from .core.serialization import (
    load,
    save,
    state_dict,
    from_state_dict,
    from_task_dir,
    serialize,
    deserialize,
)
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
)
from .generators import pathgenerator, PathGenerator
from .core.partial import (
    partial,
    param_group,
    ParameterGroup,
    Partial,
)
from .core.objects import (
    Config,
    InstanceConfig,
    copyconfig,
    setmeta,
    DependentMarker,
    Task,
    ResumableTask,
    LightweightTask,
    ObjectStore,
)
from .core.context import SerializationContext
from .core.serializers import SerializationLWTask, PathSerializationLWTask
from .core.types import Any, SubmitHook
from .launchers import Launcher
from .scheduler import (
    Scheduler,
    experiment,
    FailedExperiment,
    DirtyGitError,
    GracefulExperimentExit,
)
from .exceptions import GracefulTimeout
from .scheduler.workspace import Workspace, RunMode
from .scheduler.transient import TransientMode
from .notifications import progress, tqdm
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation
from .scheduler.base import Job
from .launcherfinder.registry import LauncherRegistry
from .experiments.configuration import DirtyGitAction


def set_launcher(launcher: Launcher):
    Workspace.CURRENT.launcher = launcher


# Get version
__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)
