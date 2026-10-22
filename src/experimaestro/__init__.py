# PEP 810 (Python 3.15+): submodules are only loaded when a name is used
__lazy_modules__ = [
    "experimaestro.actions",
    "experimaestro.annotations",
    "experimaestro.checkers",
    "experimaestro.core.arguments",
    "experimaestro.core.context",
    "experimaestro.core.objects",
    "experimaestro.core.partial",
    "experimaestro.core.serialization",
    "experimaestro.core.serializers",
    "experimaestro.core.types",
    "experimaestro.exceptions",
    "experimaestro.experiments.configuration",
    "experimaestro.generators",
    "experimaestro.launcherfinder.registry",
    "experimaestro.launchers",
    "experimaestro.mkdocs.annotations",
    "experimaestro.notifications",
    "experimaestro.scheduler",
    "experimaestro.scheduler.base",
    "experimaestro.scheduler.transient",
    "experimaestro.scheduler.workspace",
    "experimaestro.xpmutils",
]

# Annotations
from pathlib import Path
from .annotations import (
    cache,
    Array,
    TagDict,
    tag,
    tags,
    tagspath,
    stop_tags,
    STDOUT,
    STDERR,
    deprecate,
    initializer,
    config_only,
)
from .core.types import Identifier
from .core.serialization import (
    load,
    load_configs,
    load_xp_info,
    ExperimentInfo,
    save,
    state_dict,
    from_state_dict,
    from_task_dir,
    serialize,
    deserialize,
)
from .actions import Action, Interaction, CLIInteraction
from .core.arguments import (
    Param,
    Option,
    Meta,
    DataPath,
    OptionalDataPath,
    Annotated,
    Constant,
    field,
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
    sealed_set,
    DependentMarker,
    Prepare,
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
from .exceptions import GracefulTimeout, TaskCancelled
from .scheduler.workspace import Workspace, RunMode
from .scheduler.transient import TransientMode

# `progress` is also a submodule: it is loaded first, otherwise importing it
# later would replace the lazy `progress` function by the module
from . import progress as _progress_module  # noqa: F401
from .notifications import progress, tqdm
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation
from .scheduler.base import Job
from .launcherfinder.registry import LauncherRegistry
from .experiments.configuration import DirtyGitAction


def set_launcher(launcher) -> None:
    """Re-exported for backwards compatibility, see :meth:`Workspace.set_launcher`"""
    Workspace.set_launcher(launcher)


# Get version
try:
    from .version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    # Version
    "__version__",
    # Annotations
    "cache",
    "Array",
    "TagDict",
    "tag",
    "tags",
    "tagspath",
    "stop_tags",
    "STDOUT",
    "STDERR",
    "deprecate",
    "initializer",
    "config_only",
    # Types
    "Identifier",
    "Param",
    "Option",
    "Meta",
    "DataPath",
    "OptionalDataPath",
    "Annotated",
    "Constant",
    "field",
    "help",
    "Any",
    "SubmitHook",
    # Serialization
    "load",
    "load_configs",
    "load_xp_info",
    "ExperimentInfo",
    "save",
    "state_dict",
    "from_state_dict",
    "from_task_dir",
    "serialize",
    "deserialize",
    "SerializationContext",
    "SerializationLWTask",
    "PathSerializationLWTask",
    # Generators
    "pathgenerator",
    "PathGenerator",
    # Partial
    "partial",
    "param_group",
    "ParameterGroup",
    "Partial",
    # Config/Task
    "Config",
    "InstanceConfig",
    "copyconfig",
    "setmeta",
    "sealed_set",
    "DependentMarker",
    "Prepare",
    "Task",
    "ResumableTask",
    "LightweightTask",
    "ObjectStore",
    # Scheduler
    "Scheduler",
    "experiment",
    "FailedExperiment",
    "DirtyGitError",
    "GracefulExperimentExit",
    "GracefulTimeout",
    "TaskCancelled",
    "Workspace",
    "RunMode",
    "TransientMode",
    "Job",
    "set_launcher",
    # Launchers
    "Launcher",
    "LauncherRegistry",
    # Utilities
    "progress",
    "tqdm",
    "Choices",
    "DirectoryContext",
    "documentation",
    "DirtyGitAction",
    # Actions (alpha)
    "Action",
    "Interaction",
    "CLIInteraction",
    # Avoid breaking old code by re-exporting pathlib
    "Path",
]
