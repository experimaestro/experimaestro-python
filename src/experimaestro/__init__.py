# Annotations
from pathlib import Path
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
    Param,
    Option,
    Meta,
    DataPath,
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
from .exceptions import GracefulTimeout, TaskCancelled
from .scheduler.workspace import Workspace, RunMode
from .scheduler.transient import TransientMode
from .notifications import progress, tqdm
from .checkers import Choices
from .xpmutils import DirectoryContext
from .mkdocs.annotations import documentation
from .scheduler.base import Job
from .launcherfinder.registry import LauncherRegistry
from .experiments.configuration import DirtyGitAction

# Re-export set_launcher for backwards compatibility
set_launcher = Workspace.set_launcher

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
    "Annotated",
    "Constant",
    "field",
    "help",
    "Any",
    "SubmitHook",
    # Serialization
    "load",
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
    "DependentMarker",
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
    # Avoid breaking old code by re-exporting pathlib
    "Path",
]
