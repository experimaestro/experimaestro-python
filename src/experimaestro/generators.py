import inspect
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Union, TYPE_CHECKING
from experimaestro.core.arguments import ArgumentOptions, TypeAnnotation
from experimaestro.core.objects import ConfigWalkContext, Config

if TYPE_CHECKING:
    from experimaestro.core.partial import Partial


class Generator(ABC):
    """Base class for all generators"""

    def isoutput(self):
        """Returns True if this generator is a task output (e.g. generates a
        path within the job folder)"""
        return False

    @abstractmethod
    def __call__(self, context: ConfigWalkContext, config: Config): ...


class PathGenerator(Generator):
    """Generate paths within the task directory.

    Use ``PathGenerator`` with ``field(default_factory=...)`` to create
    paths relative to the task's working directory.

    Example::

        class MyTask(Task):
            output: Meta[Path] = field(default_factory=PathGenerator("results.json"))
            model: Meta[Path] = field(default_factory=PathGenerator("model.pt"))

    For shared directories across related tasks, use with partial::

        training_group = param_group("training")

        class Train(Task):
            epochs: Param[int] = field(groups=[training_group])
            checkpoint: Meta[Path] = field(
                default_factory=PathGenerator(
                    "model.pt",
                    partial=partial(exclude=[training_group])
                )
            )

    :param path: Relative path within the task directory. Can be a string,
        Path, or callable that takes (context, config) and returns a Path.
    :param partial: Optional partial for partial directory sharing.
        When provided, the path is generated in a shared partial directory.
    """

    def __init__(
        self,
        path: Union[str, Path, Callable[[ConfigWalkContext, Config], Path]] = "",
        *,
        partial: "Partial" = None,
    ):
        self.path = path
        self.partial = partial

    def __call__(self, context: ConfigWalkContext, config: Config):
        # Determine base path: partial directory or job directory
        if self.partial is not None:
            base_path = context.partial_path(self.partial, config)
        else:
            base_path = context.currentpath()

        # Generate the final path
        if inspect.isfunction(self.path):
            return base_path / self.path(context, config)
        elif self.path:
            return base_path / Path(self.path)
        else:
            return base_path

    def isoutput(self):
        return True


class pathgenerator(TypeAnnotation):
    def __init__(self, value: Union[str, Callable[[ConfigWalkContext, Config], str]]):
        self.value = value

    def annotate(self, options: ArgumentOptions):
        options.kwargs["generator"] = PathGenerator(self.value)
