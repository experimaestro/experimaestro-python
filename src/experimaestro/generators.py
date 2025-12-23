import inspect
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Union, TYPE_CHECKING
from experimaestro.core.arguments import ArgumentOptions, TypeAnnotation
from experimaestro.core.objects import ConfigWalkContext, Config

if TYPE_CHECKING:
    from experimaestro.core.subparameters import Subparameters


class Generator(ABC):
    """Base class for all generators"""

    def isoutput(self):
        """Returns True if this generator is a task output (e.g. generates a
        path within the job folder)"""
        return False

    @abstractmethod
    def __call__(self, context: ConfigWalkContext, config: Config): ...


class PathGenerator(Generator):
    """Generates a path

    Args:
        path: The relative path to generate. Can be a string, Path, or callable.
        partial: If provided, the path will be generated in a partial directory
            that is shared across configurations that differ only in the
            parameters excluded by this Subparameters instance.
    """

    def __init__(
        self,
        path: Union[str, Path, Callable[[ConfigWalkContext, Config], Path]] = "",
        *,
        partial: "Subparameters" = None,
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
