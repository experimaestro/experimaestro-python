import inspect
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Union
from experimaestro.core.arguments import ArgumentOptions, TypeAnnotation
from experimaestro.core.objects import ConfigWalkContext, Config


class Generator(ABC):
    """Base class for all generators"""

    def isoutput(self):
        """Returns True if this generator is a task output (e.g. generates a
        path within the job folder)"""
        return False

    @abstractmethod
    def __call__(self, context: ConfigWalkContext, config: Config):
        ...


class PathGenerator(Generator):
    """Generates a path"""

    def __init__(
        self, path: Union[str, Path, Callable[[ConfigWalkContext, Config], Path]]
    ):
        self.path = path

    def __call__(self, context: ConfigWalkContext, config: Config):
        if inspect.isfunction(self.path):
            path = context.currentpath() / self.path(context, config)  # type: Path
        else:
            path = context.currentpath() / Path(self.path)

        return path

    def isoutput(self):
        return True


class pathgenerator(TypeAnnotation):
    def __init__(self, value: Union[str, Callable[[ConfigWalkContext, Config], str]]):
        self.value = value

    def annotate(self, options: ArgumentOptions):
        options.kwargs["generator"] = PathGenerator(self.value)
