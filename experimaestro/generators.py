import inspect
from pathlib import Path
from typing import Callable, List, Tuple, Union
from experimaestro.core.arguments import ArgumentOptions, TypeAnnotation
from experimaestro.core.objects import GenerationContext, Config


class Generator:
    """Base class for all generators"""

    def isoutput(self):
        """Returns True if this generator is a task output (e.g. generates a path within the job folder)"""
        return False


class PathGenerator(Generator):
    """Generates a path"""

    def __init__(
        self, path: Union[str, Path, Callable[[GenerationContext, Config], Path]]
    ):
        self.path = path

    def __call__(self, context: GenerationContext, config: Config):
        if inspect.isfunction(self.path):
            path = context.currentpath() / self.path(context, config)  # type: Path
        else:
            path = context.currentpath() / Path(self.path)

        return path

    def isoutput(self):
        return True


class pathgenerator(TypeAnnotation):
    def __init__(self, value: Union[str, Callable[[GenerationContext, Config], str]]):
        self.value = value

    def annotate(self, options: ArgumentOptions):
        options.kwargs["generator"] = PathGenerator(self.value)
