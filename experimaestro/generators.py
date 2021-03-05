import inspect
from pathlib import Path
from typing import Callable, Union
from experimaestro.core.arguments import ArgumentOptions, TypeAnnotation
from experimaestro.core.objects import GenerationContext


class PathGenerator:
    """Generates a path"""

    def __init__(self, path: Union[str, Path, Callable[[GenerationContext], Path]]):
        self.path = path

    def __call__(self, context: GenerationContext):
        if inspect.isfunction(self.path):
            path = context.currentpath() / self.path(context)  # type: Path
        else:
            path = context.currentpath() / Path(self.path)

        return path


class pathgenerator(TypeAnnotation):
    def __init__(self, value):
        self.value = value

    def annotate(self, options: ArgumentOptions):
        options.kwargs["generator"] = PathGenerator(self.value)
