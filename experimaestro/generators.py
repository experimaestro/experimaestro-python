import inspect
from pathlib import Path
from typing import Callable, Union

from experimaestro.core.objects import GenerationContext


class PathGenerator:
    """Generates a path"""

    def __init__(self, path: Union[str, Path, Callable[[GenerationContext], Path]]):
        self.path = path

    def __call__(self, context: GenerationContext):
        if inspect.isfunction(self.path):
            return context.path / self.path(context)

        return context.path / self.path
