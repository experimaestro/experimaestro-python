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
            path = context.path / self.path(context)  # type: Path
        else:
            path = context.path / self.path  # type: Path

        if not context.registerpath(path):
            i = 0
            while True:
                i += 1
                newpath = path.with_suffix(f".{i}{path.suffix}")
                if context.registerpath(newpath):
                    break

            path = newpath

        return path
