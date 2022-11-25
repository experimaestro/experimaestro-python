"""Utilities exposed to users of the experimaestro API"""

from pathlib import Path
from experimaestro.core.objects import GenerationContext


class DirectoryContext(GenerationContext):
    """Special generation context used for debugging and testing"""

    def __init__(self, path: Path):
        super().__init__()
        self._path = Path(path)

    @property
    def path(self):
        return self._path


class EmptyContext(GenerationContext):
    """Special generation context used for debugging and testing"""

    @property
    def path(self):
        raise AssertionError("Empty experimaestro context does not define a path")
