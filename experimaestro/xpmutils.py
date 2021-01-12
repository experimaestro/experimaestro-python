"""Utilities exposed to users of the experimaestro API"""

from pathlib import Path
from experimaestro.core.objects import GenerationContext


class DirectoryContext(GenerationContext):
    def __init__(self, path: Path):
        self._path = Path(path)

    @property
    def path(self):
        return self._path
