from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Union
from importlib import resources
from experimaestro.compat import cached_property


class ResourcePathWrapper(PathLike):
    """Simple wrapper for resource path"""

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def create(package: str, path: Path):
        return ResourcePathWrapper(Path(*package.split(".")) / path)

    def __truediv__(self, subpath: Union[str, Path]):
        return ResourcePathWrapper(self.path / subpath)

    @cached_property
    def package(self):
        parents = [s.name for s in reversed(self.path.parents)][1:]
        return ".".join(parents)

    @cached_property
    def name(self):
        return self.path.name

    def is_file(self):
        return any(
            traversable.name == self.name and traversable.is_file()
            for traversable in resources.files(self.package).iterdir()
        )

    def __fspath__(self):
        """Return the file system path representation of the object"""
        return resources.as_file(resources.files(self.package) / self.name)

    @contextmanager
    def open(self, *args, **kwargs):
        with resources.path(self.package, self.name) as path:
            yield path.open(*args, **kwargs)
