from contextlib import contextmanager
from pathlib import Path

try:
    from pathlib import UnsupportedOperation
except ImportError:
    UnsupportedOperation = OSError
import shutil
from typing import List, Optional, Protocol, Set, Union
import os


def shallow_copy(src_path: Path, dest_path: Path):
    """Copy a directory or file, trying to use hard links if possible"""
    if src_path.is_file():
        try:
            dest_path.hardlink_to(src_path)
        except (NotImplementedError, UnsupportedOperation, OSError):
            shutil.copy(src_path, dest_path)
    else:
        if dest_path.exists():
            shutil.rmtree(dest_path)

        os.mkdir(dest_path)
        for f in src_path.iterdir():
            shallow_copy(f, dest_path / f.name)


class SerializedPath:
    """Just a container for a path that has been serialized"""

    def __init__(self, path: Path, is_folder: Optional[bool] = None):
        self.path = path
        self.is_folder = path.is_dir() if is_folder is None else is_folder


class SerializationContext:
    """Context when serializing experimaestro configurations"""

    save_directory: Optional[Path]
    var_path: List[str]
    serialized: Set[int]

    def __init__(self, *, save_directory: Optional[Path] = None):
        """Creates a new serialization context

        :param save_directory: Defines the directory where `SerializedPath` are
            stored
        """
        self.save_directory = save_directory
        self.var_path = []
        self.serialized = set()

    def serialize(self, var_path: List[str], data_path: Path) -> SerializedPath:
        if self.save_directory:
            # Creates a relative path from the configuration qualified name
            path = Path(*var_path)

            # Creates the directory if needed
            dest = self.save_directory / path
            dest.parent.mkdir(exist_ok=True, parents=True)

            # Copy, using hard links whenever possible
            shallow_copy(data_path, dest)
            return SerializedPath(path, data_path.is_dir())
        return SerializedPath(data_path)

    @contextmanager
    def push(self, varname: str):
        self.var_path.append(varname)
        yield self.var_path
        self.var_path.pop()


class SerializedPathLoader(Protocol):
    def __call__(path: Union[Path, str, SerializedPath]) -> Path:
        """Get a filesystem path from a relative path

        :param path: The relative path
        :param is_folder: Whether this path is a folder
        :return: A Path corresponding to a real file or folder
        """
        ...
