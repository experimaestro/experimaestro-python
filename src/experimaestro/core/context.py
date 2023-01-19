from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import List, Optional
import os


def shallow_copy(src_path: Path, dest_path: Path):
    """Copy a directory or file, trying to use hard links if possible"""
    if src_path.is_file():
        try:
            dest_path.hardlink_to(src_path)
        except OSError:
            shutil.copy(src_path, dest_path)
    else:
        os.mkdir(dest_path)
        for f in src_path.iterdir():
            shallow_copy(f, dest_path / f.name)


class SerializationContext:
    save_directory: Optional[Path]
    var_path: List[str]

    """Context when serializing experimaestro configurations"""

    def __init__(self, *, save_directory: Optional[Path] = None):
        self.save_directory = save_directory
        self.var_path = []

    def serialize(self, var_path: List[str], data_path):
        if self.save_directory:
            path = Path(*var_path)
            (self.save_directory / path).parent.mkdir(exist_ok=True, parents=True)
            shallow_copy(data_path, self.save_directory / path)
            return path
        return data_path

    @contextmanager
    def push(self, varname: str):
        self.var_path.append(varname)
        yield self.var_path
        self.var_path.pop()
