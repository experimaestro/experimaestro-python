from pathlib import Path
from typing import Union
import logging
import shutil

logger = logging.getLogger("xpm")


def aspath(path: Union[str, Path]):
    if isinstance(path, Path):
        return path
    return Path(path)


def cleanupdir(path: Path):
    """Remove all the tree below a path, leaving a folder"""
    if path.is_dir():
        for p in path.iterdir():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()

    path.mkdir(exist_ok=True, parents=True)
    return path
