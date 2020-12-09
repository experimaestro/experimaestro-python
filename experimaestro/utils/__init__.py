from pathlib import Path
import threading
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


class ThreadingCondition(threading.Condition):
    """Wrapper of threading.condition allowing to debug"""

    def __enter__(self) -> bool:
        import traceback

        # logger.debug("Trying to enter CV\n%s", "".join(traceback.format_stack()))

        r = super().__enter__()
        # logger.debug("CV locked")
        return r

    def __exit__(self, exc_type, exc_val, exc_tb):
        # logger.debug("Exiting CV")
        return super().__exit__(exc_type, exc_val, exc_tb)
