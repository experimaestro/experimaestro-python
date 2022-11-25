import os
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
    # Useful to debug lock problems
    TIMEOUT = float(os.environ.get("XPM_LOCK_TIMEOUT", "-1"))

    """Wrapper of threading.condition allowing to debug"""

    def __enter__(self) -> bool:
        self._acquired = self.acquire(timeout=ThreadingCondition.TIMEOUT)

        if self._acquired:
            return True

        logger.error("Failed to acquire CV (timeout %f)", ThreadingCondition.TIMEOUT)
        import faulthandler

        faulthandler.dump_traceback(all_threads=True)
        raise RuntimeError("Failed to acquire CV")

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """Property at class level"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
