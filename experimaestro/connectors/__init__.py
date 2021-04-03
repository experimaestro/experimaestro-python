"""Connectors module

This module contains :

- connectors
- process builders
- launchers

"""

import enum
from typing import Optional, Union
from pathlib import Path
from experimaestro.locking import Lock
from experimaestro.tokens import Token
import pkg_resources


class RedirectType(enum.Enum):
    INHERIT = 0
    FILE = 1
    PIPE = 2
    NONE = 3


class Redirect:
    _NONE: "Redirect"
    _INHERIT: "Redirect"

    def __init__(self, type: RedirectType, path=None, function=None):
        self.type = type
        self.path = path
        self.function = function

    @staticmethod
    def file(path: Path):
        return Redirect(RedirectType.FILE, path=path)

    @staticmethod
    def pipe(function):
        return Redirect(RedirectType.PIPE, function=function)

    @staticmethod
    def none():
        return Redirect._NONE

    @staticmethod
    def inherit():
        return Redirect._INHERIT


Redirect._NONE = Redirect(RedirectType.NONE)
Redirect._INHERIT = Redirect(RedirectType.INHERIT)


class Process:
    HANDLERS = None

    @staticmethod
    def handler(key: str):
        """Get a handler"""
        if Process.HANDLERS is None:
            Process.HANDLERS = {}
            for ep in pkg_resources.iter_entry_points(group="experimaestro.process"):
                Process.HANDLERS[ep.name] = ep.load()

        return Process.HANDLERS.get(key, None)

    def wait(self):
        """Wait until the process finishes"""
        raise NotImplementedError(f"Not implemented: {self.__class__}.wait")


class ProcessThreadError(Exception):
    """Exception thrown by the forked process, to exit properly"""

    pass


class ProcessBuilder:
    """A process builder"""

    def __init__(self):
        self.workingDirectory = None  # type: Optional[Path]
        self.stdin = Redirect.inherit()
        self.stdout = Redirect.inherit()
        self.stderr = Redirect.inherit()
        self.detach = True
        self.environ = {}
        self.command = []

    def start(self) -> Process:
        """Start the process"""
        raise NotImplementedError("Method not implemented in %s" % self.__class__)


class Connector:
    def __init__(self, localpath: Path):
        """Creates a new connector

        Arguments:
            localpath {Path} -- The working directory
        """
        self._localpath = localpath

    @property
    def localpath(self):
        if not self._localpath.is_dir():
            self._localpath.mkdir(parents=True)
        return self._localpath

    @localpath.setter
    def localpath(self, localpath: Path):
        self._localpath = localpath

    def processbuilder(self) -> ProcessBuilder:
        raise NotImplementedError()

    def lock(self, path: Path, max_delay: int = -1) -> Lock:
        """Returns a lock on a file"""
        raise NotImplementedError()

    def resolve(self, path: Path, basepath: Path = None):
        raise NotImplementedError()

    def setExecutable(self, path: Path, flag: bool):
        raise NotImplementedError()

    def createtoken(self, name: str, total: int) -> Token:
        """Returns a token in the default path for the connector"""
        raise NotImplementedError()


class Locator:
    pass


def parsepath(path: Union[str, Path]) -> Path:
    """Parse a path

    Returns a local path or a SshPath
    """
    from urllib.parse import urlparse

    if isinstance(path, Path):
        return path

    if isinstance(path, str) and path.startswith("ssh:"):
        from .ssh import SshPath

        return SshPath(path)

    return Path(path)
