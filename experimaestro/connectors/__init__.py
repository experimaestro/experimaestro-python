"""Connectors module

This module contains :

- connectors
- process builders
- launchers

"""

import enum
from typing import Dict, Optional
from pathlib import Path, PosixPath
from experimaestro.locking import Lock
from experimaestro.tokens import Token


class RedirectType(enum.Enum):
    INHERIT = 0
    FILE = 1
    PIPE = 1
    NONE = 2


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
    pass


class ProcessThreadError(Exception):
    """Exception thrown by the forked process, to exit properly"""

    pass


class ProcessBuilder:
    """A process builder
    """

    def __init__(self):
        self.workingDirectory = None
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

    def lock(self, path: Path) -> Lock:
        """Returns a lock on a file"""
        raise NotImplementedError()

    def resolve(self, path: Path, basepath: Path = None):
        raise NotImplementedError()

    def setExecutable(self, path: Path, flag: bool):
        raise NotImplementedError()

    def createtoken(self, name: str, total: int) -> Token:
        """Returns a token in the default path for the connector"""
        raise NotImplementedError()
