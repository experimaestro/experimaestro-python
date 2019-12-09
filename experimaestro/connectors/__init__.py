"""Connectors module

This module contains :

- connectors
- process builders
- launchers

"""

import enum
from typing import Dict, Optional
from pathlib import Path, PosixPath

class RedirectType(enum.Enum):
    INHERIT = 0
    FILE = 1
    PIPE = 1
    NONE = 2


class Redirect:
    NONE: "Redirect"
    INHERIT: "Redirect"

    def __init__(self, type: RedirectType, path=None, pipe=None):
        self.type = RedirectType
        self.path = path
        self.pipe = pipe

    @staticmethod
    def file(path: Path):
        return Redirect(RedirectType.FILE, path=path)

    @staticmethod
    def pipe(function):
        return Redirect(RedirectType.PIPE, pipe=function)

    @staticmethod
    def none():
        return Redirect.NONE

    @staticmethod
    def inherit():
        return Redirect.INHERIT

Redirect.NONE = Redirect(RedirectType.NONE)
Redirect.INHERIT = Redirect(RedirectType.INHERIT)

class Process: pass

class ProcessBuilder:
    """A process builder
    """
    def __init__(self, workingDirectory):
      self.workingDirectory = workingDirectory
      self.stdin = None
      self.stdout = None
      self.stderr = None
      self.detach = True
      self.environement = {}
      self.command = []

    def start(self) -> Process:
        """Start the process"""
        raise NotImplementedError("Method not implemented in %s" % self.__class__)


class Connector(): 
    def processbuilder(self) -> ProcessBuilder:
        raise NotImplementedError()

    def resolve(self, path: Path, basepath:Path=None):
        raise NotImplementedError()

    def setExecutable(self, path: Path, flag: bool):
        raise NotImplementedError()
