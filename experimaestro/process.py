"""Module containing classes for process management
"""
from pathlib import Path
import enum

class RedirectType(enum.Enum):
    INHERIT = 0
    FILE = 1
    PIPE = 1
    NONE = 2


class Redirect:
    NONE = Redirect(RedirectType.NONE)
    INHERIT = Redirect(RedirectType.INHERIT)

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

    def start(self):
        raise NotImplementedError("Method not implemented in %s" % self.__class__)