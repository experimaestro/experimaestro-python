"""Connectors module

This module contains :

- connectors
- process builders
- launchers

"""

import enum
import logging
from typing import Any, Dict, Mapping, Type, Union, Optional
from pathlib import Path
from experimaestro.utils import logger
from experimaestro.locking import Lock
from experimaestro.tokens import Token
from experimaestro.utils.asyncio import asyncThreadcheck
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


class ProcessState(enum.Enum):
    SCHEDULED = 0
    RUNNING = 1
    FINISHED = 2
    DONE = 3
    ERROR = 4

    @property
    def running(self):
        return self.value < ProcessState.FINISHED.value

    @property
    def finished(self):
        return self.value >= ProcessState.FINISHED.value


class Process:
    HANDLERS = None

    @classmethod
    def fromspec(cls, connector: "Connector", definition: Dict[str, Any]) -> "Process":
        """Rebuild a process from a specification"""
        raise NotImplementedError(f"fromspec for {cls}")

    def tospec(self) -> Dict[str, Any]:
        """Outputs a process specification"""
        raise NotImplementedError(f"tospec for {self.__class__}")

    @staticmethod
    def fromDefinition(connector: "Connector", definition: Dict[str, Any]) -> "Process":
        """Retrieves a process from a serialized definition"""
        handler_type = definition["type"]
        handler = Process.handler(handler_type)
        assert handler is not None, f"No handler of type {handler_type}"
        try:
            return handler.fromspec(connector, definition)
        except Exception as e:
            logging.exception("Could not retrieve job from specification")
            raise e

    @staticmethod
    def handler(key: str) -> Type["Process"]:
        """Get a handler"""
        if Process.HANDLERS is None:
            Process.HANDLERS = {}
            for ep in pkg_resources.iter_entry_points(group="experimaestro.process"):
                logging.debug("Adding process handler for type %s", ep.name)
                handler = ep.load()
                Process.HANDLERS[ep.name] = handler
                if handler is None:
                    logging.error("Handler of type %s is null", ep.name)

        return Process.HANDLERS.get(key, None)

    def wait(self) -> int:
        """Wait until the process finishes and returns the error code"""
        raise NotImplementedError(f"Not implemented: {self.__class__}.wait")

    async def aio_state(self) -> ProcessState:
        """Returns the job state"""
        raise NotImplementedError(f"Not implemented: {self.__class__}.aio_state")

    async def aio_isrunning(self):
        """True is the process is truly running (I/O)"""
        return (await self.aio_state()).running

    async def aio_code(self) -> Optional[int]:
        """Returns a future containing the returned code

        Returns None if the process has already finished – and no information is
        known about the process.
        """
        code = await asyncThreadcheck("aio_code", self.wait)
        logger.debug("Got return code %s for %s", code, self)
        return code

    def kill(self):
        raise NotImplementedError(f"Not implemented: {self.__class__}.kill")


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
        self.environ: Mapping[str, str] = {}
        self.command = []

    def start(self, task_mode: bool = False) -> Process:
        """Start the process

        :param task_mode: True if the process is a job script
        """
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
    if isinstance(path, Path):
        return path

    if isinstance(path, str) and path.startswith("ssh:"):
        from .ssh import SshPath

        return SshPath(path)

    return Path(path)
