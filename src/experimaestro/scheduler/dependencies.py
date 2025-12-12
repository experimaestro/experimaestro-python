"""Dependency between tasks and tokens"""

import threading
from typing import Optional, Set, TYPE_CHECKING
import asyncio
from enum import Enum
from ..utils import logger
from ..locking import Lock

if TYPE_CHECKING:
    from . import Job


class Dependents:
    """Encapsulate the access to the dependents"""

    def __init__(self):
        self.lock = threading.Lock()
        self._dependents: Set[Dependency] = set()  # as source

    def add(self, dependency):
        with self.lock:
            self._dependents.add(dependency)

    def __enter__(self):
        # Returns the set after locking
        self.lock.__enter__()
        return self._dependents

    def __exit__(self, *args):
        self.lock.__exit__(*args)


class Resource:
    def __init__(self):
        self.dependents = Dependents()


class DependencyStatus(Enum):
    WAIT = 0
    """Waiting for dependency to be available"""

    OK = 1
    """Dependency can be locked"""

    FAIL = 2
    """Dependency won't be availabe in the foreseeable future"""


class Dependency:
    """Base class for dependencies

    Static dependencies (like jobs) have a fixed state once resolved - they cannot
    go from DONE back to WAIT. This is the default behavior.
    """

    # Dependency status
    loop: asyncio.AbstractEventLoop

    def __init__(self, origin):
        # Origin and target are two resources
        self.origin = origin
        self.target: Optional["Job"] = None
        self.currentstatus = DependencyStatus.WAIT

    def is_dynamic(self) -> bool:
        """Returns True if this is a dynamic dependency (can change state)"""
        return False

    def status(self) -> DependencyStatus:
        raise NotImplementedError()

    async def aio_lock(self) -> Lock:
        """Acquire a lock on this dependency asynchronously"""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "Dep[{origin}->{target}]/{currentstatus}".format(**self.__dict__)

    def check(self):
        assert self.target is not None
        status = self.status()
        logger.debug("Dependency check: %s", self)
        if status != self.currentstatus:
            logger.debug(
                "Dependency %s is %s (was: %s)", self, status, self.currentstatus
            )
            self.target.dependencychanged(self, self.currentstatus, status)
            self.currentstatus = status


class DynamicDependency(Dependency):
    """Base class for dynamic dependencies

    Dynamic dependencies (like tokens) can change state at any time - availability
    can go from OK to WAIT and back. These require special handling during lock
    acquisition with retry logic.
    """

    def is_dynamic(self) -> bool:
        """Returns True - this is a dynamic dependency"""
        return True
