"""Dependency between tasks and tokens"""

import threading
from typing import Set
from abc import ABC, abstractmethod
from enum import Enum
from ..locking import Lock


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


class Dependency(ABC):
    """Base class for dependencies

    Static dependencies (like jobs) have a fixed state once resolved - they cannot
    go from DONE back to WAIT. This is the default behavior.
    """

    def __init__(self, origin):
        # Origin is the resource this dependency points to
        self.origin = origin
        # Target will be set by scheduler when registering the job
        self.target = None

    def is_dynamic(self) -> bool:
        """Returns True if this is a dynamic dependency (can change state)"""
        return False

    @abstractmethod
    async def aio_lock(self, timeout: float = 0) -> Lock:
        """Acquire a lock on this dependency asynchronously

        Args:
            timeout: Timeout in seconds (0 = wait indefinitely)

        Returns:
            Lock object

        Raises:
            LockError: If lock cannot be acquired within timeout
            RuntimeError: If dependency failed
        """
        pass

    def __repr__(self) -> str:
        return f"Dep[{self.origin}]"


class DynamicDependency(Dependency):
    """Base class for dynamic dependencies

    Dynamic dependencies (like tokens) can change state at any time - availability
    can go from OK to WAIT and back. These require special handling during lock
    acquisition with retry logic.
    """

    def is_dynamic(self) -> bool:
        """Returns True - this is a dynamic dependency"""
        return True
