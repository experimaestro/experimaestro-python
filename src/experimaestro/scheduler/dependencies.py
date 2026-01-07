"""Dependency between tasks and tokens"""

import threading
from typing import TYPE_CHECKING, Optional, Set
from abc import ABC, abstractmethod
from enum import Enum

from ..locking import Lock

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job


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
    """Dependency won't be available in the foreseeable future"""


class Dependency(ABC):
    origin: "Resource"
    target: Optional["Job"]

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

    Subclasses must implement:
    - _create_lock(): Create the appropriate lock object for this dependency
    """

    def is_dynamic(self) -> bool:
        """Returns True - this is a dynamic dependency"""
        return True

    @abstractmethod
    def _create_lock(self) -> "Lock":
        """Create a lock object for this dependency.

        Returns:
            Lock object (subclass of DynamicDependencyLock)
        """
        ...

    async def aio_lock(self, timeout: float = 0) -> "Lock":
        """Acquire lock on the resource with event-driven waiting.

        This is the common implementation for all dynamic dependencies.
        Uses the resource's available_condition for efficient waiting.

        Args:
            timeout: Timeout in seconds (0 = wait indefinitely)

        Returns:
            Lock object

        Raises:
            LockError: If lock cannot be acquired within timeout
        """
        from experimaestro.locking import LockError
        from experimaestro.utils.asyncio import asyncThreadcheck
        import time

        start_time = time.time()

        while True:
            try:
                lock = self._create_lock()
                lock.acquire()
                return lock
            except LockError:
                # Wait for resource availability notification
                def wait_for_available():
                    with self.origin.available_condition:
                        # Calculate remaining timeout
                        if timeout == 0:
                            wait_timeout = None  # Wait indefinitely
                        else:
                            elapsed = time.time() - start_time
                            if elapsed >= timeout:
                                return False  # Timeout exceeded
                            wait_timeout = timeout - elapsed

                        # Wait for notification
                        return self.origin.available_condition.wait(
                            timeout=wait_timeout
                        )

                # Wait in a thread (since condition is threading-based)
                result = await asyncThreadcheck(
                    "resource availability", wait_for_available
                )

                # If wait returned False, we timed out
                if result is False:
                    raise LockError(f"Timeout waiting for resource: {self.origin}")

                # Otherwise, loop back to try acquiring again
