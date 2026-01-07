from __future__ import annotations

from typing import TYPE_CHECKING, List

from experimaestro.utils.asyncio import asyncThreadcheck
from .utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.connectors import Process
    from experimaestro.scheduler.dependencies import DynamicDependency


class Lock:
    """A lock"""

    def __init__(self):
        self._level = 0
        self.detached = False

    def detach(self):
        self.detached = True

    def acquire(self):
        if self._level == 0:
            self._level += 1
            self._acquire()
        return self

    def release(self):
        if not self.detached and self._level == 1:
            self._level -= 1
            self._release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    async def __aenter__(self):
        return await asyncThreadcheck("lock (aenter)", self.__enter__)

    async def __aexit__(self, *args):
        return await asyncThreadcheck("lock (aexit)", self.__exit__, *args)

    def _acquire(self):
        raise NotImplementedError()

    def _release(self):
        raise NotImplementedError()


class LockError(Exception):
    pass


class Locks(Lock):
    """A set of locks"""

    def __init__(self):
        super().__init__()
        self.locks = []

    def append(self, lock):
        self.locks.append(lock)

    def _acquire(self):
        for lock in self.locks:
            lock.acquire()

    def _release(self):
        logger.debug("Releasing %d locks", len(self.locks))
        for lock in self.locks:
            logger.debug("[locks] Releasing %s", lock)
            lock.release()


class DynamicDependencyLock(Lock):
    """Base class for locks from dynamic dependencies with lifecycle hooks.

    Dynamic dependency locks have additional lifecycle methods that are called
    by the scheduler when a job starts and finishes. This allows locks to:
    - Persist state for recovery (e.g., write lock info to disk)
    - Clean up resources race-safely when job finishes
    - Serialize lock info to pass to job process

    Subclasses must implement to_json() to include 'module' and 'class' keys
    for dynamic deserialization in the job process.
    """

    def __init__(self, dependency: DynamicDependency):
        super().__init__()
        self.dependency = dependency

    async def aio_job_started(self, job: Job, process: Process) -> None:
        """Called when the job has started successfully.

        This is called AFTER the process has been spawned but BEFORE the
        scheduler releases the connector lock. Use this to persist lock state
        for recovery purposes.

        :param job: The job that started
        :param process: The process running the job
        """
        pass

    async def aio_job_finished(self, job: Job) -> None:
        """Called when the job has finished (success or failure).

        This is called BEFORE the lock is released. Use this for any
        pre-release cleanup that requires knowledge of job state.

        :param job: The job that finished
        """
        pass

    def to_json(self) -> dict:
        """Serialize lock info for passing to job process.

        Returns a dict with 'module' and 'class' keys for dynamic import.
        Subclasses should call super().to_json() and update with their data.

        :return: JSON-serializable dict with lock information
        """
        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
        }

    @classmethod
    def from_json(cls, data: dict) -> JobDependencyLock:
        """Deserialize lock info received in job process.

        This creates a JobDependencyLock variant - a lock that is already
        held and only needs to be released on exit.

        :param data: Dict from to_json()
        :return: JobDependencyLock instance
        """
        raise NotImplementedError(f"from_json not implemented for {cls.__name__}")


class DynamicDependencyLocks(Lock):
    """Container for dynamic dependency locks with lifecycle support.

    This container manages a collection of DynamicDependencyLock instances,
    providing batch operations for lifecycle events and serialization.
    """

    def __init__(self):
        super().__init__()
        self.locks: List[DynamicDependencyLock] = []

    def append(self, lock: DynamicDependencyLock) -> None:
        """Add a lock to the container."""
        self.locks.append(lock)

    def clear(self) -> None:
        """Clear all locks from the container (without releasing)."""
        self.locks.clear()

    def _acquire(self) -> None:
        """Acquire all locks."""
        for lock in self.locks:
            lock.acquire()

    def _release(self) -> None:
        """Release all locks."""
        logger.debug("Releasing %d dynamic dependency locks", len(self.locks))
        for lock in self.locks:
            logger.debug("[locks] Releasing %s", lock)
            lock.release()

    async def aio_job_started(self, job: Job, process: Process) -> None:
        """Notify all locks that job has started."""
        for lock in self.locks:
            await lock.aio_job_started(job, process)

    async def aio_job_finished(self, job: Job) -> None:
        """Notify all locks that job has finished."""
        for lock in self.locks:
            await lock.aio_job_finished(job)

    def to_json(self) -> List[dict]:
        """Serialize all locks for job process."""
        return [lock.to_json() for lock in self.locks]


class JobDependencyLock:
    """Lock held by job process.

    This is the job-process-side counterpart of DynamicDependencyLock.
    Created via from_json(), then acquire() is called when entering context.
    """

    def acquire(self) -> None:
        """Acquire the lock. Called when entering context."""
        pass

    def release(self) -> None:
        """Release the lock. Called when exiting context."""
        pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class _JobDependencyLocksContext:
    """Context manager for acquiring/releasing job dependency locks."""

    def __init__(self, locks: List[JobDependencyLock]):
        self._locks = locks
        self._acquired: List[JobDependencyLock] = []

    def __enter__(self):
        for lock in self._locks:
            lock.acquire()
            self._acquired.append(lock)
        return self

    def __exit__(self, *args):
        for lock in reversed(self._acquired):
            try:
                lock.release()
            except Exception:
                logger.exception("Error releasing lock %s", lock)


class JobDependencyLocks:
    """Container for locks in job process.

    Manages a collection of JobDependencyLock instances.
    Use dependency_locks() to get a context manager for acquire/release.
    """

    def __init__(self):
        self.locks: List[JobDependencyLock] = []

    def dependency_locks(self) -> _JobDependencyLocksContext:
        """Return a context manager that acquires locks on enter, releases on exit."""
        return _JobDependencyLocksContext(self.locks)

    @classmethod
    def from_json(cls, locks_data: List[dict]) -> JobDependencyLocks:
        """Create from serialized lock data.

        Each lock entry must have 'module' and 'class' keys specifying
        the DynamicDependencyLock subclass to use for deserialization.
        """
        import importlib

        instance = cls()
        for lock_data in locks_data:
            module_name = lock_data.get("module")
            class_name = lock_data.get("class")

            if module_name is None or class_name is None:
                logger.warning("Lock data missing 'module' or 'class': %s", lock_data)
                continue

            try:
                module = importlib.import_module(module_name)
                lock_class = getattr(module, class_name)
                job_lock = lock_class.from_json(lock_data)
                instance.locks.append(job_lock)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "Failed to load lock class %s.%s: %s",
                    module_name,
                    class_name,
                    e,
                )
                continue

        return instance
