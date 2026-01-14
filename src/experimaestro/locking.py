from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import json
import logging
import os.path
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Callable, Optional
import weakref

import filelock

from experimaestro.dynamic import DynamicResource

logger = logging.getLogger("xpm.locking")

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.connectors import Process
    from experimaestro.dynamic import DynamicDependency


def get_job_lock_relpath(task_id: str, identifier: str) -> Path:
    """Get the lock relative path for a job.

    Creates a unique relative path combining task_id and identifier.
    Limited to 256 characters to avoid filesystem issues.

    Args:
        task_id: The task identifier
        identifier: The job identifier (hash)

    Returns:
        Relative path in format "{task_id}@{identifier}.json"
    """
    return Path(f"{task_id}@{identifier}"[:256] + ".json")


class Lock:
    """An async lock"""

    def __init__(self):
        self._level = 0
        self.detached = False

    def detach(self):
        self.detached = True

    async def aio_acquire(self):
        if self._level == 0:
            self._level += 1
            await self._aio_acquire()
        return self

    async def aio_release(self):
        if not self.detached and self._level == 1:
            self._level -= 1
            await self._aio_release()

    async def __aenter__(self):
        await self.aio_acquire()
        return self

    async def __aexit__(self, *args):
        await self.aio_release()

    async def _aio_acquire(self):
        raise NotImplementedError()

    async def _aio_release(self):
        raise NotImplementedError()


class SyncLock:
    """A sync inter-process lock using filelock.FileLock."""

    def __init__(self, path, timeout: float = -1):
        self.path = str(path)
        self._lock = filelock.FileLock(self.path, timeout=timeout)

    @property
    def acquired(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.is_locked

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock.

        Args:
            blocking: If True, block until lock is acquired. If False, return immediately.

        Returns:
            True if lock was acquired, False otherwise.
        """
        try:
            if blocking:
                self._lock.acquire()
            else:
                self._lock.acquire(timeout=0)
            return True
        except filelock.Timeout:
            return False

    def release(self):
        """Release the lock."""
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()


class LockError(Exception):
    pass


class Locks(Lock):
    """A set of locks"""

    def __init__(self):
        super().__init__()
        self.locks = []

    def append(self, lock):
        self.locks.append(lock)

    async def _aio_acquire(self):
        for lock in self.locks:
            await lock.aio_acquire()

    async def _aio_release(self):
        logger.debug("Releasing %d locks", len(self.locks))
        for lock in self.locks:
            logger.debug("[locks] Releasing %s", lock)
            await lock.aio_release()


class DynamicDependencyLock(Lock, ABC):
    """Base class for locks from dynamic dependencies with lifecycle hooks.

    Dynamic dependency locks have additional lifecycle methods that are called
    by the scheduler when a job starts and finishes. This allows locks to:
    - Persist state for recovery (e.g., write lock info to disk)
    - Clean up resources race-safely when job finishes
    - Serialize lock info to pass to job process

    File structure (standardized):
    - {lock_folder}/informations.json: Resource-level info (e.g., token counts)
    - {lock_folder}/ipc.lock: IPC lock for inter-process coordination
    - {lock_folder}/jobs/{task_specific_path}.json: Per-job lock file

    Subclasses must implement:
    - lock_folder: Path to the lock folder

    Subclasses must implement to_json() to include 'module' and 'class' keys
    for dynamic deserialization in the job process.
    """

    dependency: "DynamicDependency"

    def __init__(self, dependency: DynamicDependency):
        super().__init__()
        self.dependency = dependency

    async def aio_acquire(self):
        """Acquire lock (runs in EventLoopThread)."""
        return await super().aio_acquire()

    async def aio_release(self):
        """Release lock (runs in EventLoopThread)."""
        return await super().aio_release()

    @property
    @abstractmethod
    def lock_folder(self) -> Path:
        """Path to the lock folder. Must be implemented by subclasses."""
        ...

    @property
    def ipc_lock_path(self) -> Path:
        """Path to the IPC lock file."""
        return self.lock_folder / "ipc.lock"

    @property
    def lock_file_path(self) -> Path:
        """Path to the lock file for the current job."""
        job = self.dependency.target
        return (
            self.lock_folder
            / "jobs"
            / get_job_lock_relpath(job.task_id, job.identifier)
        )

    async def aio_job_before_start(self, job: Job) -> None:
        """Called before the job is started.

        This is called AFTER the job directory is created but BEFORE the
        job process is spawned. Use this to set up resources needed by the job.

        :param job: The job about to start
        """
        pass

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
        self.locks: list[DynamicDependencyLock] = []

    def append(self, lock: DynamicDependencyLock) -> None:
        """Add a lock to the container."""
        self.locks.append(lock)

    def clear(self) -> None:
        """Clear all locks from the container (without releasing)."""
        self.locks.clear()

    async def _aio_acquire(self) -> None:
        """Acquire all locks."""
        for lock in self.locks:
            await lock.aio_acquire()

    async def _aio_release(self) -> None:
        """Release all locks."""
        logger.debug("Releasing %d dynamic dependency locks", len(self.locks))
        for lock in self.locks:
            logger.debug("[locks] Releasing %s", lock)
            await lock.aio_release()

    async def aio_job_before_start(self, job: Job) -> None:
        """Notify all locks before job starts."""
        for lock in self.locks:
            await lock.aio_job_before_start(job)

    async def aio_job_started(self, job: Job, process: Process) -> None:
        """Notify all locks that job has started."""
        for lock in self.locks:
            await lock.aio_job_started(job, process)

    async def aio_job_finished(self, job: Job) -> None:
        """Notify all locks that job has finished."""
        for lock in self.locks:
            await lock.aio_job_finished(job)

    def to_json(self) -> list[dict]:
        """Serialize all locks for job process."""
        return [lock.to_json() for lock in self.locks]


class JobDependencyLock:
    """Lock held by job process.

    This is the job-process-side counterpart of DynamicDependencyLock.
    Created via from_json(), then acquire() is called when entering context.

    The scheduler creates the lock file before starting the job. The job process
    verifies the lock file exists on acquire() and deletes it on release().

    Subclasses should set lock_file_path in __init__ from JSON data.
    """

    #: Path to the lock file to delete on release (set from JSON data)
    lock_file_path: Optional[Path] = None

    def verify_lock_file(self) -> None:
        """Verify the lock file exists.

        If lock_file_path is None, this is a no-op.

        Raises:
            LockError: If lock file is missing
        """
        if self.lock_file_path is not None and not self.lock_file_path.is_file():
            raise LockError(f"Lock file missing: {self.lock_file_path}")

    def acquire(self) -> None:
        """Acquire the lock. Called when entering context.

        Verifies that the scheduler created the lock file.
        """
        self.verify_lock_file()

    def release(self) -> None:
        """Release the lock and delete the lock file.

        Called when exiting context.
        """
        if self.lock_file_path is not None and self.lock_file_path.is_file():
            logger.debug("Deleting lock file: %s", self.lock_file_path)
            self.lock_file_path.unlink()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class _JobDependencyLocksContext:
    """Context manager for acquiring/releasing job dependency locks."""

    def __init__(self, locks: list[JobDependencyLock]):
        self._locks = locks
        self._acquired: list[JobDependencyLock] = []

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
        self.locks: list[JobDependencyLock] = []

    def dependency_locks(self) -> _JobDependencyLocksContext:
        """Return a context manager that acquires locks on enter, releases on exit."""
        return _JobDependencyLocksContext(self.locks)

    @classmethod
    def from_json(cls, locks_data: list[dict]) -> JobDependencyLocks:
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


# --- Generalized dynamic lock file and resource tracking ---


class DynamicLockFile(ABC):
    """Base class for files that track who holds a dynamic lock.

    Each lock file stores JSON with:
    - job_uri: Reference to the job holding the lock
    - information: Type-specific data

    Subclasses override from_information() and to_information() to
    handle type-specific data in the "information" field.
    """

    path: Path
    job_uri: Optional[str]
    process: Optional[Process]
    process_future: Optional[asyncio.Future]
    timestamp: float
    resource: "TrackedDynamicResource"
    _accounted: bool

    def __init__(self, path: Path, resource: "TrackedDynamicResource"):
        """Load lock file from disk.

        Args:
            path: Path to the lock file
            resource: The resource this lock file belongs to

        If the file doesn't exist, sets job_uri to None and calls
        from_information(None) to set defaults.
        """
        self.path = path
        self.job_uri = None
        self.process = None
        self.process_future = None
        self.timestamp = 0
        self.resource = resource
        self._accounted = False
        self._read()
        self._update_accounting()

    def _update_accounting(self) -> None:
        """Update accounting based on current valid_state."""
        if self.valid_state and not self._accounted:
            self.resource._account_lock_file(self)
            self._accounted = True
        elif not self.valid_state and self._accounted:
            self.resource._unaccount_lock_file(self)
            self._accounted = False

    def unaccount(self) -> None:
        """Force unaccount this lock file from the resource state."""
        if self._accounted:
            self.resource._unaccount_lock_file(self)
            self._accounted = False

    def _read(self) -> None:
        """Read lock file contents from disk.

        Updates job_uri, process, and timestamp from the file.
        Process info is stored in the lock file JSON under "process" key.
        """
        import os
        from experimaestro.connectors.local import LocalConnector
        from experimaestro.connectors import Process

        try:
            self.timestamp = os.path.getmtime(self.path)
            data = json.loads(self.path.read_text())
            self.job_uri = data.get("job_uri")
            self.from_information(data.get("information"))

            # Read process info from lock file JSON
            process_data = data.get("process")
            if process_data:
                connector = LocalConnector.instance()
                self.process = Process.fromDefinition(connector, process_data)
                # Resolve future if it exists
                if self.process_future is not None and not self.process_future.done():
                    self.process_future.set_result(self.process)
            else:
                self.process = None

        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is not valid JSON - use defaults
            self.job_uri = None
            self.process = None
            self.from_information(None)

    def write_process(self, process: "Process") -> None:
        """Write process info to the lock file.

        Called when the job has started and we have process info.

        Args:
            process: The process running the job
        """
        import os

        self.process = process

        # Re-read current data and add process info
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"job_uri": self.job_uri, "information": self.to_information()}

        data["process"] = process.tospec()

        logging.debug("Writing process info to lock file %s", self.path)
        with self.path.open("wt") as fp:
            json.dump(data, fp)

        self.timestamp = os.path.getmtime(self.path)

        # Resolve future if it exists
        if self.process_future is not None and not self.process_future.done():
            self.process_future.set_result(process)

        # Update accounting now that we have process info
        self._update_accounting()

    def update(self) -> bool:
        """Re-read the lock file if it has been modified or is invalid.

        Returns:
            True if the file was updated, False if unchanged or error
        """
        import os

        try:
            new_timestamp = os.path.getmtime(self.path)
            # Re-read if timestamp changed OR if we don't have a process yet
            # (the PID file might have been created since we last read)
            if new_timestamp > self.timestamp:
                self._read()
                self._update_accounting()
                return True
        except OSError:
            # The token lock is invalid
            self.job_uri = None
            self.process = None
            self._update_accounting()
        return False

    @property
    def valid_state(self):
        """Check if the lock file represents a valid held lock.

        A lock file is valid when job_uri is set.
        """
        return self.job_uri is not None

    @classmethod
    def create(
        cls,
        path: Path,
        resource: "TrackedDynamicResource",
        job_uri: str,
        information=None,
    ) -> "DynamicLockFile":
        """Create a new lock file on disk.

        Args:
            path: Path where to create the file
            resource: The resource this lock file belongs to
            job_uri: URI of the job holding the lock
            information: Type-specific data for the lock file

        Returns:
            New lock file instance
        """
        import os

        self = object.__new__(cls)
        self.path = path
        self.resource = resource
        self._accounted = False
        self.job_uri = job_uri
        self.process = None
        self.process_future = None
        self.timestamp = 0
        self.from_information(information)

        logging.debug("Writing lock file %s", path)
        data = {"job_uri": job_uri, "information": self.to_information()}
        with path.open("wt") as fp:
            json.dump(data, fp)

        self.timestamp = os.path.getmtime(path)

        # Note: process_future stays None until write_process() is called
        # async_watch() will return early if no process

        return self

    def delete(self) -> None:
        """Delete the lock file from disk."""
        if self.path.is_file():
            logging.debug("Deleting lock file %s", self.path)
            self.path.unlink()

    async def async_watch(
        self,
        on_released: Optional[Callable[[], None]] = None,
    ) -> None:
        """Watch a process and call callback when it finishes.

        Args:
            on_released: Callback when process finishes
        """
        # Wait for process if not yet available
        if self.process is None:
            if self.process_future is not None:
                logger.debug("Waiting for process future for %s", self.path)
                try:
                    self.process = await self.process_future
                except Exception as e:
                    logger.debug("Error waiting for process: %s", e)

        if self.process is None:
            logger.debug(
                "No process to watch for %s: the process has gone away?", self.path
            )
            return

        logger.debug("Watching process %s for %s", self.process, self.path)

        try:
            await self.process.aio_wait()
        except Exception as e:
            logger.warning("Error waiting for process: %s", e)

        logger.debug("Process finished for %s", self.path)
        if on_released is not None:
            await on_released()

    def from_information(self, info) -> None:
        """Set type-specific data from the "information" field.

        Override in subclasses to handle extra data.

        Args:
            info: The "information" value from the JSON file
        """
        pass

    def to_information(self):
        """Get type-specific data for the "information" field.

        Override in subclasses to include extra data.

        Returns:
            Value to store in the "information" field (JSON-serializable)
        """
        return None


class EventLoopThread(threading.Thread):
    """Singleton thread that owns the central asyncio event loop.

    This thread runs the main event loop used by:
    - The scheduler for job management
    - Lock management and file watching
    - Dynamic resource polling

    All async operations in experimaestro run in this single event loop,
    ensuring no cross-loop dispatching is needed.

    Use EventLoopThread.instance() to get the singleton instance.
    The thread is started automatically on first access.
    """

    _instance: Optional["EventLoopThread"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "EventLoopThread":
        """Get or create the singleton EventLoopThread."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """No-op: EventLoopThread is not reset as it's the central event loop.

        This method is kept for backward compatibility but does nothing.
        The event loop persists for the lifetime of the process.
        """
        pass

    def __init__(self):
        super().__init__(name="EventLoop", daemon=True)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._stopping = False

    def run(self):
        """Run the event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Set loop on AsyncEventBridge so file system events are routed here
        from experimaestro.ipc import AsyncEventBridge

        AsyncEventBridge.instance().set_loop(self._loop)

        logger.debug("EventLoopThread: started with loop %s", self._loop)
        self._ready.set()

        self._loop.run_forever()

        # Cleanup
        self._loop.close()
        logger.debug("EventLoopThread: stopped")

    def stop(self):
        """Stop the event loop thread."""
        if self._loop is not None and self._loop.is_running():
            self._stopping = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self.join(timeout=5.0)

    def wait_ready(self):
        """Wait for the thread to be ready."""
        self._ready.wait()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop, waiting for it to be ready if needed."""
        self._ready.wait()
        return self._loop

    def run_coroutine(self, coro) -> asyncio.Future:
        """Schedule a coroutine on this thread's loop and return a future.

        Can be called from any thread. Returns a concurrent.futures.Future
        that can be awaited or waited on synchronously.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


# Backward compatibility alias
LockManagerThread = EventLoopThread


class _WeakAsyncFSHandler:
    """Weak-reference async handler for filesystem events.

    Mimics the watchdog FileSystemEventHandler interface with async methods.
    Uses a weak reference to avoid preventing __del__ from being called.
    Method names use the _async suffix to distinguish from sync watchdog handlers.
    """

    def __init__(self, resource: "TrackedDynamicResource"):
        self._resource_ref = weakref.ref(resource)

    async def on_created_async(self, event):
        resource = self._resource_ref()
        if resource is not None:
            await resource.on_created_async(event)

    async def on_deleted_async(self, event):
        resource = self._resource_ref()
        if resource is not None:
            await resource.on_deleted_async(event)

    async def on_modified_async(self, event):
        resource = self._resource_ref()
        if resource is not None:
            await resource.on_modified_async(event)

    async def on_moved_async(self, event):
        resource = self._resource_ref()
        if resource is not None:
            await resource.on_moved_async(event)


class TrackedDynamicResource(DynamicResource, ABC):
    """Base class for resources with file-based lock tracking.

    Inherits from DynamicResource to provide async_wait() via ResourcePoller.

    This provides:
    - File system watching for lock files
    - IPC and thread locking
    - Condition variable for waiting on availability
    - Cache of lock files
    - Async waiting via ResourcePoller and AsyncEventBridge

    File structure:
    - {lock_folder}/informations.json: Resource-level info (e.g., token counts)
    - {lock_folder}/ipc.lock: IPC lock for inter-process coordination
    - {lock_folder}/jobs/{task_specific_path}.json: Per-job lock files

    Subclasses must implement:
    - lock_folder: Path to the lock folder (abstract property)
    - lock_file_class: The DynamicLockFile subclass to use
    - is_available(): Check if resource is available for a dependency
    - _do_acquire(): Perform acquire logic
    - _do_release(): Perform release logic
    """

    #: Subclass of DynamicLockFile to use for lock files
    lock_file_class: type[DynamicLockFile]

    @property
    @abstractmethod
    def lock_folder(self) -> Path:
        """Path to the lock folder. Must be implemented by subclasses."""
        ...

    @property
    def informations_path(self) -> Path:
        """Path to the informations.json file."""
        return self.lock_folder / "informations.json"

    @property
    def ipc_lock_path(self) -> Path:
        """Path to the IPC lock file."""
        return self.lock_folder / "ipc.lock"

    @property
    def jobs_folder(self) -> Path:
        """Path to the jobs folder containing per-job lock files."""
        return self.lock_folder / "jobs"

    def __init__(self, name: str):
        """Initialize the resource.

        Args:
            name: Human-readable name for the resource
        """
        self.name = name
        self.lock_folder.mkdir(exist_ok=True, parents=True)

        # Ensure event loop thread is running before setting up file watching
        EventLoopThread.instance().wait_ready()

        # Caches dynamic lock files objects
        self.cache: dict[str, DynamicLockFile] = {}

        # IPC lock for inter-process coordination (async only)
        self.ipc_lock = filelock.AsyncFileLock(self.ipc_lock_path)

        # Async primitives - lazily created when first used to bind to correct loop
        self.__async_lock: asyncio.Lock | None = None
        self.__available_event: asyncio.Event | None = None

        self.timestamp = os.path.getmtime(self.lock_folder)

        # Initial state update (reads existing lock files)
        # Use sync FileLock since we're in __init__
        with filelock.FileLock(self.ipc_lock_path):
            self._ipc_update()

        # Set up async file system watching
        from .ipc import ipcom

        self.watchedpath = str(self.lock_folder.absolute())
        # Use weak reference handler to avoid preventing __del__ from being called
        handler = _WeakAsyncFSHandler(self)
        self.watcher = ipcom().async_fswatch(handler, self.lock_folder, recursive=True)
        logger.debug("Watching %s", self.watchedpath)

    def __del__(self):
        if self.watcher is not None:
            logging.debug("Removing watcher on %s", self.watchedpath)
            from .ipc import ipcom

            ipcom().fsunwatch(self.watcher)
            self.watcher = None

    @property
    def _async_lock(self) -> asyncio.Lock:
        """Lazily create asyncio.Lock bound to the current event loop."""
        if self.__async_lock is None:
            self.__async_lock = asyncio.Lock()
        return self.__async_lock

    @property
    def _available_event(self) -> asyncio.Event:
        """Lazily create asyncio.Event bound to the current event loop."""
        if self.__available_event is None:
            self.__available_event = asyncio.Event()
        return self.__available_event

    # --- IPC-locked methods for reading state ---

    async def _watch_and_cleanup_async(self, key: str, lf: DynamicLockFile) -> None:
        """Watch process and cleanup when finished.

        Process was already read with IPC lock - no lock needed here.
        If process is None, the PID file wasn't found - don't cleanup
        immediately as the job might still be starting.
        """
        # Watch process - if no process, this returns immediately
        await lf.async_watch()

        # Only cleanup if we actually had a process that finished
        # If process is still None, the job might still be starting
        if lf.process is None:
            logger.debug("No process to cleanup for %s", key)
            return

        # Cleanup after process finishes (async lock only, no IPC lock)
        async with self._async_lock:
            if key in self.cache:
                del self.cache[key]
                lf.unaccount()
                lf.delete()
                self._notify_async_waiters()
        self._notify_poller()

    # --- Async event handling (called by AsyncEventBridge via _WeakAsyncFSHandler) ---

    async def on_created_or_modified_async(self, event) -> None:
        """Handle file creation/modification event asynchronously.

        Called by AsyncEventBridge when a file is created or modified in the watch folder.

        Args:
            event: Watchdog FileSystemEvent
        """
        path = Path(event.src_path)
        logger.debug(
            "Created/Modified path notification (async) %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )

        # Handle informations.json modification
        if event.src_path == str(self.informations_path):
            await self._on_information_modified_async()
            return

        if not self._is_job_lock_file(path):
            return

        # Lock first async, then IPC to avoid deadlocks
        async with self._async_lock, self.ipc_lock:
            key = self._lock_file_key(path)
            if lf := self.cache.get(key):
                # A file in cache was modified, update it
                # (accounting is handled by lf.update() -> _update_accounting())
                if lf.update():
                    self._notify_poller()
            else:
                # New file - create lock file object
                # (accounting is handled by __init__ -> _update_accounting())
                try:
                    lf = self.lock_file_class(path, self)
                except FileNotFoundError:
                    return

                self.cache[key] = lf

        # Now, wait that the associated process terminates
        asyncio.create_task(self._watch_and_cleanup_async(key, lf))

    on_modified_async = on_created_or_modified_async
    on_created_async = on_created_or_modified_async

    async def on_deleted_async(self, event) -> None:
        """Handle file deletion event asynchronously.

        Called by AsyncEventBridge when a file is deleted in the watch folder.

        Args:
            event: Watchdog FileSystemEvent
        """
        path = Path(event.src_path)
        logger.debug(
            "Deleted path notification (async) %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )

        if not self._is_job_lock_file(path):
            return

        # No IPC lock needed for deletion - just update cache
        async with self._async_lock:
            key = self._lock_file_key(path)
            if lf := self.cache.get(key):
                # A file in cache was deleted, remove it
                del self.cache[key]
                lf.unaccount()

                # Notify since we changed state
                self._notify_poller()
                self._notify_async_waiters()

    async def on_moved_async(self, event) -> None:
        """Handle file move event asynchronously.

        Called by AsyncEventBridge when a file is moved in the watch folder.
        Treats moves as a delete of src_path + create of dest_path.

        Args:
            event: Watchdog FileMovedEvent
        """
        logger.debug(
            "Moved path notification (async) %s -> %s [watched %s]",
            event.src_path,
            getattr(event, "dest_path", "?"),
            self.watchedpath,
        )

        # For moved events, we treat them as delete + create
        # Create a mock delete event for the source
        class MockDeleteEvent:
            src_path = event.src_path
            is_directory = event.is_directory

        await self.on_deleted_async(MockDeleteEvent())

        # If dest_path is in our watch folder, treat as create
        if hasattr(event, "dest_path"):

            class MockCreateEvent:
                src_path = event.dest_path
                is_directory = event.is_directory

            await self.on_created_async(MockCreateEvent())

    async def _on_information_modified_async(self) -> None:
        """Handle informations.json modification asynchronously."""
        import os

        logger.debug("Resource information modified (async): %s", self.name)
        async with self._async_lock:
            timestamp = os.path.getmtime(self.informations_path)
            if timestamp <= self.timestamp:
                logger.debug(
                    "Not reading information file [%f <= %f]",
                    timestamp,
                    self.timestamp,
                )
                return

            self._handle_information_change()
            self._notify_async_waiters()

    def _notify_async_waiters(self) -> None:
        """Notify async waiters that resource state has changed.

        Sets the async event and then clears it (pulse notification).
        """
        self._available_event.set()
        # Reset for next wait - waiters that were waiting will have been notified
        self._available_event.clear()

    async def refresh_state(self) -> None:
        """Refresh state from disk.

        This is a fallback for when file system notifications are missed.
        Called by ResourcePoller periodically from the LockManagerThread loop.
        """
        async with self._async_lock, self.ipc_lock:
            self._ipc_update()

    async def async_wait(self, timeout: float = 0) -> bool:
        """Wait asynchronously until the resource state may have changed.

        Uses both AsyncEventBridge notifications and ResourcePoller for fallback.
        Returns as soon as either notifies of a change.

        Args:
            timeout: Maximum time to wait in seconds (0 = wait indefinitely)

        Returns:
            True if notified of a change, False if timed out
        """
        from experimaestro.dynamic import ResourcePoller

        poller = ResourcePoller.instance()

        # Register with ResourcePoller as fallback
        poller_event = poller.register(self, timeout)

        # Create a task that waits for either event
        async def wait_for_either():
            # Wait for either the async event or the poller event
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._available_event.wait()),
                    asyncio.create_task(poller_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        try:
            if timeout > 0:
                try:
                    await asyncio.wait_for(wait_for_either(), timeout=timeout)
                    return True
                except asyncio.TimeoutError:
                    return False
            else:
                await wait_for_either()
                return True
        finally:
            # Event cleanup is handled by poller
            pass

    def _lock_file_key(self, path: Path) -> str:
        """Get the cache key for a lock file path.

        The key is the relative path from jobs_folder (e.g., "task_id@identifier.json").
        """
        return str(path.relative_to(self.jobs_folder))

    def _ipc_update(self) -> None:
        """Update state by reading all lock files from disk.

        Assumes IPC lock is held. Does not start process monitoring -
        that is handled by async event handlers when the loop is running.

        Lock files that were previously accounted (locks we hold) are always
        re-accounted. Other lock files are accounted only if valid_state is True.
        """
        logging.debug("Full resource state update for %s", self.name)
        old_cache = self.cache
        self.cache = {}

        # Remember which lock files were accounted (locks we hold)
        was_accounted = {key for key, lf in old_cache.items() if lf._accounted}

        # Clear accounting flags before state reset
        for lf in old_cache.values():
            lf._accounted = False

        self._reset_state()

        if self.jobs_folder.exists():
            for path in self.jobs_folder.glob("*.json"):
                key = self._lock_file_key(path)
                lf = old_cache.get(key)
                if lf is None:
                    # New lock file - accounting handled by __init__
                    lf = self.lock_file_class(path, self)
                    logging.debug("Read lock file %s", path)
                else:
                    # Existing lock file - update
                    lf.update()
                    # Re-account if not already accounted by update() AND:
                    # (1) we held this lock before, or (2) it's now valid
                    if not lf._accounted and (key in was_accounted or lf.valid_state):
                        self._account_lock_file(lf)
                        lf._accounted = True
                    logging.debug("Updated lock file from cache %s", key)

                self.cache[key] = lf

        logging.debug("Full resource state update finished for %s", self.name)

    def _is_job_lock_file(self, path: Path) -> bool:
        """Check if path is a job lock file (under jobs_folder)."""
        try:
            path.relative_to(self.jobs_folder)
            return path.suffix == ".json"
        except ValueError:
            return False

    def _notify_poller(self) -> None:
        """Notify the ResourcePoller that state has changed.

        Called after file system events to wake up async waiters.
        """
        from experimaestro.dynamic import ResourcePoller

        if ResourcePoller._instance is not None:
            ResourcePoller._instance.notify(self)

    def _handle_information_change(self) -> None:
        """Handle resource-specific information changes.

        Override in subclasses to handle changes to informations.json.
        Called after timestamp check passes. Default implementation does nothing.
        """
        pass

    @abstractmethod
    def _reset_state(self) -> None:
        """Reset resource state before re-reading lock files.

        Called at the start of _update() before iterating lock files.
        """
        pass

    @abstractmethod
    def _account_lock_file(self, lf: DynamicLockFile) -> None:
        """Account for a lock file in resource state.

        Called when a lock file is read or created.

        Args:
            lf: The lock file to account for
        """
        pass

    @abstractmethod
    def _unaccount_lock_file(self, lf: DynamicLockFile) -> None:
        """Remove a lock file from resource state accounting.

        Called when a lock file is deleted.

        Args:
            lf: The lock file to unaccount
        """
        pass

    @abstractmethod
    def is_available(self, dependency: "DynamicDependency") -> bool:
        """Check if resource is available for the given dependency.

        Args:
            dependency: The dependency requesting the resource

        Returns:
            True if resource is available
        """
        pass

    @abstractmethod
    def _do_acquire(self, dependency: "DynamicDependency") -> None:
        """Perform acquire logic for the dependency.

        Called after availability is confirmed and lock file is created.

        Args:
            dependency: The dependency acquiring the resource
        """
        pass

    @abstractmethod
    def _do_release(self, dependency: "DynamicDependency") -> None:
        """Perform release logic for the dependency.

        Called before lock file is deleted.

        Args:
            dependency: The dependency releasing the resource
        """
        pass

    def _get_job_lock_path(self, dependency: "DynamicDependency") -> Path:
        """Get the lock file path for a dependency.

        Returns path under jobs_folder: jobs/{task_id}@{identifier}.json
        """
        job = dependency.target
        return self.jobs_folder / get_job_lock_relpath(job.task_id, job.identifier)

    async def aio_acquire(self, dependency: "DynamicDependency") -> None:
        """Acquire the resource for a dependency.

        Args:
            dependency: The dependency requesting the resource

        Raises:
            LockError: If resource is not available
        """
        async with self._async_lock, self.ipc_lock:
            self._ipc_update()
            if not self.is_available(dependency):
                raise LockError(f"Resource {self.name} not available")

            # Create lock file
            lock_path = self._get_job_lock_path(dependency)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_key = self._lock_file_key(lock_path)

            lf = self.lock_file_class.create(
                lock_path,
                self,
                self._get_job_uri(dependency),
                information=self._get_lock_file_information(dependency),
            )
            self.cache[lock_key] = lf

            # _do_acquire subtracts from available count. Mark as accounted
            # so _update_accounting won't account again when process is found.
            self._do_acquire(dependency)
            lf._accounted = True

            logger.debug("Acquired %s for %s", self.name, dependency)

    async def aio_release(self, dependency: "DynamicDependency") -> None:
        """Release the resource for a dependency.

        Args:
            dependency: The dependency releasing the resource
        """
        # No IPC lock needed for deletion - just update cache and delete file
        async with self._async_lock:
            lock_path = self._get_job_lock_path(dependency)
            lock_key = self._lock_file_key(lock_path)
            lf = self.cache.get(lock_key)
            if lf is None:
                # Lock file may have been released already (e.g., job completed)
                logger.debug(
                    "Lock file not in cache for %s (%s) - may have been released already",
                    dependency,
                    lock_key,
                )
                return

            # Wait for process to be resolved if needed
            if lf.process is None and lf.process_future is not None:
                try:
                    lf.process = await lf.process_future
                except Exception:
                    pass

            logger.debug("Deleting %s from cache", lock_key)
            del self.cache[lock_key]

            # Unaccount the lock file (handles _do_release internally)
            lf.unaccount()

            self._notify_async_waiters()
            lf.delete()

    def _get_job_uri(self, dependency: "DynamicDependency") -> str:
        """Get the job URI for a dependency.

        Default implementation uses dependency.target.basepath.

        Args:
            dependency: The dependency

        Returns:
            Job URI string
        """
        return str(dependency.target.basepath)

    def _get_lock_file_information(self, dependency: "DynamicDependency"):
        """Get information to store in lock file.

        Override in subclasses to store type-specific data.

        Args:
            dependency: The dependency

        Returns:
            Information for lock file creation (JSON-serializable)
        """
        return None
