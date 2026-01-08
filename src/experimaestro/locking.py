from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os.path
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type
import weakref

import fasteners
from watchdog.events import FileSystemEventHandler

from experimaestro.utils.asyncio import asyncThreadcheck

logger = logging.getLogger("xpm.locking")

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.connectors import Process
    from experimaestro.scheduler.dependencies import DynamicDependency


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

    def to_json(self) -> List[dict]:
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

    def __init__(self, path: Path):
        """Load lock file from disk.

        Args:
            path: Path to the lock file
        """
        self.path = path
        self.job_uri = None

        last_error = None
        retries = 0
        while retries < 5:
            retries += 1
            try:
                with path.open("rt") as fp:
                    data = json.load(fp)
                    self.job_uri = data.get("job_uri")
                    self.from_information(data.get("information"))
                    return  # Success
            except FileNotFoundError:
                # File was deleted between check and read
                return
            except Exception as e:
                last_error = e
                logging.exception("Error while reading %s", self.path)
                time.sleep(0.1)
                continue

        # Exhausted retries - re-raise the last error
        if last_error is not None:
            raise last_error

    @classmethod
    def create(cls, path: Path, job_uri: str, information=None) -> "DynamicLockFile":
        """Create a new lock file on disk.

        Args:
            path: Path where to create the file
            job_uri: URI of the job holding the lock
            information: Type-specific data for the lock file

        Returns:
            New lock file instance
        """
        self = object.__new__(cls)
        self.path = path
        self.job_uri = job_uri
        self.from_information(information)

        logging.debug("Writing lock file %s", path)
        data = {"job_uri": job_uri, "information": self.to_information()}
        with path.open("wt") as fp:
            json.dump(data, fp)
        return self

    def delete(self) -> None:
        """Delete the lock file from disk."""
        if self.path.is_file():
            logging.debug("Deleting lock file %s", self.path)
            self.path.unlink()

    def watch(self, on_released: Optional[Callable[[], None]] = None) -> None:
        """Watch the job process and call callback when it finishes.

        This starts a background thread that:
        1. Waits for the job lock to be available (job started)
        2. Waits for the process to finish
        3. Deletes the lock file
        4. Calls the callback (if provided)

        Args:
            on_released: Optional callback to invoke when lock is released
        """
        if self.job_uri is None:
            return

        logger.debug("Watching process for %s (%s)", self.path, self.job_uri)
        job_path = Path(self.job_uri)
        lockpath = job_path.with_suffix(".lock")
        pidpath = job_path.with_suffix(".pid")

        def run():
            logger.debug("Locking job lock path %s", lockpath)
            process = None

            # Acquire the job lock - blocks if scheduler is still starting the job
            # Once we get the lock, the job has either started or finished
            with fasteners.InterProcessLock(lockpath):
                if not pidpath.is_file():
                    logger.debug("Job already finished (no PID file %s)", pidpath)
                else:
                    s = ""
                    while s == "":
                        s = pidpath.read_text()

                    logger.info("Loading job watcher from definition")
                    from experimaestro.connectors import Process
                    from experimaestro.connectors.local import LocalConnector

                    connector = LocalConnector.instance()
                    process = Process.fromDefinition(connector, json.loads(s))

            # Wait out of the lock
            if process is not None:
                process.wait()

            self.delete()
            if on_released is not None:
                on_released()

        threading.Thread(target=run).start()

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


class _TrackedResourceProxy(FileSystemEventHandler):
    """Weak reference proxy for file system events.

    Prevents the resource from being kept alive by the watcher.
    """

    def __init__(self, resource: "TrackedDynamicResource"):
        self._resource_ref = weakref.ref(resource)

    def on_modified(self, event):
        resource = self._resource_ref()
        if resource is not None:
            return resource.on_modified(event)

    def on_deleted(self, event):
        resource = self._resource_ref()
        if resource is not None:
            return resource.on_deleted(event)

    def on_created(self, event):
        resource = self._resource_ref()
        if resource is not None:
            return resource.on_created(event)


class TrackedDynamicResource(ABC):
    """Base class for resources with file-based lock tracking.

    This provides:
    - File system watching for lock files
    - IPC and thread locking
    - Condition variable for waiting on availability
    - Cache of lock files

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
    lock_file_class: Type[DynamicLockFile]

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

        self.cache: Dict[str, DynamicLockFile] = {}

        self.ipc_lock = fasteners.InterProcessLock(self.ipc_lock_path)
        self.lock = threading.Lock()
        self.available_condition = threading.Condition(self.lock)

        self.timestamp = os.path.getmtime(self.lock_folder)

        # Initial state update
        with self.lock, self.ipc_lock:
            self._update()

        # Set up file system watching
        from .ipc import ipcom

        self.watchedpath = str(self.lock_folder.absolute())
        self.proxy = _TrackedResourceProxy(self)
        self.watcher = ipcom().fswatch(self.proxy, self.lock_folder, recursive=True)
        logger.debug("Watching %s", self.watchedpath)

    def __del__(self):
        if self.watcher is not None:
            logging.debug("Removing watcher on %s", self.watchedpath)
            from .ipc import ipcom

            ipcom().fsunwatch(self.watcher)
            self.watcher = None

    def _lock_file_key(self, path: Path) -> str:
        """Get the cache key for a lock file path.

        The key is the relative path from jobs_folder (e.g., "task_id@identifier.json").
        """
        return str(path.relative_to(self.jobs_folder))

    def _update(self) -> None:
        """Update state by reading all lock files from disk.

        Assumes IPC lock is held.
        """
        logging.debug("Full resource state update for %s", self.name)
        old_cache = self.cache
        self.cache = {}

        self._reset_state()

        if self.jobs_folder.exists():
            for path in self.jobs_folder.glob("*.json"):
                key = self._lock_file_key(path)
                lf = old_cache.get(key)
                if lf is None:
                    lf = self.lock_file_class(path)
                    lf.watch(lambda k=key: self._on_lock_released(k))
                    logging.debug("Read lock file %s", path)
                else:
                    logging.debug("Lock file already in cache %s", key)

                self.cache[key] = lf
                self._account_lock_file(lf)

        logging.debug("Full resource state update finished for %s", self.name)

    def _on_lock_released(self, name: str) -> None:
        """Called when a watched lock is released (job finished).

        Args:
            name: Name of the lock file
        """
        with self.lock:
            if name in self.cache:
                logging.debug("Lock released (job finished): %s", name)
                lf = self.cache[name]
                del self.cache[name]
                self._unaccount_lock_file(lf)
                self.available_condition.notify_all()

    def _is_job_lock_file(self, path: Path) -> bool:
        """Check if path is a job lock file (under jobs_folder)."""
        try:
            path.relative_to(self.jobs_folder)
            return path.suffix == ".json"
        except ValueError:
            return False

    def on_deleted(self, event) -> None:
        """Handle file deletion event."""
        logger.debug(
            "Deleted path notification %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )
        path = Path(event.src_path)
        if not self._is_job_lock_file(path):
            return

        key = self._lock_file_key(path)
        if key in self.cache:
            with self.lock:
                if key in self.cache:
                    logging.debug("Deleting %s from cache (event)", key)
                    lf = self.cache[key]
                    del self.cache[key]
                    self._unaccount_lock_file(lf)
                    self.available_condition.notify_all()

    def on_created(self, event) -> None:
        """Handle file creation event."""
        logger.debug(
            "Created path notification %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )
        path = Path(event.src_path)
        if not self._is_job_lock_file(path):
            return

        try:
            key = self._lock_file_key(path)
            if key not in self.cache:
                with self.lock:
                    if key not in self.cache:
                        lf = self.lock_file_class(path)
                        lf.watch(lambda k=key: self._on_lock_released(k))
                        self.cache[key] = lf
                        self._account_lock_file(lf)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Uncaught exception in on_created handler")
            raise

    def on_modified(self, event) -> None:
        """Handle file modification event."""
        try:
            logger.debug(
                "on modified path: %s [watched %s]",
                event.src_path,
                self.watchedpath,
            )
            path = Path(event.src_path)

            # Handle informations.json modification
            if event.src_path == str(self.informations_path):
                self._on_information_modified()
                return

            # Handle job lock files
            if not self._is_job_lock_file(path):
                return

            key = self._lock_file_key(path)
            if key not in self.cache:
                with self.lock:
                    if key not in self.cache:
                        logger.debug("Lock file not in cache %s", key)
                        try:
                            lf = self.lock_file_class(path)
                            lf.watch(lambda k=key: self._on_lock_released(k))
                            self.cache[key] = lf
                            self._account_lock_file(lf)
                        except FileNotFoundError:
                            pass
        except Exception:
            logger.exception("Uncaught exception in on_modified handler")
            raise

    def _on_information_modified(self) -> None:
        """Handle informations.json modification.

        Checks timestamp to avoid duplicate processing, then calls
        _handle_information_change() for subclass-specific logic.
        """
        import os

        logger.debug("Resource information modified: %s", self.name)
        with self.lock:
            timestamp = os.path.getmtime(self.informations_path)
            if timestamp <= self.timestamp:
                logger.debug(
                    "Not reading information file [%f <= %f]",
                    timestamp,
                    self.timestamp,
                )
                return

            self._handle_information_change()

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

    def acquire(self, dependency: "DynamicDependency") -> None:
        """Acquire the resource for a dependency.

        Args:
            dependency: The dependency requesting the resource

        Raises:
            LockError: If resource is not available
        """
        with self.lock, self.ipc_lock:
            self._update()
            if not self.is_available(dependency):
                raise LockError(f"Resource {self.name} not available")

            # Create lock file
            lock_path = self._get_job_lock_path(dependency)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_key = self._lock_file_key(lock_path)

            lf = self.lock_file_class.create(
                lock_path,
                self._get_job_uri(dependency),
                information=self._get_lock_file_information(dependency),
            )
            self.cache[lock_key] = lf

            self._do_acquire(dependency)

            logger.debug("Acquired %s for %s", self.name, dependency)

    def release(self, dependency: "DynamicDependency") -> None:
        """Release the resource for a dependency.

        Args:
            dependency: The dependency releasing the resource
        """
        with self.lock, self.ipc_lock:
            self._update()

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

            logger.debug("Deleting %s from cache", lock_key)
            del self.cache[lock_key]

            self._do_release(dependency)

            self.available_condition.notify_all()
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
