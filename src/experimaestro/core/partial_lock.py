"""Partial directory locking for exclusive access.

This module provides locking for partial directories to ensure only one job
can write to a partial at a time.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Type
import filelock
import logging

from experimaestro.locking import (
    DynamicDependencyLock,
    DynamicLockFile,
    JobDependencyLock,
    TrackedDynamicResource,
)
from experimaestro.dynamic import DynamicDependency
from experimaestro.scheduler.dependencies import Resource


logger = logging.getLogger("xpm.partial")


class PartialLockFile(DynamicLockFile):
    """Lock file for partial directories.

    Stores:
    - job_uri: Reference to the job holding the lock
    - information: {"partial_name": name_of_partial}

    The lock file is created in the partial directory when a job acquires
    exclusive access, enabling recovery if the scheduler or job crashes.
    """

    partial_name: str

    def from_information(self, info) -> None:
        """Set partial name from information dict."""
        if info is None:
            # Creating a new lock file
            self.partial_name = ""
        elif isinstance(info, dict):
            self.partial_name = info.get("partial_name", "")
        else:
            raise ValueError(f"Invalid information format: {info}")

    def to_information(self) -> dict:
        """Return partial name for JSON serialization."""
        return {"partial_name": self.partial_name}


class PartialJobResource(Resource, TrackedDynamicResource):
    """Resource for partial directory locking.

    Tracks the number of jobs holding write locks on the partial directory.
    Uses file-based tracking for recovery, similar to CounterToken.

    File structure in partial directory:
    - {partial_path}/.experimaestro/locks/ipc.lock: IPC lock
    - {partial_path}/.experimaestro/locks/informations.json: {"max_write_locks": 1}
    - {partial_path}/.experimaestro/locks/jobs/{task_id}/{identifier}.json: lock file
    """

    #: Lock file class for partial lock files
    lock_file_class: Type[DynamicLockFile] = PartialLockFile

    #: Maps partial paths to PartialJobResource instances (singleton pattern)
    RESOURCES: Dict[str, "PartialJobResource"] = {}

    @staticmethod
    def forkhandler():
        """Clear resources after fork to avoid sharing state."""
        PartialJobResource.RESOURCES = {}

    @staticmethod
    def create(partial_path: Path, max_write_locks: int = 1) -> "PartialJobResource":
        """Get or create a PartialJobResource for the given path.

        This implements a singleton pattern per path to ensure only one
        resource instance exists for each partial directory.

        Args:
            partial_path: Path to the partial directory
            max_write_locks: Maximum number of concurrent write locks (default: 1)

        Returns:
            PartialJobResource for the path
        """
        key = str(partial_path)
        resource = PartialJobResource.RESOURCES.get(key)
        if resource is None:
            resource = PartialJobResource(partial_path, max_write_locks)
            PartialJobResource.RESOURCES[key] = resource
        return resource

    @property
    def lock_folder(self) -> Path:
        """Path to the lock folder within the partial directory."""
        return self._partial_path / ".experimaestro" / "locks"

    @property
    def partial_path(self) -> Path:
        """Path to the partial directory."""
        return self._partial_path

    def _write_informations(self, max_write_locks: int) -> None:
        """Write partial informations to disk."""
        self.lock_folder.mkdir(parents=True, exist_ok=True)
        with self.informations_path.open("w") as f:
            json.dump({"max_write_locks": max_write_locks}, f)

    def _read_informations(self) -> int:
        """Read max_write_locks from informations file."""
        try:
            with self.informations_path.open("r") as f:
                data = json.load(f)
                return data.get("max_write_locks", 1)
        except FileNotFoundError:
            return 1

    def __init__(self, partial_path: Path, max_write_locks: int = 1):
        """Initialize a partial job resource.

        Args:
            partial_path: Path to the partial directory
            max_write_locks: Maximum number of concurrent write locks
        """
        self._partial_path = partial_path
        self.max_write_locks = max_write_locks
        self.write_locks = 0  # Current number of write locks

        # Create informations file if needed
        self.lock_folder.mkdir(parents=True, exist_ok=True)
        if not self.informations_path.is_file():
            self._write_informations(max_write_locks)

        # Initialize base classes
        Resource.__init__(self)
        TrackedDynamicResource.__init__(self, str(partial_path))

    def __str__(self):
        return f"partial[{self._partial_path}]"

    # --- TrackedDynamicResource abstract method implementations ---

    def _reset_state(self) -> None:
        """Reset state before re-reading lock files."""
        self.max_write_locks = self._read_informations()
        self.write_locks = 0

    def _account_lock_file(self, lf: DynamicLockFile) -> None:
        """Account for a lock file - increment write lock count."""
        self.write_locks += 1

    def _unaccount_lock_file(self, lf: DynamicLockFile) -> None:
        """Unaccount a lock file - decrement write lock count."""
        self.write_locks -= 1

    def is_available(self, dependency: "PartialDependency") -> bool:
        """Check if the partial is available (write locks not exhausted)."""
        return self.write_locks < self.max_write_locks

    def _do_acquire(self, dependency: "PartialDependency") -> None:
        """Increment write lock count."""
        self.write_locks += 1
        logger.debug(
            "Partial state [acquired]: write_locks %d, max %d, path %s",
            self.write_locks,
            self.max_write_locks,
            self._partial_path,
        )

    def _do_release(self, dependency: "PartialDependency") -> None:
        """Decrement write lock count."""
        self.write_locks -= 1
        logger.debug(
            "Partial state [released]: write_locks %d, max %d, path %s",
            self.write_locks,
            self.max_write_locks,
            self._partial_path,
        )

    def _get_lock_file_information(self, dependency: "PartialDependency"):
        """Return partial name for lock file."""
        return {"partial_name": dependency.partial_name}

    def _handle_information_change(self) -> None:
        """Handle max_write_locks changes from informations.json."""
        max_write_locks = self._read_informations()
        delta = max_write_locks - self.max_write_locks
        self.max_write_locks = max_write_locks
        logger.debug(
            "Partial information modified: write_locks %d, max %d, path %s",
            self.write_locks,
            self.max_write_locks,
            self._partial_path,
        )

        # Notify waiting tasks if more locks became available
        if delta > 0:
            self.available_condition.notify_all()

    # --- Partial API ---

    def dependency(self, partial_name: str) -> "PartialDependency":
        """Create a dependency on this partial resource.

        Args:
            partial_name: Name of the partial (for symlink creation)

        Returns:
            PartialDependency for this resource
        """
        return PartialDependency(self, partial_name)


# Register fork handler to clear resources after fork
if sys.platform != "win32":
    os.register_at_fork(after_in_child=PartialJobResource.forkhandler)


class PartialJobLock(JobDependencyLock):
    """Job-side lock for a partial directory.

    Inherits from JobDependencyLock to participate in the dynamic lock lifecycle.
    Uses IPC locking (filelock.FileLock) for exclusive access.

    File structure:
    - {partial_path}/.experimaestro/locks/ipc.lock: IPC lock
    - {partial_path}/.experimaestro/locks/jobs/{task_id}/{identifier}.json: lock file

    Lifecycle:
    1. Scheduler acquires IPC lock and releases it in aio_job_started
    2. Job acquires IPC lock (blocks until scheduler releases)
    3. Job creates/updates lock file to track who holds the lock
    4. On release: IPC lock released, lock file deleted (via base class)
    """

    def __init__(self, data: dict):
        self.partial_path = Path(data["partial_path"])
        self.partial_name = data["partial_name"]
        self.job_uri = data["job_uri"]
        self.lock_folder = self.partial_path / ".experimaestro" / "locks"
        self.ipc_lock_path = self.lock_folder / "ipc.lock"
        self.lock_file_path = Path(data["lock_file_path"])
        self._lock: filelock.FileLock | None = None

    def acquire(self) -> None:
        """Acquire exclusive lock on the partial directory.

        Verifies the lock file exists (created by scheduler) and acquires
        the IPC lock for exclusive access.
        """
        # Verify lock file exists (base class)
        super().acquire()

        logger.info("Acquiring partial lock: %s", self.partial_path)

        # Acquire the IPC lock (blocking - waits for other jobs to finish)
        self._lock = filelock.FileLock(str(self.ipc_lock_path))
        self._lock.acquire()

        logger.info("Acquired partial lock: %s", self.partial_path)

    def release(self) -> None:
        """Release the partial lock and delete the lock file."""
        if self._lock is not None and self._lock.is_locked:
            logger.info("Releasing partial lock: %s", self.partial_path)
            self._lock.release()
            self._lock = None

            # Delete the lock file (handled by base class)
            super().release()


class PartialLock(DynamicDependencyLock):
    """Scheduler-side lock for a partial directory.

    Inherits from DynamicDependencyLock to participate in the dynamic lock lifecycle.
    Ensures exclusive access to a partial directory while a job is running.

    Manages lock acquisition/release through the PartialJobResource, similar to
    how CounterTokenLock manages through CounterToken.

    File structure:
    - {partial_path}/.experimaestro/locks/ipc.lock: IPC lock
    - {partial_path}/.experimaestro/locks/jobs/{task_id}/{identifier}.json: lock file
    """

    dependency: "PartialDependency"

    def __init__(self, dependency: "PartialDependency"):
        super().__init__(dependency)

    @property
    def lock_folder(self) -> Path:
        """Path to the lock folder within the partial directory."""
        return self.dependency.resource.lock_folder

    async def _aio_acquire(self):
        """Acquire exclusive lock via the resource (async)."""
        await self.dependency.resource.aio_acquire(self.dependency)

    async def _aio_release(self):
        """Release the lock via the resource (async)."""
        await self.dependency.resource.aio_release(self.dependency)

    def __str__(self):
        return f"PartialLock({self.dependency.partial_name})"

    async def aio_job_before_start(self, job) -> None:
        """Create symlink before job starts."""
        from experimaestro.scheduler.jobs import Job

        assert isinstance(job, Job)

        # Create symlink in job's .experimaestro/partials directory
        # The symlink points to the partial data directory (not the lock folder)
        partials_dir = job.experimaestro_path / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = partials_dir / self.dependency.partial_name
        if not symlink_path.exists():
            symlink_path.symlink_to(self.dependency.partial_path)
            logger.debug(
                "Created partial symlink: %s -> %s",
                symlink_path,
                self.dependency.partial_path,
            )

    def to_json(self) -> dict:
        """Serialize lock for job process."""
        data = super().to_json()
        data.update(
            {
                "partial_path": str(self.dependency.partial_path),
                "partial_name": self.dependency.partial_name,
                "lock_file_path": str(self.lock_file_path),
                "job_uri": str(self.dependency.target.basepath),
            }
        )
        return data

    @classmethod
    def from_json(cls, data: dict) -> PartialJobLock:
        """Create job-side lock from serialized data."""
        return PartialJobLock(data)


class PartialDependency(DynamicDependency):
    """A dependency on a partial directory (dynamic - availability can change).

    This ensures that only one job can write to a partial directory at a time.
    Uses PartialJobResource for state tracking and recovery.
    """

    def __init__(self, resource: PartialJobResource, partial_name: str):
        """Create a partial dependency.

        Args:
            resource: The PartialJobResource managing this partial
            partial_name: Name of the partial (used for symlink creation)
        """
        super().__init__(resource)
        self._resource = resource
        self._partial_name = partial_name

    def _create_lock(self) -> PartialLock:
        """Create a partial lock for this dependency."""
        return PartialLock(self)

    @property
    def resource(self) -> PartialJobResource:
        """The resource managing this partial."""
        return self._resource

    @property
    def partial_path(self) -> Path:
        """Path to the partial directory."""
        return self._resource.partial_path

    @property
    def partial_name(self) -> str:
        """Name of the partial."""
        return self._partial_name

    def __repr__(self) -> str:
        return f"PartialDep[{self.partial_name}]"
