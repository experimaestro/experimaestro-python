"""Tokens are special types of dependency controlling the access to
a computational resource (e.g. number of launched jobs, etc.)
"""

from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
import threading
from typing import Dict, Type

from omegaconf import DictConfig

from experimaestro.core.objects import Config
from experimaestro.launcherfinder.base import TokenConfiguration
from experimaestro.launcherfinder.registry import LauncherRegistry

from .locking import (
    DynamicDependencyLock,
    DynamicLockFile,
    JobDependencyLock,
    Lock,
    LockError,
    TrackedDynamicResource,
)
from .dynamic import DynamicDependency
from .scheduler.dependencies import Resource
import logging


logger = logging.getLogger("xpm.tokens")


class Token(Resource):
    """Base class for all token-based resources"""

    available: int


# =============================================================================
# File-based counter token
# =============================================================================


class CounterTokenJobLock(JobDependencyLock):
    """Job-side lock for counter tokens.

    Inherits from JobDependencyLock to participate in the dynamic lock lifecycle.
    On release, deletes the token lock file created by the scheduler.
    """

    def __init__(self, data: dict):
        self.token_path = Path(data["token_path"])
        self.count = data["count"]
        self.name = data["name"]
        # Set lock_file_path for base class release() to delete
        self.lock_file_path = Path(data["lock_file_path"])


class CounterTokenLock(DynamicDependencyLock):
    """Scheduler-side lock for counter token dependency.

    Inherits from DynamicDependencyLock to participate in the dynamic lock lifecycle.
    Manages token acquisition/release through the CounterToken resource.

    On serialization, passes lock file path to the job process so it can
    delete the lock file on release.
    """

    dependency: "CounterTokenDependency"

    def __init__(self, dependency: "CounterTokenDependency"):
        super().__init__(dependency)

    @property
    def lock_folder(self) -> Path:
        """Path to the token lock folder."""
        return self.dependency.token.lock_folder

    async def _aio_acquire(self):
        await self.dependency.token.aio_acquire(self.dependency)

    async def _aio_release(self):
        await self.dependency.token.aio_release(self.dependency)

    async def aio_job_started(self, job, process) -> None:
        """Called when the job has started - write process info to lock file."""
        token = self.dependency.token
        lock_key = token._lock_file_key(self.lock_file_path)

        async with token._async_lock, token.ipc_lock:
            lf = token.cache.get(lock_key)
            if lf is not None:
                lf.write_process(process)
                logger.debug("Wrote process info for %s", lock_key)

    def __str__(self):
        return "Lock(%s)" % self.dependency

    def to_json(self) -> dict:
        """Serialize lock for job process."""
        data = super().to_json()
        data.update(
            {
                "token_path": str(self.dependency.token.path),
                "count": self.dependency.count,
                "name": self.dependency.token.name,
                "lock_file_path": str(self.lock_file_path),
            }
        )
        return data

    @classmethod
    def from_json(cls, data: dict) -> CounterTokenJobLock:
        """Create job-side lock from serialized data."""
        return CounterTokenJobLock(data)


class CounterTokenDependency(DynamicDependency):
    """A dependency onto a token (dynamic - availability can change)"""

    def __init__(self, token: "CounterToken", count: int):
        super().__init__(token)
        self._token = token
        self.count = count

    def _create_lock(self) -> "Lock":
        """Create a counter token lock for this dependency."""
        return CounterTokenLock(self)

    @property
    def token(self):
        return self._token


class TokenLockFile(DynamicLockFile):
    """Lock file for counter tokens.

    The token file stores JSON with:
    - job_uri: Reference to the job holding the lock
    - information: {"count": number_of_tokens}
    """

    count: int

    def __init__(self, path: Path, resource: "TrackedDynamicResource"):
        """Load token file from disk."""
        self.count = 0
        super().__init__(path, resource)

    def from_information(self, info) -> None:
        """Set count from information dict."""
        if info is None:
            # Creating a new lock file
            self.count = 0
        elif isinstance(info, dict):
            self.count = info.get("count", 0)
        else:
            raise ValueError(f"Invalid information format: {info}")

    def to_information(self) -> dict:
        """Return count for JSON serialization."""
        return {"count": self.count}

    @classmethod
    def from_dependency(cls, dependency: "CounterTokenDependency") -> "TokenLockFile":
        """Create a token lock file from a dependency.

        This is a convenience method for testing and backward compatibility.

        Args:
            dependency: The counter token dependency

        Returns:
            New TokenLockFile instance
        """
        path = (
            dependency._token.path / "tasks" / f"{dependency.target.relmainpath}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        job_uri = str(dependency.target.basepath)
        return cls.create(
            path, dependency._token, job_uri, information={"count": dependency.count}
        )


class CounterToken(Token, TrackedDynamicResource):
    """File-based counter token

    To ensure recovery (server stopped for whatever reason), we use one folder
    per token; inside this folder (lock_folder):

    - ipc.lock is used for IPC locking (from TrackedDynamicResource)
    - informations.json contains the maximum number of tokens {"total": count}
    - jobs/{task_id}/{identifier}.json contain job-specific lock info (count, job URI)
    """

    #: Lock file class for token files
    lock_file_class: Type[DynamicLockFile] = TokenLockFile

    #: Maps token keys to CounterToken instances
    TOKENS: Dict[str, "CounterToken"] = {}

    @staticmethod
    def forkhandler():
        CounterToken.TOKENS = {}

    @staticmethod
    def create(name: str, path: Path, count: int):
        created = CounterToken.TOKENS.get(name, None)
        if created:
            logger.warning("Re-using token for path %s", path)
        else:
            created = CounterToken(name, path, count)
            CounterToken.TOKENS[name] = created
        return created

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        registry.register_token(
            "countertoken",
            DictConfig({}, key_type=str, element_type=CounterConfiguration),
        )

    @property
    def lock_folder(self) -> Path:
        """Path to the lock folder."""
        return self._path

    @property
    def path(self) -> Path:
        """Path to the token directory (alias for lock_folder)."""
        return self._path

    def _write_informations(self, total: int) -> None:
        """Write token informations to disk."""
        with self.informations_path.open("w") as f:
            json.dump({"total": total}, f)

    def _read_informations(self) -> int:
        """Read token total from informations file."""
        try:
            with self.informations_path.open("r") as f:
                data = json.load(f)
                return data.get("total", 0)
        except FileNotFoundError:
            return 0

    def __init__(self, name: str, path: Path, count: int, force=True):
        """Initialize a counter token.

        Arguments:
            name: Token name
            path: The file path of the token directory
            count: Number of tokens (overrides previous definitions)
            force: If the token has already been created, force to write the maximum
                   number of tokens
        """
        # Store path before calling super().__init__ since lock_folder needs it
        self._path = path
        self.total = count

        # Set the informations file if needed (before TrackedDynamicResource init)
        path.mkdir(exist_ok=True, parents=True)
        if force or not (path / "informations.json").is_file():
            with (path / "informations.json").open("w") as f:
                json.dump({"total": count}, f)

        # Initialize base classes - this will call _update()
        Token.__init__(self)
        TrackedDynamicResource.__init__(self, name)

    def __str__(self):
        return "token[{}]".format(self.name)

    # --- TrackedDynamicResource abstract method implementations ---

    def _reset_state(self) -> None:
        """Reset available count before re-reading lock files."""
        self.total = self._read_informations()
        self.available = self.total

    def _account_lock_file(self, lf: DynamicLockFile) -> None:
        """Subtract token count from available."""
        self.available -= lf.count

    def _unaccount_lock_file(self, lf: DynamicLockFile) -> None:
        """Add token count back to available."""
        self.available += lf.count

    def is_available(self, dependency: "CounterTokenDependency") -> bool:
        """Check if enough tokens are available."""
        return self.available >= dependency.count

    def _do_acquire(self, dependency: "CounterTokenDependency") -> None:
        """Subtract tokens from available count."""
        self.available -= dependency.count
        logger.debug(
            "Token state [acquired %d]: available %d, total %d",
            dependency.count,
            self.available,
            self.total,
        )

    def _do_release(self, dependency: "CounterTokenDependency") -> None:
        """Add tokens back to available count."""
        self.available += dependency.count
        logging.debug("%s: available %d", self, self.available)

    def _get_lock_file_information(self, dependency: "CounterTokenDependency"):
        """Return token count for lock file."""
        return {"count": dependency.count}

    # --- Token-specific event handling ---

    def _handle_information_change(self) -> None:
        """Handle token count changes from informations.json."""
        total = self._read_informations()
        delta = total - self.total
        self.total = total
        self.available += delta
        logger.debug(
            "Token information modified: available %d, total %d",
            self.available,
            self.total,
        )

        # Notify waiting tasks if tokens became available
        if delta > 0:
            self.available_condition.notify_all()

    # --- Token API ---

    def dependency(self, count):
        """Create a token dependency"""
        return CounterTokenDependency(self, count)

    def __call__(self, count, task: Config):
        """Create a token dependency and add it to the task"""
        return task.add_dependencies(self.dependency(count))


# =============================================================================
# Process level token
# =============================================================================


class ProcessCounterToken(Token):
    """Process-level token"""

    def __init__(self, count: int):
        """Creates a new

        Arguments:
            count {int} -- Number of tokens
        """
        super().__init__()

        self.count = count
        self.available = count
        self.lock = threading.Lock()

    def __str__(self):
        return "process-token()"

    def dependency(self, count):
        """Create a token dependency"""
        return CounterTokenDependency(self, count)

    def __call__(self, count, task: Config):
        """Create a token dependency and add it to the task"""
        return task.add_dependencies(self.dependency(count))

    def acquire(self, dependency: CounterTokenDependency):
        """Acquire requested token"""
        with self.lock:
            if self.available < dependency.count:
                raise LockError("No token")

            self.available -= dependency.count

    def release(self, dependency: CounterTokenDependency):
        """Release"""
        with self.lock:
            self.available += dependency.count
            logging.debug(
                "%s: releasing %d (available %d)",
                self,
                dependency.count,
                self.available,
            )


if sys.platform != "win32":
    os.register_at_fork(after_in_child=CounterToken.forkhandler)


@dataclass
class CounterConfiguration(TokenConfiguration):
    tokens: int

    def create(self, registry: "LauncherRegistry", identifier: str):
        from experimaestro.connectors.local import LocalConnector

        return LocalConnector.instance().createtoken(identifier, self.tokens)
