"""All classes related to localhost management"""

import subprocess
from typing import Optional
from pathlib import Path, WindowsPath, PosixPath
import os
import threading
from experimaestro.launcherfinder import LauncherRegistry
import filelock
import psutil

from experimaestro.locking import Lock, SyncLock

from . import (
    Connector,
    Process,
    ProcessBuilder,
    ProcessState,
    RedirectType,
    Redirect,
)
from experimaestro.tokens import Token, CounterToken
from experimaestro.utils import logger
from experimaestro.utils.psutils import aio_wait_pid


class PsutilProcess(Process):
    """Wrapper for psutil process"""

    def __init__(self, pid: int):
        self._process = psutil.Process(pid)

    def wait(self) -> Optional[int]:
        logger.debug("Waiting (psutil) for process with PID %s", self._process.pid)
        code = self._process.wait()
        logger.debug(
            "Finished to wait (psutil) for process with PID %s: code %s",
            self._process.pid,
            code,
        )
        return code

    async def aio_wait(self) -> int:
        """Asynchronously wait for process to finish (true async on Linux/macOS)"""
        logger.debug(
            "Async waiting (psutil) for process with PID %s", self._process.pid
        )
        code = await aio_wait_pid(self._process.pid)
        logger.debug(
            "Finished async wait (psutil) for process with PID %s: code %s",
            self._process.pid,
            code,
        )
        return code

    async def aio_state(self, timeout: float | None = None) -> ProcessState:
        if self._process.is_running():
            return ProcessState.RUNNING
        return ProcessState.FINISHED

    def __repr__(self):
        return f"PsUtil({self._process})"


class LocalProcess(Process):
    def __init__(self, process: subprocess.Popen):
        self._process = process

    def __repr__(self):
        return f"Process({self._process.pid})"

    def wait(self) -> int:
        logger.debug("Waiting (python) for process with PID %s", self._process.pid)
        code = self._process.wait()
        logger.debug(
            "Finished to wait (python) for process with PID %s: %s",
            self._process.pid,
            code,
        )
        return code

    async def aio_wait(self) -> int:
        """Asynchronously wait for process to finish (true async on Linux/macOS)"""
        logger.debug(
            "Async waiting (python) for process with PID %s", self._process.pid
        )
        code = await aio_wait_pid(self._process.pid)
        # Update the Popen returncode
        self._process.returncode = code
        logger.debug(
            "Finished async wait (python) for process with PID %s: %s",
            self._process.pid,
            code,
        )
        return code

    async def aio_state(self, timeout: float | None = None) -> ProcessState:
        code = self._process.poll()
        if code is None:
            return ProcessState.RUNNING

        if code == 0:
            return ProcessState.DONE

        return ProcessState.ERROR

    def tospec(self):
        return {"type": "local", "pid": self._process.pid}

    def kill(self):
        self._process.kill()

    @staticmethod
    def fromspec(connector, spec):
        pid = spec["pid"]
        try:
            return PsutilProcess(pid)
        except psutil.NoSuchProcess:
            pass

        return None


def getstream(redirect: Redirect, write: bool):
    if redirect.type == RedirectType.FILE:
        return redirect.path.open("w" if write else "r")

    if redirect.type == RedirectType.PIPE:
        return subprocess.PIPE

    if redirect.type == RedirectType.INHERIT:
        return None

    raise NotImplementedError("For %s", redirect)


class LocalProcessBuilder(ProcessBuilder):
    def start(self, task_mode=False):
        """Start the process

        :param task_mode: just ignored
        """
        stdin = getstream(self.stdin, False)
        stdout = getstream(self.stdout, True)
        stderr = getstream(self.stderr, True)

        logger.debug("Popen process")
        if self.detach:
            p = subprocess.Popen(
                self.command,
                stdin=stdin,
                stderr=stderr,
                stdout=stdout,
                env=self.environ,
                close_fds=True,
                cwd="/",
            )
        else:
            p = subprocess.Popen(
                self.command,
                stdin=stdin,
                stderr=stderr,
                stdout=stdout,
                env=self.environ,
            )

        process = LocalProcess(p)

        if self.stdout and self.stdout.type == RedirectType.PIPE:
            self.stdout.function(p.stdout)
        if self.stderr and self.stderr.type == RedirectType.PIPE:
            self.stderr.function(p.stderr)

        return process


class AsyncLock(Lock):
    """Async inter-process lock using filelock.AsyncFileLock.

    Implements the Lock interface for use as an async context manager.
    """

    def __init__(self, path, timeout: float = -1):
        Lock.__init__(self)
        self.path = str(path)
        self._lock = filelock.AsyncFileLock(self.path, timeout=timeout)

    @property
    def acquired(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.is_locked

    async def _aio_acquire(self):
        logger.debug("Locking %s (async)", self.path)
        try:
            await self._lock.acquire()
        except filelock.Timeout:
            raise threading.ThreadError("Could not acquire lock")
        logger.debug("Locked %s (async)", self.path)

    async def _aio_release(self):
        logger.debug("Unlocking %s (async)", self.path)
        await self._lock.release()


class LocalConnector(Connector):
    """Connector for executing tasks on the local machine.

    This connector handles local file system operations and process execution.
    It is the default connector used when no remote execution is needed.

    Use :meth:`instance` to get a singleton instance of the local connector.

    :param localpath: Base path for experimaestro data. Defaults to
        ``~/.local/share/experimaestro`` or the value of ``XPM_WORKDIR``
        environment variable.
    """

    INSTANCE: Connector = None

    @staticmethod
    def instance():
        if LocalConnector.INSTANCE is None:
            LocalConnector.INSTANCE = LocalConnector()
        return LocalConnector.INSTANCE

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        pass

    def __init__(self, localpath: Path = None):
        localpath = localpath
        if not localpath:
            localpath = Path(
                os.environ.get("XPM_WORKDIR", "~/.local/share/experimaestro")
            ).expanduser()
        super().__init__(localpath)

    def lock(self, path: Path, max_delay: int = -1) -> SyncLock:
        """Returns a sync lock

        Arguments:
            path {Path} -- Path of the lockfile
            max_delay {int} -- Maximum wait duration (seconds)
        """
        timeout = max_delay if max_delay > 0 else -1
        return SyncLock(path, timeout=timeout)

    def async_lock(self, path: Path, max_delay: int = -1) -> Lock:
        """Returns an async lock

        Arguments:
            path {Path} -- Path of the lockfile
            max_delay {int} -- Maximum wait duration (seconds)
        """
        timeout = max_delay if max_delay > 0 else -1
        return AsyncLock(path, timeout=timeout)

    def createtoken(self, name: str, total: int) -> Token:
        tokendir = self.localpath / "tokens"
        tokendir.mkdir(exist_ok=True, parents=True)
        return CounterToken.create(name, tokendir / ("%s.counter" % name), total)

    def processbuilder(self) -> ProcessBuilder:
        return LocalProcessBuilder()

    def resolve(self, path: Path, basepath: Path = None) -> str:
        assert isinstance(path, PosixPath) or isinstance(path, WindowsPath), (
            f"Unrecognized path {type(path)}"
        )
        if not basepath:
            return str(path.absolute())
        try:
            return str(path.relative_to(basepath))
        except ValueError:
            return str(path)

    def setExecutable(self, path: Path, flag: bool):
        os.chmod(path, 0o744)

    def __str__(self):
        return "LocalConnector"
