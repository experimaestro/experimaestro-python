"""All classes related to localhost management"""

import subprocess
from typing import Optional
from pathlib import Path, WindowsPath, PosixPath
import os
import threading
from experimaestro.launcherfinder import LauncherRegistry
from fasteners import InterProcessLock as FastenersInterProcessLock
import psutil

from asyncio import Lock

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
    if redirect.type == RedirectType.FILE and redirect.path:
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


class InterProcessLock(FastenersInterProcessLock, Lock):
    def __init__(self, path, max_delay=-1):
        FastenersInterProcessLock.__init__(self, path)
        self.max_delay = max_delay

    def __enter__(self):
        logger.debug("Locking %s", self.path)
        if not FastenersInterProcessLock.acquire(
            self, blocking=True, max_delay=self.max_delay, timeout=None
        ):
            raise threading.ThreadError("Could not acquire lock")
        logger.debug("Locked %s", self.path)
        return self

    def __aenter__(self):
        # use the synchronous __enter__ method in async context
        return self.__enter__()

    def __exit__(self, *args):
        logger.debug("Unlocking %s", self.path)
        super().__exit__(*args)

    def __aexit__(self, *args):
        # use the synchronous __exit__ method in async context
        return self.__exit__(*args)


class LocalConnector(Connector):
    INSTANCE: Optional[Connector] = None

    @staticmethod
    def instance():
        if LocalConnector.INSTANCE is None:
            LocalConnector.INSTANCE = LocalConnector()
        return LocalConnector.INSTANCE

    @staticmethod
    def init_registry(registry: LauncherRegistry):
        pass

    def __init__(self, localpath: Optional[Path] = None):
        localpath = localpath
        if not localpath:
            localpath = Path(
                os.environ.get("XPM_WORKDIR", "~/.local/share/experimaestro")
            ).expanduser()
        super().__init__(localpath)

    def lock(self, path: Path, max_delay: int = -1) -> Lock:
        """Returns a lockable path

        Arguments:
            path {Path} -- Path of the lockfile
            max_delay {int} -- Maximum wait duration (seconds)
        """
        return InterProcessLock(path, max_delay)

    def createtoken(self, name: str, total: int) -> Token:
        tokendir = self.localpath / "tokens"
        tokendir.mkdir(exist_ok=True, parents=True)
        return CounterToken.create(name, tokendir / ("%s.counter" % name), total)

    def processbuilder(self) -> ProcessBuilder:
        return LocalProcessBuilder()

    def resolve(self, path: Path, basepath: Optional[Path] = None) -> str:
        assert isinstance(path, PosixPath) or isinstance(
            path, WindowsPath
        ), f"Unrecognized path {type(path)}"
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
