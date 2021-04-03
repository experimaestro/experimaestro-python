"""All classes related to localhost management
"""

import subprocess
from pathlib import Path, WindowsPath, PosixPath
import os
import threading
import fasteners
import psutil

from experimaestro.locking import Lock

from . import (
    Connector,
    Process,
    ProcessBuilder,
    RedirectType,
    Redirect,
)
from experimaestro.tokens import Token, CounterToken
from experimaestro.utils import logger


class LocalProcess(Process):
    def __init__(self, process: subprocess.Popen):
        self._process = process

    def __repr__(self):
        return f"Process({self._process.pid})"

    def wait(self) -> int:
        code = self._process.wait()
        return code

    @staticmethod
    def fromspec(spec):
        pid = spec["pid"]
        try:
            return psutil.Process(pid)
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
    def start(self):
        """Start the process"""
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


class InterProcessLock(fasteners.InterProcessLock, Lock):
    def __init__(self, path, max_delay=-1):
        super().__init__(path)
        self.max_delay = max_delay

    def __enter__(self):
        if not super().acquire(blocking=True, max_delay=self.max_delay, timeout=None):
            raise threading.ThreadError("Could not acquire lock")
        return self

    def __exit__(self, *args):
        super().__exit__(*args)


class LocalConnector(Connector):
    INSTANCE: Connector = None

    @staticmethod
    def instance():
        if LocalConnector.INSTANCE is None:
            LocalConnector.INSTANCE = LocalConnector()
        return LocalConnector.INSTANCE

    def __init__(self, localpath: Path = None):
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

    def resolve(self, path: Path, basepath: Path = None) -> str:
        assert isinstance(path, PosixPath) or isinstance(path, WindowsPath)
        if not basepath:
            return str(path.absolute())
        try:
            return str(path.relative_to(basepath))
        except ValueError:
            return str(path)

    def setExecutable(self, path: Path, flag: bool):
        os.chmod(path, 0o744)
