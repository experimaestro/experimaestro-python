"""All classes related to localhost management
"""

import subprocess
from pathlib import Path, WindowsPath, PosixPath
import os
import subprocess
import threading
import sys
import fasteners
import psutil
from experimaestro.locking import Lock

from . import (
    Connector,
    Process,
    ProcessBuilder,
    RedirectType,
    Redirect,
    ProcessThreadError,
)
from experimaestro.tokens import Token, CounterToken
from experimaestro.utils import logger


class LocalProcess(Process):
    def __init__(self, process):
        self._process = process

    def wait(self):
        self._process.wait()


def getstream(redirect: Redirect, write: bool):
    if redirect.type == RedirectType.FILE:
        return redirect.path.open("w" if write else "r")

    if redirect.type == RedirectType.PIPE:
        raise NotImplementedError()

    if redirect.type == RedirectType.INHERIT:
        return None

    raise NotImplementedError("For %s", redirect)


class LocalProcessBuilder(ProcessBuilder):
    def start(self):
        if self.detach:
            return self.unix_daemon()
        else:
            return self.start_nodetach()

    def start_nodetach(self):
        stdin = getstream(self.stdin, False)
        stdout = getstream(self.stdout, True)
        stderr = getstream(self.stderr, True)

        # Valid values are PIPE, DEVNULL, an existing file descriptor (a positive integer), an existing file object, and None
        logger.debug("Popen process")
        return LocalProcess(
            subprocess.Popen(self.command, stdin=stdin, stderr=stderr, stdout=stdout)
        )

    def unix_daemon(self):
        # From https://stackoverflow.com/questions/6011235/run-a-program-from-python-and-have-it-continue-to-run-after-the-script-is-kille
        # do the UNIX double-fork magic, see Stevens' "Advanced
        # Programming in the UNIX Environment" for details (ISBN 0201563177)

        readpipe, writepipe = os.pipe()

        # First fork
        try:
            pid = os.fork()
            if pid > 0:
                # parent process, return and keep running
                pid = int(os.read(readpipe, 100).decode("utf-8"))
                return psutil.Process(pid)
        except OSError as e:
            logger.error("Fork #1 failed: %d (%s)" % (e.errno, e.strerror))
            raise

        os.chdir("/")
        os.setsid()

        # The second fork is done with Popen
        logger.info("(forked process) starting process")
        p = self.start_nodetach()
        os.write(writepipe, str(p._process.pid).encode("utf-8"))
        # Exit now - this will leave the process running
        raise ProcessThreadError()


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
        tokendir.mkdir(exist_ok=True)
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
