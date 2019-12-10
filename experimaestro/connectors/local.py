"""All classes related to localhost management
"""

import subprocess
from pathlib import Path, WindowsPath, PosixPath
import os
import subprocess
import threading
import fasteners
from experimaestro.locking import Lock

from . import Connector, Process, ProcessBuilder, RedirectType, Redirect
from experimaestro.utils import logger

class LocalProcess(Process):
    def __init__(self, process):
        self._process = process
    
    def wait(self):
        return self._process.wait()

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
        stdin = getstream(self.stdin, False)
        stdout = getstream(self.stdout, True)
        stderr = getstream(self.stderr, True)

        # Valid values are PIPE, DEVNULL, an existing file descriptor (a positive integer), an existing file object, and None
        return LocalProcess(subprocess.Popen(self.command, stdin=stdin, stderr=stderr, stdout=stdout))


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
    def lock(self, path: Path, max_delay: int=-1) -> Lock:
        """Returns a lockable path
        
        Arguments:
            path {Path} -- Path of the lockfile
            max_delay {int} -- Maximum wait duration (seconds)
        """
        return InterProcessLock(path, max_delay)

    def processbuilder(self) -> ProcessBuilder:
        return LocalProcessBuilder()

    def resolve(self, path: Path, basepath:Path=None) -> str:
        assert isinstance(path, PosixPath) or isinstance(path, WindowsPath)
        if not basepath:
            return str(path.absolute())
        try:
            return str(path.relative_to(basepath))
        except ValueError:
            return str(path)

    def setExecutable(self, path: Path, flag: bool):
        os.chmod(path, 0o744)
