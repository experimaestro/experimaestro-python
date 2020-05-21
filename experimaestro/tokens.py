"""Tokens are special types of dependency controlling the access to 
a computational resource (e.g. number of launched jobs, etc.)
"""

import sys
from pathlib import Path
import fasteners
import threading
import struct
import time
import os.path
from watchdog.events import FileSystemEventHandler
from typing import Dict

from .ipc import ipcom
from .locking import Lock, LockError
from .dependencies import Dependency, DependencyStatus, Resource
import logging


logger = logging.getLogger("xpm.tokens")


class Token(Resource):
    """Base class for all token-based resources"""

    pass


class CounterTokenLock(Lock):
    def __init__(self, dependency: "CounterTokenDependency"):
        super().__init__()
        self.dependency = dependency

    def _acquire(self):
        self.dependency.token.acquire(self.dependency.count)

    def _release(self):
        self.dependency.token.release(self.dependency.count)

    def __str__(self):
        return "Lock(%s)" % self.dependency


class CounterTokenDependency(Dependency):
    def __init__(self, token: "CounterToken", count: int):
        super().__init__(token)
        self._token = token
        self.count = count

    def status(self) -> DependencyStatus:
        if self.count <= self.token.available:
            return DependencyStatus.OK
        return DependencyStatus.WAIT

    def lock(self) -> "Lock":
        return CounterTokenLock(self)

    @property
    def token(self):
        return self._token


class CounterToken(Token, FileSystemEventHandler):
    """File-based counter token"""
    TOKENS: Dict[str, "CounterToken"] = {}

    """Maps paths to instances"""
    VALUES = struct.Struct("<LL")

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

    def __init__(self, name: str, path: Path, count: int):
        """[summary]
        
        Arguments:
            path {Path} -- The file path of the token file
            count {int} -- Number of tokens (overrides previous definitions)
        """
        super().__init__()
        self.path = path
        self.ipc_lock = fasteners.InterProcessLock(
            path.with_suffix(path.suffix + ".lock")
        )
        self.lock = threading.Lock()
        self.name = name

        # Watched path
        self.watchedpath = str(path.absolute())
        self.watcher = ipcom().fswatch(self, str(path.parent.absolute()))
        logger.info("Watching %s", self.watchedpath)
        # When writing, to avoid re-reading the file we just wrote
        self.timestamp = 0

        # Set the new number of tokens
        with self.lock, self.ipc_lock:
            bytes = self.path.read_bytes() if self.path.is_file() else None

            if bytes:
                total, taken = CounterToken.VALUES.unpack(bytes)
                logger.info("Read token from %s: %d/%d", self.path, total, taken)
            else:
                taken = 0
                total = count

            if total != count:
                logger.warning("Changing number of tokens from %d to %d", total, count)
                total = count

            self._write(total, taken)

        # Set the number of available tokens
        self.available = total - taken

    def __str__(self):
        return "token[{}]".format(self.name)

    def on_modified(self, event):
        logger.debug("Watched path notification %s [watched %s]", event.src_path, self.watchedpath)
        if event.src_path == self.watchedpath:
            logger.debug("Watched path modified [%s]", self.path)
            timestamp = os.path.getmtime(self.path)
            if timestamp <= self.timestamp:
                logger.debug(
                    "Not reading token file [%f <= %f]", timestamp, self.timestamp
                )
            else:
                logger.debug("Trying to read...")
                with self.lock, self.ipc_lock:
                    logger.debug("Got lock...")
                    total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
                    available = total - taken

                if available != self.available:
                    logger.info(
                        "Counter token changed: %d to %d", self.available, available
                    )
                    self.available = available
                    # Notify jobs
                    for dependency in self.dependents:
                        dependency.check()

    def dependency(self, count):
        return CounterTokenDependency(self, count)

    def _write(self, total, taken):
        # Should only be called when locked
        self.path.write_bytes(CounterToken.VALUES.pack(total, taken))
        self.timestamp = os.path.getmtime(self.path)

        logger.debug(
            "Token: wrote %d/%d to %s [%d]", total, taken, self.path, self.timestamp
        )

    def _read(self):
        # Should only be called when locked
        self.timestamp = os.path.getmtime(self.path)
        total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
        self.available = total - taken
        logger.debug("Read token information from %s: %d/%d", self.path, total, taken)
        return total, taken

    def acquire(self, count):
        """Acquire"""
        with self.lock, self.ipc_lock:
            total, taken = self._read()
            logger.debug("Token state [acquire %d]: %d, %d", count, total, taken)
            if count + taken > total:
                logger.warning("No more token available - cannot lock")
                raise LockError("No token")

            taken += count

            self._write(total, taken)
            logger.debug("Token state [acquired %d]: %d, %d", count, total, taken)

    def release(self, count):
        """Release"""
        with self.lock, self.ipc_lock:
            logger.debug("Reading token information from %s", self.path)
            total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
            logger.debug("Token state [release %d]: %d, %d", count, total, taken)
            taken -= count
            if taken < 0:
                taken = 0
                logger.error("More tokens released that taken")
            self._write(total, taken)
            logger.debug("Token state [released %d]: %d, %d", count, total, taken)
            self.available = total - taken

        # Now, check
        for dependency in self.dependents:
            dependency.check()

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    os.register_at_fork(after_in_child=CounterToken.forkhandler)
