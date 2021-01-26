"""Tokens are special types of dependency controlling the access to 
a computational resource (e.g. number of launched jobs, etc.)
"""

import sys
import psutil
from pathlib import Path
from experimaestro.core.objects import Config
import fasteners
import threading
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
        self.dependency.token.acquire(self.dependency)

    def _release(self):
        self.dependency.token.release(self.dependency)

    def __str__(self):
        return "Lock(%s)" % self.dependency


class CounterTokenDependency(Dependency):
    def __init__(self, token: "CounterToken", count: int):
        super().__init__(token)
        self._token = token
        self.count = count

    @property
    def name(self):
        return f"{self.target.identifier}.token"

    def status(self) -> DependencyStatus:
        if self.count <= self.token.available:
            return DependencyStatus.OK
        return DependencyStatus.WAIT

    def lock(self) -> "Lock":
        return CounterTokenLock(self)

    @property
    def token(self):
        return self._token


class TokenFile:
    """Represents a token file"""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rt") as fp:
            count, self.uri = [l.strip() for l in fp.readlines()]
            self.count = int(count)

    @staticmethod
    def create(dependency: CounterTokenDependency):
        path = dependency._token.path / dependency.name
        count = dependency.count
        uri = str(dependency.target.basepath)

        self = object.__new__(TokenFile)
        self.count = count
        self.uri = uri
        self.path = path
        logging.debug("Writing token file %s", path)
        with path.open("wt") as fp:
            fp.write(f"{str(count)}\n{uri}\n")
        return self

    def delete(self):
        logging.debug("Deleting token file %s", self.path)
        self.path.unlink()

    def watch(self):
        """Watch the matching process"""
        logger.debug(
            "Watching process for %s (%s, taken %d)", self.path, self.uri, self.count
        )
        path = Path(self.uri)
        lockpath = path.with_suffix(".lock")
        pidpath = path.with_suffix(".pid")

        # Watch for the job
        def run():
            logger.debug("Locking job lock path %s", lockpath)
            with fasteners.InterProcessLock(lockpath):
                if not pidpath.is_file():
                    logger.debug("Job already finished (no PID file)")
                else:
                    pid = int(pidpath.read_text())
                    logger.debug("Watching external job with PID %d", pid)
                    p = psutil.Process(pid)
                    p.wait()

                self.delete()

        threading.Thread(target=run).start()


class CounterToken(Token, FileSystemEventHandler):
    """File-based counter token

    To ensure recovery (server stopped for whatever reason), we use one folder
    per token; inside this folder:

    - token.lock is used for IPC locking
    - token.info contains the maximum number of tokens
    - TIMESTAMP.token contains (1) the number of tokens (2) the job URI
    """

    """Maps paths to instances"""
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

    def __init__(self, name: str, path: Path, count: int, force=True):
        """[summary]

        Arguments:
            path {Path} -- The file path of the token file
            count {int} -- Number of tokens (overrides previous definitions)
            force --   If the token has already been created, force to write the maximum
                       number of tokens
        """
        super().__init__()

        self.path = path
        self.path.mkdir(exist_ok=True, parents=True)

        self.cache: Dict[
            str,
        ] = {}

        self.infopath = path / "token.info"

        self.ipc_lock = fasteners.InterProcessLock(path / "token.lock")
        self.lock = threading.Lock()

        self.name = name

        # Set the new number of tokens
        with self.lock, self.ipc_lock:
            # Get the maximum number of tokens
            if force or not self.infopath.is_file():
                self.total = count
                self.infopath.write_text(str(count))

            self.timestamp = os.path.getmtime(self.path)
            self._update()

        # Watched path
        self.watchedpath = str(path.absolute())
        self.watcher = ipcom().fswatch(self, str(path.absolute()), recursive=True)
        logger.info("Watching %s", self.watchedpath)

    def _update(self):
        """Update the state by reading all the information from disk

        Assumes that the IPC lock is taken
        """
        logging.debug("Full token state update")
        self.total = int(self.infopath.read_text())
        old_cache = self.cache
        self.cache = {}
        self.available = self.total

        for path in self.path.glob("*.token"):
            tf = old_cache.get(path.name)
            if tf is None:
                tf = TokenFile(path)
                tf.watch()
                logging.debug("Read token file %s (%d)", path, tf.count)
            else:
                logging.debug(
                    "Token file already in cache %s (%d)", path.name, tf.count
                )

            self.cache[path.name] = tf
            self.available -= tf.count
        logging.debug("Full token state update finished (%d available)", self.available)

    def __str__(self):
        return "token[{}]".format(self.name)

    def on_deleted(self, event):
        logger.debug(
            "Deleted path notification %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )
        name = Path(event.src_path).name
        if name in self.cache:
            with self.lock:
                if name in self.cache:
                    fc = self.cache[name]
                    del self.cache[name]

                    self.available += fc.count
                    logger.debug(
                        "Getting back %d tokens (%d available)",
                        fc.count,
                        self.available,
                    )

            # Do not lock here (notify only)
            if self.available > 0:
                with self.dependents as dependents:
                    for dependency in dependents:
                        if self.available > 0:
                            dependency.check()

    def on_created(self, event):
        logger.debug(
            "Created path notification %s [watched %s]",
            event.src_path,
            self.watchedpath,
        )

        path = Path(event.src_path)

        try:
            if path.name.endswith(".token") and path.name not in self.cache:
                with self.lock:
                    if path.name not in self.cache:
                        tokenfile = TokenFile(path)
                        tokenfile.watch()
                        self.cache[path.name] = tokenfile
        except Exception:
            logger.exception("Uncaught exception in on_modified handler")
            raise

    def on_modified(self, event):
        try:
            logger.debug(
                "Watched path notification %s [watched %s]",
                event.src_path,
                self.watchedpath,
            )
            # logger.debug("%s", event)

            path = Path(event.src_path)

            if event.src_path == str(self.infopath):
                logger.debug("Token information modified")
                timestamp = os.path.getmtime(self.infopath)
                if timestamp <= self.timestamp:
                    logger.debug(
                        "Not reading token file [%f <= %f]", timestamp, self.timestamp
                    )

                total = int(self.infopath.read_text())
                delta = total - self.total
                self.total = total
                self.available += delta
                logger.debug(
                    "Token information modified: available %d, total %d",
                    self.available,
                    self.total,
                )

                if delta > 0 and self.available > 0:
                    with self.dependents as dependents:
                        for dependency in dependents:
                            dependency.check()

            elif path.name.endswith(".token") and path.name not in self.cache:
                with self.lock:
                    if path.name not in self.cache:
                        logger.debug("Token file not in cache %s", path.name)
                        tokenfile = TokenFile(path)
                        tokenfile.watch()
                        self.cache[path.name] = tokenfile
        except Exception:
            logger.exception("Uncaught exception in on_modified handler")
            raise

    def dependency(self, count):
        """Create a token dependency"""
        return CounterTokenDependency(self, count)

    def __call__(self, count, task: Config):
        """Create a token dependency and add it to the task"""
        return task.add_dependencies(self.dependency(count))

    def acquire(self, dependency: CounterTokenDependency):
        """Acquire requested token"""
        with self.lock, self.ipc_lock:
            self._update()
            if self.available < dependency.count:
                logger.warning(
                    "Not enough available (%d available, %d requested)",
                    self.available,
                    dependency.count,
                )
                raise LockError("No token")

            self.available -= dependency.count

            self.cache[dependency.name] = TokenFile.create(dependency)
            logger.debug(
                "Token state [acquired %d]: available %d, taken %d",
                dependency.count,
                self.available,
                self.total,
            )

    def release(self, dependency: CounterTokenDependency):
        """Release"""
        with self.lock, self.ipc_lock:
            self._update()

            tf = self.cache.get(dependency.name, None)
            if tf is None:
                logging.error(
                    "Could not find the taken token for %s (%s)",
                    dependency,
                    dependency.name,
                )
                return

            del self.cache[dependency.name]
            self.available += tf.count
            logging.debug("%s: available %d", self, self.available)
            tf.delete()

        if self.available > 0:
            with self.dependents as dependents:
                for dependency in dependents:
                    dependency.check()


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

        if self.available > 0:
            with self.dependents as dependents:
                for dependency in dependents:
                    if self.available > 0:
                        dependency.check()


if sys.platform != "win32":
    os.register_at_fork(after_in_child=CounterToken.forkhandler)
