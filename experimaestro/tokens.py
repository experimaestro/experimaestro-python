"""Tokens are special types of dependency controlling the access to 
a computational resource (e.g. number of launched jobs, etc.)
"""

from pathlib import Path
import fasteners
import threading
import struct
from .locking import Lock
from .dependencies import Dependency, DependencyStatus, Resource
from .utils import logger


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

class CounterToken(Token): 
    """File-based counter token"""

    VALUES = struct.Struct("<LL")

    def __init__(self, name: str, path: Path, count: int):
        """[summary]
        
        Arguments:
            path {Path} -- The file path of the token file
            count {int} -- Number of tokens (overrides previous definitions)
        """
        super().__init__()
        self.path = path
        self.lock = fasteners.InterProcessLock(path)
        self.name = name
        

        # Set the new number of tokens
        with self.lock:
            bytes = self.path.read_bytes()

            if bytes:
                logger.info("Reading token from %s", self.path)
                total, taken = CounterToken.VALUES.unpack(bytes)
            else:
                taken = 0
                total = count

            if total != count:
                logger.info("Changing number of tokens from %d to %d", total, count)
                total = count
                
            self.path.write_bytes(CounterToken.VALUES.pack(total, taken))

        # Set the number of available tokens
        self.available = total - taken

    def __str__(self):
        return "token[{}]".format(self.name)

    def dependency(self, count):
        return CounterTokenDependency(self, count)

    def acquire(self, count):
        """Acquire"""
        with self.lock:
            total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
            logger.info("Token state [acquire %d]: %d, %d", count, total, taken)
            if  count + taken > total:
                return False

            taken += count

            self.path.write_bytes(CounterToken.VALUES.pack(total, taken))
            self.available = total - taken
            logger.info("Token state [acquired %d]: %d, %d", count, total, taken)

            total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
            logger.info("Token state [acquired/w %d]: %d, %d", count, total, taken)

    def release(self, count):
        """Release"""
        with self.lock:
            total, taken = CounterToken.VALUES.unpack(self.path.read_bytes())
            logger.info("Token state [release %d]: %d, %d", count, total, taken)
            taken -= count
            if taken < 0:
                taken = 0
                logger.error("More tokens released that taken")
            
            self.path.write_bytes(CounterToken.VALUES.pack(total, taken))
            logger.info("Token state [released %d]: %d, %d", count, total, taken)
            self.available = total - taken

        if self.available > 0:
            # Notify jobs
            for dependency in self.dependents:
                dependency.check()
