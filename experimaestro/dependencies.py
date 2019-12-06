"""Dependency between tasks and tokens"""

from pathlib import Path
from .utils import logger

class Dependency(): 
    # Dependency status
    STATUS_WAIT = 0
    STATUS_OK = 1
    STATUS_FAIL = 2

    def __init__(self, origin):
        self.origin = origin
        self.target = None
        self.currentstatus = Dependency.STATUS_WAIT

    def status(self) -> int:
        raise NotImplementedError()

    def check(self):
        status = self.status()
        if status != self.currentstatus:
            logger.info("Dependency %s is %s (was: %s)", self, status, self.currentstatus)
            self.target.dependencychanged(self, self.currentstatus, status)
            self.currentstatus = status

class Lock: pass

class LockError(Exception): pass

# --- Token dependencies

class Token(): pass

class CounterToken(Token):
    def __init__(self, path: Path, tokens: int=-1):
        self.path = path
        self.tokens = tokens

    def createDependency(self, count: int):
        return Dependency(self)