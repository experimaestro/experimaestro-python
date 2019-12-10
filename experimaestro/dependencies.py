"""Dependency between tasks and tokens"""

from pathlib import Path
from enum import Enum
from .utils import logger

class DependencyStatus(Enum):
    """Waiting for dependency to be available"""
    WAIT = 0
    """Dependency is available"""
    OK = 1
    """Dependency won't be available"""
    FAIL = 2

class Dependency(): 
    # Dependency status

    def __init__(self, origin):
        # Origin and target are two resources
        self.origin = origin
        self.target = None
        self.currentstatus = DependencyStatus.WAIT

    def status(self) -> DependencyStatus:
        raise NotImplementedError()

    def lock(self) -> "Lock":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "Dep[{origin}->{target}]/{currentstatus}".format(**self.__dict__)

    def check(self):
        status = self.status()
        logger.debug("Dependency check: %s", self)
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