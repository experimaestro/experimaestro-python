from experimaestro.utils.asyncio import asyncThreadcheck
from .utils import logger


class Lock:
    """A lock"""

    def __init__(self):
        self._level = 0
        self.detached = False

    def detach(self):
        self.detached = True

    def acquire(self):
        if self._level == 0:
            self._level += 1
            self._acquire()
        return self

    def release(self):
        if not self.detached and self._level == 1:
            self._level -= 1
            self._release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    async def __aenter__(self):
        return await asyncThreadcheck("lock (aenter)", self.__enter__)

    async def __aexit__(self, *args):
        return await asyncThreadcheck("lock (aexit)", self.__exit__, *args)

    def _acquire(self):
        raise NotImplementedError()

    def _release(self):
        raise NotImplementedError()


class LockError(Exception):
    pass


class Locks(Lock):
    """A set of locks"""

    def __init__(self):
        super().__init__()
        self.locks = []

    def append(self, lock):
        self.locks.append(lock)

    def _acquire(self):
        for lock in self.locks:
            lock.acquire()

    def _release(self):
        logger.debug("Releasing %d locks", len(self.locks))
        for lock in self.locks:
            logger.debug("[locks] Releasing %s", lock)
            lock.release()
