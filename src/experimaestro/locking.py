from asyncio import Lock
from .utils import logger


class LockError(Exception):
    pass


class Locks(Lock):
    """A set of locks that can be acquired/released together"""

    def __init__(self):
        super().__init__()
        self.locks = []

    def append(self, lock):
        """Add a lock to the collection"""
        self.locks.append(lock)

    async def acquire(self):
        """Acquire all locks in order"""
        if not self.locked():
            for lock in self.locks:
                await lock.acquire()
            self._acquired = True
        await super().acquire()
        return self

    def release(self):
        """Release all locks in reverse order"""
        if self.locked():
            # if not self.detached and self._acquired:
            logger.debug("Releasing %d locks", len(self.locks))
            # Release in reverse order to prevent deadlocks
            for lock in reversed(self.locks):
                logger.debug("[locks] Releasing %s", lock)
                lock.release()
            super().release()

    async def __aenter__(self):
        await super().__aenter__()
        return self
