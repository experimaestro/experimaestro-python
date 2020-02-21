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
            self._acquire()
            self._level += 1
        return self

    def release(self):
        if not self.detached and self._level == 1:
            self._release()
            self._level -= 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

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
        for lock in self.locks:
            logger.debug("Releasing %s", lock)
            lock.release()
