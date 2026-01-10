"""IPC utilities"""

from typing import Optional
from pathlib import Path
import os
import sys
import logging
from .utils import logger
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from watchdog.events import FileSystemEventHandler


class IPCom:
    """IPC async thread"""

    INSTANCE: Optional["IPCom"] = None
    # Testing mode: use polling observer with small interval
    TESTING_MODE: bool = False
    POLLING_INTERVAL: float = 0.01

    def __init__(self):
        if IPCom.TESTING_MODE:
            from watchdog.observers.polling import PollingObserver

            self.observer = PollingObserver(timeout=IPCom.POLLING_INTERVAL)
        else:
            self.observer = Observer()
        self.observer.start()
        self.pid = os.getpid()

    @classmethod
    def set_testing_mode(cls, enabled: bool = True, polling_interval: float = 0.01):
        """Enable testing mode with polling observer

        Args:
            enabled: Whether to enable testing mode
            polling_interval: Polling interval in seconds (default 0.01)
        """
        cls.TESTING_MODE = enabled
        cls.POLLING_INTERVAL = polling_interval
        # Reset instance to apply new settings
        if cls.INSTANCE is not None:
            cls.INSTANCE.observer.stop()
            cls.INSTANCE.observer.join(timeout=5)
            cls.INSTANCE = None

    def fswatch(
        self, watcher: FileSystemEventHandler, path: Path, recursive=False
    ) -> ObservedWatch:
        if not self.observer.is_alive():
            logging.error("Observer is not alive")

        return self.observer.schedule(
            watcher, str(path.absolute()), recursive=recursive
        )

    def fsunwatch(self, watcher):
        self.observer.unschedule(watcher)


def fork_childhandler():
    if IPCom.INSTANCE:
        logger.warning(
            "Removing IPCom instance in child process (watchers won't be copied)"
        )
        IPCom.INSTANCE = None


if sys.platform != "win32":
    os.register_at_fork(after_in_child=fork_childhandler)


def ipcom():
    # If multiprocessing, remove the IPCom instance if this does not
    # belong to our process
    if IPCom.INSTANCE is not None and IPCom.INSTANCE.pid != os.getpid():
        fork_childhandler()

    if IPCom.INSTANCE is None:
        IPCom.INSTANCE = IPCom()
        logger.info("Started IPCom instance (%s)", id(IPCom.INSTANCE))
        # IPCom.INSTANCE.start()
    return IPCom.INSTANCE
