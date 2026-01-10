"""IPC utilities"""

from enum import Enum
from typing import Optional
from pathlib import Path
import os
import sys
import logging
from .utils import logger
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from watchdog.events import FileSystemEventHandler


class WatcherType(str, Enum):
    """Available filesystem watcher types"""

    AUTO = "auto"
    """Use the best available watcher for the platform (default)"""

    POLLING = "polling"
    """Platform-independent polling (works on network mounts)"""

    INOTIFY = "inotify"
    """Linux inotify (Linux 2.6.13+ only)"""

    FSEVENTS = "fsevents"
    """macOS FSEvents (macOS only)"""

    KQUEUE = "kqueue"
    """BSD/macOS kqueue (less scalable for deep directories)"""

    WINDOWS = "windows"
    """Windows API (Windows only)"""


def _create_observer(watcher_type: WatcherType, polling_interval: float = 1.0):
    """Create an observer of the specified type

    Args:
        watcher_type: The type of filesystem watcher to use
        polling_interval: Polling interval in seconds (for polling watcher)

    Returns:
        An observer instance

    Raises:
        ImportError: If the requested watcher type is not available on this platform
        ValueError: If watcher_type is invalid
    """
    match watcher_type:
        case WatcherType.AUTO:
            return Observer()

        case WatcherType.POLLING:
            from watchdog.observers.polling import PollingObserver

            return PollingObserver(timeout=polling_interval)

        case WatcherType.INOTIFY:
            from watchdog.observers.inotify import InotifyObserver

            return InotifyObserver()

        case WatcherType.FSEVENTS:
            from watchdog.observers.fsevents import FSEventsObserver

            return FSEventsObserver()

        case WatcherType.KQUEUE:
            from watchdog.observers.kqueue import KqueueObserver

            return KqueueObserver()

        case WatcherType.WINDOWS:
            from watchdog.observers.read_directory_changes import WindowsApiObserver

            return WindowsApiObserver()

        case _:
            raise ValueError(f"Unknown watcher type: {watcher_type}")


class IPCom:
    """IPC async thread"""

    INSTANCE: Optional["IPCom"] = None
    # Testing mode: use polling observer with small interval
    TESTING_MODE: bool = False
    POLLING_INTERVAL: float = 0.01
    # Watcher type configuration
    WATCHER_TYPE: WatcherType = WatcherType.AUTO

    def __init__(self):
        if IPCom.TESTING_MODE:
            from watchdog.observers.polling import PollingObserver

            self.observer = PollingObserver(timeout=IPCom.POLLING_INTERVAL)
        else:
            self.observer = _create_observer(IPCom.WATCHER_TYPE, IPCom.POLLING_INTERVAL)
        self.observer.start()
        self.pid = os.getpid()

    @classmethod
    def set_watcher_type(cls, watcher_type: WatcherType, polling_interval: float = 1.0):
        """Set the filesystem watcher type

        Args:
            watcher_type: The type of watcher to use
            polling_interval: Polling interval in seconds (for polling watcher)

        Note:
            This must be called before the first IPCom instance is created.
            If an instance already exists, it will be reset.
        """
        cls.WATCHER_TYPE = watcher_type
        cls.POLLING_INTERVAL = polling_interval
        # Reset instance to apply new settings
        if cls.INSTANCE is not None:
            cls.INSTANCE.observer.stop()
            cls.INSTANCE.observer.join(timeout=5)
            cls.INSTANCE = None
        logger.info("Set watcher type to %s", watcher_type.value)

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
