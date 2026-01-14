"""IPC utilities"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Optional, Callable, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import os
import sys
import logging
import threading

from .utils import logger
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from watchdog.events import FileSystemEventHandler, FileSystemEvent

if TYPE_CHECKING:
    pass


# Type for async event handlers - should have on_created_async, on_deleted_async,
# on_modified_async, on_moved_async methods that take a FileSystemEvent and return a coroutine
AsyncEventHandler = Any  # Duck-typed: must have async on_*_async methods


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


class AsyncEventBridge:
    """Bridge watchdog filesystem events to asyncio event loop.

    This class allows filesystem events from the watchdog thread to be
    processed in an asyncio event loop. Handlers are registered for specific
    paths and events are posted via call_soon_threadsafe.

    Usage:
        bridge = AsyncEventBridge()
        bridge.set_loop(asyncio.get_running_loop())

        async def handler(event_type: str, src_path: str):
            print(f"Event: {event_type} on {src_path}")

        bridge.register_handler("/some/path", handler)
    """

    _instance: Optional["AsyncEventBridge"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "AsyncEventBridge":
        """Get or create the singleton AsyncEventBridge."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._loop = None
                cls._instance._handlers.clear()
            cls._instance = None

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        # Maps watched path -> list of async handlers
        self._handlers: Dict[str, List[AsyncEventHandler]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio loop to post events to.

        This must be called from the async context before events can be
        processed. Typically called during scheduler startup.

        Args:
            loop: The asyncio event loop to use
        """
        with self._lock:
            self._loop = loop
        logger.debug("AsyncEventBridge: set event loop %s", loop)

    def register_handler(
        self, path: str | Path, handler: AsyncEventHandler
    ) -> Callable[[], None]:
        """Register an async handler for filesystem events at a path.

        Args:
            path: The path being watched (used as key for routing events)
            handler: Async function called with (event_type, src_path, **kwargs)

        Returns:
            Unregister function - call to remove the handler
        """
        path_str = str(Path(path).absolute())

        with self._lock:
            if path_str not in self._handlers:
                self._handlers[path_str] = []
            self._handlers[path_str].append(handler)

        logger.debug("AsyncEventBridge: registered handler for %s", path_str)

        def unregister():
            with self._lock:
                if path_str in self._handlers:
                    try:
                        self._handlers[path_str].remove(handler)
                        if not self._handlers[path_str]:
                            del self._handlers[path_str]
                    except ValueError:
                        pass

        return unregister

    def post_event(
        self,
        watched_path: str | Path,
        event: FileSystemEvent,
    ) -> None:
        """Post a filesystem event from watchdog thread to asyncio loop.

        This method is thread-safe and can be called from the watchdog thread.

        Args:
            watched_path: The root path being watched (for handler lookup)
            event: The watchdog FileSystemEvent
        """
        with self._lock:
            loop = self._loop
            watched_path_str = str(Path(watched_path).absolute())
            handlers = self._handlers.get(watched_path_str, [])[:]

        if not handlers:
            return

        if loop is None:
            logger.debug(
                "AsyncEventBridge: no loop set, dropping event %s on %s",
                event.event_type,
                event.src_path,
            )
            return

        # Post to asyncio loop
        for handler in handlers:
            try:
                loop.call_soon_threadsafe(
                    lambda h=handler, e=event: asyncio.create_task(
                        self._call_handler(h, e)
                    )
                )
            except RuntimeError:
                # Loop might be closed
                logger.debug("AsyncEventBridge: loop closed, dropping event")

    async def _call_handler(
        self, handler: AsyncEventHandler, event: FileSystemEvent
    ) -> None:
        """Call the appropriate async handler method based on event type."""
        try:
            method_name = f"on_{event.event_type}_async"
            method = getattr(handler, method_name, None)
            if method is not None:
                result = method(event)
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            logger.exception(
                "AsyncEventBridge: error in handler for %s event on %s",
                event.event_type,
                event.src_path,
            )


class AsyncFileSystemEventHandler(FileSystemEventHandler):
    """Watchdog event handler that posts events to AsyncEventBridge.

    This handler is used with watchdog to bridge filesystem events to asyncio.
    Events are posted to the bridge which calls the corresponding async methods
    (on_created, on_deleted, on_modified, on_moved) on registered handlers.
    """

    def __init__(self, watched_path: str | Path, bridge: AsyncEventBridge):
        """Initialize the handler.

        Args:
            watched_path: The root path being watched
            bridge: The AsyncEventBridge to post events to
        """
        super().__init__()
        self.watched_path = str(Path(watched_path).absolute())
        self.bridge = bridge

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.bridge.post_event(self.watched_path, event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.bridge.post_event(self.watched_path, event)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.bridge.post_event(self.watched_path, event)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.bridge.post_event(self.watched_path, event)


class AsyncObservedWatch:
    """Wrapper for async filesystem watch that handles cleanup.

    Combines the watchdog ObservedWatch with the AsyncEventBridge unregister function.
    """

    def __init__(self, watch: ObservedWatch, unregister: Callable[[], None]):
        self.watch = watch
        self.unregister = unregister


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

    def async_fswatch(
        self, handler: AsyncEventHandler, path: Path, recursive=False
    ) -> "AsyncObservedWatch":
        """Watch a path and call an async handler for filesystem events.

        This is the async equivalent of fswatch(). It sets up:
        1. A watchdog observer for filesystem events
        2. An AsyncEventBridge to route events to the async handler

        Args:
            handler: Async function called with (event_type, src_path, **kwargs)
            path: Path to watch
            recursive: Whether to watch subdirectories

        Returns:
            AsyncObservedWatch that can be passed to fsunwatch()
        """
        if not self.observer.is_alive():
            logging.error("Observer is not alive")

        # Register async handler with the event bridge
        bridge = AsyncEventBridge.instance()
        unregister = bridge.register_handler(path, handler)

        # Create watchdog handler that posts to the bridge
        fs_handler = AsyncFileSystemEventHandler(path, bridge)

        # Schedule with watchdog
        watch = self.observer.schedule(
            fs_handler, str(path.absolute()), recursive=recursive
        )

        return AsyncObservedWatch(watch, unregister)

    def fsunwatch(self, watcher):
        if isinstance(watcher, AsyncObservedWatch):
            watcher.unregister()
            self.observer.unschedule(watcher.watch)
        else:
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
