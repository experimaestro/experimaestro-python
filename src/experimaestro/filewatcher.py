"""Centralized file watching service

This module provides a unified API for all file monitoring in experimaestro,
replacing the scattered implementations across ipc.py, scheduler/polling.py,
and various other modules.

Key components:
- FileWatcherService: Singleton owning one watchdog Observer + one polling thread
- DirectoryWatch: Resource handle for directory watching with adaptive polling
- FileFollower: Async file follower (like tail -f)
- AsyncWatch: Handle for async filesystem watching
- PolledFile: Per-file adaptive state with Polyak averaging

Thread model:
1. Main thread -- user code
2. EventLoopThread -- asyncio loop for scheduler/locking
3. Watchdog thread -- owned by Observer (inside FileWatcherService)
4. Polling thread -- adaptive polling loop (inside FileWatcherService)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from io import TextIOWrapper
from typing import Any, Callable, TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

if TYPE_CHECKING:
    pass

logger = logging.getLogger("xpm.filewatcher")

# Type for async event handlers - duck-typed: must have async on_*_async methods
AsyncEventHandler = Any


# =============================================================================
# WatcherType (moved from ipc.py)
# =============================================================================


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
    """Create an observer of the specified type"""
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


# =============================================================================
# AdaptivePoller — generic adaptive polling with watchdog reliability tracking
# =============================================================================


@dataclass
class AdaptivePoller:
    """Adaptive polling scheduler with watchdog reliability tracking.

    Tracks how reliably watchdog detects changes and adjusts the polling
    interval accordingly using Polyak (exponential moving) averaging:

    - When watchdog is reliable → poll less frequently
    - When watchdog misses changes (poll detects them) → poll more frequently
    - When changes happen rapidly → poll more frequently
    - When nothing changes for a while → slow down polling

    This class is generic: it owns no I/O or check logic. The caller is
    responsible for performing the actual check and calling the appropriate
    notification method based on who detected the change.
    """

    min_interval: float = 0.5
    max_interval: float = 30.0

    # State
    watchdog_reliability: float = 0.5
    estimated_change_interval: float = 5.0
    poll_interval: float = 0.5
    next_poll: float = 0.0
    last_change_time: float = field(default_factory=time.time)

    # Polyak averaging parameters
    _polyak_alpha: float = field(default=0.3, repr=False)
    _reliability_alpha: float = field(default=0.2, repr=False)

    def schedule_next(self) -> None:
        """Schedule the next poll time."""
        self.next_poll = time.time() + self.poll_interval

    def _update_change_interval(self) -> None:
        """Update estimated change interval using Polyak averaging."""
        now = time.time()
        observed = now - self.last_change_time
        observed = max(0.1, min(observed, self.max_interval * 2))
        self.estimated_change_interval = (
            self._polyak_alpha * observed
            + (1 - self._polyak_alpha) * self.estimated_change_interval
        )
        self.last_change_time = now

    def _compute_poll_interval(self) -> None:
        """Compute poll interval based on reliability and change frequency."""
        base = max(self.min_interval, self.estimated_change_interval * 0.5)
        self.poll_interval = min(
            base + (self.max_interval - base) * self.watchdog_reliability,
            self.max_interval,
        )

    def on_poll_detected_change(self) -> None:
        """Called when POLLING detected a change (watchdog missed it)."""
        self._update_change_interval()
        self.watchdog_reliability = (
            self._reliability_alpha * 0.0
            + (1 - self._reliability_alpha) * self.watchdog_reliability
        )
        self._compute_poll_interval()
        self.schedule_next()

    def on_watchdog_detected_change(self) -> None:
        """Called when WATCHDOG detected a change."""
        self._update_change_interval()
        self.watchdog_reliability = (
            self._reliability_alpha * 1.0
            + (1 - self._reliability_alpha) * self.watchdog_reliability
        )
        self._compute_poll_interval()
        self.schedule_next()

    def on_no_activity(self) -> None:
        """Called when no changes detected during poll."""
        self.estimated_change_interval = min(
            self.estimated_change_interval * 1.2, self.max_interval * 2
        )
        self._compute_poll_interval()
        self.schedule_next()

    @property
    def is_due(self) -> bool:
        """Whether this poller is due for a check."""
        return time.time() >= self.next_poll


# =============================================================================
# PolledFile — file-specific polling using AdaptivePoller
# =============================================================================


@dataclass
class PolledFile:
    """State for a file being watched with adaptive polling fallback.

    Combines file-specific state (path, last_size) with an AdaptivePoller
    that manages the polling schedule and watchdog reliability tracking.
    """

    path: Path
    last_size: int = 0
    poller: AdaptivePoller = field(default_factory=AdaptivePoller)

    # --- Delegate properties for backward compatibility ---

    @property
    def poll_interval(self) -> float:
        return self.poller.poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        self.poller.poll_interval = value

    @property
    def next_poll(self) -> float:
        return self.poller.next_poll

    @next_poll.setter
    def next_poll(self, value: float) -> None:
        self.poller.next_poll = value

    @property
    def watchdog_reliability(self) -> float:
        return self.poller.watchdog_reliability

    @watchdog_reliability.setter
    def watchdog_reliability(self, value: float) -> None:
        self.poller.watchdog_reliability = value

    @property
    def estimated_change_interval(self) -> float:
        return self.poller.estimated_change_interval

    @estimated_change_interval.setter
    def estimated_change_interval(self, value: float) -> None:
        self.poller.estimated_change_interval = value

    @property
    def MIN_INTERVAL(self) -> float:
        return self.poller.min_interval

    @MIN_INTERVAL.setter
    def MIN_INTERVAL(self, value: float) -> None:
        self.poller.min_interval = value

    @property
    def MAX_INTERVAL(self) -> float:
        return self.poller.max_interval

    @MAX_INTERVAL.setter
    def MAX_INTERVAL(self, value: float) -> None:
        self.poller.max_interval = value

    def schedule_next(self) -> None:
        self.poller.schedule_next()

    def _compute_poll_interval(self) -> None:
        self.poller._compute_poll_interval()

    def on_poll_detected_change(self) -> None:
        """Called when POLLING detected a change (watchdog missed it)."""
        self.poller.on_poll_detected_change()

    def on_watchdog_detected_change(self) -> None:
        """Called when WATCHDOG detected a change."""
        self.poller.on_watchdog_detected_change()

    def on_no_activity(self) -> None:
        """Called when no changes detected during poll."""
        self.poller.on_no_activity()

    def update_size(self) -> bool | None:
        """Update the last known size.

        Returns:
            True if size changed, False if unchanged, None if file was deleted.
        """
        try:
            if not self.path.exists():
                return None
            current_size = self.path.stat().st_size
            if current_size != self.last_size:
                self.last_size = current_size
                return True
            return False
        except OSError:
            return False

    # Keep old method name for compatibility
    def on_activity(self) -> None:
        """Deprecated: use on_poll_detected_change or on_watchdog_detected_change"""
        self.on_poll_detected_change()


# =============================================================================
# Callback types
# =============================================================================

FileChangeCallback = Callable[[Path], None]
FileDeletedCallback = Callable[[Path], None]
FileFilter = Callable[[Path], bool]


# =============================================================================
# AsyncEventBridge (moved from ipc.py)
# =============================================================================


class AsyncEventBridge:
    """Bridge watchdog filesystem events to asyncio event loop.

    Routes events from the watchdog thread to async handlers via
    call_soon_threadsafe.
    """

    _instance: AsyncEventBridge | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> AsyncEventBridge:
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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._handlers: dict[str, list[AsyncEventHandler]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio loop to post events to."""
        with self._lock:
            self._loop = loop
        logger.debug("AsyncEventBridge: set event loop %s", loop)

    def register_handler(
        self, path: str | Path, handler: AsyncEventHandler
    ) -> Callable[[], None]:
        """Register an async handler for filesystem events at a path.

        Returns:
            Unregister function
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
        """Post a filesystem event from watchdog thread to asyncio loop."""
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

        for handler in handlers:
            try:
                loop.call_soon_threadsafe(
                    lambda h=handler, e=event: asyncio.create_task(
                        self._call_handler(h, e)
                    )
                )
            except RuntimeError:
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
    """Watchdog event handler that posts events to AsyncEventBridge."""

    def __init__(self, watched_path: str | Path, bridge: AsyncEventBridge):
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


# =============================================================================
# DirectoryWatch
# =============================================================================


class DirectoryWatch:
    """Resource handle for directory watching with watchdog + adaptive polling.

    Provides callbacks for file changes, creations, and deletions within
    a watched directory. Uses adaptive polling (via PolledFile) as fallback
    when watchdog misses events.

    Use as a context manager or call close() when done.
    """

    def __init__(
        self,
        service: FileWatcherService,
        path: Path,
        *,
        recursive: bool = False,
        file_filter: FileFilter | None = None,
        on_change: FileChangeCallback | None = None,
        on_created: FileChangeCallback | None = None,
        on_deleted: FileDeletedCallback | None = None,
        min_poll_interval: float = 0.5,
        max_poll_interval: float = 30.0,
        enable_tailing: bool = False,
        max_open_files: int = 128,
    ):
        self._service = service
        self._path = path
        self._recursive = recursive
        self._file_filter = file_filter or (lambda p: True)
        self._on_change = on_change
        self._on_created = on_created
        self._on_deleted = on_deleted
        self._min_poll_interval = min_poll_interval
        self._max_poll_interval = max_poll_interval

        self._files: dict[Path, PolledFile] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._watchdog_watch: ObservedWatch | None = None
        self._handler: _DirectoryWatchHandler | None = None

        # Tailed file pool for efficient line reading
        self._tailed_pool: TailedFilePool | None = (
            TailedFilePool(max_open=max_open_files) if enable_tailing else None
        )

        # Directory scanning poller: detects new files when watchdog misses
        # on_created events (e.g. NFS, GPFS, Lustre)
        self._dir_poller = AdaptivePoller(
            min_interval=min_poll_interval,
            max_interval=max_poll_interval,
        )
        self._dir_poller.schedule_next()
        self._known_files: set[Path] = set()

        # Set up watchdog
        self._setup_watchdog()

    def _setup_watchdog(self) -> None:
        """Register with the service's observer."""
        self._handler = _DirectoryWatchHandler(self)
        self._path.mkdir(parents=True, exist_ok=True)
        self._watchdog_watch = self._service._observer.schedule(
            self._handler, str(self._path.absolute()), recursive=self._recursive
        )
        logger.debug(
            "DirectoryWatch: watching %s (recursive=%s)", self._path, self._recursive
        )

    def add_file(self, path: Path) -> None:
        """Add a file to be polled."""
        if not self._file_filter(path):
            return

        # Track in known files so directory polling doesn't re-discover it
        self._known_files.add(path)

        with self._lock:
            if path in self._files:
                return
            try:
                size = path.stat().st_size if path.exists() else 0
            except OSError:
                size = 0

            polled = PolledFile(
                path=path,
                last_size=size,
                poller=AdaptivePoller(
                    min_interval=self._min_poll_interval,
                    max_interval=self._max_poll_interval,
                    poll_interval=self._min_poll_interval,
                ),
            )
            polled.schedule_next()
            self._files[path] = polled
            logger.debug("DirectoryWatch: tracking %s", path)

    def remove_file(self, path: Path) -> None:
        """Stop watching a file."""
        with self._lock:
            self._files.pop(path, None)
        if self._tailed_pool is not None:
            self._tailed_pool.remove(path)

    def notify_change(self, path: Path) -> None:
        """Notify that watchdog detected a change (updates reliability)."""
        with self._lock:
            if path in self._files:
                polled = self._files[path]
                polled.update_size()
                polled.on_watchdog_detected_change()

    def set_poll_interval(
        self,
        min_interval: float | None = None,
        max_interval: float | None = None,
    ) -> None:
        """Change polling interval bounds."""
        if min_interval is not None:
            self._min_poll_interval = min_interval
        if max_interval is not None:
            self._max_poll_interval = max_interval
        with self._lock:
            for polled in self._files.values():
                polled.MIN_INTERVAL = self._min_poll_interval
                polled.MAX_INTERVAL = self._max_poll_interval
                polled._compute_poll_interval()

    # --- Tailing API ---

    def read_new_lines(self, path: Path) -> list[str]:
        """Read complete new lines from a file via the tailed file pool.

        Requires enable_tailing=True at construction time.
        Returns list of complete lines (without trailing newline).
        Incomplete final lines (no trailing \\n) are NOT returned.
        """
        if self._tailed_pool is None:
            raise RuntimeError(
                "Tailing not enabled. Use enable_tailing=True when creating the watch."
            )
        return self._tailed_pool.read_new_lines(path)

    def set_tail_position(self, path: Path, pos: int) -> None:
        """Set the read position for a tailed file."""
        if self._tailed_pool is None:
            raise RuntimeError("Tailing not enabled.")
        self._tailed_pool.set_position(path, pos)

    def get_tail_position(self, path: Path) -> int:
        """Get the current read position for a tailed file."""
        if self._tailed_pool is None:
            raise RuntimeError("Tailing not enabled.")
        return self._tailed_pool.get_position(path)

    def remove_tail(self, path: Path) -> None:
        """Stop tailing a file and close its FD."""
        if self._tailed_pool is not None:
            self._tailed_pool.remove(path)

    def _handle_file_change(self, path: Path, from_watchdog: bool = False) -> None:
        """Handle a file change (from watchdog or poll)."""
        with self._lock:
            polled = self._files.get(path)
            if polled:
                polled.update_size()
                if from_watchdog:
                    polled.on_watchdog_detected_change()
                else:
                    polled.on_poll_detected_change()

        if self._on_change:
            try:
                self._on_change(path)
            except Exception:
                logger.exception("Error in change callback for %s", path)

    def poll(self) -> float:
        """Poll all tracked files and scan directory for new files.

        Returns time until next poll.
        """
        now = time.time()
        next_wake = now + 1.0

        # Directory scan: detect new files that watchdog missed
        if self._dir_poller.is_due:
            self._scan_for_new_files()

        if self._dir_poller.next_poll < next_wake:
            next_wake = self._dir_poller.next_poll

        with self._lock:
            files_snapshot = list(self._files.values())

        for polled in files_snapshot:
            if polled.poller.is_due:
                changed = polled.update_size()
                if changed is None:
                    # File was deleted — trigger deletion callback
                    self.remove_file(polled.path)
                    if self._on_deleted:
                        try:
                            self._on_deleted(polled.path)
                        except Exception:
                            logger.exception(
                                "Error in deleted callback for %s", polled.path
                            )
                elif changed:
                    polled.on_poll_detected_change()
                    if self._on_change:
                        try:
                            self._on_change(polled.path)
                        except Exception:
                            logger.exception(
                                "Error in poll callback for %s", polled.path
                            )
                else:
                    polled.on_no_activity()

            if polled.next_poll < next_wake:
                next_wake = polled.next_poll

        return max(0.1, next_wake - time.time())

    def _scan_for_new_files(self) -> None:
        """Scan the watched directory for new files not yet tracked.

        Uses the adaptive _dir_poller to adjust scan frequency based on
        whether watchdog is reliably detecting new file creation.
        On NFS/network filesystems, watchdog often misses on_created
        events, so the poller will gradually increase scan frequency.
        """
        try:
            if self._recursive:
                current_files = set(p for p in self._path.rglob("*") if p.is_file())
            else:
                current_files = set(p for p in self._path.iterdir() if p.is_file())
        except OSError:
            self._dir_poller.on_no_activity()
            return

        # Filter to matching files
        current_files = {p for p in current_files if self._file_filter(p)}

        new_files = current_files - self._known_files
        self._known_files = current_files

        if new_files:
            # Poll discovered new files → watchdog missed them
            self._dir_poller.on_poll_detected_change()
            for path in sorted(new_files):
                logger.debug("Directory poll discovered new file: %s", path)
                if self._on_created:
                    try:
                        self._on_created(path)
                    except Exception:
                        logger.exception("Error in created callback for %s", path)
                self.add_file(path)
                self._handle_file_change(path, from_watchdog=False)
        else:
            self._dir_poller.on_no_activity()

    def close(self) -> None:
        """Stop watching and release resources."""
        if self._closed:
            return
        self._closed = True

        if self._watchdog_watch is not None:
            try:
                self._service._observer.unschedule(self._watchdog_watch)
            except Exception:
                pass
            self._watchdog_watch = None

        with self._lock:
            self._files.clear()

        if self._tailed_pool is not None:
            self._tailed_pool.close_all()

        self._service._unregister_watch(self)
        logger.debug("DirectoryWatch: closed for %s", self._path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class _DirectoryWatchHandler(FileSystemEventHandler):
    """Internal watchdog handler for DirectoryWatch."""

    def __init__(self, watch: DirectoryWatch):
        super().__init__()
        self._watch_ref = weakref.ref(watch)

    def _get_watch(self) -> DirectoryWatch | None:
        return self._watch_ref()

    def on_modified(self, event):
        if event.is_directory:
            return
        watch = self._get_watch()
        if watch and watch._file_filter(Path(event.src_path)):
            logger.debug("Watchdog on_modified: %s", event.src_path)
            watch._handle_file_change(Path(event.src_path), from_watchdog=True)

    def on_created(self, event):
        if event.is_directory:
            return
        watch = self._get_watch()
        if watch and watch._file_filter(Path(event.src_path)):
            path = Path(event.src_path)
            logger.debug("Watchdog on_created: %s", path)
            # Watchdog detected creation → increase directory poller reliability
            watch._dir_poller.on_watchdog_detected_change()
            if watch._on_created:
                try:
                    watch._on_created(path)
                except Exception:
                    logger.exception("Error in created callback for %s", path)
            watch.add_file(path)
            watch._handle_file_change(path, from_watchdog=True)

    def on_deleted(self, event):
        if event.is_directory:
            return
        watch = self._get_watch()
        if watch and watch._file_filter(Path(event.src_path)):
            path = Path(event.src_path)
            watch.remove_file(path)
            if watch._on_deleted:
                try:
                    watch._on_deleted(path)
                except Exception:
                    logger.exception("Error in deleted callback for %s", path)


# =============================================================================
# AsyncWatch
# =============================================================================


class AsyncWatch:
    """Handle for async filesystem watching.

    Combines a watchdog ObservedWatch with an AsyncEventBridge unregister
    function. Close to release resources.
    """

    def __init__(
        self,
        service: FileWatcherService,
        watch: ObservedWatch,
        unregister: Callable[[], None],
    ):
        self._service = service
        self._watch = watch
        self._unregister = unregister
        self._closed = False

    def close(self) -> None:
        """Stop watching and release resources."""
        if self._closed:
            return
        self._closed = True

        self._unregister()
        try:
            self._service._observer.unschedule(self._watch)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


# =============================================================================
# FileFollower
# =============================================================================


class FileFollower:
    """Follows a file for new lines (like tail -f).

    Can be used as an async iterator yielding complete lines,
    or via read_new() / read_tail() for raw content access.
    """

    INITIAL_TAIL_SIZE = 256 * 1024  # 256KB

    def __init__(
        self,
        path: Path,
        *,
        poll_interval: float = 0.5,
        from_end: bool = False,
    ):
        self._path = path
        self._poll_interval = poll_interval
        self._position = 0
        self._size = 0
        self._closed = False
        self._line_buffer = ""

        self._update_size()

        if from_end:
            self._position = self._size

    def _update_size(self) -> None:
        """Update known file size."""
        try:
            self._size = self._path.stat().st_size
        except OSError:
            self._size = 0

    def read_tail(self, max_bytes: int = INITIAL_TAIL_SIZE) -> str:
        """Read the last N bytes of the file."""
        if not self._path.exists():
            return ""

        self._update_size()
        if self._size == 0:
            return ""

        try:
            with open(self._path, "r", errors="replace") as f:
                start_pos = max(0, self._size - max_bytes)
                f.seek(start_pos)
                if start_pos > 0:
                    f.readline()  # Skip partial line
                content = f.read()
                self._position = f.tell()
                return content
        except Exception:
            return ""

    def read_new(self) -> str:
        """Read any new content since last read."""
        if not self._path.exists():
            return ""

        self._update_size()

        # File was truncated
        if self._size < self._position:
            self._position = 0

        if self._position >= self._size:
            return ""

        try:
            with open(self._path, "r", errors="replace") as f:
                f.seek(self._position)
                content = f.read()
                self._position = f.tell()
                return content
        except Exception:
            return ""

    async def __aiter__(self):
        """Yield complete lines as they appear."""
        while not self._closed:
            new_content = self.read_new()
            if new_content:
                self._line_buffer += new_content
                while "\n" in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split("\n", 1)
                    yield line
            else:
                await asyncio.sleep(self._poll_interval)

    def close(self) -> None:
        """Stop following."""
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


# =============================================================================
# FileWatcherService
# =============================================================================


class FileWatcherService:
    """Singleton managing one watchdog Observer + one polling thread.

    Replaces IPCom entirely. All filesystem watching goes through this service.
    """

    _instance: FileWatcherService | None = None
    _instance_lock = threading.Lock()

    # Configuration (class-level, set before first instance)
    _watcher_type: WatcherType = WatcherType.AUTO
    _polling_interval: float = 1.0
    _testing_mode: bool = False

    @classmethod
    def instance(cls) -> FileWatcherService:
        """Get or create the singleton instance."""
        if cls._instance is not None and cls._instance._pid != os.getpid():
            # Fork detected
            logger.warning("Removing FileWatcherService instance in child process")
            cls._instance = None

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. For testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    @classmethod
    def configure(
        cls,
        watcher_type: WatcherType = WatcherType.AUTO,
        polling_interval: float = 1.0,
        testing_mode: bool = False,
    ) -> None:
        """Configure the service. Must be called before first use.

        If an instance already exists, it will be reset.
        """
        cls._watcher_type = watcher_type
        cls._polling_interval = polling_interval
        cls._testing_mode = testing_mode

        # Reset existing instance to apply new settings
        if cls._instance is not None:
            cls.reset()

        logger.info(
            "FileWatcherService configured: watcher_type=%s, polling_interval=%s, "
            "testing_mode=%s",
            watcher_type.value,
            polling_interval,
            testing_mode,
        )

    def __init__(self):
        self._pid = os.getpid()

        # Create observer
        if self._testing_mode:
            from watchdog.observers.polling import PollingObserver

            self._observer = PollingObserver(timeout=self._polling_interval)
        else:
            self._observer = _create_observer(
                self._watcher_type, self._polling_interval
            )
        self._observer.start()

        # Polling thread state
        self._directory_watches: list[DirectoryWatch] = []
        self._watches_lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        logger.debug("FileWatcherService started (pid=%d)", self._pid)

    def _ensure_poll_thread(self) -> None:
        """Start the polling thread if not already running."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="filewatcher-poll", daemon=True
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            next_sleep = 1.0

            with self._watches_lock:
                watches = list(self._directory_watches)

            for watch in watches:
                try:
                    t = watch.poll()
                    if t < next_sleep:
                        next_sleep = t
                except Exception:
                    logger.exception("Error in poll loop")

            self._stop_event.wait(timeout=next_sleep)

    def _unregister_watch(self, watch: DirectoryWatch) -> None:
        """Remove a DirectoryWatch from the polling list."""
        with self._watches_lock:
            try:
                self._directory_watches.remove(watch)
            except ValueError:
                pass

    def _shutdown(self) -> None:
        """Stop observer and polling thread."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
            self._poll_thread = None

        try:
            self._observer.stop()
            self._observer.join(timeout=5)
        except Exception:
            pass

        with self._watches_lock:
            # Copy and clear to avoid deadlock: clear() may drop the last
            # reference to a DirectoryWatch, triggering __del__ -> close() ->
            # _unregister_watch() which tries to acquire _watches_lock.
            old_watches = self._directory_watches[:]
            self._directory_watches.clear()
        # References released outside the lock
        del old_watches

    # --- Public API ---

    def watch_directory(
        self,
        path: Path,
        *,
        recursive: bool = False,
        file_filter: FileFilter | None = None,
        on_change: FileChangeCallback | None = None,
        on_created: FileChangeCallback | None = None,
        on_deleted: FileDeletedCallback | None = None,
        min_poll_interval: float = 0.5,
        max_poll_interval: float = 30.0,
        enable_tailing: bool = False,
        max_open_files: int = 128,
    ) -> DirectoryWatch:
        """Create a DirectoryWatch for the given path.

        Args:
            enable_tailing: If True, enable file tailing via TailedFilePool.
                Use read_new_lines() to read complete lines from watched files.
            max_open_files: Maximum open FDs for tailing (only when enable_tailing=True).

        Returns a DirectoryWatch handle. Call close() or use as context manager
        when done.
        """
        watch = DirectoryWatch(
            self,
            path,
            recursive=recursive,
            file_filter=file_filter,
            on_change=on_change,
            on_created=on_created,
            on_deleted=on_deleted,
            min_poll_interval=min_poll_interval,
            max_poll_interval=max_poll_interval,
            enable_tailing=enable_tailing,
            max_open_files=max_open_files,
        )

        with self._watches_lock:
            self._directory_watches.append(watch)

        self._ensure_poll_thread()
        return watch

    def follow_file(
        self,
        path: Path,
        *,
        poll_interval: float = 0.5,
        from_end: bool = False,
    ) -> FileFollower:
        """Create a FileFollower for the given path."""
        return FileFollower(path, poll_interval=poll_interval, from_end=from_end)

    def async_watch(
        self,
        handler: AsyncEventHandler,
        path: Path,
        recursive: bool = False,
    ) -> AsyncWatch:
        """Watch a path and call an async handler for filesystem events.

        This is the async equivalent of watch_directory(). Sets up:
        1. A watchdog observer for filesystem events
        2. An AsyncEventBridge to route events to the async handler

        Returns:
            AsyncWatch handle. Call close() when done.
        """
        if not self._observer.is_alive():
            logger.error("Observer is not alive")

        bridge = AsyncEventBridge.instance()
        unregister = bridge.register_handler(path, handler)

        fs_handler = AsyncFileSystemEventHandler(path, bridge)

        watch = self._observer.schedule(
            fs_handler, str(path.absolute()), recursive=recursive
        )

        return AsyncWatch(self, watch, unregister)

    def fswatch(
        self, watcher: FileSystemEventHandler, path: Path, recursive: bool = False
    ) -> ObservedWatch:
        """Low-level watchdog schedule. For callers that manage their own handler."""
        if not self._observer.is_alive():
            logger.error("Observer is not alive")
        return self._observer.schedule(
            watcher, str(path.absolute()), recursive=recursive
        )

    def fsunwatch(self, watch: ObservedWatch) -> None:
        """Low-level watchdog unschedule."""
        self._observer.unschedule(watch)


# =============================================================================
# TailedFilePool
# =============================================================================


@dataclass
class _TailedFile:
    """An open file handle kept for tailing."""

    handle: TextIOWrapper
    last_read_time: float


class TailedFilePool:
    """Manages a bounded pool of open file descriptors for tailing.

    When `max_open` is reached, the least-recently-read file is closed.
    Closed files can still be read on-demand (open/seek/read/close), but
    won't keep an FD open until they become active enough to re-enter
    the pool.
    """

    def __init__(self, max_open: int = 128):
        self._max_open = max_open
        self._open_files: dict[Path, _TailedFile] = {}  # path -> open handle
        self._positions: dict[Path, int] = {}  # all tracked positions
        self._lock = threading.Lock()

    def read_new_lines(self, path: Path) -> list[str]:
        """Read complete new lines from a file, keeping FD open if within limit.

        Returns list of complete lines (without trailing newline).
        Incomplete final lines (no trailing newline) are NOT returned
        (position is not advanced past them).
        """
        with self._lock:
            pos = self._positions.get(path, 0)
            tailed = self._open_files.get(path)

            # Check for file truncation or deletion
            try:
                file_size = path.stat().st_size
            except OSError:
                # File deleted or inaccessible
                self._close_file_locked(path)
                self._positions.pop(path, None)
                return []

            if file_size < pos:
                # File was truncated - reset position
                pos = 0
                if tailed:
                    self._close_file_locked(path)
                    tailed = None

            if tailed:
                # Read from open FD
                handle = tailed.handle
                try:
                    handle.seek(pos)
                    lines, new_pos = self._read_complete_lines(handle, pos)
                    self._positions[path] = new_pos
                    tailed.last_read_time = time.time()
                    return lines
                except OSError:
                    # FD went bad (e.g., file deleted and recreated)
                    self._close_file_locked(path)
                    self._positions.pop(path, None)
                    return []
            else:
                # One-shot read or promote to pool
                try:
                    handle = open(path, "r", errors="replace")  # noqa: SIM115
                except OSError:
                    self._positions.pop(path, None)
                    return []

                try:
                    handle.seek(pos)
                    lines, new_pos = self._read_complete_lines(handle, pos)
                    self._positions[path] = new_pos

                    # Try to keep this FD open if there's room
                    if len(self._open_files) < self._max_open:
                        self._open_files[path] = _TailedFile(
                            handle=handle,
                            last_read_time=time.time(),
                        )
                    else:
                        # Evict LRU and take its slot
                        self._evict_one_locked()
                        self._open_files[path] = _TailedFile(
                            handle=handle,
                            last_read_time=time.time(),
                        )
                    return lines
                except OSError:
                    handle.close()
                    self._positions.pop(path, None)
                    return []

    @staticmethod
    def _read_complete_lines(
        handle: TextIOWrapper, start_pos: int
    ) -> tuple[list[str], int]:
        """Read complete lines from handle. Returns (lines, new_position).

        Only lines ending with \\n are returned. The position is not
        advanced past an incomplete trailing line.
        """
        lines: list[str] = []
        last_complete_pos = start_pos

        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # Incomplete line - don't advance position
                break
            lines.append(line.rstrip("\n"))
            last_complete_pos = handle.tell()

        return lines, last_complete_pos

    def get_position(self, path: Path) -> int:
        """Get current read position for a path."""
        with self._lock:
            return self._positions.get(path, 0)

    def set_position(self, path: Path, pos: int) -> None:
        """Set read position (e.g. when replaying from beginning or skipping)."""
        with self._lock:
            self._positions[path] = pos
            # If FD is open, we don't need to close it - next read will seek

    def remove(self, path: Path) -> None:
        """Stop tracking and close FD for a path."""
        with self._lock:
            self._close_file_locked(path)
            self._positions.pop(path, None)

    def close_all(self) -> None:
        """Close all open file descriptors."""
        with self._lock:
            for path in list(self._open_files):
                self._close_file_locked(path)
            self._positions.clear()

    @property
    def open_count(self) -> int:
        """Number of currently open file descriptors."""
        with self._lock:
            return len(self._open_files)

    def _evict_one_locked(self) -> None:
        """Close least-recently-read file. Must hold self._lock."""
        if not self._open_files:
            return
        # Find the file with the oldest last_read_time
        oldest_path = min(
            self._open_files, key=lambda p: self._open_files[p].last_read_time
        )
        self._close_file_locked(oldest_path)

    def _close_file_locked(self, path: Path) -> None:
        """Close an open FD (position is preserved). Must hold self._lock."""
        tailed = self._open_files.pop(path, None)
        if tailed:
            try:
                tailed.handle.close()
            except OSError:
                pass


# =============================================================================
# Fork handler
# =============================================================================


def _fork_childhandler():
    if FileWatcherService._instance is not None:
        logger.warning(
            "Removing FileWatcherService instance in child process "
            "(watchers won't be copied)"
        )
        FileWatcherService._instance = None
    # Also reset AsyncEventBridge
    AsyncEventBridge._instance = None


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_fork_childhandler)
