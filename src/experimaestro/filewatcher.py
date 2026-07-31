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

# Upper bound on descriptors any single pool/source will hold. Half of a very
# large RLIMIT_NOFILE (some systems report ~1M) is not a useful target: past a
# few thousand the bookkeeping costs more than the descriptors save.
MAX_OPEN_FILES_CEILING = 8192
MIN_OPEN_FILES = 64


def default_max_open_files() -> int:
    """Descriptor budget for one pool/source: half the process limit.

    Half, rather than all, because the budget is shared with everything else
    experimaestro opens -- SSH connectors, RPyC sockets, job log handles.
    Clamped at both ends: RLIMIT_NOFILE is unbounded on some hosts and as low
    as 256 on a stock macOS.
    """
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):
        return 128

    if soft is None or soft < 0 or soft == getattr(resource, "RLIM_INFINITY", -1):
        return MAX_OPEN_FILES_CEILING
    return max(MIN_OPEN_FILES, min(soft // 2, MAX_OPEN_FILES_CEILING))


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

    #: While something changed within `hot_window` seconds, the interval is
    #: capped at `hot_max_interval`. None disables the cap entirely.
    #:
    #: Without it `min_interval` is not really the floor: the interval is
    #: driven by `estimated_change_interval * 0.5`, which starts at 2.5s and
    #: grows on every idle poll, so a freshly tracked file is polled seconds
    #: apart no matter how small `min_interval` is. That is the right
    #: behaviour for a dormant file and the wrong one for a job actively
    #: writing progress, which is exactly the case users watch.
    hot_window: float = 30.0
    hot_max_interval: float | None = None

    # State
    watchdog_reliability: float = 0.5
    estimated_change_interval: float = 5.0
    poll_interval: float = 0.5
    next_poll: float = 0.0
    last_change_time: float = field(default_factory=time.time)

    # Polyak averaging parameters
    _polyak_alpha: float = field(default=0.3, repr=False)
    _reliability_alpha: float = field(default=0.2, repr=False)

    @property
    def is_hot(self) -> bool:
        """Whether this file changed recently enough to warrant fast polling."""
        return (time.time() - self.last_change_time) < self.hot_window

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
        interval = min(
            base + (self.max_interval - base) * self.watchdog_reliability,
            self.max_interval,
        )
        if self.hot_max_interval is not None and self.is_hot:
            interval = min(interval, self.hot_max_interval)
        self.poll_interval = interval

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

#: Builds a ChangeSource for a DirectoryWatch. Passing these explicitly
#: replaces the default backends, which is how tests substitute simulated
#: sources (blind, coalescing, flaky) for the real ones.
ChangeSourceFactory = Callable[["DirectoryWatch"], "ChangeSource"]


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
# ChangeSource — pluggable change-notification backends
# =============================================================================


class ChangeSource:
    """A backend that tells a DirectoryWatch when watched files change.

    Two kinds coexist:

    * **event sources** (watchdog today) push notifications from the OS. They
      are fast when they work, but silently blind in some environments -- e.g.
      inotify reports nothing at all for a file appended to by another host on
      a shared filesystem, and macOS FSEvents coalesces rapid appends into a
      single notification delivered only once writing stops.
    * the **poll source**, which always works and is therefore the backstop.

    DirectoryWatch measures how often each event source actually beats the
    poller (see AdaptivePoller) and lets the poller back off accordingly. That
    is why no part of the system needs to know which filesystem it is on: an
    event source that never fires simply keeps its reliability at zero and the
    poller keeps doing the work.

    Sources report through the DirectoryWatch._source_* methods, which own all
    the bookkeeping (reliability, tracking, user callbacks). A source itself
    only decides *when* to report.
    """

    #: Identifies the source in logs and reliability bookkeeping.
    name: str = "source"

    #: False for the polling backstop. Detections through the poller prove
    #: that push notification *failed*, so they must not raise reliability.
    is_event_source: bool = True

    def __init__(self, watch: DirectoryWatch):
        # Weak, so a source (which the watchdog observer keeps alive through
        # its handler) can never keep its DirectoryWatch from being collected.
        self._watch_ref = weakref.ref(watch)
        self._stopped = False

    @property
    def _watch(self) -> DirectoryWatch | None:
        """The watch to report to, or None if this source must stay silent.

        Returning None once stopped is what makes stop() take effect
        immediately. Releasing the underlying OS resource can be asynchronous
        -- unscheduling a watchdog watch is deferred to the poll thread to
        avoid a finalizer deadlock -- so events may still arrive afterwards
        and must be dropped rather than reported.
        """
        if self._stopped:
            return None
        watch = self._watch_ref()
        if watch is None or watch._closed:
            return None
        return watch

    @classmethod
    def available(cls, service: FileWatcherService) -> bool:
        """Whether this source can be used at all on this platform/config."""
        return True

    def start(self) -> None:
        """Begin reporting changes."""

    def stop(self) -> None:
        """Stop reporting and release resources.

        Subclasses must call super().stop(): it is what silences the source.
        """
        self._stopped = True

    def add_file(self, path: Path) -> None:
        """Called when a file becomes tracked. Per-file sources arm here."""

    def remove_file(self, path: Path) -> None:
        """Called when a file stops being tracked."""

    def tick(self) -> float | None:
        """Advance a source that needs driving from the poll thread.

        Returns the number of seconds until it wants ticking again, or None if
        it is purely event-driven and never needs the poll thread.
        """
        return None


class WatchdogSource(ChangeSource):
    """Event source backed by the shared watchdog Observer."""

    name = "watchdog"

    def __init__(self, watch: DirectoryWatch):
        super().__init__(watch)
        self._handler: _WatchdogSourceHandler | None = None
        self._observed: ObservedWatch | None = None

    def start(self) -> None:
        watch = self._watch
        if watch is None:
            return
        self._handler = _WatchdogSourceHandler(self)
        self._observed = watch._service._observer.schedule(
            self._handler, str(watch._path.absolute()), recursive=watch._recursive
        )
        logger.debug(
            "WatchdogSource: watching %s (recursive=%s)", watch._path, watch._recursive
        )

    def stop(self) -> None:
        super().stop()
        if self._observed is None:
            return
        watch = self._watch_ref()
        if watch is not None:
            # Deferred to the poll thread to avoid a finalizer deadlock on the
            # watchdog observer lock (see _defer_unschedule).
            watch._service._defer_unschedule(self._observed)
        self._observed = None


class _WatchdogSourceHandler(FileSystemEventHandler):
    """Internal watchdog handler translating events into source reports."""

    def __init__(self, source: WatchdogSource):
        super().__init__()
        self._source = source

    def _resolve(self, event) -> tuple[WatchdogSource, DirectoryWatch, Path] | None:
        # Ignore events once the watch is closed: the underlying watchdog watch
        # is unscheduled asynchronously by the poll thread, so events may still
        # arrive briefly after close().
        if event.is_directory:
            return None
        watch = self._source._watch
        if watch is None:
            return None
        path = Path(event.src_path)
        if not watch._file_filter(path):
            return None
        return self._source, watch, path

    def on_modified(self, event):
        resolved = self._resolve(event)
        if resolved is not None:
            source, watch, path = resolved
            logger.debug("Watchdog on_modified: %s", path)
            watch._source_changed(source, path)

    def on_created(self, event):
        resolved = self._resolve(event)
        if resolved is not None:
            source, watch, path = resolved
            logger.debug("Watchdog on_created: %s", path)
            watch._source_created(source, path)

    def on_deleted(self, event):
        resolved = self._resolve(event)
        if resolved is not None:
            source, watch, path = resolved
            watch._source_deleted(source, path)


class KqueueSource(ChangeSource):
    """Event source using BSD/macOS kqueue on the watched files and directory.

    Registers EVFILT_VNODE with NOTE_WRITE|NOTE_EXTEND (plus DELETE/RENAME),
    which has two properties the alternatives lack on this platform:

    * it does not coalesce. macOS FSEvents merges rapid appends into a single
      notification delivered only once writing *stops*, which for a job
      appending progress lines means no notification while it matters.
    * it cannot be tripped by a local reader. inotify's mask includes IN_OPEN
      and IN_CLOSE_NOWRITE, so merely reading a file emits events; these flags
      fire only on modification.

    kqueue needs an open descriptor per watched file, so registration is capped
    at `max_registered` and evicts the least-recently-changed file. Anything
    colder simply falls back to the poll backstop, which is why the cap is a
    performance knob and not a correctness one.

    Descriptors are this source's own rather than borrowed from TailedFilePool:
    tailing is optional, and the pool evicts on *read* recency, which would let
    an unrelated component silently deregister a file that is still being
    written.
    """

    name = "kqueue"

    def __init__(self, watch: DirectoryWatch, max_registered: int | None = None):
        super().__init__(watch)
        self._max_registered = (
            watch._max_open_files if max_registered is None else max_registered
        )
        self._kq = None
        self._dir_fd: int | None = None
        self._fds: dict[Path, int] = {}
        self._paths: dict[int, Path] = {}
        self._last_active: dict[Path, float] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def available(cls, service: FileWatcherService) -> bool:
        import select

        return hasattr(select, "kqueue")

    # --- registration ---

    def _vnode_event(self, fd: int):
        import select

        return select.kevent(
            fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=(
                select.KQ_NOTE_WRITE
                | select.KQ_NOTE_EXTEND
                | select.KQ_NOTE_DELETE
                | select.KQ_NOTE_RENAME
            ),
        )

    def add_file(self, path: Path) -> None:
        if self._watch is None or self._kq is None:
            return
        with self._lock:
            if path in self._fds:
                self._last_active[path] = time.time()
                return
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                return  # not there yet; the directory event will bring it back
            try:
                self._kq.control([self._vnode_event(fd)], 0, 0)
            except OSError:
                os.close(fd)
                return
            self._fds[path] = fd
            self._paths[fd] = path
            self._last_active[path] = time.time()
            self._evict_if_needed()

    def remove_file(self, path: Path) -> None:
        with self._lock:
            self._close_locked(path)
            self._last_active.pop(path, None)

    def _evict_if_needed(self) -> None:
        """Drop the least-recently-changed registrations. Must hold _lock."""
        while len(self._fds) > self._max_registered:
            coldest = min(self._fds, key=lambda p: self._last_active.get(p, 0.0))
            logger.debug("KqueueSource: evicting %s (budget reached)", coldest)
            self._close_locked(coldest)

    def _close_locked(self, path: Path) -> None:
        """Deregistration is implicit: closing the FD removes its kevent."""
        fd = self._fds.pop(path, None)
        if fd is None:
            return
        self._paths.pop(fd, None)
        try:
            os.close(fd)
        except OSError:
            pass

    # --- lifecycle ---

    def start(self) -> None:
        import select

        watch = self._watch
        if watch is None:
            return
        self._kq = select.kqueue()
        try:
            self._dir_fd = os.open(watch._path, os.O_RDONLY)
            # NOTE_WRITE on a directory fires when an entry is added or
            # removed: this is how rotation to a new file is noticed.
            self._kq.control([self._vnode_event(self._dir_fd)], 0, 0)
        except OSError:
            logger.warning("KqueueSource: cannot watch directory %s", watch._path)
            self._dir_fd = None

        self._thread = threading.Thread(
            target=self._loop, name=f"kqueue-{watch._path.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        super().stop()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        with self._lock:
            for path in list(self._fds):
                self._close_locked(path)
            self._last_active.clear()
        if self._dir_fd is not None:
            try:
                os.close(self._dir_fd)
            except OSError:
                pass
            self._dir_fd = None
        if self._kq is not None:
            try:
                self._kq.close()
            except OSError:
                pass
            self._kq = None

    def _loop(self) -> None:
        import select

        while not self._stop_event.is_set():
            try:
                events = self._kq.control(None, 16, 0.5)
            except OSError:
                break
            except ValueError:
                break  # kqueue closed underneath us

            watch = self._watch
            if watch is None:
                break

            for event in events:
                if event.ident == self._dir_fd:
                    for path in watch._scan_directory():
                        watch._source_created(self, path)
                    continue

                with self._lock:
                    path = self._paths.get(event.ident)
                if path is None:
                    continue

                if event.fflags & (select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME):
                    watch._source_deleted(self, path)
                else:
                    with self._lock:
                        self._last_active[path] = time.time()
                    watch._source_changed(self, path)


class PollSource(ChangeSource):
    """The always-available backstop: adaptive stat-based polling.

    Owns the per-file PolledFile state and the directory-scan poller. Both
    adapt independently, because the two channels fail independently -- macOS
    FSEvents, for instance, reports file *creation* promptly while coalescing
    *appends* for seconds.
    """

    name = "poll"
    is_event_source = False

    def __init__(self, watch: DirectoryWatch):
        super().__init__(watch)
        self.files: dict[Path, PolledFile] = {}
        self.lock = threading.Lock()

        # Directory scanning poller: detects new files when an event source
        # misses creations (e.g. NFS, GPFS, Lustre).
        self.dir_poller = AdaptivePoller(
            min_interval=watch._min_poll_interval,
            max_interval=watch._max_poll_interval,
            hot_window=watch._hot_window,
            hot_max_interval=watch._hot_poll_interval,
        )
        self.dir_poller.schedule_next()

    def add_file(self, path: Path) -> None:
        watch = self._watch
        if watch is None:
            return

        with self.lock:
            if path in self.files:
                return
            try:
                size = path.stat().st_size if path.exists() else 0
            except OSError:
                size = 0

            polled = PolledFile(
                path=path,
                last_size=size,
                poller=AdaptivePoller(
                    min_interval=watch._min_poll_interval,
                    max_interval=watch._max_poll_interval,
                    poll_interval=watch._min_poll_interval,
                    hot_window=watch._hot_window,
                    hot_max_interval=watch._hot_poll_interval,
                ),
            )
            polled.schedule_next()
            self.files[path] = polled
            logger.debug("DirectoryWatch: tracking %s", path)

    def remove_file(self, path: Path) -> None:
        with self.lock:
            self.files.pop(path, None)

    def tick(self) -> float:
        """Poll all tracked files and scan the directory for new ones."""
        watch = self._watch
        if watch is None:
            return 1.0

        now = time.time()
        next_wake = now + 1.0

        if self.dir_poller.is_due:
            self._scan_for_new_files(watch)
        if self.dir_poller.next_poll < next_wake:
            next_wake = self.dir_poller.next_poll

        with self.lock:
            files_snapshot = list(self.files.values())

        for polled in files_snapshot:
            if polled.poller.is_due:
                changed = polled.update_size()
                if changed is None:
                    # File was deleted — trigger deletion callback
                    watch._source_deleted(self, polled.path)
                elif changed:
                    watch._source_changed(self, polled.path)
                else:
                    polled.on_no_activity()

            if polled.next_poll < next_wake:
                next_wake = polled.next_poll

        return max(0.1, next_wake - time.time())

    def _scan_for_new_files(self, watch: DirectoryWatch) -> None:
        """Discover files an event source failed to report as created.

        The enumeration itself lives on DirectoryWatch, since kqueue does the
        same scan when the directory reports a change. Only the adaptive
        scheduling of it belongs to the poller.
        """
        new_files = watch._scan_directory()
        if not new_files:
            self.dir_poller.on_no_activity()
            return

        # Poll discovered new files → the event sources missed them. Recorded
        # once for the batch, unlike the per-file event-source case.
        self.dir_poller.on_poll_detected_change()
        for path in new_files:
            logger.debug("Directory poll discovered new file: %s", path)
            watch._source_created(self, path)


# =============================================================================
# DirectoryWatch
# =============================================================================


class DirectoryWatch:
    """Resource handle for directory watching.

    Provides callbacks for file changes, creations, and deletions within a
    watched directory. The actual detection is delegated to a list of
    ChangeSource backends: an always-present PollSource backstop plus whatever
    event sources are available. DirectoryWatch owns the bookkeeping the
    sources feed -- reliability tracking, file tracking, user callbacks -- so
    adding a backend does not touch this class.

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
        hot_poll_interval: float = 0.2,
        hot_window: float = 30.0,
        enable_tailing: bool = False,
        max_open_files: int | None = None,
        source_factories: list[ChangeSourceFactory] | None = None,
    ):
        # Set first: __del__ -> close() must be safe even if __init__ raises.
        self._closed = False
        self._sources: list[ChangeSource] = []
        self._poll_source: PollSource | None = None

        self._service = service
        self._path = path
        self._recursive = recursive
        self._file_filter = file_filter or (lambda p: True)
        self._on_change = on_change
        self._on_created = on_created
        self._on_deleted = on_deleted
        self._min_poll_interval = min_poll_interval
        self._max_poll_interval = max_poll_interval
        self._hot_window = hot_window
        # A file cannot be considered hot and yet polled slower than the
        # configured ceiling.
        self._hot_poll_interval = min(hot_poll_interval, max_poll_interval)
        self._max_open_files = (
            default_max_open_files() if max_open_files is None else max_open_files
        )

        self._lock = threading.RLock()
        self._known_files: set[Path] = set()

        # Tailed file pool for efficient line reading
        self._tailed_pool: TailedFilePool | None = (
            TailedFilePool(max_open=self._max_open_files) if enable_tailing else None
        )

        self._poll_source = PollSource(self)
        self._setup_sources(source_factories)

    def _setup_sources(
        self, source_factories: list[ChangeSourceFactory] | None
    ) -> None:
        """Build and start the change sources.

        The PollSource backstop is always first: it is what keeps the watch
        correct when every event source turns out to be blind.
        """
        # The watched directory must exist before any source looks at it, and
        # that is the watch's concern rather than any one backend's.
        self._path.mkdir(parents=True, exist_ok=True)

        sources: list[ChangeSource] = [self._poll_source]
        if source_factories is None:
            # kqueue where available, watchdog otherwise. They are not stacked:
            # on macOS the watchdog backend is FSEvents, whose append channel
            # kqueue strictly dominates, and running both would double the
            # callbacks for no gain.
            if KqueueSource.available(self._service):
                sources.append(KqueueSource(self))
            elif WatchdogSource.available(self._service):
                sources.append(WatchdogSource(self))
        else:
            sources.extend(factory(self) for factory in source_factories)

        self._sources = sources
        for source in sources:
            source.start()
        logger.debug(
            "DirectoryWatch: watching %s (recursive=%s) via %s",
            self._path,
            self._recursive,
            ", ".join(s.name for s in sources),
        )

    # --- State owned by the poll source, exposed for compatibility ---

    @property
    def _files(self) -> dict[Path, PolledFile]:
        return self._poll_source.files

    @property
    def _dir_poller(self) -> AdaptivePoller:
        return self._poll_source.dir_poller

    # --- Directory enumeration, shared by every discovering source ---

    def _scan_directory(self) -> list[Path]:
        """Enumerate the directory, returning matching files not yet known.

        Shared because more than one source discovers creations: the poll
        backstop scans on an adaptive schedule, while kqueue scans only when
        the directory itself reports a change. Sorted so that rotated files
        (``…-1.jsonl``, ``…-2.jsonl``) are always reported in order, which
        consumers rely on to track the newest file per entity.
        """
        try:
            if self._recursive:
                current = {p for p in self._path.rglob("*") if p.is_file()}
            else:
                current = {p for p in self._path.iterdir() if p.is_file()}
        except OSError:
            return []

        current = {p for p in current if self._file_filter(p)}
        with self._lock:
            new_files = current - self._known_files
            self._known_files = current
        return sorted(new_files)

    # --- Source reporting API ---

    def _source_changed(self, source: ChangeSource, path: Path) -> None:
        """A source reports that `path` changed."""
        if self._closed:
            return
        polled = self._poll_source.files.get(path)
        if polled:
            polled.update_size()
            if source.is_event_source:
                polled.on_watchdog_detected_change()
            else:
                polled.on_poll_detected_change()

        if self._on_change:
            try:
                self._on_change(path)
            except Exception:
                logger.exception("Error in change callback for %s", path)

    def _source_created(self, source: ChangeSource, path: Path) -> None:
        """A source reports that `path` was created."""
        if self._closed:
            return
        if source.is_event_source:
            # The poll source records its own directory-scan reliability once
            # per batch, so only event sources credit themselves here.
            self._poll_source.dir_poller.on_watchdog_detected_change()

        if self._on_created:
            try:
                self._on_created(path)
            except Exception:
                logger.exception("Error in created callback for %s", path)

        self.add_file(path)
        self._source_changed(source, path)

    def _source_deleted(self, source: ChangeSource, path: Path) -> None:
        """A source reports that `path` was deleted."""
        self.remove_file(path)
        if self._on_deleted:
            try:
                self._on_deleted(path)
            except Exception:
                logger.exception("Error in deleted callback for %s", path)

    # --- Public API ---

    def add_file(self, path: Path) -> None:
        """Add a file to be watched."""
        if not self._file_filter(path):
            return
        # Known, so a directory scan does not re-report it as newly created.
        with self._lock:
            self._known_files.add(path)
        for source in self._sources:
            source.add_file(path)
        self._service.wake()

    def remove_file(self, path: Path) -> None:
        """Stop watching a file."""
        for source in self._sources:
            source.remove_file(path)
        if self._tailed_pool is not None:
            self._tailed_pool.remove(path)

    def notify_change(self, path: Path) -> None:
        """Notify that an event source detected a change (updates reliability).

        Does not fire the change callback: this is the low-level reliability
        hook, for callers driving detection themselves.
        """
        with self._poll_source.lock:
            polled = self._poll_source.files.get(path)
            if polled is not None:
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
        with self._poll_source.lock:
            for polled in self._poll_source.files.values():
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

    def poll(self) -> float:
        """Drive every source that needs ticking from the poll thread.

        Returns the number of seconds until the next tick is wanted.
        """
        next_sleep = 1.0
        for source in list(self._sources):
            try:
                wanted = source.tick()
            except Exception:
                logger.exception("Error ticking change source %s", source.name)
                continue
            if wanted is not None and wanted < next_sleep:
                next_sleep = wanted
        return max(0.1, next_sleep)

    def close(self) -> None:
        """Stop watching and release resources."""
        if self._closed:
            return
        self._closed = True

        for source in self._sources:
            try:
                source.stop()
            except Exception:
                logger.exception("Error stopping change source %s", source.name)

        if self._poll_source is not None:
            self._poll_source.files.clear()

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
        # Deferred to the poll thread to avoid a finalizer deadlock on the
        # watchdog observer lock (see _defer_unschedule).
        self._service._defer_unschedule(self._watch)

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
        # Interrupts the poll loop's sleep. Without it a watch or file added
        # while the loop is idling waits out the remaining sleep -- up to a
        # second -- before it is polled for the first time.
        self._wake_event = threading.Event()

        # Watchdog watches queued for unscheduling by the poll thread.
        # See _defer_unschedule for why this indirection exists.
        self._pending_unschedule: list[ObservedWatch] = []
        self._pending_lock = threading.Lock()

        # Start the poll thread eagerly so deferred unschedules are always
        # drained, even when only async_watch()/fswatch() are used.
        self._ensure_poll_thread()

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

    def _defer_unschedule(self, watch: ObservedWatch) -> None:
        """Queue a watchdog watch to be unscheduled by the poll thread.

        Called from close()/__del__ paths. Calling observer.unschedule()
        inline can deadlock: a cyclic-GC finalizer (e.g. LockState.__del__)
        may run inside another thread's emitter bootstrap while
        observer._lock is held by schedule(), and unschedule() blocks on that
        same lock. Routing all unschedules through the poll thread keeps them
        out of finalizer contexts and breaks the cycle.
        """
        with self._pending_lock:
            self._pending_unschedule.append(watch)

    def _drain_pending_unschedule(self) -> None:
        """Unschedule watches queued via _defer_unschedule."""
        with self._pending_lock:
            if not self._pending_unschedule:
                return
            pending = self._pending_unschedule
            self._pending_unschedule = []
        for watch in pending:
            try:
                self._observer.unschedule(watch)
            except Exception:
                pass

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            self._drain_pending_unschedule()
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

            self._wake_event.wait(timeout=next_sleep)
            self._wake_event.clear()

    def _unregister_watch(self, watch: DirectoryWatch) -> None:
        """Remove a DirectoryWatch from the polling list."""
        with self._watches_lock:
            try:
                self._directory_watches.remove(watch)
            except ValueError:
                pass

    def wake(self) -> None:
        """Interrupt the poll loop's sleep so new work is picked up at once."""
        self._wake_event.set()

    def _shutdown(self) -> None:
        """Stop observer and polling thread."""
        self._stop_event.set()
        self._wake_event.set()
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
        hot_poll_interval: float = 0.2,
        hot_window: float = 30.0,
        enable_tailing: bool = False,
        max_open_files: int | None = None,
        source_factories: list[ChangeSourceFactory] | None = None,
    ) -> DirectoryWatch:
        """Create a DirectoryWatch for the given path.

        Args:
            enable_tailing: If True, enable file tailing via TailedFilePool.
                Use read_new_lines() to read complete lines from watched files.
            hot_poll_interval: Ceiling on the poll interval for a file that
                changed within hot_window seconds. Without it min_poll_interval
                is not the effective floor -- see AdaptivePoller.hot_window.
            hot_window: How long after a change a file counts as hot.
            max_open_files: Maximum open FDs for tailing and for kqueue
                registration. Defaults to half the process limit.
            source_factories: Replaces the default change sources. The
                PollSource backstop is always present regardless; these are the
                event sources layered on top. Mainly for tests that simulate a
                blind or coalescing backend.

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
            hot_poll_interval=hot_poll_interval,
            hot_window=hot_window,
            enable_tailing=enable_tailing,
            max_open_files=max_open_files,
            source_factories=source_factories,
        )

        with self._watches_lock:
            self._directory_watches.append(watch)

        self._ensure_poll_thread()
        self.wake()
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
