"""Adaptive file watching with polling fallback

This module provides a hybrid file watching system that combines watchdog
for immediate notifications with adaptive polling as a fallback.

Key components:
- PolledFile: State for a single watched file with adaptive reliability tracking
- FileWatcher: Manages watchdog + adaptive polling for reliable change detection

The system uses Polyak averaging (exponential moving average) to track:
1. watchdog_reliability: How reliably watchdog detects changes (0.0 to 1.0)
2. estimated_change_interval: Expected time between file changes

Adaptation behavior:
- When watchdog detects changes, reliability increases → poll less often
- When polling detects changes (watchdog missed), reliability decreases → poll more
- Polling interval = base_interval + (max - base) * reliability
- Base interval is half the estimated change interval

This ensures efficient resource usage when watchdog works well, while
providing reliable fallback when watchdog misses events (e.g., network
filesystems, certain filesystem types, or high-load scenarios).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler

logger = logging.getLogger("xpm.polling")


@dataclass
class PolledFile:
    """State for a file being watched with polling fallback

    Uses adaptive polling with Polyak averaging to track:
    - watchdog_reliability: How reliably watchdog detects changes (0.0-1.0)
    - estimated_change_interval: Expected time between file changes

    The polling interval adapts based on both metrics:
    - When watchdog is reliable, poll less frequently
    - When file changes rapidly, poll more frequently
    - Uses exponential moving average (Polyak) for smooth adaptation

    Attributes:
        path: Path to the file
        last_size: Last known file size (for detecting changes)
        last_change_time: time.time() when last change was detected
        poll_interval: Current polling interval (adapts based on activity)
        next_poll: time.time() when next poll should happen
        watchdog_reliability: 0.0 (unreliable) to 1.0 (reliable)
        estimated_change_interval: Polyak average of time between changes
    """

    path: Path
    last_size: int = 0
    last_change_time: float = field(default_factory=time.time)
    poll_interval: float = 0.5  # Start with fast polling
    next_poll: float = 0.0

    # Adaptive tracking using Polyak averaging
    watchdog_reliability: float = 0.5  # Start neutral
    estimated_change_interval: float = 5.0  # Initial estimate

    # Polling interval bounds
    MIN_INTERVAL: float = field(default=0.5, repr=False)
    MAX_INTERVAL: float = field(default=30.0, repr=False)

    # Polyak averaging parameters
    POLYAK_ALPHA: float = field(default=0.3, repr=False)  # Weight for new observations
    RELIABILITY_ALPHA: float = field(
        default=0.2, repr=False
    )  # Slower adaptation for reliability

    def schedule_next(self) -> None:
        """Schedule the next poll time"""
        self.next_poll = time.time() + self.poll_interval

    def _update_change_interval(self) -> None:
        """Update estimated change interval using Polyak averaging"""
        now = time.time()
        observed_interval = now - self.last_change_time
        # Clamp observed interval to reasonable bounds
        observed_interval = max(0.1, min(observed_interval, self.MAX_INTERVAL * 2))
        self.estimated_change_interval = (
            self.POLYAK_ALPHA * observed_interval
            + (1 - self.POLYAK_ALPHA) * self.estimated_change_interval
        )
        self.last_change_time = now

    def _compute_poll_interval(self) -> None:
        """Compute poll interval based on reliability and change frequency

        The interval is computed as:
        - Base: half the estimated change interval (catch changes promptly)
        - Scaled up by watchdog reliability (trust watchdog more = poll less)
        """
        # Base interval: poll at roughly half the change rate to catch changes
        base = max(self.MIN_INTERVAL, self.estimated_change_interval * 0.5)
        # Scale up based on watchdog reliability
        # reliability=0 -> poll_interval = base
        # reliability=1 -> poll_interval = MAX_INTERVAL
        self.poll_interval = min(
            base + (self.MAX_INTERVAL - base) * self.watchdog_reliability,
            self.MAX_INTERVAL,
        )

    def on_poll_detected_change(self) -> None:
        """Called when POLLING detected a change (watchdog missed it)

        Decreases watchdog reliability since it missed this change.
        Updates change interval estimate.
        """
        self._update_change_interval()
        # Decrease reliability (Polyak toward 0)
        self.watchdog_reliability = (
            self.RELIABILITY_ALPHA * 0.0
            + (1 - self.RELIABILITY_ALPHA) * self.watchdog_reliability
        )
        self._compute_poll_interval()
        self.schedule_next()

    def on_watchdog_detected_change(self) -> None:
        """Called when WATCHDOG detected a change

        Increases watchdog reliability since it caught this change.
        Updates change interval estimate.
        """
        self._update_change_interval()
        # Increase reliability (Polyak toward 1)
        self.watchdog_reliability = (
            self.RELIABILITY_ALPHA * 1.0
            + (1 - self.RELIABILITY_ALPHA) * self.watchdog_reliability
        )
        self._compute_poll_interval()
        self.schedule_next()

    def on_no_activity(self) -> None:
        """Called when no changes detected during poll

        When nothing changes, we can afford to poll less often.
        Reliability stays the same (no new information about watchdog).
        """
        # Increase estimated change interval slowly (file is quiet)
        self.estimated_change_interval = min(
            self.estimated_change_interval * 1.2, self.MAX_INTERVAL * 2
        )
        self._compute_poll_interval()
        self.schedule_next()

    def update_size(self) -> bool:
        """Update the last known size and return True if changed"""
        try:
            if not self.path.exists():
                return False
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


# Callback types
FileChangeCallback = Callable[[Path], None]
FileDeletedCallback = Callable[[Path], None]
FileFilter = Callable[[Path], bool]


class FileWatcher:
    """Hybrid file watcher combining watchdog with adaptive polling

    Provides reliable file change detection by using:
    1. Watchdog for immediate filesystem event notifications
    2. Adaptive polling as a fallback with Polyak-averaged reliability

    The system learns from experience:
    - When watchdog catches changes: reliability increases, poll less often
    - When polling catches changes: reliability decreases, poll more often
    - When files are quiet: change interval estimate increases, poll less

    Polling interval is computed from:
    - Estimated change frequency (poll at half the change rate)
    - Watchdog reliability (higher reliability = longer intervals)
    """

    def __init__(
        self,
        on_change: FileChangeCallback,
        on_deleted: FileDeletedCallback | None = None,
        file_filter: FileFilter | None = None,
        min_interval: float = 0.5,
        max_interval: float = 30.0,
    ):
        """Initialize the watcher

        Args:
            on_change: Callback for file modifications (watchdog or poll)
            on_deleted: Optional callback for file deletions
            file_filter: Optional filter to check if a path should be watched
            min_interval: Minimum polling interval in seconds
            max_interval: Maximum polling interval in seconds
        """
        self.on_change = on_change
        self.on_deleted = on_deleted
        self.file_filter = file_filter or (lambda p: True)
        self.min_interval = min_interval
        self.max_interval = max_interval

        self._files: dict[Path, PolledFile] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._watches: list = []
        self._handler: FileSystemEventHandler | None = None

    def add_file(self, path: Path) -> None:
        """Add a file to be watched

        Args:
            path: Path to the file to watch
        """
        if not self.file_filter(path):
            return

        with self._lock:
            if path in self._files:
                return  # Already tracking

            try:
                size = path.stat().st_size if path.exists() else 0
            except OSError:
                size = 0

            polled = PolledFile(
                path=path,
                last_size=size,
                poll_interval=self.min_interval,
            )
            polled.MIN_INTERVAL = self.min_interval
            polled.MAX_INTERVAL = self.max_interval
            polled.schedule_next()
            self._files[path] = polled
            logger.debug(
                "Started watching %s (poll interval=%.1fs)", path, polled.poll_interval
            )

    def remove_file(self, path: Path) -> None:
        """Stop watching a file

        Args:
            path: Path to stop watching
        """
        with self._lock:
            if path in self._files:
                del self._files[path]
                logger.debug("Stopped watching %s", path)

    def notify_change(self, path: Path) -> None:
        """Notify that a file was changed externally (e.g., from another watchdog)

        This signals that watchdog is working, so polling interval can increase.
        Does NOT call the callback (caller should handle that).

        Args:
            path: Path that changed
        """
        with self._lock:
            if path in self._files:
                polled = self._files[path]
                polled.update_size()
                polled.on_watchdog_detected_change()

    def watch_directory(self, directory: Path, recursive: bool = True) -> None:
        """Start watching a directory with watchdog

        Args:
            directory: Directory to watch
            recursive: Whether to watch subdirectories
        """
        from experimaestro.ipc import ipcom

        directory.mkdir(parents=True, exist_ok=True)

        if self._handler is None:
            self._handler = self._create_handler()

        ipc = ipcom()
        watch = ipc.fswatch(self._handler, directory, recursive=recursive)
        self._watches.append(watch)
        logger.debug("Started watchdog for %s (recursive=%s)", directory, recursive)

    def _create_handler(self) -> FileSystemEventHandler:
        """Create the watchdog event handler"""
        watcher = self

        class WatchdogHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if watcher.file_filter(path):
                    watcher._handle_file_change(path, from_watchdog=True)

            def on_created(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if watcher.file_filter(path):
                    # Add to tracked files and process
                    watcher.add_file(path)
                    watcher._handle_file_change(path, from_watchdog=True)

            def on_deleted(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if watcher.file_filter(path):
                    watcher.remove_file(path)
                    if watcher.on_deleted:
                        try:
                            watcher.on_deleted(path)
                        except Exception:
                            logger.exception("Error in deleted callback for %s", path)

        return WatchdogHandler()

    def _handle_file_change(self, path: Path, from_watchdog: bool = False) -> None:
        """Handle a file change (from watchdog or poll)

        Args:
            path: Path that changed
            from_watchdog: True if triggered by watchdog, False if by polling
        """
        with self._lock:
            polled = self._files.get(path)
            if polled:
                # Update size tracking
                polled.update_size()
                # Adjust polling based on source
                if from_watchdog:
                    # Watchdog is working - can poll less frequently
                    polled.on_watchdog_detected_change()
                else:
                    # Polling detected change - watchdog missed it
                    polled.on_poll_detected_change()

        # Call the change callback
        try:
            self.on_change(path)
        except Exception:
            logger.exception("Error in change callback for %s", path)

    def start(self) -> None:
        """Start the polling thread"""
        if self._thread is not None:
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="file-watcher-poll", daemon=True
        )
        self._thread.start()
        logger.debug("Started polling thread")

    def stop(self) -> None:
        """Stop watching and polling"""
        # Stop polling thread
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

        # Stop watchdog watches
        if self._watches:
            from experimaestro.ipc import ipcom

            ipc = ipcom()
            for watch in self._watches:
                try:
                    ipc.fsunwatch(watch)
                except Exception:
                    pass
            self._watches.clear()

        self._handler = None

        with self._lock:
            self._files.clear()
        logger.debug("Stopped file watcher")

    def __del__(self) -> None:
        """Ensure thread is stopped when object is garbage collected"""
        self.stop()

    def _poll_loop(self) -> None:
        """Main polling loop - runs in background thread"""
        while not self._stop_event.is_set():
            now = time.time()
            next_wake = now + 1.0  # Default wake time if no files

            with self._lock:
                files_snapshot = list(self._files.values())

            for polled in files_snapshot:
                if polled.next_poll <= now:
                    # Time to poll this file
                    changed = polled.update_size()
                    if changed:
                        # Polling detected change - watchdog missed it
                        polled.on_poll_detected_change()
                        try:
                            self.on_change(polled.path)
                        except Exception:
                            logger.exception(
                                "Error in poll callback for %s", polled.path
                            )
                    else:
                        polled.on_no_activity()

                # Track earliest next poll time
                if polled.next_poll < next_wake:
                    next_wake = polled.next_poll

            # Sleep until next poll is due (or stop event)
            sleep_time = max(0.1, next_wake - time.time())
            self._stop_event.wait(timeout=sleep_time)
