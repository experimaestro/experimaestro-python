"""Adaptive file watching with polling fallback

This module provides a hybrid file watching system that combines watchdog
for immediate notifications with adaptive polling as a fallback.

Key components:
- PolledFile: State for a single watched file
- FileWatcher: Manages watchdog + adaptive polling for reliable change detection

The system works as follows:
1. Watchdog provides immediate notifications when available
2. Polling runs in the background with adaptive intervals (0.5s to 30s)
3. When watchdog fires, polling interval is reset to minimum
4. When files are idle, polling interval gradually increases
5. This ensures reliable change detection even when watchdog misses events
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

    Attributes:
        path: Path to the file
        last_size: Last known file size (for detecting changes)
        last_activity: time.time() when last update was seen
        poll_interval: Current polling interval (adapts based on activity)
        next_poll: time.time() when next poll should happen
    """

    path: Path
    last_size: int = 0
    last_activity: float = field(default_factory=time.time)
    poll_interval: float = 0.5  # Start with fast polling
    next_poll: float = 0.0

    # Polling interval bounds
    MIN_INTERVAL: float = field(default=0.5, repr=False)
    MAX_INTERVAL: float = field(default=30.0, repr=False)
    INTERVAL_MULTIPLIER: float = field(default=1.5, repr=False)

    def schedule_next(self) -> None:
        """Schedule the next poll time"""
        self.next_poll = time.time() + self.poll_interval

    def on_activity(self) -> None:
        """Called when file has changed - reset to fast polling"""
        self.last_activity = time.time()
        self.poll_interval = self.MIN_INTERVAL
        self.schedule_next()

    def on_no_activity(self) -> None:
        """Called when no changes - increase interval up to max"""
        self.poll_interval = min(
            self.poll_interval * self.INTERVAL_MULTIPLIER, self.MAX_INTERVAL
        )
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


# Callback types
FileChangeCallback = Callable[[Path], None]
FileDeletedCallback = Callable[[Path], None]
FileFilter = Callable[[Path], bool]


class FileWatcher:
    """Hybrid file watcher combining watchdog with adaptive polling

    Provides reliable file change detection by using:
    1. Watchdog for immediate filesystem event notifications
    2. Adaptive polling as a fallback (0.5s to 30s based on activity)

    When watchdog works correctly, polling serves as verification.
    When watchdog misses events, polling catches the changes.

    The polling interval adapts:
    - Resets to minimum (0.5s) when activity is detected
    - Gradually increases to maximum (30s) when files are idle
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

        This resets the polling interval to minimum for faster follow-up polls,
        but does NOT call the callback (caller should handle that).

        Args:
            path: Path that changed
        """
        with self._lock:
            if path in self._files:
                polled = self._files[path]
                polled.update_size()
                polled.on_activity()

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
                # Reset to fast polling since file is active
                polled.on_activity()

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
                        polled.on_activity()
                        # Call callback (file changed but watchdog missed it)
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
