"""Adaptive file synchronization for remote monitoring

Provides background rsync with adaptive intervals based on change frequency.
- Minimum interval: 10 seconds (to avoid overloading)
- Maximum interval: 5 minutes (to ensure eventual updates)
- Adapts based on whether changes are detected
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("xpm.remote.sync")

# Sync interval limits (in seconds)
MIN_SYNC_INTERVAL = 10.0
MAX_SYNC_INTERVAL = 300.0  # 5 minutes
INITIAL_SYNC_INTERVAL = 15.0

# Interval adjustment factors
SPEEDUP_FACTOR = 0.7  # When changes detected, reduce interval
SLOWDOWN_FACTOR = 1.5  # When no changes, increase interval


class AdaptiveSynchronizer:
    """Background synchronizer with adaptive intervals

    Syncs a remote path periodically, adjusting the interval based on
    whether changes are detected.
    """

    def __init__(
        self,
        sync_func: Callable[[str], Optional[Path]],
        remote_path: str,
        name: str = "",
        on_sync_start: Optional[Callable[[], None]] = None,
        on_sync_complete: Optional[Callable[[Path], None]] = None,
        on_sync_error: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the synchronizer

        Args:
            sync_func: Function to call for syncing (e.g., state_provider.sync_path)
            remote_path: Remote path to sync
            name: Human-readable name for logging (e.g., "job:task_id" or "service:tensorboard")
            on_sync_start: Callback when sync starts
            on_sync_complete: Callback when sync completes with local path
            on_sync_error: Callback when sync fails with error message
        """
        self.sync_func = sync_func
        self.remote_path = remote_path
        self.name = name or remote_path
        self.on_sync_start = on_sync_start
        self.on_sync_complete = on_sync_complete
        self.on_sync_error = on_sync_error
        self._syncing = False

        self._interval = INITIAL_SYNC_INTERVAL
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track file modification times to detect changes
        self._last_mtimes: dict[str, float] = {}
        self._local_path: Optional[Path] = None

    @property
    def interval(self) -> float:
        """Current sync interval"""
        return self._interval

    @property
    def local_path(self) -> Optional[Path]:
        """Local path after sync (None if not synced yet)"""
        return self._local_path

    @property
    def syncing(self) -> bool:
        """Whether a sync is currently in progress"""
        return self._syncing

    def start(self) -> None:
        """Start background syncing"""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info(
            "[%s] Started adaptive sync (path: %s)", self.name, self.remote_path
        )

    def stop(self) -> None:
        """Stop background syncing"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("[%s] Stopped adaptive sync", self.name)

    def sync_now(self) -> Optional[Path]:
        """Perform an immediate sync (blocking)

        Returns:
            Local path if successful, None otherwise
        """
        return self._do_sync()

    def _sync_loop(self) -> None:
        """Background sync loop"""
        # Do initial sync immediately
        self._do_sync()

        while self._running:
            # Wait for interval or stop signal
            if self._stop_event.wait(timeout=self._interval):
                break  # Stop requested

            if not self._running:
                break

            self._do_sync()

    def _do_sync(self) -> Optional[Path]:
        """Perform a single sync operation"""
        try:
            self._syncing = True
            if self.on_sync_start:
                self.on_sync_start()

            logger.info("[%s] Starting rsync for %s...", self.name, self.remote_path)
            start_time = time.time()
            local_path = self.sync_func(self.remote_path)
            logger.info("[%s] sync_func returned: %s", self.name, local_path)
            sync_duration = time.time() - start_time

            if local_path:
                self._local_path = local_path

                # Check for changes
                has_changes = self._check_for_changes(local_path)

                # Adjust interval based on changes
                if has_changes:
                    # Changes detected - sync more frequently
                    self._interval = max(
                        MIN_SYNC_INTERVAL,
                        self._interval * SPEEDUP_FACTOR,
                    )
                    # But ensure we don't sync faster than the sync takes
                    self._interval = max(self._interval, sync_duration * 2)
                    logger.info(
                        "[%s] Rsync completed in %.1fs (changes detected), "
                        "next sync in %.1fs",
                        self.name,
                        sync_duration,
                        self._interval,
                    )
                else:
                    # No changes - sync less frequently
                    self._interval = min(
                        MAX_SYNC_INTERVAL,
                        self._interval * SLOWDOWN_FACTOR,
                    )
                    logger.info(
                        "[%s] Rsync completed in %.1fs (no changes), "
                        "next sync in %.1fs",
                        self.name,
                        sync_duration,
                        self._interval,
                    )

                if self.on_sync_complete:
                    self.on_sync_complete(local_path)

                return local_path
            else:
                logger.warning("[%s] Rsync returned no path", self.name)
                if self.on_sync_error:
                    self.on_sync_error("Sync returned no path")
                return None

        except Exception as e:
            logger.warning("[%s] Rsync failed: %s", self.name, e)
            # On error, slow down to avoid hammering
            self._interval = min(MAX_SYNC_INTERVAL, self._interval * 2)
            logger.info(
                "[%s] Next sync in %.1fs (after error)", self.name, self._interval
            )

            if self.on_sync_error:
                self.on_sync_error(str(e))

            return None
        finally:
            self._syncing = False

    def _check_for_changes(self, local_path: Path) -> bool:
        """Check if any files in the synced directory have changed

        Returns:
            True if changes detected, False otherwise
        """
        has_changes = False
        current_mtimes: dict[str, float] = {}

        try:
            # Check all files in the directory
            if local_path.is_dir():
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        key = str(file_path)
                        try:
                            mtime = file_path.stat().st_mtime
                            current_mtimes[key] = mtime

                            if key not in self._last_mtimes:
                                # New file
                                has_changes = True
                            elif self._last_mtimes[key] != mtime:
                                # File modified
                                has_changes = True
                        except OSError:
                            pass
            elif local_path.is_file():
                key = str(local_path)
                try:
                    mtime = local_path.stat().st_mtime
                    current_mtimes[key] = mtime

                    if key not in self._last_mtimes:
                        has_changes = True
                    elif self._last_mtimes[key] != mtime:
                        has_changes = True
                except OSError:
                    pass

            # Check for deleted files
            if set(self._last_mtimes.keys()) != set(current_mtimes.keys()):
                has_changes = True

            self._last_mtimes = current_mtimes

        except Exception as e:
            logger.warning("Error checking for changes: %s", e)

        return has_changes

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
