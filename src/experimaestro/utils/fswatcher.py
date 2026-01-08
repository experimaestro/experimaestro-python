"""File system watcher utilities

Workarounds for platform-specific file watching limitations.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("xpm.fswatcher")

# Marker file for macOS FSEvents workaround
# See https://github.com/experimaestro/experimaestro-python/issues/154
# FSEvents doesn't reliably detect SQLite WAL file changes
DB_CHANGE_MARKER = ".db_changed"
DB_CHANGE_MARKER_DEBOUNCE_SECONDS = 1.0


class FSEventsMarkerWorkaround:
    """Workaround for FSEvents not detecting SQLite WAL file changes on macOS

    On macOS, FSEvents doesn't reliably detect SQLite WAL file modifications.
    This class schedules a marker file touch after a delay. If the file system
    detects the database change before the delay (via cancel()), the touch is
    skipped.

    Multiple rapid DB writes reset the timer, so touch happens only once after
    DB_CHANGE_MARKER_DEBOUNCE_SECONDS of inactivity.

    Usage:
        workaround = FSEventsMarkerWorkaround(db_dir)

        # After each DB write:
        workaround.schedule_touch()

        # When FSEvents detects a change:
        workaround.cancel()

        # On cleanup:
        workaround.stop()

    See https://github.com/experimaestro/experimaestro-python/issues/154
    """

    def __init__(
        self,
        db_dir: Path,
        debounce_seconds: float = DB_CHANGE_MARKER_DEBOUNCE_SECONDS,
    ):
        """Initialize the workaround

        Args:
            db_dir: Directory containing the database files
            debounce_seconds: Delay before touching the marker file
        """
        self._db_dir = db_dir
        self._debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._enabled = sys.platform == "darwin"

    def schedule_touch(self) -> None:
        """Schedule a marker file touch after the delay

        First call schedules a touch in debounce_seconds. Subsequent calls
        within that window are ignored. After the touch happens, the next
        call can schedule another touch.

        This provides rate limiting: touch happens at most once per
        debounce_seconds interval.
        """
        if not self._enabled:
            return

        with self._lock:
            # If timer already running, ignore this event
            if self._timer is not None:
                logger.debug("Marker file touch already scheduled, ignoring")
                return

            # Schedule new touch
            self._timer = threading.Timer(
                self._debounce_seconds,
                self._do_touch,
            )
            self._timer.daemon = True
            self._timer.start()
            logger.debug("Scheduled marker file touch in %.1fs", self._debounce_seconds)

    def cancel(self) -> None:
        """Cancel any pending marker file touch

        Call this when FSEvents successfully detected a database change,
        meaning the workaround is not needed for this change.
        """
        if not self._enabled:
            return

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
                logger.debug("Cancelled pending marker file touch")

    def stop(self) -> None:
        """Stop and cleanup (call on shutdown)"""
        self.cancel()

    def _do_touch(self) -> None:
        """Actually touch the marker file (called by timer)"""
        with self._lock:
            self._timer = None

        if not self._enabled:
            return

        try:
            marker_path = self._db_dir / DB_CHANGE_MARKER
            marker_path.touch()
            logger.debug("Touched marker file %s for FSEvents workaround", marker_path)
        except Exception as e:
            logger.warning("Failed to touch marker file: %s", e)
