"""Reconnection policy of the remote (SSH) state provider

When the connection drops, the client tries to reconnect at most
``max_retries`` times within a sliding time window of ``window`` seconds
(:class:`~experimaestro.settings.ReconnectSettings`). The attempts are counted
over the window and not per outage, so that a connection that keeps dropping
right after reconnecting is eventually given up as well. The delay before an
attempt doubles with each failed attempt of the outage, up to ``max_delay``.
"""

import threading
from collections import deque

from experimaestro.settings import ReconnectSettings


class ReconnectPolicy:
    """Decides whether and when to try reconnecting (thread-safe)"""

    def __init__(self, settings: ReconnectSettings | None = None):
        self.settings = settings if settings is not None else ReconnectSettings()
        #: Times of the attempts made within the window (monotonic clock)
        self._attempts: deque[float] = deque()
        #: Failed attempts of the current outage
        self.failures = 0
        self._lock = threading.Lock()

    @property
    def max_retries(self) -> int:
        return self.settings.max_retries

    def _prune(self, now: float) -> None:
        while self._attempts and self._attempts[0] <= now - self.settings.window:
            self._attempts.popleft()

    def attempts(self, now: float) -> int:
        """Number of attempts made within the window"""
        with self._lock:
            self._prune(now)
            return len(self._attempts)

    def next_delay(self, now: float) -> float | None:
        """Delay before the next attempt, or None when no attempt is left"""
        with self._lock:
            self._prune(now)
            if len(self._attempts) >= self.settings.max_retries:
                return None
            return min(self.settings.delay * 2**self.failures, self.settings.max_delay)

    def attempted(self, now: float, *, success: bool) -> None:
        """Records an attempt"""
        with self._lock:
            self._attempts.append(now)
            self.failures = 0 if success else self.failures + 1
