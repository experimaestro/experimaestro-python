"""Dynamic resources and dependencies that can be waited on asynchronously."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from weakref import WeakSet

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.locking import Lock

logger = logging.getLogger("xpm.dynamic")

# Polling configuration
POLL_INTERVAL_INITIAL = 0.1  # seconds
POLL_INTERVAL_MAX = float(os.environ.get("XPM_POLL_INTERVAL_MAX", "30.0"))
POLL_INTERVAL_MULTIPLIER = 1.5


class DynamicResource(ABC):
    """Abstract base class for resources that can be waited on asynchronously.

    Subclasses must implement async_wait() which waits until the resource
    state may have changed.
    """

    @abstractmethod
    async def async_wait(self, timeout: float = 0) -> bool:
        """Wait asynchronously until the resource state may have changed.

        Args:
            timeout: Maximum time to wait in seconds (0 = wait indefinitely)

        Returns:
            True if notified of a change, False if timed out
        """
        ...

    @abstractmethod
    def refresh_state(self) -> None:
        """Refresh resource state from underlying storage.

        Called by ResourcePoller. Should update internal state and
        notify waiters as appropriate.
        """
        ...


class ResourcePoller(threading.Thread):
    """Single thread that polls all DynamicResource instances.

    This consolidates polling into one thread instead of creating threads
    per wait operation. Resources register themselves and get polled at
    intervals determined by exponential backoff.
    """

    _instance: Optional["ResourcePoller"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ResourcePoller":
        """Get or create the singleton ResourcePoller."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                # Clear waiters to stop processing
                with cls._instance._lock:
                    cls._instance._waiters.clear()
                    cls._instance._resources.clear()
            cls._instance = None

    def __init__(self):
        super().__init__(daemon=True, name="ResourcePoller")
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        # Resources waiting to be polled (weak references)
        self._resources: WeakSet[DynamicResource] = WeakSet()

        # Async waiters: resource_id -> list of (asyncio.Event, loop, deadline)
        self._waiters: dict[
            int, list[tuple[asyncio.Event, asyncio.AbstractEventLoop, Optional[float]]]
        ] = {}

    def register(
        self,
        resource: DynamicResource,
        loop: asyncio.AbstractEventLoop,
        timeout: float = 0,
    ) -> asyncio.Event:
        """Register a resource for polling and return an event to wait on.

        Args:
            resource: The resource to poll
            loop: The asyncio event loop to notify
            timeout: Timeout in seconds (0 = no timeout)

        Returns:
            asyncio.Event that will be set when resource changes or timeout
        """
        event = asyncio.Event()
        deadline = time.time() + timeout if timeout > 0 else None
        resource_id = id(resource)

        with self._lock:
            self._resources.add(resource)
            if resource_id not in self._waiters:
                self._waiters[resource_id] = []
            self._waiters[resource_id].append((event, loop, deadline))
            self._cv.notify()

        return event

    def _notify_waiters(self, resource: DynamicResource) -> None:
        """Notify all waiters for a resource."""
        resource_id = id(resource)
        with self._lock:
            waiters = self._waiters.pop(resource_id, [])

        for event, loop, _ in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # Loop might be closed
                pass

    def notify(self, resource: DynamicResource) -> None:
        """Notify that a resource's state has changed.

        Called by resources when they detect a state change (e.g., via watchdog).
        This wakes up any waiters for this resource immediately.
        """
        self._notify_waiters(resource)

    def _check_timeouts(self) -> Optional[float]:
        """Check for timed out waiters and return time until next timeout."""
        now = time.time()
        next_timeout = float("inf")

        with self._lock:
            for resource_id, waiters in list(self._waiters.items()):
                remaining = []
                for event, loop, deadline in waiters:
                    if deadline is not None and now >= deadline:
                        # Timed out - notify with event set
                        try:
                            loop.call_soon_threadsafe(event.set)
                        except RuntimeError:
                            pass
                    else:
                        remaining.append((event, loop, deadline))
                        if deadline is not None:
                            next_timeout = min(next_timeout, deadline - now)

                if remaining:
                    self._waiters[resource_id] = remaining
                else:
                    self._waiters.pop(resource_id, None)

        return next_timeout if next_timeout != float("inf") else None

    def run(self):
        """Main polling loop."""
        poll_interval = POLL_INTERVAL_INITIAL

        while True:
            # Get resources to poll
            with self._lock:
                resources = list(self._resources)
                has_waiters = bool(self._waiters)

            if not has_waiters:
                # No active waiters, wait for registration
                with self._lock:
                    self._cv.wait(timeout=1.0)
                poll_interval = POLL_INTERVAL_INITIAL
                continue

            # Poll each resource
            for resource in resources:
                try:
                    resource.refresh_state()
                    self._notify_waiters(resource)
                except Exception:
                    logger.exception("Error polling resource %s", resource)

            # Check timeouts
            next_timeout = self._check_timeouts()

            # Calculate sleep time
            sleep_time = poll_interval
            if next_timeout is not None:
                sleep_time = min(sleep_time, next_timeout)

            # Sleep with ability to wake up on new registration
            with self._lock:
                self._cv.wait(timeout=max(0.01, sleep_time))

            # Increase poll interval (exponential backoff)
            poll_interval = min(
                poll_interval * POLL_INTERVAL_MULTIPLIER, POLL_INTERVAL_MAX
            )


class DynamicDependency(ABC):
    """Base class for dynamic dependencies.

    Dynamic dependencies (like tokens) can change state at any time - availability
    can go from OK to WAIT and back. These require special handling during lock
    acquisition with retry logic.

    The origin must be a DynamicResource that supports async_wait().

    Subclasses must implement:
    - _create_lock(): Create the appropriate lock object for this dependency
    """

    origin: DynamicResource
    target: Optional["Job"]

    def __init__(self, origin: DynamicResource):
        self.origin = origin
        self.target = None

    def is_dynamic(self) -> bool:
        """Returns True - this is a dynamic dependency."""
        return True

    @abstractmethod
    def _create_lock(self) -> Lock:
        """Create a lock object for this dependency.

        Returns:
            Lock object (subclass of DynamicDependencyLock)
        """
        ...

    async def aio_lock(self, timeout: float = 0) -> Lock:
        """Acquire lock on the resource with async waiting.

        Uses the resource's async_wait() for efficient waiting without threads.

        Args:
            timeout: Timeout in seconds (0 = wait indefinitely)

        Returns:
            Lock object

        Raises:
            LockError: If lock cannot be acquired within timeout
        """
        from experimaestro.locking import LockError

        start_time = time.time()

        while True:
            try:
                lock = self._create_lock()
                lock.acquire()
                return lock
            except LockError:
                # Calculate remaining timeout
                if timeout > 0:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise LockError(f"Timeout waiting for resource: {self.origin}")
                else:
                    remaining = 0  # Wait indefinitely

                # Wait for resource state to change
                await self.origin.async_wait(timeout=remaining)

    def __repr__(self) -> str:
        return f"DynamicDep[{self.origin}]"
