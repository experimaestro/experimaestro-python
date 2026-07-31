"""Caching of state provider read queries

State providers answer the same questions over and over: the UIs re-query
after every state event, and each query costs a filesystem scan (workspace
provider) or a blocking RPC round-trip (SSH provider).

Read queries are therefore cached by default and only recomputed when:

- an event tells the provider that the underlying state changed (see
  :meth:`OfflineStateProvider.apply_event`),
- a mutation is performed through the provider (kill, clean, delete),
- the caller explicitly asks for a full refresh (``refresh=True``).

Entries are tagged with the experiment they derive from, so an event about
one experiment does not invalidate the queries of the others.  Queries that
span the whole workspace (experiment list, orphan jobs, ...) are tagged with
``None`` and are dropped whenever any entity appears or disappears.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

#: Returned by :meth:`QueryCache.get` when there is no usable entry — ``None``
#: cannot be used since it is a legitimate cached value (e.g. ``get_experiment``)
MISSING: Any = object()


@dataclass
class CacheEntry:
    """A cached query result"""

    value: Any
    """The value returned by the query"""

    experiment_id: Optional[str]
    """Experiment this entry derives from (None for workspace-wide queries)"""

    timestamp: float
    """Monotonic time at which the entry was stored"""


class QueryCache:
    """Thread-safe cache of state provider query results

    Args:
        ttl: Maximum age (in seconds) of an entry before it is recomputed.
            ``None`` (the default) means entries only expire through
            :meth:`invalidate` — appropriate when every state change is
            reported by an event.
    """

    def __init__(self, ttl: Optional[float] = None):
        self.ttl = ttl
        self._entries: dict[Hashable, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, copy: Optional[Callable[[Any], Any]] = None) -> Any:
        """Return the cached value for `key`, or :data:`MISSING`

        Args:
            copy: Applied to the value before returning it, while the lock is
                held — cached lists and dicts are updated in place when events
                arrive, so the copy has to be atomic with respect to them
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return MISSING

            if self.ttl is not None and (time.monotonic() - entry.timestamp) > self.ttl:
                del self._entries[key]
                self.misses += 1
                return MISSING

            self.hits += 1
            return copy(entry.value) if copy is not None else entry.value

    def entries(self) -> list[tuple[Hashable, CacheEntry]]:
        """Snapshot of the cached entries, to decide what an event changes"""
        with self._lock:
            return list(self._entries.items())

    def update(self, key: Hashable, updater: Callable[[Any], None]) -> None:
        """Apply `updater` to a cached value in place

        Does nothing if the entry is gone. Used to keep cached answers up to
        date when an event carries enough information to extend them, instead
        of dropping them and reading the state again.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                updater(entry.value)

    def drop(self, key: Hashable) -> None:
        """Drop a single entry"""
        with self._lock:
            self._entries.pop(key, None)

    def put(self, key: Hashable, value: Any, *, experiment_id: Optional[str]) -> None:
        """Store `value` for `key`, tagged with the experiment it derives from"""
        with self._lock:
            self._entries[key] = CacheEntry(
                value=value, experiment_id=experiment_id, timestamp=time.monotonic()
            )

    def invalidate(self, experiment_id: Optional[str] = None) -> None:
        """Drop cached queries

        Args:
            experiment_id: When given, drops the queries of that experiment
                together with the workspace-wide ones (an experiment gaining
                or losing a job also changes the global lists).  When ``None``,
                drops everything.
        """
        with self._lock:
            if experiment_id is None:
                self._entries.clear()
                return

            for key, entry in list(self._entries.items()):
                if entry.experiment_id in (experiment_id, None):
                    del self._entries[key]

    def invalidate_global(self) -> None:
        """Drop the workspace-wide queries, keeping the per-experiment ones

        Used when the set of experiments changes: the experiment list and the
        workspace-wide job queries have to be recomputed, but what is cached
        about each experiment is still valid.
        """
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry.experiment_id is None:
                    del self._entries[key]

    def clear(self) -> None:
        """Drop every entry and reset the statistics"""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def freeze(value: Any) -> Hashable:
    """Make a query argument usable as part of a cache key"""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, set)):
        return tuple(freeze(v) for v in value)
    return value


__all__ = ["MISSING", "CacheEntry", "QueryCache", "freeze"]
