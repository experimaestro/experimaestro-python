"""In-memory preparation resources.

`Prepare` configs (defined in ``experimaestro.core.objects.config``) declare
an in-process preparation step. When a Task references a Prepare instance in
its params, ``ConfigMixin.submit`` discovers it and attaches a
``PrepareDependency`` so the scheduler awaits ``prepare()`` before running
the task. Unlike ``Job``, a ``PrepareResource`` has no on-disk footprint
(no workdir, no ``.done``, no entry under ``jobs/``); idempotence is owned
by ``prepare()`` itself.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict

from experimaestro.locking import Lock
from .dependencies import Dependency, Resource

if TYPE_CHECKING:
    from ..core.objects.config import Prepare


logger = logging.getLogger("xpm.prepare")


class PrepareLock(Lock):
    """No-op lock — prep state lives in ``PrepareResource._executed``."""

    async def _aio_acquire(self):
        return None

    async def _aio_release(self):
        return None


class PrepareResource(Resource):
    """In-memory resource that runs ``Prepare.prepare()`` exactly once.

    Dedup is by identifier hex: two ``Prepare`` configs with the same
    identifier share one ``PrepareResource`` and one execution. State is
    purely in-memory — restarting the Python process re-runs ``prepare()``,
    but the underlying tool is expected to be idempotent (e.g., datamaestro
    checks the local cache).
    """

    #: identifier hex (``config.__xpm__.identifier.main.hex()``) → singleton
    RESOURCES: Dict[str, "PrepareResource"] = {}

    def __init__(self, config: "Prepare"):
        super().__init__()
        self.config = config
        self._executed = False
        # asyncio.Lock is created lazily — at first await — so it binds to the
        # event loop that's actually running prepare(), not whatever loop
        # happened to exist at construction time.
        self._lock: asyncio.Lock | None = None

    @classmethod
    def for_config(cls, config: "Prepare") -> "PrepareResource":
        """Return the singleton ``PrepareResource`` for this config's identifier."""
        key = config.__xpm__.identifier.main.hex()
        existing = cls.RESOURCES.get(key)
        if existing is not None:
            return existing
        resource = cls(config)
        cls.RESOURCES[key] = resource
        return resource

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton registry. Used by tests."""
        cls.RESOURCES.clear()

    def dependency(self) -> "PrepareDependency":
        return PrepareDependency(self)

    async def aio_ensure_prepared(self) -> None:
        """Run ``prepare()`` once; concurrent callers wait, later callers no-op."""
        if self._executed:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._executed:
                return
            logger.debug("Running prepare() for %s", self.config)
            # Run prepare() off the event loop so blocking downloads don't
            # stall the scheduler.
            await asyncio.to_thread(self.config.prepare)
            self._executed = True

    def __str__(self) -> str:
        return f"prepare[{self.config.__xpm__.identifier.main.hex()[:8]}]"


class PrepareDependency(Dependency):
    """Static dependency on a ``PrepareResource`` — triggers ``prepare()`` on await."""

    def is_dynamic(self) -> bool:
        return False

    async def aio_lock(self, timeout: float = 0) -> PrepareLock:
        await self.origin.aio_ensure_prepared()
        return PrepareLock()
