"""Running the preparation declared by a ``Prepare`` config.

`Prepare` configs (defined in ``experimaestro.core.objects.config``) declare a
preparation step, which can run two ways.

**In-process** — the default. When a Task references a Prepare instance in its
params, ``ConfigInformation.submit`` discovers it and attaches a
``PrepareDependency`` so the scheduler awaits ``prepare()`` before running the
task. A ``PrepareResource`` has no on-disk footprint (no workdir, no ``.done``,
no entry under ``jobs/``); idempotence is owned by ``prepare()`` itself.

**As a job** — when the Prepare was submitted. ``PrepareTask`` wraps it into a
real task, so the preparation gets a launcher, a resource request, logs and
retries. The Prepare config remains a plain parameter of the tasks that
reference it: only the wrapper is a task, so identifiers are unaffected by the
choice between the two modes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from experimaestro.locking import Lock
from ..core.arguments import Param

from ..core.objects.config import Prepare, ResumableTask
from .dependencies import Dependency, Resource

if TYPE_CHECKING:
    from .jobs import Job


logger = logging.getLogger("xpm.prepare")


class PrepareTask(ResumableTask):
    """Runs a ``Prepare`` config's ``prepare()`` as a job.

    Resumable: a preparation killed by a walltime limit is restarted with its
    directory preserved, which matches the way preparations are expected to be
    idempotent and to pick up where a warm cache left them.
    """

    __xpmid__ = "experimaestro.prepare"

    target: Param[Prepare]
    """The configuration to prepare"""

    def __xpm_output_valid__(self) -> bool | None:
        return bool(self.target.is_prepared())

    def execute(self):
        # Re-checked here and not only in the driver: two experiments may race
        # on the same target, and the winner's work makes ours unnecessary.
        if self.target.is_prepared():
            logger.info("%s is already prepared, nothing to do", self.target)
            return
        self.target.prepare()


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
    unless the config implements ``is_prepared()``.

    A resource whose Prepare was submitted is backed by a job instead: it
    then hands out a dependency on that job, and ``prepare()`` never runs in
    the driver. Identifier keying is what makes the two identifier-equal
    instances that ``prepare_dataset("...")`` returns on two calls share the
    one job.
    """

    #: identifier hex (``config.__xpm__.identifier.main.hex()``) → singleton
    RESOURCES: dict[str, "PrepareResource"] = {}

    def __init__(self, config: "Prepare"):
        super().__init__()
        self.config = config
        self._executed = False

        #: The job preparing this config, when it was submitted
        self.job: "Job" | None = None
        # asyncio.Lock is created lazily — at first await — so it binds to the
        # event loop that's actually running prepare(), not whatever loop
        # happened to exist at construction time.
        self._lock: asyncio.Lock | None = None

    @staticmethod
    def _key(config: "Prepare") -> str:
        return config.__xpm__.identifier.main.hex()

    @classmethod
    def for_config(cls, config: "Prepare") -> "PrepareResource":
        """Return the singleton ``PrepareResource`` for this config's identifier."""
        key = cls._key(config)
        existing = cls.RESOURCES.get(key)
        if existing is not None:
            return existing
        resource = cls(config)
        cls.RESOURCES[key] = resource
        return resource

    @classmethod
    def reset(cls) -> None:
        """Clear the registry. Used by tests."""
        cls.RESOURCES.clear()

    def dependency(self) -> Dependency:
        """The dependency a task referencing this Prepare must hold."""
        if self.job is not None:
            from .jobs import JobDependency

            return JobDependency(self.job)
        return PrepareDependency(self)

    def _is_prepared(self) -> bool:
        """Whether ``is_prepared()`` is implemented and answers True."""
        value_type = self.config.__xpmtype__.value_type
        if value_type.is_prepared is Prepare.is_prepared:
            return False
        try:
            return bool(self.config.is_prepared())
        except Exception:
            logger.exception(
                "Error while checking whether %s is prepared - preparing it",
                self.config,
            )
            return False

    async def aio_ensure_prepared(self) -> None:
        """Run ``prepare()`` once; concurrent callers wait, later callers no-op."""
        if self._executed:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._executed:
                return
            # Both calls go off the event loop: a blocking download must not
            # stall the scheduler, and neither must a check that stats a
            # network filesystem.
            if await asyncio.to_thread(self._is_prepared):
                logger.debug("%s is already prepared", self.config)
            else:
                logger.debug("Running prepare() for %s", self.config)
                await asyncio.to_thread(self.config.prepare)
            self._executed = True

    def __str__(self) -> str:
        return f"prepare[{self._key(self.config)[:8]}]"


class PrepareDependency(Dependency):
    """Static dependency on a ``PrepareResource`` — triggers ``prepare()`` on await."""

    def is_dynamic(self) -> bool:
        return False

    async def aio_lock(self, timeout: float = 0) -> PrepareLock:
        await self.origin.aio_ensure_prepared()
        return PrepareLock()
