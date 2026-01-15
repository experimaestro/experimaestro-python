"""Done handler worker for processing job completion in a dedicated thread pool."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experimaestro.scheduler.jobs import Job
    from experimaestro.scheduler import JobState

logger = logging.getLogger("xpm.done_handler")


class DoneHandlerWorker:
    """Worker for processing job completion in a dedicated thread pool.

    This worker handles the final steps of job completion:
    1. Call job's done_handler (processes task outputs)
    2. Write final status file (while holding lock if provided)
    3. Remove job from scheduler's waiting jobs
    4. Resolve the job's final_state future

    The worker uses a thread pool to avoid blocking the main event loop
    during potentially slow I/O operations.
    """

    _instance: Optional["DoneHandlerWorker"] = None

    @classmethod
    def instance(cls) -> "DoneHandlerWorker":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="DoneHandler"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for callbacks."""
        self._loop = loop

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)

    def submit(
        self,
        job: "Job",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Submit a job for done handler processing.

        Args:
            job: The job that has completed
            loop: Event loop for callbacks (uses stored loop if not provided)
        """
        # Ensure loop is set (may be reset between tests)
        if loop is not None:
            self._loop = loop
        self._executor.submit(self._process_job, job)

    def _process_job(self, job: "Job") -> None:
        """Process job completion in the thread pool.

        Args:
            job: The job to process
        """
        try:
            logger.debug("Processing done handler for job %s", job.identifier[:8])

            # Skip processing for UNSCHEDULED jobs (transient jobs that weren't needed)
            from experimaestro.scheduler import JobState

            if job.state == JobState.UNSCHEDULED:
                logger.debug(
                    "Skipping done handler for unscheduled job %s", job.identifier[:8]
                )
            else:
                # Call job's done_handler (processes task outputs)
                job.done_handler()

                # Write final status while holding job lock
                from filelock import FileLock

                lock_path = job.lockpath
                with FileLock(lock_path):
                    try:
                        job.status_path.parent.mkdir(parents=True, exist_ok=True)
                        job.status_path.write_text(json.dumps(job.state_dict()))
                    except Exception as e:
                        logger.warning(
                            "Failed to write final status for job %s: %s",
                            job.identifier[:8],
                            e,
                        )

            # Remove from scheduler waitingjobs
            if job.scheduler:
                job.scheduler.waitingjobs.discard(job)

            # Resolve final_state future
            if job._final_state is not None and self._loop is not None:
                logger.debug(
                    "Resolving final_state future for job %s", job.identifier[:8]
                )
                self._loop.call_soon_threadsafe(
                    self._resolve_future, job._final_state, job.state
                )
            else:
                logger.warning(
                    "Cannot resolve future for job %s: _final_state=%s, _loop=%s",
                    job.identifier[:8],
                    job._final_state is not None,
                    self._loop is not None,
                )

            logger.debug("Done handler completed for job %s", job.identifier[:8])

            # Notify exit condition in the event loop
            if self._loop is not None and job.scheduler is not None:
                asyncio.run_coroutine_threadsafe(
                    self._notify_exit_condition(job.scheduler.exitCondition), self._loop
                )

        except Exception as e:
            logger.exception(
                "Error in done handler for job %s: %s", job.identifier[:8], e
            )
            # Still try to resolve the future even on error
            if job._final_state is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._resolve_future, job._final_state, job.state
                )

    async def _notify_exit_condition(self, condition: asyncio.Condition) -> None:
        """Notify the exit condition (must be called from event loop)."""
        async with condition:
            condition.notify_all()

    def _resolve_future(self, future: asyncio.Future, state: "JobState") -> None:
        """Resolve a future with the given state (must be called from event loop)."""
        if not future.done():
            logger.debug("Setting future result to %s", state)
            future.set_result(state)
        else:
            logger.debug("Future already done, not setting result")
