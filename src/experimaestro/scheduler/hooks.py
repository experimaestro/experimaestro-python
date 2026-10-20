"""Job hooks: work tied to a job's submission and to the end of its runs

The lifecycle of a job run is: process exited → state recorded → **hooks
run** → **settled** (:attr:`Job.settled`). Experiments wait for the settled
future of their jobs, and for any work a hook tracks with
:meth:`experiment.track`, before exiting.
"""

import asyncio
from typing import TYPE_CHECKING

from experimaestro.scheduler.interfaces import JobState
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.experiment import experiment
    from experimaestro.scheduler.jobs import Job


class JobHook:
    """Base class for job hooks

    Both methods run on the scheduler loop.
    """

    def on_submitted(self, job: "Job", xp: "experiment", request: "Job") -> None:
        """Called each time the job is submitted within an experiment

        :param job: the job kept by the scheduler
        :param xp: the experiment the job is submitted to
        :param request: the job built by this submission; it is ``job`` for a
            first submission, and carries the settings of this submission
            (e.g. watched outputs) otherwise
        """

    async def on_finished(self, job: "Job") -> None:
        """Called when a run of the job is over and its state recorded

        The job is settled once this returns (for every hook): any work that
        must finish before the experiment exits must either be awaited here or
        tracked with :meth:`experiment.track`.
        """


class TaskOutputsHook(JobHook):
    """Dynamic task outputs: watch them, and process them all at the end"""

    def on_submitted(self, job, xp, request):
        if request is not job:
            job.watched_outputs.extend(request.watched_outputs)
        for watched in request.watched_outputs:
            watched.job = job
            xp.watch_output(watched)

    async def on_finished(self, job):
        await job.aio_drain_task_outputs()


class OnCompletedHook(JobHook):
    """Runs the :meth:`~experimaestro.Task.on_completed` callbacks"""

    def on_submitted(self, job, xp, request):
        for callback in request.config.__xpm__.on_completed_callbacks:
            job.on_completed(callback, xp)

    async def on_finished(self, job):
        if job.scheduler_state != JobState.DONE:
            return

        futures = []
        for xp, callback in job.take_on_completed():
            if xp not in job.experiments:
                # Fall back on the most recent experiment of the job
                xp = job.experiments[-1]
            futures.append(xp.run_callback(callback))
        await asyncio.gather(*futures)


class ArchiveHook(JobHook):
    """Archives successful jobs to the attached folders (beta)

    The archive is not awaited (the job is settled without it), but it is
    tracked, so experiments wait for it before exiting (and before removing
    transient job directories).
    """

    async def on_finished(self, job):
        # Only on successful completion, and only if the per-job opt-out
        # wasn't set. Skip transient jobs whose source dir will be deleted at
        # experiment exit — archiving them would be wasted work (or worse, a
        # race with the cleanup pass).
        if (
            job.scheduler_state != JobState.DONE
            or not job.backup
            or job.workspace is None
            or job.transient.should_remove
        ):
            return

        from experimaestro.scheduler.folders import archive_job_async

        try:
            futures = archive_job_async(job, job.workspace.folders)
        except Exception:
            logger.exception("Failed to schedule folder archive for %s", job)
            return

        for future in futures:
            for xp in job.experiments:
                xp.track(asyncio.wrap_future(future))


def default_hooks() -> list[JobHook]:
    return [TaskOutputsHook(), OnCompletedHook(), ArchiveHook()]
