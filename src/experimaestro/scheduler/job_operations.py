"""User operations on jobs and experiments: kill and delete

Shared by the TUI, the web UI and the CLI so that they accept and refuse the
same operations and report them with the same messages. Each operation returns
an :class:`Outcome`, which the interfaces display as is (notification, toast or
printed line).

Interfaces that ask for a confirmation first call the matching ``check_*``
function: it returns the refusal (if any) without doing anything, so that no
confirmation is asked for an operation that would be refused.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from experimaestro.scheduler.interfaces import BaseJob

if TYPE_CHECKING:
    from experimaestro.scheduler.state_provider import StateProvider

logger = logging.getLogger("xpm.operations")

#: ``info`` for a success, ``warning`` for a refused operation (nothing was
#: done) and ``error`` for a failure
Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class Outcome:
    severity: Severity
    message: str

    @property
    def ok(self) -> bool:
        return self.severity == "info"


def _count(done: int, total: int, failed: int, verb: str) -> Outcome:
    if failed:
        return Outcome("warning", f"{verb} {done} of {total} jobs ({failed} failed)")
    return Outcome("info", f"{verb} {done} of {total} jobs")


# --- Kill


def check_kill_job(job: BaseJob) -> Outcome | None:
    """Refusal to kill a job, if any"""
    # Live jobs: the scheduler lifecycle state can be ahead of the execution
    # state (e.g. scheduled in a SLURM queue)
    if not (job.state.running() or job.scheduler_state.running()):
        return Outcome("warning", f"Job {job.identifier} is not running")
    return None


def kill_job(provider: "StateProvider", job: BaseJob) -> Outcome:
    """Kill a running job"""
    if refusal := check_kill_job(job):
        return refusal
    try:
        provider.kill_job(job, perform=True)
    except Exception as e:
        logger.warning("Failed to kill job %s: %s", job.identifier, e)
        return Outcome("error", f"Failed to kill job {job.identifier}: {e}")
    return Outcome("info", f"Job {job.identifier} killed")


def kill_jobs(
    provider: "StateProvider",
    jobs: list[BaseJob],
    *,
    nothing: str = "No running jobs",
) -> Outcome:
    """Kill the running jobs among ``jobs``"""
    running = [job for job in jobs if check_kill_job(job) is None]
    if not running:
        return Outcome("warning", nothing)
    killed = sum(kill_job(provider, job).ok for job in running)
    return _count(killed, len(running), len(running) - killed, "Killed")


def check_kill_experiment(
    provider: "StateProvider", experiment_id: str
) -> Outcome | None:
    """Refusal to kill an experiment, if any"""
    if not provider.get_running_jobs(experiment_id):
        return Outcome("warning", "No running jobs in experiment")
    return None


def kill_experiment(provider: "StateProvider", experiment_id: str) -> Outcome:
    """Kill the running jobs of an experiment"""
    return kill_jobs(
        provider,
        provider.get_running_jobs(experiment_id),
        nothing="No running jobs in experiment",
    )


# --- Delete


def check_delete_job(job: BaseJob) -> Outcome | None:
    """Refusal to delete a job, if any"""
    if job.state and not (job.state.finished() or job.state.is_unscheduled()):
        return Outcome(
            "warning",
            f"Cannot delete job {job.identifier} while it is {job.state.name.lower()}",
        )
    return None


def delete_job(
    provider: "StateProvider", job: BaseJob, *, cleanup: bool = True
) -> Outcome:
    """Delete a finished job

    Args:
        cleanup: Also remove the partial directories the deletion leaves
            orphaned (callers deleting several jobs do it once at the end)
    """
    if refusal := check_delete_job(job):
        return refusal
    try:
        ok, message = provider.delete_job_safely(job, perform=True)
    except Exception as e:
        logger.warning("Failed to delete job %s: %s", job.identifier, e)
        ok, message = False, f"Failed to delete job {job.identifier}: {e}"
    if ok and cleanup:
        provider.cleanup_orphan_partials(perform=True)
    return Outcome("info" if ok else "error", message)


def delete_jobs(
    provider: "StateProvider",
    jobs: list[BaseJob],
    *,
    nothing: str = "No jobs to delete",
) -> Outcome:
    """Delete the finished jobs among ``jobs``"""
    deletable = [job for job in jobs if check_delete_job(job) is None]
    if not deletable:
        return Outcome("warning", nothing)
    deleted = sum(delete_job(provider, job, cleanup=False).ok for job in deletable)
    if deleted:
        provider.cleanup_orphan_partials(perform=True)
    return _count(deleted, len(deletable), len(deletable) - deleted, "Deleted")


def check_delete_experiment(
    provider: "StateProvider", experiment_id: str
) -> Outcome | None:
    """Refusal to delete an experiment, if any"""
    if provider.is_experiment_running(experiment_id):
        return Outcome("warning", "Cannot delete: experiment is still running")
    return None


def delete_experiment(provider: "StateProvider", experiment_id: str) -> Outcome:
    """Delete an experiment (its job directories are kept)"""
    if refusal := check_delete_experiment(provider, experiment_id):
        return refusal
    ok, message = provider.delete_experiment(experiment_id, delete_jobs=False)
    return Outcome("info" if ok else "error", message)
