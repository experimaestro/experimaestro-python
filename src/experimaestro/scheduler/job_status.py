"""Job status ordering shared by the TUI and the web UI

Both interfaces sort and group jobs by status with :func:`status_rank`, so the
same jobs come out in the same order. The web server sends the rank with each
job (``statusRank``) instead of the frontend keeping its own table.
"""

from experimaestro.scheduler.interfaces import BaseExperiment, BaseJob, JobState
from experimaestro.scheduler.state_provider import StateProvider

#: Status order (lower first): active jobs, then pending, failed and done ones
STATUS_ORDER = {
    "running": 0,
    "scheduled": 1,
    "ready": 2,
    "waiting": 3,
    "error": 4,
    "done": 5,
    "unscheduled": 6,
    "transient": 7,
}

#: Order of failure reasons within the error status, most actionable first
FAILURE_ORDER = {
    "TIMEOUT": 0,  # Might just need a retry
    "MEMORY": 1,  # Might need resource adjustment
    "REJECTED_TIMELIMIT": 2,  # Time limit exceeded the partition max
    "REJECTED_OTHER": 3,  # Other rejection reason
    "DEPENDENCY": 4,  # Need to fix the upstream job first
    "FAILED": 5,  # Generic failure
}

#: Description of each failure reason, shown by the TUI and the web UI
FAILURE_DESCRIPTIONS = {
    "DEPENDENCY": "a dependency failed",
    "FAILED": "the job failed",
    "MEMORY": "out of memory",
    "TIMEOUT": "timed out",
    "CANCELLED": "cancelled",
    "REJECTED_TIMELIMIT": "rejected: time limit above the partition maximum",
    "REJECTED_OTHER": "rejected by the launcher",
    "MISSING_OUTPUTS": "outputs missing (probably purged)",
    "DELETED": "job data deleted",
}

#: Rank of unknown statuses and failure reasons (sorted last)
UNKNOWN_RANK = 99


def display_state(job: BaseJob, experiment: BaseExperiment | None = None) -> JobState:
    """State shown for a job: its execution state resolved against the scheduler

    With an experiment, uses the state the experiment tracks for the job (as
    :meth:`StateProvider.get_resolved_state`); without one, the job's own
    scheduler state, which is the same for a live experiment.
    """
    if experiment is not None:
        state, _ = StateProvider.get_resolved_state(job, experiment)
        return state
    return job.scheduler_state.resolve(job.state) or job.state


def failure_description(state: JobState | None) -> str | None:
    """Description of the failure reason of a state (None if there is none)"""
    reason = state.failure_reason if state is not None else None
    if reason is None:
        return None
    return FAILURE_DESCRIPTIONS.get(reason.name, reason.name.lower())


def has_started(state: JobState | None) -> bool:
    """Whether a job in this state has started (and may have written logs)

    Jobs that are not submitted yet, wait for their dependencies or are queued
    by the launcher (scheduled) have not; their log files, if any, are those of
    an earlier run.
    """
    if state is None:
        return False
    return not (state.notstarted() or state == JobState.SCHEDULED)


def status_rank(state: JobState | None) -> int:
    """Sort rank of a job state (lower comes first)

    Error states are further ordered by failure reason.
    """
    if state is None:
        return UNKNOWN_RANK * 100
    rank = STATUS_ORDER.get(state.name, UNKNOWN_RANK) * 100
    if state.is_error():
        reason = state.failure_reason
        rank += FAILURE_ORDER.get(reason.name, UNKNOWN_RANK) if reason else UNKNOWN_RANK
    return rank


def retries_summary(job: BaseJob, state: JobState | None = None) -> str | None:
    """Retries of a resumable job, e.g. ``1/3`` (None if it has none)

    Only timeouts are retried; a job that timed out with no retry left is not
    restarted, which the summary says.

    :param state: the displayed state of the job (defaults to its state)
    """
    if not job.resumable or (job.retry_count == 0 and job.max_retries is None):
        return None
    if job.max_retries is None:
        return str(job.retry_count)
    summary = f"{job.retry_count}/{job.max_retries}"
    state = state or job.state
    if (
        state is not None
        and state.is_error()
        and state.failure_reason is not None
        and state.failure_reason.name == "TIMEOUT"
        and job.retry_count >= job.max_retries
    ):
        summary += " — exhausted, not restarted"
    return summary
