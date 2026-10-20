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
