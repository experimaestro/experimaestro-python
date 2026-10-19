"""Tests for the job status order shared by the TUI and the web UI"""

from experimaestro.scheduler.interfaces import (
    JobFailureStatus,
    JobState,
    JobStateError,
)
from experimaestro.scheduler.job_status import display_state, status_rank
from experimaestro.scheduler.state_provider import MockExperiment, MockJob


def _job(state: str) -> MockJob:
    return MockJob(
        identifier="j1",
        task_id="pkg.Task",
        path=None,
        state=state,
        starttime=None,
        endtime=None,
        progress=[],
        updated_at="",
    )


def test_status_rank_order():
    states = [
        JobState.DONE,
        JobState.WAITING,
        JobState.RUNNING,
        JobState.ERROR,
        JobState.TRANSIENT,
        JobState.READY,
        JobState.SCHEDULED,
        JobState.UNSCHEDULED,
    ]
    assert [s.name for s in sorted(states, key=status_rank)] == [
        "running",
        "scheduled",
        "ready",
        "waiting",
        "error",
        "done",
        "unscheduled",
        "transient",
    ]


def test_status_rank_failure_reasons():
    """Errors are ordered by failure reason, unknown reasons last"""
    timeout = JobStateError(JobFailureStatus.TIMEOUT)
    failed = JobStateError(JobFailureStatus.FAILED)
    unknown = JobStateError()
    assert status_rank(timeout) < status_rank(failed) < status_rank(unknown)
    assert status_rank(unknown) < status_rank(JobState.DONE)


def test_display_state_uses_experiment(tmp_path):
    """The state tracked by the experiment wins when more progressed"""
    job = _job("waiting")
    assert display_state(job).name == "waiting"

    experiment = MockExperiment(tmp_path, "run", job_states={"j1": JobState.RUNNING})
    assert display_state(job, experiment).name == "running"
