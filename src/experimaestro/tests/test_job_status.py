"""Tests for the job and experiment statuses shared by the TUI and the web UI"""

import pytest

from experimaestro.carbon.base import CarbonAggregateData, CarbonImpactData
from experimaestro.scheduler.interfaces import (
    ExperimentStatus,
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


@pytest.mark.parametrize(
    "status,total,finished,failed,expected",
    [
        (ExperimentStatus.DONE, 2, 1, 1, "failed"),
        (ExperimentStatus.RUNNING, 2, 1, 0, "running"),
        (ExperimentStatus.DONE, 2, 2, 0, "done"),
        # Not running any more, but some jobs never finished
        (ExperimentStatus.FAILED, 2, 1, 0, "stopped"),
        (ExperimentStatus.DONE, 0, 0, 0, "empty"),
    ],
)
def test_experiment_status_category(
    tmp_path, status, total, finished, failed, expected
):
    experiment = MockExperiment(
        tmp_path,
        "run",
        status=status,
        total_jobs=total,
        finished_jobs=finished,
        failed_jobs=failed,
    )
    assert experiment.status_category == expected


def test_experiment_carbon_summary(tmp_path):
    """The latest run of each job is preferred over all the runs"""
    latest = CarbonAggregateData(co2_kg=1.0)
    total = CarbonAggregateData(co2_kg=3.0)
    experiment = MockExperiment(tmp_path, "run")
    assert experiment.carbon_summary is None

    experiment = MockExperiment(
        tmp_path, "run", carbon_impact=CarbonImpactData(sum=total, latest=latest)
    )
    assert experiment.carbon_summary is latest

    experiment = MockExperiment(
        tmp_path, "run", carbon_impact=CarbonImpactData(sum=total)
    )
    assert experiment.carbon_summary is total
