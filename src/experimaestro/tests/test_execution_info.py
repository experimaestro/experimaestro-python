"""Tests of the record telling where a job runs"""

import json

import pytest

from experimaestro.cli.filter import FilterContext, FilterVocabulary, createFilter
from experimaestro.scheduler.execution import ExecutionInfo, expand_nodelist
from experimaestro.scheduler.interfaces import JobState
from experimaestro.scheduler.state_provider import MockJob
from experimaestro.scheduler.state_status import EventBase, JobStateChangedEvent


@pytest.mark.parametrize(
    "nodelist,expected",
    [
        ("", []),
        ("node1", ["node1"]),
        ("node1,gpu5", ["node1", "gpu5"]),
        ("node[01-03]", ["node01", "node02", "node03"]),
        ("node[1,7-8],gpu5", ["node1", "node7", "node8", "gpu5"]),
        ("r[1-2]n[09-10]", ["r1n09", "r1n10", "r2n09", "r2n10"]),
        ("node[9-10]-ib", ["node9-ib", "node10-ib"]),
    ],
)
def test_expand_nodelist(nodelist, expected):
    assert expand_nodelist(nodelist) == expected


def test_current_from_slurm_environment():
    info = ExecutionInfo.current(
        {
            "SLURM_JOB_NODELIST": "node[07-08]",
            "SLURMD_NODENAME": "node08",
            "SLURM_JOB_PARTITION": "gpu",
            "SLURM_CPUS_ON_NODE": "8",
            "SLURM_JOB_GPUS": "2,3",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    # The node running the job process comes first
    assert info == ExecutionInfo(
        hosts=["node08", "node07"], partition="gpu", cpus=8, gpus=["2", "3"]
    )
    assert info.describe() == "node08, node07 (partition gpu, 8 CPUs, GPUs 2,3)"


def test_current_outside_slurm():
    info = ExecutionInfo.current({"CUDA_VISIBLE_DEVICES": "0"})
    assert len(info.hosts) == 1 and info.hosts[0]
    assert (info.partition, info.cpus, info.gpus) == (None, None, ["0"])
    assert info.describe() == f"{info.hosts[0]} (GPU 0)"

    assert ExecutionInfo(hosts=["laptop"]).describe() == "laptop"


def test_state_dict_roundtrip():
    info = ExecutionInfo(hosts=["node07"], partition="gpu", cpus=1, gpus=[])
    assert ExecutionInfo.from_state_dict(info.state_dict()) == info
    assert ExecutionInfo.from_state_dict(None) is None
    assert ExecutionInfo.from_state_dict("node07") is None


def _job(tmp_path, state: str = "scheduled") -> MockJob:
    path = tmp_path / "jobs" / "pkg.Learn" / "abc"
    path.mkdir(parents=True, exist_ok=True)
    return MockJob(
        identifier="abc",
        task_id="pkg.Learn",
        path=path,
        state=state,
        starttime=None,
        endtime=None,
        progress=[],
        updated_at="",
    )


def test_job_status_keeps_execution(tmp_path):
    """The record is written in status.json and read back"""
    job = _job(tmp_path)
    assert "execution" not in job.state_dict()

    job.set_state(JobState.RUNNING)
    job.execution = ExecutionInfo(hosts=["node07"], partition="gpu")
    job.write_status()

    loaded = MockJob.from_disk(job.path, job.task_id, job.identifier)
    assert loaded.execution == job.execution
    copy = MockJob.from_state_dict(job.state_dict(), tmp_path)
    assert copy.execution == job.execution

    # A new run forgets where the previous one ran
    loaded._clear_transient_fields()
    assert loaded.execution is None
    copy.update_from_state_dict(loaded.state_dict())
    assert copy.execution is None


def test_running_event_sets_execution(tmp_path):
    """The record travels with the "running" event; later events keep it"""
    job = _job(tmp_path)
    info = ExecutionInfo(hosts=["node07"], cpus=4)
    event = JobStateChangedEvent(
        job_id="abc", state="running", execution=info.state_dict()
    )
    # As read from an event file or received from a remote server
    job.apply_event(EventBase.from_dict(json.loads(event.to_json())))
    assert job.execution == info

    job.apply_event(JobStateChangedEvent(job_id="abc", state="done"))
    assert job.execution == info
    assert job.state_changed_event().execution == info.state_dict()


def test_filter_by_host_and_partition(tmp_path):
    job = _job(tmp_path, "running")
    job.execution = ExecutionInfo(hosts=["node07", "node08"], partition="gpu")
    unknown = _job(tmp_path / "other", "waiting")
    ctx = FilterContext()

    def matches(query: str) -> list[bool]:
        filter_fn = createFilter(query, ctx)
        return [filter_fn(job), filter_fn(unknown)]

    assert matches("@host = 'node08'") == [True, False]
    assert matches("@host = 'node09'") == [False, False]
    assert matches("@host ~ 'node0[78]'") == [True, False]
    assert matches("@host in ['node01', 'node07']") == [True, False]
    assert matches("@host not in ['node07']") == [False, True]
    assert matches("@partition = 'gpu'") == [True, False]

    vocabulary = FilterVocabulary()
    vocabulary.add_jobs([job, unknown], {})
    assert vocabulary.values("@host") == ["node07", "node08"]
    assert vocabulary.values("@partition") == ["gpu"]
