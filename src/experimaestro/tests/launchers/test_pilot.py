"""SLURM pilot allocations (with the fake SLURM commands of ``bin/``)"""

import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from experimaestro import Meta, Param, PathGenerator, ResumableTask, Task, field
from experimaestro.connectors import Process
from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.slurm import PilotOptions, SlurmLauncher
from experimaestro.launchers.slurm import base as slurm_base
from experimaestro.launchers.slurm.base import SlurmLauncherInformation
from experimaestro.launchers.slurm.pilot import (
    DEADLINE_ENV,
    PilotProcess,
    PoolDirectory,
    Worker,
    list_pilots,
    parse_slurm_duration,
    stop_pilots,
)
from experimaestro.scheduler import JobState
from experimaestro.scheduler.jobs import Job
from experimaestro.scheduler.interfaces import FailureDetails, JobFailureStatus
from experimaestro.tests.utils import TemporaryExperiment

pytestmark = pytest.mark.launchers

BINPATH = Path(__file__).parent / "bin"


def make_launcher(tmp_path: Path, **pilot) -> SlurmLauncher:
    """A SLURM launcher (fake commands) running jobs in pilots"""
    binpath = tmp_path / "slurm-bin" / "bin"
    (tmp_path / "slurm-bin" / "slurm").mkdir(parents=True)
    shutil.copytree(BINPATH, binpath)
    options = (
        dict(time=60, idle_timeout=3, margin=1, kill_wait=5, poll_interval=0.1) | pilot
    )
    return SlurmLauncher(
        connector=LocalConnector.instance(),
        binpath=binpath,
        interval=0.05,
        pilot=PilotOptions(**options),
    ).config(time="0:20")


def job_of(task: Any) -> Job:
    return task.__xpm__.job


def pool_dirs(workdir: Path) -> list[Path]:
    return list((workdir / "pilots").iterdir())


def pilots(pool: Path) -> list[str]:
    return [p.stem for p in (pool / "pilots").glob("*.json")]


def claims(pool: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in (pool / "entries").glob("*.claim")]


class Write(Task):
    value: Param[int]
    out: Meta[Path] = field(default_factory=PathGenerator("out.txt"))

    def execute(self):
        self.out.write_text(str(self.value))


class Add(Task):
    """Depends on the previous tasks (the pilot runs them one after another)"""

    inputs: Param[list[Write]]
    out: Meta[Path] = field(default_factory=PathGenerator("out.txt"))

    def execute(self):
        total = sum(int(x.out.read_text()) for x in self.inputs)
        self.out.write_text(str(total))


class Wait(Task):
    """Signals that it started, then waits"""

    started: Meta[Path] = field(default_factory=PathGenerator("started"))

    def execute(self):
        self.started.touch()
        time.sleep(60)


@pytest.mark.timeout(60)
def test_pilot_runs_several_jobs(tmp_path: Path):
    """Jobs (and the jobs depending on them) run in the same pilot"""
    launcher = make_launcher(tmp_path, max_pilots=1)
    workdir = tmp_path / "xp"
    with TemporaryExperiment("pilot", workdir=workdir, timeout_multiplier=6):
        writes = [Write.C(value=i).submit(launcher=launcher) for i in range(3)]
        add = Add.C(inputs=writes).submit(launcher=launcher)
        job = job_of(add)
        assert job.wait() == JobState.DONE
        assert add.out.read_text() == "3"

    (pool,) = pool_dirs(workdir)
    assert len(pilots(pool)) == 1
    assert {c["pilot"] for c in claims(pool)} == set(pilots(pool))
    assert len(claims(pool)) == 4


@pytest.mark.timeout(60)
def test_pilot_job_time_limit(tmp_path: Path):
    """The pilot stops a job at its time limit: the job timed out"""
    launcher = make_launcher(tmp_path).config(time="0:02")
    with TemporaryExperiment("pilot", workdir=tmp_path / "xp", timeout_multiplier=6):
        task = Wait.C().submit(launcher=launcher)
        job = job_of(task)
        state = job.wait()
        assert state.failure_reason == JobFailureStatus.TIMEOUT
        details = FailureDetails.from_path(job.path, job.name)
        assert (
            details is not None and details.message == "Job time limit reached (pilot)"
        )


class SlowFirst(ResumableTask):
    """Times out on its first run, then completes"""

    attempts: Meta[Path] = field(default_factory=PathGenerator("attempts"))

    def execute(self):
        attempt = int(self.attempts.read_text()) + 1 if self.attempts.exists() else 1
        self.attempts.write_text(str(attempt))
        remaining = self.remaining_time()
        assert remaining is not None and remaining <= 2
        if attempt == 1:
            time.sleep(60)


@pytest.mark.timeout(60)
def test_pilot_resumable_retry(tmp_path: Path):
    """A resumable task stopped at its time limit is retried in a pilot"""
    launcher = make_launcher(tmp_path).config(time="0:02")
    with TemporaryExperiment("pilot", workdir=tmp_path / "xp", timeout_multiplier=6):
        task = SlowFirst.C().submit(launcher=launcher, max_retries=2)
        job = job_of(task)
        assert job.wait() == JobState.DONE
        assert job.retry_count == 1
        assert task.attempts.read_text() == "2"


@pytest.mark.timeout(60)
def test_pilot_kill_and_reattach(tmp_path: Path):
    """A running job can be found back from its process specification, and
    killed"""
    launcher = make_launcher(tmp_path)
    with TemporaryExperiment("pilot", workdir=tmp_path / "xp", timeout_multiplier=6):
        task = Wait.C().submit(launcher=launcher)
        job = job_of(task)
        while not task.started.exists():
            assert not job.state.finished(), job.state
            time.sleep(0.1)

        spec = json.loads(job.pidpath.read_text())
        assert spec["type"] == "slurm-pilot"
        process = Process.fromDefinition(LocalConnector.instance(), spec)
        assert isinstance(process, PilotProcess)

        # Shown by the UIs and the CLI
        detail = Process.describe_definition(spec)
        assert detail is not None and detail.startswith("running in pilot SLURM job")
        (pilot,) = list_pilots(tmp_path / "xp")[0]
        assert pilot.status == "running"
        assert pilot.job == str(job.path)

        process.kill()
        state = job.wait()
        assert state.failure_reason == JobFailureStatus.CANCELLED
        assert Process.fromDefinition(LocalConnector.instance(), spec) is None


@pytest.mark.timeout(60)
def test_pilot_too_long(tmp_path: Path):
    """A job longer than a pilot is submitted with sbatch"""
    launcher = make_launcher(tmp_path, time=60).config(time="2:00")
    with TemporaryExperiment("pilot", workdir=tmp_path / "xp", timeout_multiplier=6):
        task = Write.C(value=1).submit(launcher=launcher)
        job = job_of(task)
        assert job.wait() == JobState.DONE
    assert not (tmp_path / "xp" / "pilots").exists()


def test_pilot_config():
    """config() keeps the pilot options, unless told otherwise"""
    launcher = SlurmLauncher(pilot=PilotOptions(time="1:00:00"))
    assert launcher.config(time="10").pilot == launcher.pilot
    assert launcher.config(pilot=None).pilot is None
    assert PilotOptions(time="1-00:00:00").seconds == 24 * 3600


@pytest.mark.parametrize(
    "value,seconds",
    [
        (4, 240),
        ("4", 240),
        ("4:30", 270),
        ("1:00:00", 3600),
        ("1-2", 26 * 3600),
        ("1-2:03", 26 * 3600 + 180),
        ("1-2:03:04", 26 * 3600 + 184),
    ],
)
def test_parse_slurm_duration(value, seconds):
    assert parse_slurm_duration(value) == seconds


def test_remaining_time_deadline(monkeypatch):
    """A job run by a pilot uses its own deadline"""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(slurm_base, "_slurm_job_end_time", None)
    monkeypatch.setenv(DEADLINE_ENV, str(time.time() + 100))
    remaining = SlurmLauncherInformation().remaining_time()
    assert remaining is not None and 98 < remaining <= 100


@pytest.mark.timeout(60)
def test_pilot_cancel_queued(tmp_path: Path):
    """A job waiting for a pilot is cancelled without starting"""
    # No pilot can be submitted: the job stays in the queue
    launcher = make_launcher(tmp_path, max_pilots=0)
    with TemporaryExperiment("pilot", workdir=tmp_path / "xp", timeout_multiplier=6):
        task = Write.C(value=1).submit(launcher=launcher)
        job = job_of(task)
        while not job.pidpath.exists():
            time.sleep(0.1)
        spec = json.loads(job.pidpath.read_text())
        detail = Process.describe_definition(spec)
        assert detail is not None and detail.startswith("waiting for a pilot")
        assert len(list_pilots(tmp_path / "xp")[1]) == 1

        process = Process.fromDefinition(LocalConnector.instance(), spec)
        assert process is not None
        process.kill()
        state = job.wait()
        assert state.failure_reason == JobFailureStatus.CANCELLED
        details = FailureDetails.from_path(job.path, job.name)
        assert details is not None
        assert details.message == "Cancelled before it started"


@pytest.mark.timeout(60)
def test_pilot_stops_after_experiment(tmp_path: Path):
    """An idle pilot stops soon after no experiment uses its pool, well
    before its idle timeout"""
    launcher = make_launcher(tmp_path, idle_timeout=600, linger=1)
    workdir = tmp_path / "xp"
    with TemporaryExperiment("pilot", workdir=workdir, timeout_multiplier=6):
        assert job_of(Write.C(value=1).submit(launcher=launcher)).wait() == (
            JobState.DONE
        )
    (pool,) = pool_dirs(workdir)
    (pilot_id,) = pilots(pool)
    deadline = time.time() + 20
    while not (pool / "pilots" / f"{pilot_id}.stopped").exists():
        assert time.time() < deadline, "the pilot did not stop"
        time.sleep(0.1)


def test_pilot_stop_command(tmp_path: Path):
    """Pilots can be asked to stop (CLI)"""
    d = PoolDirectory(tmp_path / "pilots" / "pool")
    d.mkdirs()
    (d.pilots / "p1.json").write_text(json.dumps({"slurm_id": "12"}))
    (d.pilots / "p2.json").write_text(json.dumps({"slurm_id": "13"}))
    (d.pilots / "p2.stopped").write_text("{}")
    assert [p.id for p in stop_pilots(tmp_path)] == ["p1"]
    assert (d.pilots / "p1.stop").exists()
    statuses = {p.id: p.status for p in list_pilots(tmp_path)[0]}
    assert statuses == {"p1": "pending", "p2": "stopped"}


def test_pilot_claim_exclusive(tmp_path: Path):
    """A job is taken by one pilot only"""
    d = PoolDirectory(tmp_path / "pool")
    d.mkdirs()
    (d.queue / "job.json").write_text(json.dumps({"time": 10, "cwd": "/"}))
    workers = [
        Worker(
            d.path,
            f"p{i}",
            time_limit=100,
            idle_timeout=10,
            margin=1,
            kill_wait=1,
            poll_interval=1,
            linger=10,
        )
        for i in range(2)
    ]
    first, second = (w.claim() for w in workers)
    assert first is not None and first[0] == "job"
    assert second is None
    assert json.loads(d.claim_path("job").read_text())["pilot"] == "p0"
    assert d.pending() == []


def test_process_detail_web():
    """The process details are sent to the web UI"""
    from experimaestro.webui.websocket import _process_to_frontend

    process = _process_to_frontend({"pid": "x", "type": "slurm-pilot", "detail": "d"})
    assert process is not None and process["detail"] == "d"
