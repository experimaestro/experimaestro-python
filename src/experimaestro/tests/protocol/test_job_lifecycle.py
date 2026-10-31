"""Job process lifecycle (see filesystem-protocol.md: I1, I2, I3, I7, I10)"""

import json
import os
import signal
import subprocess
import sys

import pytest

from experimaestro.tests.utils import is_posix

from .conftest import JobFiles, wait_for
from .tasks import CatchCancel, Controllable, Failing, Quick

pytestmark = pytest.mark.skipif(not is_posix(), reason="POSIX signals and flock")


def test_killed_job_is_restarted(actors):
    """I7: a job killed while running is run again by the next experiment"""
    task = Quick.C(x=1)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task, crashpoints="job.running=kill")
    actors.wait_exit(xp)
    assert not job.done.exists()
    assert job.runs == 0

    xp = actors.experiment(task)
    assert actors.wait_exit(xp) == 0, actors.logs()
    assert job.done.exists()
    assert job.runs == 1


def test_killed_after_failed_marker(actors):
    """I3, I7: a job killed right after writing ``.failed`` is seen as failed"""
    task = Failing.C(x=1)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task, crashpoints="job.failed_written=kill")
    actors.wait_exit(xp)
    assert json.loads(job.failed.read_text())["reason"] == "failed"
    assert not job.done.exists()
    assert job.lock_is_free()


def test_failure_marker_before_lock_release(actors):
    """I2: on failure, ``.failed`` exists when the job lock is released"""
    task = Failing.C(x=2)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task, crashpoints="job.lock_released=pause")
    actors.wait_reached("job.lock_released", xp)
    assert job.failed.exists()


@pytest.mark.deviation("D-02")
def test_success_marker_before_lock_release(actors):
    """I2: on success, ``.done`` exists when the job lock is released

    Otherwise another scheduler can take the lock, see neither a pid file nor
    a marker, and start the job again.
    """
    task = Quick.C(x=3)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task, crashpoints="job.lock_released=pause")
    actors.wait_reached("job.lock_released", xp)

    assert job.lock_is_free()
    assert job.done.exists() or job.pid.exists(), (
        "job lock released with neither .done nor .pid: a new run can start"
    )


@pytest.mark.deviation("D-02")
def test_no_second_run_after_success(actors):
    """I1: a job that succeeded is not run a second time by a concurrent
    experiment that starts it while the first run is ending"""
    task = Quick.C(x=4)
    job = JobFiles.of(actors.workdir, task)

    first = actors.experiment(task, crashpoints="job.lock_released=pause")
    actors.wait_reached("job.lock_released", first)

    # Another experiment (another name: no experiment lock in common)
    # submits the same job while the first run is between the lock release
    # and the .done marker
    second = actors.experiment(task, name="other")
    actors.wait_exit(second)
    actors.resume("job.lock_released")
    actors.wait_exit(first)

    assert job.done.exists()
    assert job.runs == 1


@pytest.mark.deviation("D-03")
def test_cancelled_job_is_not_done(actors):
    """I3: a cancelled job is never marked done, even if the task catches
    the cancellation and exits normally"""
    task = CatchCancel.C(x=1)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task)
    wait_for(job.touch_path().exists, "the task to start")
    os.kill(job.process_pid(), signal.SIGTERM)
    actors.wait_exit(xp)

    assert job.failed.exists()
    assert json.loads(job.failed.read_text())["reason"] == "cancelled"
    assert not job.done.exists()


@pytest.mark.deviation("D-04")
def test_slave_rank_does_not_release_token(actors):
    """I10: a token is held until the master process ends, whatever the
    other ranks of the job do"""
    task = Controllable.C(x=1)
    job = JobFiles.of(actors.workdir, task)
    token_path = actors.workdir / "token"

    xp = actors.experiment(
        task, token={"name": "protocol", "path": str(token_path), "count": 1}
    )
    wait_for(job.touch_path().exists, "the task to start")
    holders = list(token_path.rglob("*@*.json"))
    assert len(holders) == 1, holders

    # Another rank of the same job (as with srun / torchrun) ends first
    subprocess.run(
        [sys.executable, str(job.py)],
        cwd=job.path,
        env={**os.environ, "RANK": "1"},
        check=True,
        capture_output=True,
    )

    assert holders[0].exists(), "token released while the master still runs"
    job.release()
    assert actors.wait_exit(xp) == 0, actors.logs()
