"""Monitors, cleanup and CLI acting while jobs and experiments run
(see filesystem-protocol.md: I1, I5, I6)"""

import shutil

import pytest

from experimaestro.cli.workspace import _orphans
from experimaestro.locking import create_file_lock
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.tests.utils import is_posix

import filelock

from .conftest import XP_NAME, JobFiles, wait_for
from .tasks import Controllable, Quick

pytestmark = pytest.mark.skipif(not is_posix(), reason="POSIX signals and flock")


@pytest.mark.deviation("D-01")
def test_cleanup_job_lock_excludes_running_job(actors):
    """I1, I5: the job lock taken by the cleanup is the one held by a
    running job"""
    task = Controllable.C(x=1)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task)
    wait_for(job.touch_path().exists, "the task to start")
    assert not job.lock_is_free()

    # Lock taken by the stale pid and job event cleanup (cleanup.py)
    lock = create_file_lock(job.path / ".experimaestro" / f"{job.name}.lock", 0)
    try:
        lock.acquire()
    except filelock.Timeout:
        pass
    else:
        lock.release()
        pytest.fail("the cleanup job lock can be taken while the job runs")
    finally:
        job.release()
        actors.wait_exit(xp)


@pytest.mark.deviation("D-08")
def test_orphan_clean_spares_running_job(actors):
    """I5: ``workspace orphans --clean`` never deletes a running job

    The job becomes an orphan when the user removes its experiment by hand
    while it still runs.
    """
    task = Controllable.C(x=2)
    job = JobFiles.of(actors.workdir, task)

    xp = actors.experiment(task)
    wait_for(job.touch_path().exists, "the task to start")
    actors.job_pids.add(job.process_pid())
    xp.kill()
    xp.wait()
    shutil.rmtree(actors.workdir / "experiments" / XP_NAME)

    _orphans(actors.workdir, clean=True, size=False, include_folders=True)
    assert job.path.is_dir(), "running job deleted"


@pytest.mark.deviation("D-11")
def test_monitor_does_not_write_into_starting_run(actors):
    """A monitor never writes the status of a run being created"""
    xp = actors.experiment(Quick.C(x=3), crashpoints="experiment.run_dir_created=pause")
    actors.wait_reached("experiment.run_dir_created", xp)

    (run_dir,) = (
        path
        for path in (actors.workdir / "experiments" / XP_NAME).iterdir()
        if path.is_dir()
    )
    provider = WorkspaceStateProvider(actors.workdir)
    provider.get_experiments()
    assert not (run_dir / "status.json").exists(), "written by the monitor"

    actors.resume("experiment.run_dir_created")
    assert actors.wait_exit(xp) == 0, actors.logs()


@pytest.mark.deviation("D-07")
def test_monitor_sees_crashed_run_as_not_running(actors):
    """I5: a monitor started after an experiment crashed does not show it as
    running"""
    xp = actors.experiment(Quick.C(x=4), crashpoints="experiment.job_submitted=kill")
    actors.wait_exit(xp)

    provider = WorkspaceStateProvider(actors.workdir)
    assert not provider.is_experiment_running(XP_NAME)
