"""Tokens and folder archives (see filesystem-protocol.md: I7, I10)"""

import os
import shutil
import time

import pytest

from experimaestro.scheduler import folders
from experimaestro.scheduler.folders import _archive_job, recover_from_folders
from experimaestro.settings import FolderMode, FolderSettings
from experimaestro.tests.test_folders import TASK_ID, _build_job
from experimaestro.tests.utils import is_posix
from experimaestro.utils.crashpoints import CrashPointReached, activate

from .conftest import JobFiles, wait_for
from .tasks import Controllable


@pytest.mark.skipif(not is_posix(), reason="POSIX signals and flock")
def test_token_kept_by_running_job_after_scheduler_crash(actors):
    """I10: a token stays held by a running job whose scheduler died before
    recording the job process in the token"""
    token = {"name": "protocol", "path": str(actors.workdir / "token"), "count": 1}
    running, waiting = Controllable.C(x=1), Controllable.C(x=2)
    running_job = JobFiles.of(actors.workdir, running)
    waiting_job = JobFiles.of(actors.workdir, waiting)

    first = actors.experiment(
        running, token=token, crashpoints="scheduler.job_launched=kill"
    )
    actors.wait_exit(first)
    wait_for(running_job.touch_path().exists, "the first task to start")

    # The holder file is older than the orphan delay (30 s)
    (holder,) = (actors.workdir / "token").rglob("*@*.json")
    old = time.time() - 120
    os.utime(holder, (old, old))

    second = actors.experiment(waiting, token=token, name="other")
    try:
        wait_for(waiting_job.touch_path().exists, "the second task", multiplier=1)
    except TimeoutError:
        pass  # Still waiting for the token, as it should
    else:
        pytest.fail("the token was given to a second job while the first runs")
    finally:
        running_job.release()
        waiting_job.release()
        actors.wait_exit(second)


@pytest.fixture
def move_folder(tmp_path):
    job = _build_job(tmp_path / "workspace")
    folder = FolderSettings(path=tmp_path / "archive", mode=FolderMode.MOVE)
    yield job, folder
    activate(None)


def test_move_crash_before_symlink_is_recovered(move_folder):
    """I7: a crash between the removal of the job directory and its
    replacement by a symlink is recovered from the archive"""
    job, folder = move_folder
    activate({"folders.source_removed": "raise"})
    with pytest.raises(CrashPointReached):
        _archive_job(job, folder)
    activate({})
    assert not job.src.exists()

    recover_from_folders(job.rel, job.src.parents[1], [folder])
    assert (job.src / f"{job.name}.done").exists()
    assert (job.src / "result.txt").read_text() == "hello\n"


@pytest.mark.deviation("D-21")
def test_move_crash_during_removal_is_recovered(move_folder, monkeypatch):
    """I7: a crash while the job directory is being removed (MOVE mode) is
    recovered from the archive"""
    job, folder = move_folder

    def partial_rmtree(path, *args, **kwargs):
        # Removal order is arbitrary: the marker goes first, then a crash
        (path / f"{job.name}.done").unlink()
        raise CrashPointReached("during rmtree")

    monkeypatch.setattr(folders.shutil, "rmtree", partial_rmtree)
    with pytest.raises(CrashPointReached):
        _archive_job(job, folder)
    monkeypatch.setattr(folders.shutil, "rmtree", shutil.rmtree)

    recover_from_folders(job.rel, job.src.parents[1], [folder])
    assert (job.src / f"{job.name}.done").exists(), (
        "the job directory lost its marker and was not recovered"
    )
    assert TASK_ID in str(job.rel)
