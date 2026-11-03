"""Tests for the removal of job directories (``scheduler/liveness.py``)"""

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from experimaestro.cli import cli
from experimaestro.scheduler import liveness
from experimaestro.scheduler.interfaces import BaseJob
from experimaestro.scheduler.liveness import (
    is_removal_leftover,
    lock_is_held,
    remove_job_directory,
    remove_removal_leftovers,
)
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.tests.test_workspace_state_provider import create_orphan_job


def _job(path: Path):
    return SimpleNamespace(
        path=path,
        identifier=path.name,
        lockpath=BaseJob.get_lockfile(path, "learn"),
        pidfile=path / "learn.pid",
    )


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    (workspace / ".__experimaestro__").touch()
    return workspace


def test_remove_job_directory_after_releasing_the_lock(tmp_path, monkeypatch):
    """The directory is removed once the lock (a file inside it) is closed

    On NFS, removing a directory holding an open file fails (the file is
    renamed ``.nfsXXXX``): the job directory is renamed under the lock and
    removed after the lock is released.
    """
    workspace = _workspace(tmp_path)
    job_path = create_orphan_job(workspace, "pkg.Learn", "job1")
    (job_path / "data").mkdir()
    (job_path / "data" / "model.bin").write_bytes(b"x")
    job = _job(job_path)

    removed = []
    rmtree = liveness.shutil.rmtree

    def check_rmtree(path, *args, **kwargs):
        # The job path is free and the lock released when removing
        assert not job_path.exists()
        assert is_removal_leftover(Path(path))
        assert not lock_is_held(Path(path) / job.lockpath.name)
        removed.append(path)
        rmtree(path, *args, **kwargs)

    monkeypatch.setattr(liveness.shutil, "rmtree", check_rmtree)
    remove_job_directory(job)

    assert len(removed) == 1
    assert list(job_path.parent.iterdir()) == []


def test_removal_leftovers(tmp_path, monkeypatch):
    """A directory that could not be removed is ignored by the orphan scan
    and removed later"""
    workspace = _workspace(tmp_path)
    job_path = create_orphan_job(workspace, "pkg.Learn", "job1")

    def failing_rmtree(path, *args, **kwargs):
        raise OSError(39, "Directory not empty")

    with monkeypatch.context() as m:
        m.setattr(liveness.shutil, "rmtree", failing_rmtree)
        remove_job_directory(_job(job_path))  # logged, does not raise

    assert not job_path.exists()
    (leftover,) = job_path.parent.iterdir()
    assert is_removal_leftover(leftover)
    assert WorkspaceStateProvider(workspace).get_orphan_jobs() == []

    assert remove_removal_leftovers(workspace / "jobs") == 1
    assert list(job_path.parent.iterdir()) == []


def test_cli_orphans_clean_continues_after_a_failure(tmp_path, monkeypatch):
    """``workspace orphans --clean`` removes the other orphans when one fails"""
    workspace = _workspace(tmp_path)
    paths = [create_orphan_job(workspace, "pkg.Learn", f"job{i}") for i in range(3)]
    remove = liveness.remove_job_directory

    def remove_or_fail(job):
        if job.path.name == "job1":
            raise OSError(39, "Directory not empty")
        remove(job)

    monkeypatch.setattr(liveness, "remove_job_directory", remove_or_fail)
    result = CliRunner().invoke(
        cli, ["--workdir", str(workspace), "workspace", "orphans", "--clean"]
    )

    assert result.exit_code == 0, result.output
    assert "failed" in result.output
    assert "1 orphan job(s) could not be removed" in result.output
    assert [p.exists() for p in paths] == [False, True, False]
