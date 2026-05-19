"""Tests for the auxiliary folders feature (beta) — issue #55.

These tests exercise the file-level logic in
``experimaestro.scheduler.folders`` directly so they don't need a
running scheduler. Integration with the scheduler (post-DONE archive
trigger) is exercised by the existing job lifecycle tests by virtue of
``Workspace.folders`` defaulting to empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experimaestro.locking import create_file_lock
from experimaestro.scheduler.folders import (
    _archive_job,
    _Job,
    _sweep_stale_tmp,
    _tmp_lock_path,
    archive_job_async,
    recover_from_folders,
    wait_for_pending,
)
from experimaestro.settings import FolderMode, FolderSettings


TASK_ID = "my.module.MyTask"
JOB_ID = "deadbeef" + "0" * 56  # 64 hex chars


def _build_job(workspace: Path, name: str = "MyTask") -> _Job:
    """Create a fake job directory with the usual artifacts."""
    src = workspace / "jobs" / TASK_ID / JOB_ID
    (src / ".experimaestro").mkdir(parents=True, exist_ok=True)

    # User payload
    (src / "result.txt").write_text("hello\n")
    (src / "params.json").write_text(json.dumps({"version": 3, "value": 1}))
    (src / f"{name}.done").write_text("")
    (src / f"{name}.out").write_text("stdout\n")

    # Stuff that must NOT be copied
    (src / f"{name}.pid").write_text("99999")
    (src / f"{name}.lock").write_text("")
    (src / ".scheduler").mkdir()
    (src / ".scheduler" / "junk").write_text("noise")

    # Stuff that must be preserved
    (src / ".experimaestro" / "task-outputs.jsonl").write_text(
        '{"event": "checkpoint", "step": 1}\n'
    )

    return _Job(src=src, rel=Path(TASK_ID) / JOB_ID, name=name)


class TestBackup:
    def test_backup_copies_payload(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        dest = _archive_job(job, folder)

        assert dest == tmp_path / "backup" / "jobs" / TASK_ID / JOB_ID
        assert dest.is_dir()
        assert (dest / "result.txt").read_text() == "hello\n"
        assert (dest / "params.json").exists()
        assert (dest / "MyTask.done").exists()
        assert (dest / "MyTask.out").read_text() == "stdout\n"
        assert (dest / ".experimaestro" / "task-outputs.jsonl").exists()

        # Source unchanged
        assert job.src.is_dir()
        assert not job.src.is_symlink()

    def test_backup_skips_internal_files(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        dest = _archive_job(job, folder)

        assert not (dest / "MyTask.pid").exists()
        assert not (dest / "MyTask.lock").exists()
        assert not (dest / ".scheduler").exists()

    def test_backup_is_atomic(self, tmp_path: Path):
        """The .tmp.<hash> staging dir should not survive a successful run."""
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        _archive_job(job, folder)

        siblings = list((tmp_path / "backup" / "jobs" / TASK_ID).iterdir())
        # Only the final job dir should remain — no .tmp.* lying around.
        assert all(not s.name.startswith(".tmp.") for s in siblings)

    def test_backup_idempotent_when_dest_exists(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        # Pre-populate the destination
        dest_root = tmp_path / "backup" / "jobs" / TASK_ID / JOB_ID
        dest_root.mkdir(parents=True)
        (dest_root / "previous.txt").write_text("old")

        result = _archive_job(job, folder)

        assert result == dest_root
        # Existing content preserved (we don't overwrite)
        assert (dest_root / "previous.txt").read_text() == "old"
        # And no copy happened
        assert not (dest_root / "result.txt").exists()


class TestMove:
    def test_move_replaces_source_with_symlink(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "fast", mode=FolderMode.MOVE)

        dest = _archive_job(job, folder)

        assert dest.is_dir()
        assert (dest / "result.txt").read_text() == "hello\n"

        # Original location is now a symlink pointing at the dest
        assert job.src.is_symlink()
        assert job.src.resolve() == dest.resolve()
        # And reads transparently work
        assert (job.src / "result.txt").read_text() == "hello\n"


class TestRecovery:
    def test_recover_copies_from_backup(self, tmp_path: Path):
        # Set up a "backup-only" layout (no primary dir)
        backup_root = tmp_path / "backup"
        src = backup_root / "jobs" / TASK_ID / JOB_ID
        src.mkdir(parents=True)
        (src / "result.txt").write_text("recovered\n")

        primary = tmp_path / "primary" / "jobs"
        folder = FolderSettings(path=backup_root, mode=FolderMode.BACKUP)

        dest = recover_from_folders(Path(TASK_ID) / JOB_ID, primary, [folder])

        assert dest == primary / TASK_ID / JOB_ID
        assert (dest / "result.txt").read_text() == "recovered\n"

    def test_recover_no_op_when_primary_exists(self, tmp_path: Path):
        primary = tmp_path / "primary" / "jobs"
        existing = primary / TASK_ID / JOB_ID
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("primary\n")

        backup_root = tmp_path / "backup"
        src = backup_root / "jobs" / TASK_ID / JOB_ID
        src.mkdir(parents=True)
        (src / "keep.txt").write_text("backup\n")

        folder = FolderSettings(path=backup_root, mode=FolderMode.BACKUP)

        dest = recover_from_folders(Path(TASK_ID) / JOB_ID, primary, [folder])

        assert dest == existing
        # We didn't overwrite the primary
        assert (existing / "keep.txt").read_text() == "primary\n"

    def test_recover_returns_none_when_not_found(self, tmp_path: Path):
        primary = tmp_path / "primary" / "jobs"
        folder = FolderSettings(path=tmp_path / "empty", mode=FolderMode.BACKUP)

        dest = recover_from_folders(Path(TASK_ID) / JOB_ID, primary, [folder])

        assert dest is None


class TestWorker:
    def test_async_archive_completes(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        # Fake "Job" object exposing just what archive_job_async needs
        class FakeJob:
            path = job.src
            relpath = job.rel
            name = job.name

        archive_job_async(FakeJob(), [folder])
        assert wait_for_pending(timeout=10.0)

        dest = tmp_path / "backup" / "jobs" / TASK_ID / JOB_ID
        assert (dest / "result.txt").read_text() == "hello\n"

    def test_use_mode_is_skipped_by_archive(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "elsewhere", mode=FolderMode.USE)

        class FakeJob:
            path = job.src
            relpath = job.rel
            name = job.name

        archive_job_async(FakeJob(), [folder])
        assert wait_for_pending(timeout=2.0)

        # Nothing was copied
        assert not (tmp_path / "elsewhere").exists()


class TestAutoRecovery:
    """Auto-recovery on submit: a job archived in a folder should be
    picked up transparently when the experiment is re-run."""

    def test_recovery_helper_restores_job(self, tmp_path: Path):
        # Simulate a prior backup
        backup_root = tmp_path / "backup"
        backup_src = backup_root / "jobs" / TASK_ID / JOB_ID
        backup_src.mkdir(parents=True)
        (backup_src / "result.txt").write_text("from-backup\n")
        (backup_src / "MyTask.done").write_text("")

        # Primary workspace is empty
        primary_jobs = tmp_path / "primary" / "jobs"
        folder = FolderSettings(path=backup_root, mode=FolderMode.BACKUP)

        dest = recover_from_folders(Path(TASK_ID) / JOB_ID, primary_jobs, [folder])

        assert dest is not None
        assert (dest / "MyTask.done").exists()
        assert (dest / "result.txt").read_text() == "from-backup\n"


class TestStaleTmpSweep:
    """Crash recovery: leftover ``.tmp.*`` directories from a previous
    crashed process must be cleaned up automatically. A staging dir
    whose owner is still running (= holds the lock) must NOT be
    touched, to avoid corrupting an in-flight archive on a peer
    process or thread."""

    def test_sweep_removes_unlocked_tmp(self, tmp_path: Path):
        """An unlocked tmp dir = previous process crashed = safe to delete."""
        parent = tmp_path / "jobs" / TASK_ID
        parent.mkdir(parents=True)
        stale = parent / ".tmp.abc123"
        stale.mkdir()
        (stale / "leftover").write_text("crash")
        # No lock file present (the crashed process never created one,
        # or its lock file was cleaned up). Either way, sweep clears.

        _sweep_stale_tmp(parent)
        assert not stale.exists()

    def test_sweep_removes_tmp_with_unheld_lock(self, tmp_path: Path):
        """A tmp dir whose lock file exists but is unlocked = also stale."""
        parent = tmp_path / "jobs" / TASK_ID
        parent.mkdir(parents=True)
        stale = parent / ".tmp.abc123"
        stale.mkdir()
        # Create the lock file but don't hold it — simulates a process
        # that crashed between creating the lock file and acquiring it.
        _tmp_lock_path(stale).touch()

        _sweep_stale_tmp(parent)
        assert not stale.exists()
        assert not _tmp_lock_path(stale).exists()

    def test_sweep_keeps_locked_tmp(self, tmp_path: Path):
        """A tmp dir whose lock is currently held = in-progress, leave it alone."""
        parent = tmp_path / "jobs" / TASK_ID
        parent.mkdir(parents=True)
        fresh = parent / ".tmp.def456"
        fresh.mkdir()

        lock = create_file_lock(_tmp_lock_path(fresh), timeout=-1)
        lock.acquire()
        try:
            _sweep_stale_tmp(parent)
            assert fresh.exists()
        finally:
            lock.release()

    def test_archive_sweeps_before_staging(self, tmp_path: Path):
        """An _archive_job call should auto-clean stale tmp dirs in the
        destination's parent. We seed a stale tmp for a DIFFERENT job
        hash and verify it's gone after archiving."""
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        # Seed a stale tmp from a "previous crashed run" of another job
        # in the same task_id sub-tree (no lock = orphaned).
        backup_parent = tmp_path / "backup" / "jobs" / TASK_ID
        backup_parent.mkdir(parents=True)
        stale = backup_parent / ".tmp.someoldhash"
        stale.mkdir()
        (stale / "junk").write_text("crash")

        _archive_job(job, folder)

        # Stale tmp gone; real archive present
        assert not stale.exists()
        assert (backup_parent / JOB_ID / "result.txt").exists()

    def test_archive_skips_when_peer_holds_lock(self, tmp_path: Path):
        """If another process is currently archiving the same job, our
        attempt should fail-fast (timeout=0) rather than racing on the
        same staging dir."""
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        # Simulate the in-flight peer: acquire the lock that
        # _archive_job will try to take.
        dest_root = tmp_path / "backup" / "jobs" / TASK_ID / JOB_ID
        dest_root.parent.mkdir(parents=True)
        tmp_dir = dest_root.parent / f".tmp.{dest_root.name}"
        lock = create_file_lock(_tmp_lock_path(tmp_dir), timeout=-1)
        lock.acquire()
        try:
            result = _archive_job(job, folder)
            assert result is None
            # Destination not written
            assert not dest_root.exists()
        finally:
            lock.release()


class TestDrain:
    """The exit drain helper must block until every queued archive is
    fully copied to its destination — otherwise transient cleanup or
    Python exit could race with an in-flight copy."""

    def test_wait_blocks_until_archive_visible(self, tmp_path: Path):
        job = _build_job(tmp_path)
        folder = FolderSettings(path=tmp_path / "backup", mode=FolderMode.BACKUP)

        class FakeJob:
            path = job.src
            relpath = job.rel
            name = job.name

        dest_root = tmp_path / "backup" / "jobs" / TASK_ID / JOB_ID
        assert not dest_root.exists()

        archive_job_async(FakeJob(), [folder])

        # The contract: after wait_for_pending() returns True, the
        # destination must be fully materialised on disk.
        assert wait_for_pending(timeout=10.0)
        assert dest_root.is_dir()
        assert (dest_root / "result.txt").exists()


class TestSettingsDeprecation:
    """alt_workspaces should emit a DeprecationWarning when read."""

    def test_alt_workspaces_warns_and_maps_to_use(self, tmp_path: Path):
        from experimaestro.scheduler.workspace import Workspace
        from experimaestro.settings import Settings, WorkspaceSettings

        other = tmp_path / "other-workspace"
        other.mkdir()

        ws_settings = WorkspaceSettings(id="primary", path=tmp_path / "primary")
        ws_settings.alt_workspaces = [str(other)]

        settings = Settings(workspaces=[ws_settings])
        # Reset the once-per-process beta banner so this test can observe it
        Workspace._folders_beta_warned = False

        ws = Workspace(settings, ws_settings, launcher=None)

        with pytest.warns(DeprecationWarning):
            folders = ws.folders

        assert len(folders) == 1
        assert folders[0].mode == FolderMode.USE
        assert folders[0].path == other
