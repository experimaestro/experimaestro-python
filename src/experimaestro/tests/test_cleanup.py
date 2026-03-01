"""Tests for workspace cleanup, in particular that active job events are not deleted."""

import json
import os
from pathlib import Path


from experimaestro.locking import create_file_lock
from experimaestro.scheduler.cleanup import _check_orphaned_job_events


TASK_ID = "my.module.MyTask"
JOB_ID = "abc123def456"
SCRIPTNAME = "MyTask"


def _create_event_file(workspace: Path, task_id: str, job_id: str) -> Path:
    """Create a fake event file in .events/jobs/{task_id}/"""
    events_dir = workspace / ".events" / "jobs" / task_id
    events_dir.mkdir(parents=True, exist_ok=True)
    event_file = events_dir / f"event-{job_id}-0.jsonl"
    event_file.write_text(
        json.dumps({"type": "job_state_changed", "job_id": job_id, "state": "running"})
        + "\n"
    )
    return event_file


def _create_job_dir(workspace: Path, task_id: str, job_id: str) -> Path:
    """Create a job directory with .experimaestro subdirectory."""
    job_path = workspace / "jobs" / task_id / job_id
    xpm_dir = job_path / ".experimaestro"
    xpm_dir.mkdir(parents=True, exist_ok=True)
    return job_path


def _write_status_json(job_path: Path, *, events_count: int | None = None):
    """Write a minimal status.json."""
    status = {
        "job_id": JOB_ID,
        "task_id": TASK_ID,
        "state": "RUNNING",
        "path": str(job_path),
    }
    if events_count is not None:
        status["events_count"] = events_count
    status_path = job_path / ".experimaestro" / "status.json"
    status_path.write_text(json.dumps(status))


def _write_pid_file(job_path: Path, pid: int):
    """Write a PID file simulating a running job process."""
    pid_path = job_path / f"{SCRIPTNAME}.pid"
    pid_path.write_text(json.dumps({"type": "local", "pid": pid}))


def _write_done_marker(job_path: Path):
    """Write a .done marker file."""
    done_path = job_path / f"{SCRIPTNAME}.done"
    done_path.write_text("")


def _write_failed_marker(job_path: Path):
    """Write a .failed marker file."""
    failed_path = job_path / f"{SCRIPTNAME}.failed"
    failed_path.write_text(json.dumps({"reason": "FAILED"}))


class TestCleanupSkipsActiveJobs:
    """Tests that _check_orphaned_job_events does not delete event files
    for jobs that are currently active (running or starting)."""

    def test_skips_when_lock_held(self, tmp_path: Path):
        """Event files should NOT be deleted when the job lock is held."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)

        # Hold the lock (simulating a running job process)
        lock_path = job_path / ".experimaestro" / f"{SCRIPTNAME}.lock"
        lock = create_file_lock(lock_path)
        with lock:
            _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), "Event file was deleted while job lock was held"

    def test_skips_when_pid_running(self, tmp_path: Path):
        """Event files should NOT be deleted when PID file shows a running process."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        # Use current process PID (which is definitely running)
        _write_pid_file(job_path, os.getpid())

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job process is running"
        )

    def test_skips_when_no_terminal_markers(self, tmp_path: Path):
        """Event files should NOT be deleted when no .done/.failed markers exist.

        This handles the gap between the scheduler releasing the job lock
        and the job process acquiring it (steps 4-5 of the lifecycle).
        """
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        # No .done, no .failed, no lock, no PID — job may be starting

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job has no terminal markers (may be starting)"
        )

    def test_skips_when_job_dir_missing(self, tmp_path: Path):
        """Event files should NOT be deleted when the job directory doesn't exist.

        The job directory may not yet exist during very early startup.
        """
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        # No job directory created

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job directory doesn't exist (may be starting)"
        )

    def test_cleans_when_job_done(self, tmp_path: Path):
        """Event files SHOULD be deleted when the job is finished (.done exists)."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_done_marker(job_path)

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert not event_file.exists(), (
            "Event file should have been deleted for a finished job"
        )

    def test_cleans_when_job_failed(self, tmp_path: Path):
        """Event files SHOULD be deleted when the job has failed (.failed exists)."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_failed_marker(job_path)

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert not event_file.exists(), (
            "Event file should have been deleted for a failed job"
        )
