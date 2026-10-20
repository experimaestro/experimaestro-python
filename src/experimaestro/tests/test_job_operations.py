"""Tests for the shared job operations, beyond the UI feature tests

The UI feature tests (``tests/ui/``) cover the operations as the TUI and the
web UI use them; these cover the CLI and the live scheduler.
"""

import asyncio
import threading
from types import SimpleNamespace

from click.testing import CliRunner

from experimaestro.cli import cli
from experimaestro.scheduler.base import Scheduler
from experimaestro.scheduler.interfaces import JobState
from experimaestro.tests.test_workspace_state_provider import create_orphan_job
from experimaestro.utils.disk_usage import folder_size, format_size


def test_cli_stray_delete(tmp_path):
    """``workspace stray --delete --perform`` deletes the finished orphan jobs"""
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    (workspace / ".__experimaestro__").touch()
    job_path = create_orphan_job(workspace, "pkg.Learn", "orphan1")

    result = CliRunner().invoke(
        cli,
        ["--workdir", str(workspace), "workspace", "stray", "--delete", "--perform"],
    )

    assert result.exit_code == 0, result.output
    assert "DELETED" in result.output
    assert not job_path.exists()


class _Process:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


class _Job:
    """A live job re-attached from an earlier run: no process handle"""

    identifier = "job1"
    scheduler_state = JobState.RUNNING

    def __init__(self, process):
        self._process = None
        self._pid_file_process = process

    async def aio_process(self):
        # Job.aio_process falls back to the PID file
        return self._pid_file_process


def test_scheduler_kill_uses_pid_file_process():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        process = _Process()
        job = _Job(process)
        scheduler = SimpleNamespace(jobs={job.identifier: job}, loop=loop)

        assert Scheduler.kill_job(scheduler, job, perform=True)
        assert process.killed
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_folder_size(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "data").write_bytes(b"x" * 10_000)
    # Disk usage: at least the data, rounded up to whole blocks
    assert folder_size(tmp_path) >= 10_000
    assert folder_size(tmp_path / "missing") == 0


def test_format_size():
    assert format_size(512) == "512B"
    assert format_size(1536) == "1.5KB"
    assert format_size(3 * 1024**3) == "3.0GB"
