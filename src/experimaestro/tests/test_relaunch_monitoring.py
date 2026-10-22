"""Monitoring a job relaunched after it failed

A monitor (``WorkspaceStateProvider``, as used by the TUI and the web UI) kept
open while a failed job is submitted again must follow the new run: its
progress and its process, without restarting the monitor.

The progress of the new run (lower than the one the failed run reached) is
followed by the provider; the reported "progress ignored after a relaunch" is
not reproduced at this level (UI layers or the SSH provider remain to check).
"""

import json
import time
from pathlib import Path

import pytest

from experimaestro import Meta, Task, progress
from experimaestro.scheduler.interfaces import JobState
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

from .utils import TemporaryExperiment

pytestmark = pytest.mark.tasks


class FailOnce(Task):
    """Reports progress and waits for the test; fails on its first run"""

    control: Meta[Path]

    def execute(self):
        first = not (self.control / "failed").exists()
        run = 1 if first else 2
        progress(0.6 if first else 0.1)
        (self.control / f"running{run}").touch()
        while not (self.control / f"release{run}").exists():
            time.sleep(0.05)
        if first:
            (self.control / "failed").touch()
            raise RuntimeError("first run fails")


def wait_for(condition, timeout: float = 30.0, message: str = "") -> None:
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, f"timeout: {message}"
        time.sleep(0.1)


def pid(job) -> int:
    pid_file = job.path / f"{job.task_id.rsplit('.', 1)[-1]}.pid"
    wait_for(pid_file.exists, message="pid file")
    return json.loads(pid_file.read_text())["pid"]


def test_relaunched_job_is_followed(tmp_path):
    workdir, control = tmp_path / "workdir", tmp_path / "control"
    workdir.mkdir()
    control.mkdir()
    provider = WorkspaceStateProvider(workdir, no_cleanup=True)

    def monitored():
        return provider.get_job(job.task_id, job.identifier)

    def check(progress: float):
        """The monitor shows the progress and the process of the current run"""
        expected_pid = pid(job)
        wait_for(
            lambda: monitored() is not None and monitored().state.running(),
            message="running state",
        )
        wait_for(
            lambda: [p.progress for p in monitored().progress] == [progress],
            message=f"progress {progress} (got {monitored().progress})",
        )
        info = provider.get_process_info(monitored())
        assert info is not None and info.pid == expected_pid and info.running
        assert monitored().getprocess().tospec()["pid"] == expected_pid
        # The serialized job (web UI updates, SSH answers) has the same process
        assert monitored().state_dict()["process"]["pid"] == expected_pid

    try:
        # The monitor follows the first run (as a user looking at the job)
        with TemporaryExperiment("relaunch", workdir=workdir, timeout_multiplier=6):
            job = FailOnce.C(control=control).submit().__xpm__.job
            try:
                wait_for((control / "running1").exists, message="first run")
                check(0.6)
            finally:
                (control / "release1").touch()
        wait_for(lambda: monitored().state == JobState.ERROR, message="failed state")

        # Relaunch, keeping the monitor open
        with TemporaryExperiment("relaunch", workdir=workdir, timeout_multiplier=6):
            FailOnce.C(control=control).submit()
            try:
                wait_for((control / "running2").exists, message="relaunch")
                check(0.1)
            finally:
                (control / "release2").touch()
    finally:
        provider.close()
