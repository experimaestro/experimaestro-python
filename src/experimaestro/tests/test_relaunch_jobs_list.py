"""Jobs of an experiment relaunched while a monitor is open

A monitor (``WorkspaceStateProvider``, or the SSH client of a remote
workspace, as used by the web UI) answers the jobs of an experiment from its
cache. After a relaunch, it must show the jobs of the new run with their
states, as a monitor started then would: a job that failed in the previous
run and is queued again (SLURM) is shown as scheduled, or as waiting for its
dependencies, not as failed; a job done before counts as finished in the new
run.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from experimaestro import Meta, Param, Task
from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.slurm import SlurmLauncher
from experimaestro.scheduler.remote.client import SSHStateProviderClient
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.webui.websocket import serialize_job

from .utils import TemporaryExperiment, wait_for

pytestmark = pytest.mark.tasks

NAME = "relaunch-list"

#: Mock SLURM commands (see launchers/bin/sbatch)
SLURM_BIN = Path(__file__).parent / "launchers" / "bin"


class FailFirst(Task):
    """Fails while the control file ``ok-{name}`` does not exist"""

    name: Param[str]
    control: Meta[Path]

    def execute(self):
        if not (self.control / f"ok-{self.name}").exists():
            raise RuntimeError("not ok")


class Block(Task):
    """Runs until the control file ``release-{name}`` exists"""

    name: Param[str]
    control: Meta[Path]

    def execute(self):
        while not (self.control / f"release-{self.name}").exists():
            time.sleep(0.05)


def submit(control: Path, name: str, launcher=None, after=None):
    """Submits a job tagged with its name ("block" and "held" run until
    released), after the job `after` if given"""
    task: Any = Block if name in ("block", "held") else FailFirst
    config: Any = task.C(name=name, control=control).tag("name", name)
    if after is not None:
        config.add_dependencies(after.__xpm__.dependency())
    return config.submit(launcher=launcher)


#: First run, killed while "block" runs (argv: workdir, control)
KILLED_RUN = """
import sys
from pathlib import Path
from typing import Any
from experimaestro.tests.test_relaunch_jobs_list import NAME, submit
from experimaestro.tests.utils import TemporaryExperiment

workdir, control = Path(sys.argv[1]), Path(sys.argv[2])
with TemporaryExperiment(NAME, workdir=workdir, timeout_multiplier=20) as xp:
    for name in ("ok", "retried", "dropped", "block"):
        submit(control, name)
    xp.wait()
"""


class LocalClient(SSHStateProviderClient):
    """SSH client running the monitor server locally (no SSH)"""

    def _ssh_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "experimaestro",
            "experiments",
            "--workdir",
            self.remote_workspace,
            "monitor-server",
        ]


def workspace_provider(workdir: Path) -> StateProvider:
    return WorkspaceStateProvider(workdir, no_cleanup=True)


def ssh_provider(workdir: Path) -> StateProvider:
    # Marks the working directory (created by the first experiment otherwise)
    (workdir / ".__experimaestro__").touch()
    client = LocalClient(host="localhost", remote_workspace=str(workdir))
    client.connect()
    return client


def shown(provider: StateProvider) -> dict[str, str]:
    """Jobs shown by the web UI for the current run: name -> status

    The jobs and their tags are read separately: a job submitted in between
    is named by its identifier (and the caller waits for the next answer)
    """
    experiment = provider.get_experiment(NAME)
    tags = provider.get_tags_map(NAME)
    return {
        tags.get(job.identifier, {}).get("name", job.identifier): serialize_job(
            job, experiment=experiment
        )["status"]
        for job in provider.get_jobs(NAME)
    }


def fresh_view(workdir: Path) -> dict[str, str]:
    """Jobs shown by a monitor started now"""
    provider = WorkspaceStateProvider(workdir, no_cleanup=True)
    try:
        return shown(provider)
    finally:
        provider.close()


def finished_run(workdir: Path, control: Path, provider) -> dict[str, str]:
    """First run, which ends: "ok" succeeds, "retried" and "dropped" fail"""
    with TemporaryExperiment(NAME, workdir=workdir, timeout_multiplier=6) as xp:
        for name in ("ok", "retried", "dropped"):
            submit(control, name)
        xp.wait()
    return {"ok": "done", "retried": "error", "dropped": "error"}


def killed_run(workdir: Path, control: Path, provider) -> dict[str, str]:
    """First run, killed while "block" runs (the run is not finalized)"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}
    process = subprocess.Popen(
        [sys.executable, "-c", KILLED_RUN, str(workdir), str(control)], env=env
    )
    expected = {"ok": "done", "retried": "error", "dropped": "error"}
    try:
        wait_for(
            lambda: shown(provider) == {**expected, "block": "running"},
            what="first run",
        )
    finally:
        process.kill()
        process.wait()
        (control / "release-block").touch()
    return {**expected, "block": "done"}


@pytest.fixture
def slurm(tmp_path) -> tuple[SlurmLauncher, Path]:
    """Mock SLURM launcher, and the file holding its jobs pending"""
    shutil.copytree(SLURM_BIN, tmp_path / "bin")
    (tmp_path / "slurm").mkdir()
    launcher = SlurmLauncher(
        connector=LocalConnector.instance(),
        binpath=tmp_path / "bin",
        interval=0.01,
    )
    return launcher, tmp_path / "slurm" / "hold"


def counts(provider: StateProvider) -> tuple[int, int, int]:
    """Jobs of the current run: total, finished and failed"""
    exp = provider.get_experiment(NAME)
    assert exp is not None
    return exp.total_jobs, exp.finished_jobs, exp.failed_jobs


@pytest.mark.parametrize("queued", ["scheduled", "waiting"])
@pytest.mark.parametrize("first_run", [finished_run, killed_run])
@pytest.mark.parametrize("make_provider", [workspace_provider, ssh_provider])
def test_relaunch_shows_the_jobs_of_the_new_run(
    tmp_path, slurm, make_provider, first_run, queued
):
    workdir, control = tmp_path / "workdir", tmp_path / "control"
    workdir.mkdir()
    control.mkdir()
    launcher, hold = slurm
    provider = make_provider(workdir)
    # As the web UI page showing the experiment
    page = object()
    provider.watch(page, {NAME})
    (control / "ok-ok").touch()

    try:
        expected = first_run(workdir, control, provider)
        wait_for(lambda: shown(provider) == expected, what=f"first run {expected}")
        first_run_id = provider.get_current_run(NAME)

        # Relaunch, the monitor kept open: "dropped" is not submitted anymore,
        # and "retried" is either queued by SLURM, or waits for "held" (queued
        # by SLURM)
        (control / "ok-retried").touch()
        (control / "release-held").touch()
        hold.touch()
        with TemporaryExperiment(NAME, workdir=workdir, timeout_multiplier=6) as xp:
            try:
                submit(control, "ok")
                if queued == "scheduled":
                    submit(control, "retried", launcher=launcher)
                    expected = {"ok": "done", "retried": "scheduled"}
                else:
                    held = submit(control, "held", launcher=launcher)
                    submit(control, "retried", after=held)
                    expected = {"ok": "done", "retried": "waiting", "held": "scheduled"}
                wait_for(
                    lambda: provider.get_current_run(NAME) != first_run_id,
                    what="new run",
                )
                # A monitor started now shows the jobs of the new run...
                wait_for(
                    lambda: fresh_view(workdir) == expected,
                    what=f"new run {expected} (monitor started now)",
                )
                # ... and so does the monitor kept open, with the counts of
                # the new run ("ok" is finished)
                total = len(expected)
                try:
                    wait_for(
                        lambda: (
                            (shown(provider), counts(provider))
                            == (expected, (total, 1, 0))
                        ),
                        what="new run",
                    )
                finally:
                    assert (shown(provider), counts(provider)) == (
                        expected,
                        (total, 1, 0),
                    )
            finally:
                hold.unlink()
            xp.wait()

        expected = {name: "done" for name in expected}
        wait_for(
            lambda: (
                (shown(provider), counts(provider)) == (expected, (total, total, 0))
            ),
            what="new run end",
        )
    finally:
        hold.unlink(missing_ok=True)
        provider.close()
