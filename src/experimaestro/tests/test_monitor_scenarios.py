"""What a monitor shows while experiments run, fail and are relaunched

Each scenario runs real experiments (tasks that succeed, fail or block on
demand, see :class:`Step`) and checks, after each step, the jobs of the
current run as shown by the UIs (status resolved with
:func:`~experimaestro.scheduler.job_status.display_state`, and the counts of
the run):

- by a monitor opened before the first run and kept open, reading the
  workspace directly (``WorkspaceStateProvider``) or through SSH (the monitor
  server run locally);
- by a monitor started at that step, which reads everything from the files.

A difference between the two points to the event handling or the caches of
the monitor kept open; a fresh monitor showing a wrong state points to what is
written on disk or to how it is read.
"""

import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from experimaestro import Meta, Param, Task, experiment
from experimaestro.scheduler.experiment import FailedExperiment
from experimaestro.scheduler.job_status import display_state
from experimaestro.scheduler.state_provider import StateProvider
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

from .test_relaunch_jobs_list import LocalClient
from .utils import get_test_timeout, wait_for

pytestmark = pytest.mark.tasks


class Step(Task):
    """Does what the control file ``{control}/{name}`` says when it runs

    ``ok`` succeeds, ``fail`` fails, ``block`` waits until the file says
    something else (and then does it).
    """

    name: Param[str]
    control: Meta[Path]

    def execute(self):
        path = self.control / self.name
        while (action := path.read_text()) == "block":
            time.sleep(0.02)
        if action != "ok":
            raise RuntimeError(f"{self.name}: {action}")


#: Jobs of a run: name -> names of the jobs it depends on
Jobs = dict[str, list[str]]


class Workspace:
    """A workspace shared by the scenarios

    Each scenario names its experiments and jobs with its own prefix, so the
    scenarios do not see each other; the monitors still have the experiments
    of the previous scenarios to deal with.
    """

    def __init__(self, path: Path, make_provider):
        self.workdir = path / "workdir"
        self.control = path / "control"
        self.workdir.mkdir()
        self.control.mkdir()
        # Marks the working directory (created by the first experiment
        # otherwise): a monitor can be opened before any run
        (self.workdir / ".__experimaestro__").touch()
        self.make_provider = make_provider


class Scenario:
    """Runs experiments in a workspace, and checks what monitors show

    Experiments and jobs are named in the scenario without its prefix.
    """

    def __init__(self, workdir: Path, control: Path, prefix: str, monitor=None):
        self.workdir = workdir
        self.control = control
        self.prefix = prefix
        self.monitor = monitor
        # As the UI page showing the experiments of the scenario
        self.page = object()
        self.watched: set[str] = set()

    def n(self, name: str) -> str:
        """Name in the workspace"""
        return f"{self.prefix}-{name}"

    def close(self):
        self.monitor.unwatch(self.page)

    def watch(self, *names: str):
        self.watched.update(self.n(name) for name in names)
        self.monitor.watch(self.page, set(self.watched))

    def set(self, **actions: str):
        """Sets what the jobs do (``ok``, ``fail`` or ``block``)"""
        for name, action in actions.items():
            (self.control / self.n(name)).write_text(action)

    # --- Runs

    def submit(self, jobs: Jobs) -> None:
        """Submits jobs in the current experiment (in order)"""
        for name, after in jobs.items():
            config: Any = Step.C(name=self.n(name), control=self.control)
            config.tag("name", name)
            for dep in after:
                config.add_dependencies(self.submitted[dep].__xpm__.dependency())
            self.submitted[name] = config.submit()

    @contextmanager
    def running(self, xp_name: str, jobs: Jobs) -> Iterator[experiment]:
        """Runs an experiment (in this process) submitting ``jobs``

        Leaving the block waits for the jobs, a failure being expected
        """
        self.submitted: dict[str, Any] = {}
        try:
            with experiment(
                self.workdir, self.n(xp_name), no_environmental_impact=True
            ) as xp:
                self.submit(jobs)
                yield xp
        except FailedExperiment:
            pass

    def run(self, xp_name: str, jobs: Jobs) -> None:
        """Runs an experiment until its jobs end"""
        with self.running(xp_name, jobs):
            pass

    @contextmanager
    def in_process(self, xp_name: str, jobs: Jobs, *, kill: bool) -> Iterator[None]:
        """Runs an experiment in another process during the block

        When leaving the block, the process is killed (the run is not
        finalized, the jobs still running continue) or waited for.
        """
        jobs_arg = ";".join(f"{name}:{','.join(after)}" for name, after in jobs.items())
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experimaestro.tests.test_monitor_scenarios",
                str(self.workdir),
                str(self.control),
                self.prefix,
                xp_name,
                jobs_arg,
            ],
            env=env,
        )
        try:
            yield
            if not kill:
                process.wait(timeout=get_test_timeout(10))
        finally:
            process.kill()
            process.wait()

    def killed_run(self, xp_name: str, jobs: Jobs):
        """Runs an experiment in another process, killed when leaving the block"""
        return self.in_process(xp_name, jobs, kill=True)

    # --- Views

    def view(self, provider: StateProvider, xp_name: str) -> dict[str, Any]:
        """Jobs of the current run (name -> status) and the counts of the run"""
        xp_name = self.n(xp_name)
        provider_experiment = provider.get_experiment(xp_name)
        if provider_experiment is None:
            return {"jobs": None}
        tags = provider.get_tags_map(xp_name)
        jobs = {}
        for job in provider.get_jobs(xp_name):
            state = display_state(job, provider_experiment)
            status = state.name
            if state.is_error() and state.failure_reason is not None:
                status = f"error:{state.failure_reason.name}"
            name = tags.get(job.identifier, {}).get("name", job.identifier)
            jobs[name] = status
        return {
            "jobs": jobs,
            "counts": (
                provider_experiment.total_jobs,
                provider_experiment.finished_jobs,
                provider_experiment.failed_jobs,
            ),
        }

    def fresh_view(self, xp_name: str) -> dict[str, Any]:
        provider = WorkspaceStateProvider(self.workdir, no_cleanup=True)
        try:
            return self.view(provider, xp_name)
        finally:
            provider.close()

    @staticmethod
    def expected(jobs: dict[str, str]) -> dict[str, Any]:
        statuses = list(jobs.values())
        return {
            "jobs": jobs,
            "counts": (
                len(statuses),
                statuses.count("done"),
                sum(s.startswith("error") for s in statuses),
            ),
        }

    def check(self, xp_name: str, what: str, **jobs: str):
        """Checks that the monitors show ``jobs`` (name -> status) for the
        current run of ``xp_name``

        Waits for both views (events are processed asynchronously, and the
        files are written as the run goes), then compares them at once, so
        that a failure shows whether the monitor started now (which reads the
        files) agrees.
        """
        expected = self.expected(jobs)

        def views() -> dict[str, Any]:
            return {
                "kept open": self.view(self.monitor, xp_name),
                "started now": self.fresh_view(xp_name),
            }

        target = {"kept open": expected, "started now": expected}
        try:
            # Shorter than the test timeout, so that a failure shows the views
            wait_for(
                lambda: views() == target,
                what=what,
                timeout=get_test_timeout(3),
                interval=0.1,
            )
        except AssertionError:
            pass
        assert views() == target, what


def workspace_provider(workdir: Path) -> StateProvider:
    return WorkspaceStateProvider(workdir, no_cleanup=True)


def ssh_provider(workdir: Path) -> StateProvider:
    client = LocalClient(host="localhost", remote_workspace=str(workdir))
    client.connect()
    return client


@pytest.fixture(
    scope="module", params=[workspace_provider, ssh_provider], ids=["direct", "ssh"]
)
def workspace(request, tmp_path_factory):
    return Workspace(tmp_path_factory.mktemp("workspace"), request.param)


@pytest.fixture
def scenario(request, workspace: Workspace):
    """A scenario, with its monitor opened before its first run

    The monitor is opened by each test: the file watchers of the process are
    closed after each test (see conftest.py)
    """
    prefix = re.sub(r"[^a-zA-Z0-9]+", "_", request.node.name).strip("_")
    monitor = workspace.make_provider(workspace.workdir)
    scenario = Scenario(workspace.workdir, workspace.control, prefix, monitor)
    yield scenario
    scenario.close()
    monitor.close()


# --- Failed dependencies


def test_failed_dependency(scenario: Scenario):
    """A job whose dependency fails is shown as failed (because of it)"""
    scenario.watch("xp")
    scenario.set(a="fail", b="ok", c="ok")
    scenario.run("xp", {"a": [], "b": ["a"], "c": []})
    scenario.check("xp", "end", a="error:FAILED", b="error:DEPENDENCY", c="done")


def test_failed_dependency_chain(scenario: Scenario):
    """All the jobs downstream of a failed job are failed"""
    scenario.watch("xp")
    scenario.set(a="fail", b="ok", c="ok", d="ok")
    scenario.run("xp", {"a": [], "b": ["a"], "c": ["b"], "d": ["b", "c"]})
    scenario.check(
        "xp",
        "end",
        a="error:FAILED",
        b="error:DEPENDENCY",
        c="error:DEPENDENCY",
        d="error:DEPENDENCY",
    )


def test_failed_dependency_followed(scenario: Scenario):
    """The monitor follows a job waiting for a dependency that then fails"""
    scenario.watch("xp")
    scenario.set(a="block", b="ok", c="ok")
    with scenario.running("xp", {"a": [], "b": ["a"], "c": []}):
        scenario.check("xp", "a runs", a="running", b="waiting", c="done")
        scenario.set(a="fail")
    scenario.check("xp", "end", a="error:FAILED", b="error:DEPENDENCY", c="done")


def test_failed_dependency_of_a_job_that_failed_before(scenario: Scenario):
    """A job that failed in a previous run, and that the dependency failure
    prevents from running now, is shown as failed because of the dependency
    (not with the failure of the previous run)"""
    scenario.watch("xp")
    scenario.set(a="fail", b="fail")
    scenario.run("xp", {"b": []})
    scenario.check("xp", "first run", b="error:FAILED")

    scenario.set(b="ok")
    scenario.run("xp", {"a": [], "b": ["a"]})
    scenario.check("xp", "second run", a="error:FAILED", b="error:DEPENDENCY")


def test_failed_dependency_again(scenario: Scenario):
    """Relaunching with the dependency still failing, then fixed"""
    scenario.watch("xp")
    scenario.set(a="fail", b="ok")
    jobs = {"a": [], "b": ["a"]}
    scenario.run("xp", jobs)
    scenario.check("xp", "first run", a="error:FAILED", b="error:DEPENDENCY")

    scenario.run("xp", jobs)
    scenario.check("xp", "second run", a="error:FAILED", b="error:DEPENDENCY")

    scenario.set(a="ok")
    scenario.run("xp", jobs)
    scenario.check("xp", "third run", a="done", b="done")


def test_failed_dependency_in_a_killed_run(scenario: Scenario):
    """A run killed after a dependency failed (not finalized)"""
    scenario.watch("xp")
    scenario.set(a="fail", b="ok", c="block")
    with scenario.killed_run("xp", {"a": [], "b": ["a"], "c": []}):
        scenario.check("xp", "run", a="error:FAILED", b="error:DEPENDENCY", c="running")
    scenario.set(c="ok")
    scenario.check(
        "xp", "after the kill", a="error:FAILED", b="error:DEPENDENCY", c="done"
    )


# --- Relaunching


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"a": []}, id="fewer"),
        pytest.param({"a": [], "b": [], "c": [], "d": []}, id="more"),
        pytest.param({"c": [], "d": []}, id="other"),
    ],
)
def test_relaunch_changes_the_jobs(scenario: Scenario, second: Jobs):
    """The jobs shown are those of the new run"""
    scenario.watch("xp")
    scenario.set(a="ok", b="ok", c="ok", d="ok")
    scenario.run("xp", {"a": [], "b": []})
    scenario.check("xp", "first run", a="done", b="done")

    scenario.run("xp", second)
    scenario.check("xp", "second run", **{name: "done" for name in second})


def test_relaunch_shown_while_running(scenario: Scenario):
    """The jobs of a new run are shown as they are submitted and run"""
    scenario.watch("xp")
    scenario.set(a="ok", b="fail", c="block")
    scenario.run("xp", {"a": [], "b": []})
    scenario.check("xp", "first run", a="done", b="error:FAILED")

    scenario.set(b="block")
    with scenario.running("xp", {"a": [], "b": [], "c": ["b"]}):
        scenario.check("xp", "second run", a="done", b="running", c="waiting")
        scenario.set(b="ok")
        scenario.check("xp", "b done", a="done", b="done", c="running")
        scenario.set(c="ok")
    scenario.check("xp", "end", a="done", b="done", c="done")


def test_many_relaunches(scenario: Scenario):
    """Back-to-back runs with changing jobs, checked after each run"""
    scenario.watch("xp")
    scenario.set(a="ok", b="ok", c="fail", d="ok", e="ok")
    runs: list[tuple[Jobs, dict[str, str]]] = [
        ({"a": [], "b": []}, {"a": "done", "b": "done"}),
        (
            {"a": [], "c": [], "d": ["c"]},
            {"a": "done", "c": "error:FAILED", "d": "error:DEPENDENCY"},
        ),
        ({"e": []}, {"e": "done"}),
        (
            {"a": [], "b": [], "c": [], "d": ["c"], "e": []},
            {
                "a": "done",
                "b": "done",
                "c": "error:FAILED",
                "d": "error:DEPENDENCY",
                "e": "done",
            },
        ),
    ]
    for index, (jobs, expected) in enumerate(runs):
        scenario.run("xp", jobs)
        scenario.check("xp", f"run {index}", **expected)


def test_relaunches_not_followed_in_between(scenario: Scenario):
    """Runs following each other faster than the monitor reads them: the
    monitor ends showing the last one"""
    scenario.watch("xp")
    scenario.set(a="ok", b="ok", c="ok", d="fail", e="ok")
    scenario.run("xp", {"a": [], "b": [], "c": []})
    scenario.run("xp", {"a": []})
    scenario.run("xp", {"b": [], "d": [], "e": ["d"]})
    scenario.check("xp", "last run", b="done", d="error:FAILED", e="error:DEPENDENCY")


def test_relaunch_after_a_killed_run(scenario: Scenario):
    """A run killed while a job runs, relaunched with other jobs"""
    scenario.watch("xp")
    scenario.set(a="ok", b="block", c="ok")
    with scenario.killed_run("xp", {"a": [], "b": []}):
        scenario.check("xp", "killed run", a="done", b="running")
    scenario.set(b="ok")
    scenario.check("xp", "killed run, b done", a="done", b="done")

    scenario.run("xp", {"a": [], "c": []})
    scenario.check("xp", "relaunch", a="done", c="done")


def test_monitor_opened_during_the_run(scenario: Scenario):
    """A monitor opened while a relaunched experiment runs"""
    scenario.set(a="fail", b="ok")
    scenario.run("xp", {"a": [], "b": ["a"]})

    scenario.set(a="block")
    with scenario.running("xp", {"a": [], "b": ["a"], "c": []}):
        scenario.set(c="ok")
        # The monitor kept open did not watch the experiment until now
        scenario.watch("xp")
        scenario.check("xp", "second run", a="running", b="waiting", c="done")
        scenario.set(a="fail")
    scenario.check("xp", "end", a="error:FAILED", b="error:DEPENDENCY", c="done")


# --- Several experiments


def test_job_shared_by_two_experiments(scenario: Scenario):
    """Each experiment shows the state of the job in its own run"""
    scenario.watch("xp1", "xp2")
    scenario.set(a="fail", b="ok", c="block")
    # In xp1, b cannot run (a failed)
    scenario.run("xp1", {"a": [], "b": ["a"]})
    scenario.check("xp1", "xp1", a="error:FAILED", b="error:DEPENDENCY")

    # In xp2 (run by another process, as experiments usually are), b waits
    # for c
    with scenario.in_process("xp2", {"c": [], "b": ["c"]}, kill=False):
        scenario.check("xp2", "xp2 runs", c="running", b="waiting")
        scenario.check(
            "xp1", "xp1 while xp2 runs", a="error:FAILED", b="error:DEPENDENCY"
        )
        scenario.set(c="ok")
    scenario.check("xp2", "xp2 end", c="done", b="done")


def test_two_experiments_relaunched(scenario: Scenario):
    """Relaunching an experiment does not change what another one shows"""
    scenario.watch("xp1", "xp2")
    scenario.set(a="ok", b="ok", c="ok")
    scenario.run("xp1", {"a": [], "b": []})
    scenario.run("xp2", {"b": [], "c": []})
    scenario.check("xp1", "xp1", a="done", b="done")
    scenario.check("xp2", "xp2", b="done", c="done")

    scenario.run("xp1", {"a": []})
    scenario.check("xp1", "xp1 relaunched", a="done")
    scenario.check("xp2", "xp2 after xp1 relaunched", b="done", c="done")


# --- Killed run (run in another process by Scenario.killed_run)


def main(workdir: str, control: str, prefix: str, xp_name: str, jobs_arg: str):
    scenario = Scenario(Path(workdir), Path(control), prefix)
    jobs: Jobs = {}
    for item in jobs_arg.split(";"):
        name, _, after = item.partition(":")
        jobs[name] = [dep for dep in after.split(",") if dep]
    with scenario.running(xp_name, jobs):
        pass


if __name__ == "__main__":
    # Not this module's ``__main__`` copy: the jobs must run the tasks of
    # experimaestro.tests.test_monitor_scenarios
    from experimaestro.tests.test_monitor_scenarios import main as run_main

    run_main(*sys.argv[1:])
