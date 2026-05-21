"""Tests for the ``Prepare`` config and its in-memory resource machinery."""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import pytest

from experimaestro import Param, Prepare, RunMode, Task
from experimaestro.scheduler.jobs import JobDependency
from experimaestro.scheduler.prepare import PrepareDependency, PrepareResource
from experimaestro.tests.utils import TemporaryExperiment


pytestmark = pytest.mark.tasks


# --- Test-state shared across processes within one test ---------------------

#: Module-level call counter for FakePrep.prepare(), keyed by ``name``.
PREP_CALLS: Dict[str, int] = {}

#: Module-level call timestamps for concurrency tests.
PREP_TIMES: Dict[str, tuple[float, float]] = {}

#: Lock to protect counter mutations across threads.
_PREP_LOCK = threading.Lock()


@pytest.fixture(autouse=True)
def _reset_prepare_state():
    """Clear the PrepareResource singleton and the call counters between tests."""
    PrepareResource.reset()
    PREP_CALLS.clear()
    PREP_TIMES.clear()
    yield
    PrepareResource.reset()
    PREP_CALLS.clear()
    PREP_TIMES.clear()


# --- Test types -------------------------------------------------------------


class FakePrep(Prepare):
    """A Prepare config whose ``prepare()`` increments a global counter."""

    name: Param[str]

    def prepare(self, *args, **kwargs) -> None:
        with _PREP_LOCK:
            PREP_CALLS[self.name] = PREP_CALLS.get(self.name, 0) + 1


class SlowFakePrep(Prepare):
    """A Prepare config whose ``prepare()`` sleeps so concurrency can be observed."""

    name: Param[str]
    sleep: Param[float]

    def prepare(self, *args, **kwargs) -> None:
        start = time.monotonic()
        time.sleep(self.sleep)
        end = time.monotonic()
        with _PREP_LOCK:
            PREP_TIMES[self.name] = (start, end)
            PREP_CALLS[self.name] = PREP_CALLS.get(self.name, 0) + 1


class TaskUsingPrep(Task):
    prep: Param[FakePrep]

    def execute(self):
        pass


class TaskUsingSlowPrep(Task):
    prep: Param[SlowFakePrep]

    def execute(self):
        pass


class InnerTask(Task):
    def execute(self):
        pass


class OuterTaskWithPrepAndTask(Task):
    prep: Param[FakePrep]
    inner: Param[InnerTask]

    def execute(self):
        pass


# --- Helpers ----------------------------------------------------------------


def _prepare_deps(task: Task) -> List[PrepareDependency]:
    return [
        dep
        for dep in task.__xpm__.job.dependencies
        if isinstance(dep, PrepareDependency)
    ]


# --- Core tests -------------------------------------------------------------


@pytest.mark.parametrize("run_mode", [RunMode.NORMAL, RunMode.PREPARE])
def test_prepare_runs_in_both_modes(run_mode):
    """In both NORMAL and PREPARE modes, prepare() must run before the experiment ends."""
    with TemporaryExperiment("prepare-both-modes", run_mode=run_mode):
        prep = FakePrep.C(name="foo")
        task = TaskUsingPrep.C(prep=prep)
        task.submit()

        # Auto-discovery attached a PrepareDependency to the task's job.
        assert len(_prepare_deps(task)) == 1

    assert PREP_CALLS == {"foo": 1}, (
        f"Expected prepare() to run exactly once in {run_mode}, got {PREP_CALLS!r}"
    )


def test_prepare_dedupes_across_tasks_normal():
    """Two NORMAL tasks referencing the same Prepare share one prepare() call."""
    with TemporaryExperiment("prepare-dedup-normal", run_mode=RunMode.NORMAL):
        prep = FakePrep.C(name="shared")
        TaskUsingPrep.C(prep=prep).submit()
        TaskUsingPrep.C(prep=prep).submit()

    assert PREP_CALLS == {"shared": 1}


def test_prepare_dedupes_across_tasks_prepare_mode():
    """In PREPARE mode, two tasks referencing the same Prepare share one prepare() call."""
    with TemporaryExperiment("prepare-dedup-prepare", run_mode=RunMode.PREPARE):
        prep = FakePrep.C(name="shared")
        TaskUsingPrep.C(prep=prep).submit()
        TaskUsingPrep.C(prep=prep).submit()

    assert PREP_CALLS == {"shared": 1}


def test_prepare_dedupes_by_identifier_not_object_identity():
    """Two FakePrep instances with the same params should share one execution."""
    with TemporaryExperiment("prepare-dedup-id", run_mode=RunMode.PREPARE):
        TaskUsingPrep.C(prep=FakePrep.C(name="same")).submit()
        TaskUsingPrep.C(prep=FakePrep.C(name="same")).submit()

    assert PREP_CALLS == {"same": 1}


def test_prepares_run_concurrently():
    """Distinct Prepare configs should not serialize each other."""
    with TemporaryExperiment("prepare-concurrent", run_mode=RunMode.NORMAL):
        prep_a = SlowFakePrep.C(name="A", sleep=0.3)
        prep_b = SlowFakePrep.C(name="B", sleep=0.3)
        TaskUsingSlowPrep.C(prep=prep_a).submit()
        TaskUsingSlowPrep.C(prep=prep_b).submit()

    assert set(PREP_TIMES.keys()) == {"A", "B"}
    a_start, a_end = PREP_TIMES["A"]
    b_start, b_end = PREP_TIMES["B"]
    # The two intervals must overlap.
    overlap = min(a_end, b_end) - max(a_start, b_start)
    assert overlap > 0, (
        f"Expected A and B prepare() calls to overlap; intervals were "
        f"A=[{a_start:.3f},{a_end:.3f}] B=[{b_start:.3f},{b_end:.3f}]"
    )


def test_prepare_skipped_in_dry_run():
    """DRY_RUN must not invoke prepare()."""
    with TemporaryExperiment("prepare-dry-run", run_mode=RunMode.DRY_RUN):
        prep = FakePrep.C(name="ignored")
        TaskUsingPrep.C(prep=prep).submit()

    assert PREP_CALLS == {}


def test_prepare_skipped_in_generate_only():
    """GENERATE_ONLY must not invoke prepare() (no surprise downloads)."""
    with TemporaryExperiment("prepare-generate", run_mode=RunMode.GENERATE_ONLY):
        prep = FakePrep.C(name="ignored")
        TaskUsingPrep.C(prep=prep).submit()

    assert PREP_CALLS == {}


def test_prepare_leaves_no_on_disk_residue():
    """Prepare configs must not create entries under jobs/."""
    with TemporaryExperiment("prepare-no-disk", run_mode=RunMode.PREPARE) as xp:
        prep = FakePrep.C(name="diskless")
        task = TaskUsingPrep.C(prep=prep)
        task.submit()
        jobspath = xp.workspace.jobspath

    # Verify no directory was created using the FakePrep type's identifier.
    fake_prep_type_id = str(FakePrep.__xpmtype__.identifier)
    assert not (jobspath / fake_prep_type_id).exists(), (
        f"Expected no Prepare workdir under {jobspath}, found one for {fake_prep_type_id}"
    )


def test_prepare_attaches_dependency_in_normal_mode():
    """The submitted task must have a PrepareDependency wired up in NORMAL mode."""
    with TemporaryExperiment("prepare-dep-wired", run_mode=RunMode.DRY_RUN):
        prep = FakePrep.C(name="wired")
        task = TaskUsingPrep.C(prep=prep)
        task.submit()

        deps = _prepare_deps(task)
        assert len(deps) == 1
        assert isinstance(deps[0], PrepareDependency)
        # The PrepareDependency's resource matches the singleton lookup.
        expected = PrepareResource.for_config(prep)
        assert deps[0].origin is expected


def test_no_double_attachment_when_task_also_in_params():
    """Having a Task-typed param alongside a Prepare must not duplicate prep deps."""
    with TemporaryExperiment("prepare-with-task-dep", run_mode=RunMode.DRY_RUN):
        inner = InnerTask.C().submit()
        prep = FakePrep.C(name="combo")
        outer = OuterTaskWithPrepAndTask.C(prep=prep, inner=inner)
        outer.submit()

        prep_deps = _prepare_deps(outer)
        assert len(prep_deps) == 1

        # The Task-typed param still becomes a JobDependency.
        job_deps = [
            dep
            for dep in outer.__xpm__.job.dependencies
            if isinstance(dep, JobDependency)
        ]
        assert len(job_deps) == 1
