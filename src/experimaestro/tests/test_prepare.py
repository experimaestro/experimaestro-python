"""Tests for the ``Prepare`` config and its in-memory resource machinery."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from experimaestro import Param, Prepare, RunMode, Task
from experimaestro.scheduler.jobs import JobDependency
from experimaestro.scheduler.prepare import (
    PrepareDependency,
    PrepareResource,
    PrepareTask,
)
from experimaestro.tests.utils import TemporaryExperiment


pytestmark = pytest.mark.tasks


# --- Test-state shared across processes within one test ---------------------

#: Module-level call counter for FakePrep.prepare(), keyed by ``name``.
PREP_CALLS: dict[str, int] = {}

#: Module-level call timestamps for concurrency tests.
PREP_TIMES: dict[str, tuple[float, float]] = {}

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


class CheckedPrep(Prepare):
    """A Prepare that can run as a job: its state is a marker file on disk.

    ``prepare()`` may run in a worker process, so calls are counted through
    ``log`` rather than through a module-level counter.
    """

    marker: Param[Path]
    log: Param[Path]

    def is_prepared(self) -> bool:
        return self.marker.is_file()

    def prepare(self, *args, **kwargs) -> None:
        with self.log.open("a") as out:
            out.write("prepared\n")
        self.marker.write_text("prepared")


class TaskUsingCheckedPrep(Task):
    prep: Param[CheckedPrep]
    touch: Param[Path]
    variant: Param[int]
    """Distinguishes two tasks that only differ by needing to be run again

    (paths do not take part in identifiers)
    """

    def execute(self):
        assert self.prep.marker.is_file(), "Task ran before its Prepare"
        self.touch.write_text("done")


class InnerTask(Task):
    def execute(self):
        pass


class OuterTaskWithPrepAndTask(Task):
    prep: Param[FakePrep]
    inner: Param[InnerTask]

    def execute(self):
        pass


# --- Helpers ----------------------------------------------------------------


def _prepare_deps(task: Task) -> list[PrepareDependency]:
    return [
        dep
        for dep in task.__xpm__.job.dependencies
        if isinstance(dep, PrepareDependency)
    ]


def _job_deps(task: Task) -> list[JobDependency]:
    return [
        dep for dep in task.__xpm__.job.dependencies if isinstance(dep, JobDependency)
    ]


def _prepare_calls(log: Path) -> int:
    """How many times ``CheckedPrep.prepare()`` ran, across processes."""
    if not log.is_file():
        return 0
    return len([line for line in log.read_text().splitlines() if line])


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


# --- Prepare as a job -------------------------------------------------------


class PreparedFiles:
    """Paths shared by a Prepare and the assertions about it."""

    def __init__(self, tmp_path: Path):
        self.data = tmp_path / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.marker = self.data / "marker"
        self.log = self.data / "prepare.log"
        self.workdir = tmp_path / "workdir"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def config(self) -> CheckedPrep:
        return CheckedPrep.C(marker=self.marker, log=self.log)

    def touch_path(self, variant: int = 0) -> Path:
        return self.data / f"task-ran-{variant}"

    @property
    def calls(self) -> int:
        return _prepare_calls(self.log)


@pytest.fixture
def files(tmp_path: Path) -> PreparedFiles:
    return PreparedFiles(tmp_path)


def _run(
    files: PreparedFiles,
    name: str,
    *,
    submit_prepare: bool = True,
    variant: int = 0,
):
    """Run one experiment preparing then using ``files``."""
    PrepareResource.reset()
    with TemporaryExperiment(name, workdir=files.workdir):
        prep = files.config()
        if submit_prepare:
            prep = prep.submit()
        task = TaskUsingCheckedPrep.C(
            prep=prep, touch=files.touch_path(variant), variant=variant
        )
        task.submit()
        return _prepare_job(prep), task


def _prepare_job(prep: CheckedPrep):
    return PrepareResource.for_config(prep).job


def _seed_prepare_done(files: PreparedFiles) -> "Path":
    """Leave behind the .done of a previous run, without the data it stands for.

    The scheduler is a process-wide singleton whose job registry only resets
    between tests, so a second experiment in the same test would not go
    through the disk state at all. Seeding the marker is what a previous
    *process* would have left.
    """
    PrepareResource.reset()
    with TemporaryExperiment("seed", workdir=files.workdir, run_mode=RunMode.DRY_RUN):
        prep = files.config()
        prep.submit()
        job = _prepare_job(prep)

    job.path.mkdir(parents=True, exist_ok=True)
    job.donepath.touch()
    assert not files.marker.exists(), "The seeded state must not include the data"
    return job.donepath


def test_prepare_submit_runs_as_job(files):
    """A submitted Prepare runs as a job, and its dependent waits for it."""
    job, task = _run(files, "prepare-as-job")

    assert files.calls == 1
    assert files.marker.is_file()
    assert files.touch_path().is_file(), "The dependent task did not run"

    # The dependency is a plain job dependency, not an in-process one
    assert _prepare_deps(task) == []
    assert len(_job_deps(task)) == 1

    # ... and the preparation ran in its own job directory
    assert job.donepath.is_file()
    assert job.stdout.is_file(), "The preparation did not run as a process"


def test_prepare_job_adopts_existing_output(files):
    """No .done but the output is there: adopt it instead of preparing again."""
    files.marker.write_text("prepared by someone else")

    job, _ = _run(files, "prepare-adopt")

    assert files.calls == 0, "prepare() ran even though the output was present"
    assert files.touch_path().is_file(), "The dependent task did not run"
    assert job.donepath.is_file(), "The existing output was not adopted"
    assert not job.stdout.exists(), "The preparation ran although it had nothing to do"


def test_prepare_job_reruns_when_output_disappears(files):
    """A .done marker that no longer reflects reality must not be trusted."""
    _seed_prepare_done(files)

    _run(files, "prepare-stale")

    assert files.calls == 1, "prepare() did not run despite a stale .done marker"
    assert files.marker.is_file()
    assert files.touch_path().is_file()


def test_prepare_job_skipped_when_nothing_needs_it(files):
    """Being transient, an unneeded preparation is not redone even if invalid."""
    donepath = _seed_prepare_done(files)

    PrepareResource.reset()
    with TemporaryExperiment("prepare-unneeded", workdir=files.workdir):
        files.config().submit()

    assert files.calls == 0, "prepare() ran although no task needed it"
    assert not donepath.exists(), "The stale marker should have been invalidated"


def test_prepare_job_dedupes_identifier_equal_instances(files):
    """Submitting one instance covers the identifier-equal ones."""
    PrepareResource.reset()
    with TemporaryExperiment("prepare-job-dedup", workdir=files.workdir):
        files.config().submit()

        # A distinct instance, built the way a second prepare_dataset() call would
        other = files.config()
        task = TaskUsingCheckedPrep.C(prep=other, touch=files.touch_path(), variant=0)
        task.submit()

        assert _prepare_deps(task) == [], "The second instance prepared in-process"
        assert len(_job_deps(task)) == 1

    assert files.calls == 1


@pytest.mark.parametrize("run_mode", [RunMode.DRY_RUN, RunMode.GENERATE_ONLY])
def test_submitted_prepare_not_run_in_simulation_modes(files, run_mode):
    """No surprise downloads when the experiment is only being simulated."""
    PrepareResource.reset()
    with TemporaryExperiment(
        "prepare-simulated", workdir=files.workdir, run_mode=run_mode
    ):
        prep = files.config().submit()
        TaskUsingCheckedPrep.C(prep=prep, touch=files.touch_path(), variant=0).submit()

    assert files.calls == 0
    assert not files.marker.exists()


def test_submitted_prepare_runs_in_prepare_mode(files):
    """PREPARE mode schedules nothing, so a submitted Prepare runs in-process."""
    PrepareResource.reset()
    with TemporaryExperiment(
        "prepare-mode", workdir=files.workdir, run_mode=RunMode.PREPARE
    ) as xp:
        prep = files.config().submit()
        TaskUsingCheckedPrep.C(prep=prep, touch=files.touch_path(), variant=0).submit()
        jobspath = xp.workspace.jobspath

    assert files.calls == 1, "PREPARE mode did not prepare the submitted config"
    assert files.marker.is_file()
    assert not jobspath.exists(), "PREPARE mode must not write under jobs/"


def test_prepare_dependencies_go_to_the_job(files):
    """Dependencies declared on the Prepare (tokens) end up on the job."""
    PrepareResource.reset()
    with TemporaryExperiment("prepare-token", workdir=files.workdir) as xp:
        token = xp.token("prepare-test-token", 1)
        prep = files.config().add_dependencies(token.dependency(1))
        prep.submit()

        job = _prepare_job(prep)
        assert any(dep.origin is token for dep in job.dependencies), (
            f"The token did not reach the prepare job: {job.dependencies}"
        )

    assert files.calls == 0, "The transient preparation ran although unneeded"


def test_prepare_task_rechecks_on_the_worker(files):
    """The worker re-checks: another experiment may have won the race."""
    files.marker.write_text("prepared by the winner of the race")

    task = PrepareTask.C(target=files.config()).instance()
    task.execute()

    assert files.calls == 0, "prepare() ran although the target was already prepared"


def test_prepare_submit_requires_is_prepared():
    """A Prepare that cannot check itself must not become a job."""
    with TemporaryExperiment("prepare-no-check", run_mode=RunMode.DRY_RUN):
        with pytest.raises(ValueError, match="is_prepared"):
            FakePrep.C(name="uncheckable").submit()


def test_prepare_inprocess_honours_is_prepared(files):
    """The in-process path skips prepare() when the output is already there."""
    files.marker.write_text("prepared by someone else")

    _run(files, "prepare-inprocess-check", submit_prepare=False)

    assert files.calls == 0
    assert files.touch_path().is_file()
