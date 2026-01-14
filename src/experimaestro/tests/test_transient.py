"""Tests for transient task functionality"""

from tempfile import TemporaryDirectory
from typing import List, Optional
from experimaestro import Config, Task, Param, TransientMode
from experimaestro.scheduler import JobState
from experimaestro.scheduler.base import Scheduler
from .utils import TemporaryExperiment


class TransientTask(Task):
    """A simple task that can be marked as transient"""

    x: Param[int]

    def execute(self):
        print(f"TransientTask x={self.x}")  # noqa: T201


class DependentTask(Task):
    """A task that depends on TransientTask"""

    deps: Param[List[TransientTask]]

    def execute(self):
        print(f"DependentTask with {len(self.deps)} deps")  # noqa: T201


class SingleDependentTask(Task):
    """A task that depends on a single TransientTask"""

    dep: Param[TransientTask]

    def execute(self):
        print(f"SingleDependentTask with dep x={self.dep.x}")  # noqa: T201


class ChainableTask(Task):
    """A task that can depend on any Config (for chain testing)"""

    dep: Param[Optional[Config]]

    def execute(self):
        print("ChainableTask executed")  # noqa: T201


def test_transient_with_dependents():
    """Transient task should run when it has dependents"""
    with TemporaryExperiment("transient_with_deps", timeout_multiplier=12):
        # Submit transient task
        a = TransientTask.C(x=1).submit(transient=TransientMode.TRANSIENT)

        # Submit dependent tasks
        b1 = SingleDependentTask.C(dep=a).submit()
        b2 = SingleDependentTask.C(dep=a).submit()

    # Transient task should have run because b1 and b2 depend on it
    assert a.__xpm__.job.state == JobState.DONE
    assert b1.__xpm__.job.state == JobState.DONE
    assert b2.__xpm__.job.state == JobState.DONE


def test_transient_without_dependents():
    """Transient task should be skipped when it has no dependents"""
    with TemporaryExperiment("transient_no_deps", timeout_multiplier=12):
        # Submit transient task with no dependents
        a = TransientTask.C(x=1).submit(transient=TransientMode.TRANSIENT)

    # Transient task should remain UNSCHEDULED since it was skipped
    assert a.__xpm__.job.state == JobState.UNSCHEDULED


def test_transient_remove_mode():
    """Transient task with REMOVE mode should have its directory removed"""
    with TemporaryExperiment("transient_remove", timeout_multiplier=12):
        # Submit transient task with REMOVE mode
        a = TransientTask.C(x=1).submit(transient=TransientMode.REMOVE)

        # Submit a dependent task
        b = SingleDependentTask.C(dep=a).submit()

        # Store the job path before the experiment ends
        job_path = a.__xpm__.job.path

    # Both tasks should complete
    assert a.__xpm__.job.state == JobState.DONE
    assert b.__xpm__.job.state == JobState.DONE

    # The transient task's directory should be removed
    assert not job_path.exists(), f"Job path {job_path} should have been removed"


def test_transient_remove_without_dependents():
    """Transient task with REMOVE mode and no dependents should be skipped"""
    with TemporaryExperiment("transient_remove_no_deps", timeout_multiplier=12):
        # Submit transient task with REMOVE mode and no dependents
        a = TransientTask.C(x=1).submit(transient=TransientMode.REMOVE)

    # Task should remain UNSCHEDULED since it was skipped
    assert a.__xpm__.job.state == JobState.UNSCHEDULED


def test_transient_mode_merge_none_wins():
    """When resubmitting, NONE mode should win over transient modes"""
    with TemporaryExperiment("transient_merge_none", timeout_multiplier=12):
        # Submit with TRANSIENT mode first
        a1 = TransientTask.C(x=1).submit(transient=TransientMode.TRANSIENT)

        # Resubmit same task with NONE mode
        a2 = TransientTask.C(x=1).submit(transient=TransientMode.NONE)

        # They should be the same job
        assert a1.__xpm__.job is a2.__xpm__.job

    # The job should have run (NONE mode takes precedence)
    assert a1.__xpm__.job.state == JobState.DONE
    assert a1.__xpm__.job.transient == TransientMode.NONE


def test_transient_mode_merge_transient_wins_over_remove():
    """When resubmitting, TRANSIENT mode should win over REMOVE mode"""
    with TemporaryExperiment("transient_merge_transient", timeout_multiplier=12):
        # Submit with REMOVE mode first
        a1 = TransientTask.C(x=2).submit(transient=TransientMode.REMOVE)

        # Resubmit same task with TRANSIENT mode
        a2 = TransientTask.C(x=2).submit(transient=TransientMode.TRANSIENT)

        # They should be the same job
        assert a1.__xpm__.job is a2.__xpm__.job

    # The transient mode should be merged to TRANSIENT (more conservative)
    assert a1.__xpm__.job.transient == TransientMode.TRANSIENT

    # Job should be UNSCHEDULED since no non-transient dependent exists
    assert a1.__xpm__.job.state == JobState.UNSCHEDULED


def test_transient_chain():
    """Chain of transient tasks should work correctly"""
    with TemporaryExperiment("transient_chain", timeout_multiplier=12):
        # Create a chain: a -> b -> c where a and b are transient
        a = TransientTask.C(x=1).submit(transient=TransientMode.TRANSIENT)
        b = ChainableTask.C(dep=a).submit(transient=TransientMode.TRANSIENT)
        c = ChainableTask.C(dep=b).submit()  # Non-transient

    # All tasks should run because c needs b which needs a
    assert a.__xpm__.job.state == JobState.DONE
    assert b.__xpm__.job.state == JobState.DONE
    assert c.__xpm__.job.state == JobState.DONE


def test_transient_chain_all_transient():
    """Chain of all transient tasks - all should be skipped"""
    with TemporaryExperiment("transient_chain_all", timeout_multiplier=12):
        # Create a chain: a -> b where both are transient
        a = TransientTask.C(x=1).submit(transient=TransientMode.TRANSIENT)
        b = ChainableTask.C(dep=a).submit(transient=TransientMode.TRANSIENT)

    # Both should be skipped since there's no non-transient job at the end
    # b has no dependents → UNSCHEDULED
    # a is transient, and b (its only dependent) is also transient and never runs
    # so a is never started via ensure_started() → UNSCHEDULED
    assert a.__xpm__.job.state == JobState.UNSCHEDULED
    assert b.__xpm__.job.state == JobState.UNSCHEDULED


def test_transient_resubmit_within_experiment():
    """When resubmitting transient job within same experiment, state should reflect final mode"""
    with TemporaryExperiment("transient_resubmit", timeout_multiplier=12):
        # Submit as transient first (will be skipped)
        a1 = TransientTask.C(x=3).submit(transient=TransientMode.TRANSIENT)

        # Resubmit as non-transient - should trigger run
        a2 = TransientTask.C(x=3).submit(transient=TransientMode.NONE)

        # They should be the same job
        assert a1.__xpm__.job is a2.__xpm__.job

    # Job should be done (NONE mode triggers run)
    assert a1.__xpm__.job.state == JobState.DONE
    assert a1.__xpm__.job.transient == TransientMode.NONE


def test_transient_remove_then_transient_across_experiments():
    """Test transient behavior across experiments with shared workspace.

    First experiment: A(REMOVE) -> B, both run, A's directory removed at end.
    Second experiment: A(TRANSIENT) -> B, B already done, A should not run.
    """
    with TemporaryDirectory(prefix="xpm_transient_") as workdir:
        # First experiment: A with REMOVE, B depends on A
        with TemporaryExperiment(
            "transient_across", timeout_multiplier=12, workdir=workdir
        ):
            a1 = TransientTask.C(x=10).submit(transient=TransientMode.REMOVE)
            b1 = SingleDependentTask.C(dep=a1).submit()

            # Store paths for later checks
            a_job_path = a1.__xpm__.job.path

        # Both should complete
        assert a1.__xpm__.job.state == JobState.DONE
        assert b1.__xpm__.job.state == JobState.DONE

        # A's directory should be removed (REMOVE mode)
        assert not a_job_path.exists(), "A's directory should be removed after REMOVE"

        # Clear the scheduler to simulate a fresh experiment
        Scheduler.instance().jobs.clear()

        # Second experiment: same workspace, A with TRANSIENT, B depends on A
        with TemporaryExperiment(
            "transient_across", timeout_multiplier=12, workdir=workdir
        ):
            a2 = TransientTask.C(x=10).submit(transient=TransientMode.TRANSIENT)
            b2 = SingleDependentTask.C(dep=a2).submit()

        # B should be DONE (already completed from previous run - donepath exists)
        assert b2.__xpm__.job.state == JobState.DONE

        # A should be UNSCHEDULED (transient with no need to run since B is done)
        assert a2.__xpm__.job.state == JobState.UNSCHEDULED
        # Verify A was never started (aio_start never called)
        assert a2.__xpm__.job.starttime is None, "A should not have been started"
        # Verify no job folder was created (check for .experimaestro subdir)
        assert not a2.__xpm__.job.path.exists(), (
            "A's job folder should not have been created"
        )
