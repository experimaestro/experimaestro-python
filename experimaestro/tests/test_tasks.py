# --- Task and types definitions

from pathlib import Path
import pytest

from experimaestro import *
from experimaestro.tools.jobs import fix_deprecated
from experimaestro.scheduler import FailedExperiment, JobState

from .utils import TemporaryDirectory, TemporaryExperiment, get_times

from .tasks.all import *
from . import restart
from .definitions_types import IntegerTask, FloatTask


def test_task_types():
    with TemporaryExperiment("simple") as xp:
        assert IntegerTask(value=5).submit().__xpm__.job.wait() == JobState.DONE
        assert FloatTask(value=5.1).submit().__xpm__.job.wait() == JobState.DONE


def test_simple_task():
    with TemporaryDirectory(prefix="xpm", suffix="helloworld") as workdir:
        assert isinstance(workdir, Path)
        with TemporaryExperiment("helloworld", workdir=workdir, maxwait=1000):
            # Submit the tasks
            hello = Say(word="hello").submit()
            world = Say(word="world").submit()

            # Concat will depend on the two first tasks
            concat = Concat(strings=[hello, world]).submit()

        assert concat.__xpm__.job.state == JobState.DONE
        assert Path(concat.__xpm__.stdout()).read_text() == "HELLO WORLD\n"


def test_not_submitted():
    """A not submitted task should not be accepted as an argument"""
    with TemporaryExperiment("helloworld", maxwait=2):
        hello = Say(word="hello")
        with pytest.raises(ValueError):
            Concat(strings=[hello])


def test_fail_simple():
    """Failing task... should fail"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failing", maxwait=1000):  # FIXME: should wait less!
            fail = Fail().submit()
            fail.__unwrap__().touch()


def test_foreign_type():
    """When the argument real type is in an non imported module"""
    with TemporaryExperiment("foreign_type", maxwait=2):
        # Submit the tasks
        from .tasks.foreign import ForeignClassB2

        b = ForeignClassB2(x=1, y=2)
        a = ForeignTaskA(b=b).submit()

        assert a.__xpm__.job.wait() == JobState.DONE
        assert a.__xpm__.stdout().read_text().strip() == "1"


def test_fail_dep():
    """Failing task... should cancel dependent"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failingdep"):
            fail = Fail().submit()
            dep = FailConsumer(fail=fail).submit()
            fail.__unwrap__().touch()

    assert fail.__xpm__.job.state == JobState.ERROR
    assert dep.__xpm__.job.state == JobState.ERROR


def test_unknown_attribute():
    """No check when setting attributes while executing"""
    with TemporaryExperiment("unknown"):
        method = SetUnknown().submit()

    assert method.__xpm__.job.wait() == JobState.DONE


def test_function():
    with TemporaryExperiment("function"):
        method = Method(a=1).submit()

    assert method.__xpm__.job.wait() == JobState.DONE


@pytest.mark.skip()
def test_done():
    """Checks that we do not run an already done job"""
    pass


def restart_function(xp):
    restart.Restart().submit()


@pytest.mark.parametrize("terminate", restart.TERMINATES_FUNC)
def test_restart(terminate):
    """Restarting the experiment should take back running tasks"""
    restart.restart(terminate, restart_function)


def test_submitted_twice():
    """Check that a job cannot be submitted twice within the same experiment"""
    with TemporaryExperiment("duplicate", maxwait=10) as xp:
        task1 = SimpleTask(x=1).submit()
        task2 = SimpleTask(x=1).submit()
        assert task1 is task2, f"{id(task1)} != {id(task2)}"


def test_configcache():
    """Test a configuration cache"""

    with TemporaryExperiment("configcache", maxwait=10) as xp:
        task = CacheConfigTask(data=CacheConfig()).submit()

    assert task.__xpm__.job.wait() == JobState.DONE


def test_subparams():
    """Test sub-parameters that allow a main task to contain shared data

    This is useful when e.g. training with a different number of epochs
    """
    from .tasks.subparams import Task

    with TemporaryExperiment("subparam", maxwait=10) as xp:
        task100 = Task(epoch=100, x=1).submit()
        task200 = Task(epoch=200, x=1).submit()

        xp.wait()

        t100, t200 = get_times(task100), get_times(task200)
        assert t100 > t200 or t200 > t100


# ---- Deprecation


class NewConfig(Config):
    __xpmid__ = "new"


@deprecate
class DeprecatedConfig(NewConfig):
    __xpmid__ = "deprecated"


class OldConfig(NewConfig):
    __xpmid__ = "deprecated"


class TaskWithDeprecated(Task):
    p: Param[NewConfig]

    def execute(self):
        pass


def checknewpaths(task_new, task_old_path):
    task_new_path = task_new.__xpm__.job.path  # type: Path

    assert task_new_path.exists(), f"New path {task_new_path} should exist"
    assert task_new_path.is_symlink(), f"New path {task_new_path} should be a symlink"

    assert task_new_path.resolve() == task_old_path


def test_tasks_deprecated_inner():
    """Test that when submitting the task, the computed idenfitier is the one of the new class"""
    with TemporaryExperiment("deprecated") as xp:
        # Check that paths are really different first
        task_new = TaskWithDeprecated(p=NewConfig()).submit(dryrun=True)
        task_old = TaskWithDeprecated(p=OldConfig()).submit(dryrun=True)
        task_deprecated = TaskWithDeprecated(p=DeprecatedConfig()).submit(dryrun=True)

        assert (
            task_new.__xpm__.stdout() != task_old.__xpm__.stdout()
        ), "Old and new path should be different"
        assert (
            task_new.__xpm__.stdout() == task_deprecated.__xpm__.stdout()
        ), "Deprecated path should be the same as non deprecated"

        # OK, now check that automatic linking is performed
        task_old = TaskWithDeprecated(p=OldConfig()).submit()
        task_old.__xpm__.wait()
        task_old_path = task_old.__xpm__.stdout().parent

        # Fix deprecated
        OldConfig.__xpmtype__.deprecate()
        fix_deprecated(xp.workspace.path, True)

        checknewpaths(task_new, task_old_path)


class NewTask(Task):
    x: Param[int]

    def execute(self):
        pass


class OldTask(NewTask):
    __xpmid__ = "deprecated"


@deprecate
class DeprecatedTask(NewTask):
    __xpmid__ = "deprecated"


def test_tasks_deprecated():
    """Test that when submitting the task, the computed idenfitier is the one of the new class"""
    with TemporaryExperiment("deprecated", maxwait=100) as xp:
        # Check that paths are really different first
        task_new = NewTask(x=1).submit(dryrun=True)
        task_old = OldTask(x=1).submit(dryrun=True)
        task_deprecated = DeprecatedTask(x=1).submit(dryrun=True)

        assert (
            task_new.__xpm__.stdout() != task_old.__xpm__.stdout()
        ), "Old and new path should be different"
        assert (
            task_new.__xpm__.stdout() == task_deprecated.__xpm__.stdout()
        ), "Deprecated path should be the same as non deprecated"

        # OK, now check that automatic linking is performed
        task_old = OldTask(x=1).submit()
        task_old.__xpm__.wait()
        task_old_path = task_old.__xpm__.stdout().parent

        # Fix deprecated
        OldTask.__xpmtype__.deprecate()
        fix_deprecated(xp.workspace.path, True)

        checknewpaths(task_new, task_old_path)
