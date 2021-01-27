# --- Task and types definitions

from pathlib import Path
import pytest
import signal

from experimaestro import *
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
        assert Path(concat.stdout()).read_text() == "HELLO WORLD\n"


def test_not_submitted():
    """A not submitted task should not be accepted as an argument"""
    with TemporaryExperiment("helloworld", maxwait=2):
        hello = Say(word="hello")
        with pytest.raises(ValueError):
            Concat(strings=[hello])


def test_fail():
    """Failing task... should fail"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failing", maxwait=2):
            fail = Fail().submit()
            fail.touch()


def test_foreign_type():
    """When the argument real type is in an non imported module"""
    with TemporaryExperiment("foreign_type", maxwait=2):
        # Submit the tasks
        from .tasks.foreign import ForeignClassB2

        b = ForeignClassB2(x=1, y=2)
        a = ForeignTaskA(b=b).submit()

        assert a.__xpm__.job.wait() == JobState.DONE
        assert a.stdout().read_text().strip() == "1"


def test_fail_dep():
    """Failing task... should cancel dependent"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failingdep"):
            fail = Fail().submit()
            dep = FailConsumer(fail=fail).submit()
            fail.touch()

    assert fail.__xpm__.job.wait() == JobState.ERROR
    assert dep.__xpm__.job.wait() == JobState.ERROR


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
