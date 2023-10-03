# --- Task and types definitions

from pathlib import Path
import pytest
import logging
from experimaestro import Config, deprecate, Task, Param
from experimaestro.scheduler.workspace import RunMode
from experimaestro.tools.jobs import fix_deprecated
from experimaestro.scheduler import FailedExperiment, JobState
from experimaestro import SubmitHook, Job, Launcher, LightweightTask

from .utils import TemporaryDirectory, TemporaryExperiment

from .tasks.all import (
    Concat,
    ForeignTaskA,
    Fail,
    FailConsumer,
    SetUnknown,
    Method,
    Say,
    SimpleTask,
    CacheConfig,
    CacheConfigTask,
)
from . import restart
from .definitions_types import IntegerTask, FloatTask


def test_task_types():
    with TemporaryExperiment("simple"):
        assert IntegerTask(value=5).submit().__xpm__.job.wait() == JobState.DONE
        assert FloatTask(value=5.1).submit().__xpm__.job.wait() == JobState.DONE


def test_simple_task():
    with TemporaryDirectory(prefix="xpm", suffix="helloworld") as workdir:
        assert isinstance(workdir, Path)
        with TemporaryExperiment("helloworld", workdir=workdir, maxwait=20):
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


def test_fail_simple():
    """Failing task... should fail"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failing", maxwait=20):
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
    with TemporaryExperiment("duplicate", maxwait=20):
        task1 = SimpleTask(x=1).submit()
        task2 = SimpleTask(x=1).submit()
        assert task1 is task2, f"{id(task1)} != {id(task2)}"


def test_configcache():
    """Test a configuration cache"""

    with TemporaryExperiment("configcache", maxwait=20):
        task = CacheConfigTask(data=CacheConfig()).submit()

    assert task.__xpm__.job.wait() == JobState.DONE


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
    """Test that when submitting the task, the computed identifier is the one of
    the new class"""
    with TemporaryExperiment("deprecated", maxwait=0) as xp:
        # --- Check that paths are really different first
        task_new = TaskWithDeprecated(p=NewConfig()).submit(run_mode=RunMode.DRY_RUN)
        task_old = TaskWithDeprecated(p=OldConfig()).submit(run_mode=RunMode.DRY_RUN)
        task_deprecated = TaskWithDeprecated(p=DeprecatedConfig()).submit(
            run_mode=RunMode.DRY_RUN
        )

        logging.debug("New task ID: %s", task_new.__xpm__.identifier.all.hex())
        logging.debug("Old task ID: %s", task_old.__xpm__.identifier.all.hex())
        logging.debug(
            "Old task (with deprecated flag): %s",
            task_deprecated.__xpm__.identifier.all.hex(),
        )
        assert (
            task_new.stdout() != task_old.stdout()
        ), "Old and new path should be different"

        assert (
            task_new.stdout() == task_deprecated.stdout()
        ), "Deprecated path should be the same as non deprecated"

        # --- Now check that automatic linking is performed

        # Run old task with deprecated configuration
        task_old = TaskWithDeprecated(p=OldConfig()).submit()
        task_old.wait()
        task_old_path = task_old.stdout().parent

        # Fix deprecated
        OldConfig.__xpmtype__.deprecate()
        fix_deprecated(xp.workspace.path, True, False)

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
    """Test that when submitting the task, the computed identifier is the one of
    the new class"""
    with TemporaryExperiment("deprecated", maxwait=20) as xp:
        # Check that paths are really different first
        task_new = NewTask(x=1).submit(run_mode=RunMode.DRY_RUN)
        task_old = OldTask(x=1).submit(run_mode=RunMode.DRY_RUN)
        task_deprecated = DeprecatedTask(x=1).submit(run_mode=RunMode.DRY_RUN)

        assert (
            task_new.stdout() != task_old.stdout()
        ), "Old and new path should be different"
        assert (
            task_new.stdout() == task_deprecated.stdout()
        ), "Deprecated path should be the same as non deprecated"

        # OK, now check that automatic linking is performed
        task_old = OldTask(x=1).submit()
        task_old.wait()
        task_old_path = task_old.stdout().parent

        # Fix deprecated
        OldTask.__xpmtype__.deprecate()
        fix_deprecated(xp.workspace.path, True, False)

        checknewpaths(task_new, task_old_path)


class needs_java(SubmitHook):
    def __init__(self, version: int):
        self.version = version

    def spec(self):
        return self.version

    def process(self, job: Job, launcher: Launcher):
        job.environ["JAVA_HOME"] = "THE_JAVA_HOME"


@needs_java(11)
class HookedTask(Task):
    def execute(self):
        pass


def test_task_submit_hook():
    result = HookedTask().submit(run_mode=RunMode.DRY_RUN)
    assert (
        result.__xpm__.task.__xpm__.job.environ.get("JAVA_HOME", None)
        == "THE_JAVA_HOME"
    )


# --- Test for lightweight tasks


class LightweightConfig(Config):
    def __post_init__(self) -> None:
        self.data = 0


class LightweightTask(LightweightTask):
    x: Param[LightweightConfig]

    def execute(self) -> None:
        self.x.data += 1


class MyLightweightTask(Task):
    x: Param[LightweightConfig]

    def execute(self) -> None:
        assert self.x.data == 1


def test_task_lightweight():
    with TemporaryExperiment("lightweight", maxwait=20):
        x = LightweightConfig()
        lwtask = LightweightTask(x=x)
        assert (
            MyLightweightTask(x=x).add_pretasks(lwtask).submit().__xpm__.job.wait()
            == JobState.DONE
        ), "Pre-tasks should be executed"

        x_2 = LightweightConfig()
        lwtask_2 = LightweightTask(x=x)
        assert (
            MyLightweightTask(x=x_2.add_pretasks(lwtask_2))
            .add_pretasks(lwtask_2)
            .submit()
            .__xpm__.job.wait()
            == JobState.DONE
        ), "Pre-tasks should be run just once"


def test_task_lightweight_init():
    with TemporaryExperiment("lightweight_init", maxwait=20):
        x = LightweightConfig()
        lwtask = LightweightTask(x=x)
        assert (
            MyLightweightTask(x=x).submit(init_tasks=[lwtask]).__xpm__.job.wait()
            == JobState.DONE
        ), "Init tasks should be executed"
