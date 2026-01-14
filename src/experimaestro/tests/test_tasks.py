# --- Task and types definitions

import sys
import time
from pathlib import Path
import pytest
import logging
from experimaestro import (
    Config,
    Task,
    Param,
    ResumableTask,
    Meta,
    field,
    PathGenerator,
)
from experimaestro.scheduler.workspace import RunMode
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
        assert IntegerTask.C(value=5).submit().__xpm__.job.wait() == JobState.DONE
        assert FloatTask.C(value=5.1).submit().__xpm__.job.wait() == JobState.DONE


def test_simple_task():
    with TemporaryDirectory(prefix="xpm", suffix="helloworld") as workdir:
        assert isinstance(workdir, Path)
        with TemporaryExperiment("helloworld", workdir=workdir, timeout_multiplier=9):
            # Submit the tasks
            hello = Say.C(word="hello").submit()
            world = Say.C(word="world").submit()

            # Concat will depend on the two first tasks
            concat = Concat.C(strings=[hello, world]).submit()

        assert concat.__xpm__.job.state == JobState.DONE
        assert Path(concat.stdout()).read_text() == "HELLO WORLD\n"


def test_not_submitted():
    """A not submitted task should not be accepted as an argument"""
    with TemporaryExperiment("helloworld"):
        hello = Say.C(word="hello")
        with pytest.raises(ValueError):
            Concat.C(strings=[hello])


def test_fail_simple():
    """Failing task... should fail"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failing", timeout_multiplier=9):
            fail = Fail.C().submit()
            fail.touch()


def test_foreign_type():
    """When the argument real type is in an non imported module"""
    with TemporaryExperiment("foreign_type"):
        # Submit the tasks
        from .tasks.foreign import ForeignClassB2

        b = ForeignClassB2.C(x=1, y=2)
        a = ForeignTaskA.C(b=b).submit()

        assert a.__xpm__.job.wait() == JobState.DONE
        assert a.stdout().read_text().strip() == "1"


def test_fail_dep():
    """Failing task... should cancel dependent"""
    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("failingdep"):
            fail = Fail.C().submit()
            dep = FailConsumer.C(fail=fail).submit()
            fail.touch()

    assert fail.__xpm__.job.state == JobState.ERROR
    assert dep.__xpm__.job.state == JobState.ERROR


def test_unknown_attribute():
    """No check when setting attributes while executing"""
    with TemporaryExperiment("unknown"):
        method = SetUnknown.C().submit()

    assert method.__xpm__.job.wait() == JobState.DONE


def test_function():
    with TemporaryExperiment("function"):
        method = Method.C(a=1).submit()

    assert method.__xpm__.job.wait() == JobState.DONE


@pytest.mark.skip()
def test_done():
    """Checks that we do not run an already done job"""
    pass


def restart_function(xp):
    restart.Restart.C().submit()


@pytest.mark.parametrize("terminate", restart.TERMINATES_FUNC)
def test_restart(terminate):
    """Restarting the experiment should take back running tasks"""
    restart.restart(terminate, restart_function)


def test_submitted_twice():
    """Check that a job cannot be submitted twice within the same experiment"""
    with TemporaryExperiment("duplicate", timeout_multiplier=9):
        task1 = SimpleTask.C(x=1)
        o1 = task1.submit()

        task2 = SimpleTask.C(x=1)
        o2 = task2.submit()

        print(o1)  # noqa: T201
        assert o1.task is not o2.task
        assert task1.__xpm__.job is task2.__xpm__.job, f"{id(task1)} != {id(task2)}"


def test_configcache():
    """Test a configuration cache"""

    with TemporaryExperiment("configcache", timeout_multiplier=9):
        task = CacheConfigTask.C(data=CacheConfig.C()).submit()

    assert task.__xpm__.job.wait() == JobState.DONE


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
    result = HookedTask.C().submit(run_mode=RunMode.DRY_RUN)
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


def test_task_lightweight_init():
    with TemporaryExperiment("lightweight_init", timeout_multiplier=9):
        x = LightweightConfig.C()
        lwtask = LightweightTask.C(x=x)
        assert (
            MyLightweightTask.C(x=x).submit(init_tasks=[lwtask]).__xpm__.job.wait()
            == JobState.DONE
        ), "Init tasks should be executed"


# --- Test for resumable task resubmission


class ControllableResumableTask(ResumableTask):
    """A resumable task that can be controlled via files"""

    control_file: Meta[Path] = field(default_factory=PathGenerator("control"))

    def execute(self):
        # Wait for control file
        while not self.control_file.is_file():
            time.sleep(0.1)

        # Read control: "fail" to exit with error, "complete" to succeed
        action = self.control_file.read_text().strip()
        self.control_file.unlink()

        if action == "fail":
            sys.exit(1)


def test_resumable_task_resubmit():
    """Test resubmitting a failed ResumableTask within the same experiment"""
    with TemporaryExperiment("resumable_resubmit", timeout_multiplier=12):
        task1 = ControllableResumableTask.C()
        task1.submit()

        # Tell task to fail
        task1.control_file.parent.mkdir(parents=True, exist_ok=True)
        task1.control_file.write_text("fail")

        # Wait for the job to fail
        job = task1.__xpm__.job
        assert job.wait() == JobState.ERROR, "Job should have failed"

        # Resubmit by creating a new instance with same parameters
        task2 = ControllableResumableTask.C()
        task2.submit()

        # Tell task to complete
        task2.control_file.write_text("complete")

        # Wait for the resubmitted job to complete
        assert task2.__xpm__.job.wait() == JobState.DONE


def test_resumable_task_resubmit_across_experiments():
    """Test resubmitting a failed ResumableTask across two experiment instances"""
    with TemporaryDirectory(prefix="xpm", suffix="resubmit_across") as workdir:
        # First experiment: task fails
        try:
            with TemporaryExperiment(
                "resubmit_across", timeout_multiplier=6, workdir=workdir
            ):
                task1 = ControllableResumableTask.C()
                task1.submit()

                # Tell task to fail
                task1.control_file.parent.mkdir(parents=True, exist_ok=True)
                task1.control_file.write_text("fail")
        except Exception as e:
            logging.info("First experiment ended (expected): %s", e)

        # Second experiment: task completes
        with TemporaryExperiment(
            "resubmit_across", timeout_multiplier=12, workdir=workdir
        ):
            task2 = ControllableResumableTask.C()
            task2.submit()

            # Tell task to complete
            task2.control_file.write_text("complete")

            # Wait for the resubmitted job to complete
            assert task2.__xpm__.job.wait() == JobState.DONE


def test_task_resubmit_across_experiments():
    """Test resubmitting a completed task across two experiment instances"""
    with TemporaryDirectory(prefix="xpm", suffix="resubmit_across") as workdir:
        # First experiment: task completes
        with TemporaryExperiment(
            "resubmit_across", timeout_multiplier=12, workdir=workdir
        ):
            task1 = ControllableResumableTask.C()
            task1.submit()

            # Tell task to complete
            task1.control_file.parent.mkdir(parents=True, exist_ok=True)
            task1.control_file.write_text("complete")

            assert task1.__xpm__.job.wait() == JobState.DONE

        # Second experiment: resubmit completed task (uses same workdir)
        with TemporaryExperiment(
            "resubmit_across", timeout_multiplier=12, workdir=workdir
        ):
            task2 = ControllableResumableTask.C()
            task2.submit()

            # Task should recognize it's already done
            assert task2.__xpm__.job.wait() == JobState.DONE
