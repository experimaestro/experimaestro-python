# --- Task and types definitions

import json
import signal
import sys
import time
from pathlib import Path
import pytest
import logging
import psutil
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
from experimaestro.scheduler import FailedExperiment, JobState, JobFailureStatus
from experimaestro.scheduler.jobs import JobStateError
from experimaestro.scheduler.interfaces import JobState as JobStateClass
from experimaestro import SubmitHook, Job, Launcher, LightweightTask

from .utils import TemporaryDirectory, TemporaryExperiment, is_posix

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

# Mark all tests in this module as task tests (depends on identifier)
pytestmark = [
    pytest.mark.tasks,
    pytest.mark.dependency(
        name="mod_tasks", depends=["mod_identifier"], scope="session"
    ),
]


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
        # Wait for control file with non-empty content
        # This avoids race conditions where the file exists but is empty
        while True:
            if self.control_file.is_file():
                action = self.control_file.read_text().strip()
                if action:  # Only proceed if file has content
                    break
            time.sleep(0.1)

        self.control_file.unlink()

        if action == "fail":
            sys.exit(1)


def test_resumable_task_resubmit():
    """Test that resubmitting a failed ResumableTask stays failed unless max_retries increased"""
    with TemporaryExperiment("resumable_resubmit", timeout_multiplier=12):
        logging.info("# Submission 1/3")
        task1 = ControllableResumableTask.C()
        task1.submit(max_retries=1)

        # Tell task to fail
        task1.control_file.parent.mkdir(parents=True, exist_ok=True)
        task1.control_file.write_text("fail")

        # Wait for the job to fail
        job = task1.__xpm__.job
        assert job.wait() == JobState.ERROR, "Job should have failed"

        # Resubmit with same max_retries - should stay failed
        task2 = ControllableResumableTask.C()
        logging.info("# Submission 2/3")
        task2.submit(max_retries=1)

        # Even though we tell it to complete, it should not run
        task2.control_file.write_text("complete")

        # Job should still be in error state (not restarted)
        assert task2.__xpm__.job.wait() == JobState.ERROR

        # Resubmit with higher max_retries - should run and complete
        task3 = ControllableResumableTask.C()
        logging.info("# Submission 3/3")
        task3.submit(max_retries=2)

        # Tell task to complete
        task3.control_file.write_text("complete")

        # Now it should complete
        assert task3.__xpm__.job.wait() == JobState.DONE


def test_resumable_task_resubmit_across_experiments():
    """Test resubmitting a failed ResumableTask across experiments requires higher max_retries"""
    with TemporaryDirectory(prefix="xpm", suffix="resubmit_across") as workdir:
        # First experiment: task fails
        try:
            with TemporaryExperiment(
                "resubmit_across", timeout_multiplier=6, workdir=workdir
            ):
                task1 = ControllableResumableTask.C()
                task1.submit(max_retries=1)

                # Tell task to fail
                task1.control_file.parent.mkdir(parents=True, exist_ok=True)
                task1.control_file.write_text("fail")
        except Exception as e:
            logging.info("First experiment ended (expected): %s", e)

        # Second experiment: resubmit with same max_retries - should stay failed
        try:
            with TemporaryExperiment(
                "resubmit_across", timeout_multiplier=6, workdir=workdir
            ):
                task2 = ControllableResumableTask.C()
                task2.submit(max_retries=1)

                task2.control_file.write_text("complete")

                # Should still be error (retry_count exhausted)
                assert task2.__xpm__.job.wait() == JobState.ERROR
        except FailedExperiment:
            # Expected - experiment raises because job is in error state
            pass

        # Third experiment: resubmit with higher max_retries - should complete
        with TemporaryExperiment(
            "resubmit_across", timeout_multiplier=12, workdir=workdir
        ):
            task3 = ControllableResumableTask.C()
            task3.submit(max_retries=2)

            # Tell task to complete
            task3.control_file.write_text("complete")

            # Now it should complete
            assert task3.__xpm__.job.wait() == JobState.DONE


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


def test_resubmit_preserves_status():
    """Test that resubmitting a completed job doesn't modify status.json"""
    with TemporaryDirectory(prefix="xpm", suffix="preserve_status") as workdir:
        # First experiment: task completes
        with TemporaryExperiment(
            "preserve_status", timeout_multiplier=12, workdir=workdir
        ):
            task1 = ControllableResumableTask.C()
            task1.submit()

            # Tell task to complete
            task1.control_file.parent.mkdir(parents=True, exist_ok=True)
            task1.control_file.write_text("complete")

            assert task1.__xpm__.job.wait() == JobState.DONE

            # Get the status.json path and record its modification time and content
            status_path = task1.__xpm__.job.status_path
            assert status_path.exists(), "status.json should exist after job completion"
            original_mtime = status_path.stat().st_mtime
            original_content = status_path.read_text()

        # Second experiment: resubmit completed task
        with TemporaryExperiment(
            "preserve_status", timeout_multiplier=12, workdir=workdir
        ):
            task2 = ControllableResumableTask.C()
            task2.submit()

            assert task2.__xpm__.job.wait() == JobState.DONE

            # Verify status.json was not modified
            new_mtime = status_path.stat().st_mtime
            new_content = status_path.read_text()

            assert new_mtime == original_mtime, (
                f"status.json mtime changed: {original_mtime} -> {new_mtime}"
            )
            assert new_content == original_content, "status.json content changed"


# --- Tests for JobState.from_path with cancelled reason ---


def test_job_state_from_path_json_cancelled(tmp_path):
    """Test JobState.from_path reads JSON format with cancelled reason"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(
        json.dumps(
            {"code": 15, "reason": "cancelled", "message": "Job terminated by SIGTERM"}
        )
    )

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.CANCELLED


# --- Tests for graceful termination with TaskCancelled ---


class CancellableTask(Task):
    """Task that catches TaskCancelled and does cleanup"""

    started_file: Meta[Path] = field(default_factory=PathGenerator("started"))
    cleanup_file: Meta[Path] = field(default_factory=PathGenerator("cleanup_done"))
    reraise: Param[bool] = field(default=True)

    def execute(self):
        from experimaestro import TaskCancelled

        # Signal that we started
        self.started_file.write_text("started")

        try:
            # Wait forever until cancelled
            while True:
                time.sleep(0.1)
        except TaskCancelled as e:
            # Do cleanup
            self.cleanup_file.write_text(
                f"cleanup at remaining_time={e.remaining_time}"
            )
            if self.reraise:
                raise  # Re-raise to let framework handle
            # Otherwise task completes "normally" after catching the exception


MAX_CANCELLATION_WAIT = 50  # 5 seconds max wait


def _run_cancellation_test(task, experiment_name: str):
    """Helper to run cancellation test logic.

    Returns (job, process) for verification after experiment exits.
    """
    p = None
    job = task.__xpm__.job

    # Wait for task to start
    counter = 0
    while not task.started_file.is_file():
        time.sleep(0.1)
        counter += 1
        if counter >= MAX_CANCELLATION_WAIT:
            pytest.fail("Timeout waiting for task to start")

    # Get the task process PID
    jobinfo = json.loads(job.pidpath.read_text())
    pid = int(jobinfo["pid"])
    p = psutil.Process(pid)

    logging.info("Task started with PID %d", pid)

    # Send SIGTERM to the task process
    p.send_signal(signal.SIGTERM)

    # Wait for the cleanup file to appear
    counter = 0
    while not task.cleanup_file.is_file():
        time.sleep(0.1)
        counter += 1
        if counter >= MAX_CANCELLATION_WAIT:
            pytest.fail("Timeout waiting for cleanup file")

    # Verify cleanup was done
    cleanup_content = task.cleanup_file.read_text()
    assert "cleanup at remaining_time=" in cleanup_content

    # Wait for process to exit
    try:
        p.wait(timeout=5)
    except psutil.TimeoutExpired:
        p.kill()
        pytest.fail("Task process did not exit after SIGTERM")

    return job, p


def _verify_cancelled_job(job):
    """Verify that job was marked as cancelled."""
    failed_path = job.path / f"{job.scriptname}.failed"
    counter = 0
    while not failed_path.is_file():
        time.sleep(0.1)
        counter += 1
        if counter >= MAX_CANCELLATION_WAIT:
            pytest.fail("Timeout waiting for .failed file")

    failed_content = json.loads(failed_path.read_text())
    assert failed_content["reason"] == "cancelled", (
        f"Expected reason 'cancelled', got {failed_content}"
    )

    # Verify that .done file was NOT created
    done_path = job.path / f"{job.scriptname}.done"
    assert not done_path.is_file(), ".done should not exist for cancelled task"


@pytest.mark.skipif(not is_posix(), reason="Signal handling only works on POSIX")
def test_graceful_termination_with_cleanup():
    """Test that task can catch TaskCancelled and do cleanup before termination"""
    p = None
    job = None
    try:
        with pytest.raises(FailedExperiment):
            with TemporaryExperiment("graceful_termination", timeout_multiplier=9):
                task = CancellableTask.C(reraise=True)
                task.submit()
                job, p = _run_cancellation_test(task, "graceful_termination")

        assert job is not None
        _verify_cancelled_job(job)

    finally:
        if p is not None and p.is_running():
            logging.warning("Force killing task process %d", p.pid)
            p.kill()


@pytest.mark.skipif(not is_posix(), reason="Signal handling only works on POSIX")
def test_graceful_termination_no_reraise():
    """Test that task catching TaskCancelled without re-raising still marks job as cancelled"""
    p = None
    job = None
    try:
        with pytest.raises(FailedExperiment):
            with TemporaryExperiment(
                "graceful_termination_no_reraise", timeout_multiplier=9
            ):
                task = CancellableTask.C(reraise=False)
                task.submit()
                job, p = _run_cancellation_test(task, "graceful_termination_no_reraise")

        assert job is not None
        _verify_cancelled_job(job)

    finally:
        if p is not None and p.is_running():
            logging.warning("Force killing task process %d", p.pid)
            p.kill()
