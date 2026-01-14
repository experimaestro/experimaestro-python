"""Tests for ResumableTask with timeout retry logic"""

import json
from pathlib import Path
from experimaestro import field, ResumableTask, Task, Param, GracefulTimeout
from experimaestro.scheduler.workspace import RunMode
from experimaestro.scheduler import JobState, JobFailureStatus
from experimaestro.scheduler.jobs import JobStateError
from experimaestro.scheduler.interfaces import JobState as JobStateClass
from experimaestro.connectors import Process, ProcessState
from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.direct import DirectLauncher
from experimaestro.commandline import CommandLineJob
from .utils import TemporaryExperiment


class MockTimeoutCommandLineJob(CommandLineJob):
    """CommandLineJob that simulates timeouts based on attempt count"""

    def __init__(self, *args, timeout_count=0, checkpoint_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_count = timeout_count
        self.test_checkpoint_file = checkpoint_file

    async def aio_run(self):
        """Override to simulate timeout behavior"""
        # Execute the actual task in a thread
        from experimaestro.utils.asyncio import asyncThreadcheck

        await asyncThreadcheck("execute", self.config.execute)

        # Return a mock process that simulates timeout based on attempt count
        return MockTimeoutProcess(self.test_checkpoint_file, self.timeout_count)

    async def aio_process(self):
        """No existing process"""
        return None


class MockTimeoutProcess(Process):
    """Process that returns TIMEOUT for first N attempts"""

    def __init__(self, checkpoint_file: Path, timeout_count: int):
        self.checkpoint_file = checkpoint_file
        self.timeout_count = timeout_count

    def wait(self) -> int:
        # Always return success (code 0) - timeout detection is in get_job_state
        return 0

    async def aio_state(self, timeout: float | None = None) -> ProcessState:
        return ProcessState.DONE

    def get_job_state(self, code: int) -> "JobState":
        """Return TIMEOUT for first timeout_count attempts"""
        # Read attempt count from checkpoint
        attempt = 1
        if self.checkpoint_file.exists():
            attempt = int(self.checkpoint_file.read_text())

        # Return TIMEOUT for first timeout_count attempts
        if attempt <= self.timeout_count:
            return JobStateError(JobFailureStatus.TIMEOUT)

        return JobState.DONE


class MockTimeoutLauncher(DirectLauncher):
    """Launcher that creates jobs simulating timeouts"""

    def __init__(self, timeout_count: int, checkpoint_file: Path):
        super().__init__(LocalConnector())
        self.timeout_count = timeout_count
        self.checkpoint_file = checkpoint_file


# Monkey-patch the task type to use our mock job
# This is done by overriding the task factory
def create_mock_timeout_task(timeout_count: int, checkpoint_file: Path):
    """Create a task type that uses MockTimeoutCommandLineJob"""

    def mock_job_factory(commandline):
        class MockCommandLineTask:
            def __init__(self, commandline):
                self.commandline = commandline

            def __call__(
                self,
                pyobject,
                *,
                launcher=None,
                workspace=None,
                run_mode=None,
                max_retries=None,
            ):
                return MockTimeoutCommandLineJob(
                    self.commandline,
                    pyobject,
                    launcher=launcher,
                    workspace=workspace,
                    run_mode=run_mode,
                    max_retries=max_retries,
                    timeout_count=timeout_count,
                    checkpoint_file=checkpoint_file,
                )

        return MockCommandLineTask(commandline)

    return mock_job_factory


class CountingResumableTask(ResumableTask):
    """Resumable task that counts execution attempts"""

    checkpoint: Param[Path]

    def execute(self):
        # Count attempts in checkpoint file
        attempt = 1
        if self.checkpoint.exists():
            attempt = int(self.checkpoint.read_text()) + 1

        self.checkpoint.write_text(str(attempt))


class SimpleResumableTask(ResumableTask):
    """Simple resumable task for testing"""

    def execute(self):
        # This would normally contain checkpoint logic
        pass


class SimpleNonResumableTask(Task):
    """Simple non-resumable task for testing"""

    def execute(self):
        pass


def test_resumable_task_has_resumable_flag():
    """Test that ResumableTask instances are correctly identified"""
    with TemporaryExperiment("resumable_flag"):
        launcher = DirectLauncher(LocalConnector())

        # Submit resumable task
        resumable = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN
        )
        assert resumable.__xpm__.job.resumable is True

        # Submit non-resumable task
        non_resumable = SimpleNonResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN
        )
        assert non_resumable.__xpm__.job.resumable is False


def test_max_retries_default():
    """Test that default max_retries is 3"""
    with TemporaryExperiment("max_retries_default"):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN
        )

        # Default should be 3 (from workspace settings)
        assert task.__xpm__.job.max_retries == 3
        assert task.__xpm__.job.retry_count == 0


def test_max_retries_custom():
    """Test that custom max_retries parameter is respected"""
    with TemporaryExperiment("max_retries_custom"):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN, max_retries=5
        )

        assert task.__xpm__.job.max_retries == 5
        assert task.__xpm__.job.retry_count == 0


def test_max_retries_zero():
    """Test that max_retries=0 is allowed (no retries)"""
    with TemporaryExperiment("max_retries_zero"):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN, max_retries=0
        )

        assert task.__xpm__.job.max_retries == 0
        assert task.__xpm__.job.retry_count == 0


def test_resumable_task_succeeds_after_timeouts():
    """Test that a resumable task retries and succeeds after timeouts"""
    with TemporaryExperiment("resumable_timeout_success", timeout_multiplier=9) as xp:
        checkpoint_file = xp.workspace.path / "checkpoint.txt"
        launcher = DirectLauncher(LocalConnector())

        # Create task config
        task_config = CountingResumableTask.C(checkpoint=checkpoint_file)

        # Create mock job directly
        job = MockTimeoutCommandLineJob(
            None,  # commandline not used
            task_config,
            workspace=xp.workspace,
            launcher=launcher,
            max_retries=5,
            timeout_count=2,
            checkpoint_file=checkpoint_file,
        )

        # Set the job on the config
        task_config.__xpm__.job = job

        # Submit to scheduler
        from experimaestro.scheduler import experiment

        experiment.CURRENT.submit(job)

        # Should succeed after 3 attempts (2 timeouts + 1 success)
        state = job.wait()
        assert state == JobState.DONE
        assert job.retry_count == 2
        # Checkpoint should show 3 executions
        assert int(checkpoint_file.read_text()) == 3


def test_resumable_task_fails_after_max_retries():
    """Test that a resumable task fails after exceeding max_retries"""
    from experimaestro.scheduler import FailedExperiment
    import pytest

    with pytest.raises(FailedExperiment):
        with TemporaryExperiment("resumable_timeout_fail", timeout_multiplier=9) as xp:
            checkpoint_file = xp.workspace.path / "checkpoint.txt"
            launcher = DirectLauncher(LocalConnector())

            # Create task config
            task_config = CountingResumableTask.C(checkpoint=checkpoint_file)

            # Create mock job directly
            job = MockTimeoutCommandLineJob(
                None,  # commandline not used
                task_config,
                workspace=xp.workspace,
                launcher=launcher,
                max_retries=3,
                timeout_count=10,
                checkpoint_file=checkpoint_file,
            )

            # Set the job on the config
            task_config.__xpm__.job = job

            # Submit to scheduler
            from experimaestro.scheduler import experiment

            experiment.CURRENT.submit(job)

            # Wait for job to complete (will fail)
            state = job.wait()

    # Verify job failed correctly
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.TIMEOUT
    # retry_count should be max_retries + 1 (initial attempt + 3 retries = 4 total)
    assert job.retry_count == 4
    # Checkpoint should show 4 executions
    assert int(checkpoint_file.read_text()) == 4


# =============================================================================
# Tests for JobState.from_path and .failed file format
# =============================================================================


def test_job_state_from_path_json_timeout(tmp_path):
    """Test JobState.from_path reads JSON format with timeout reason"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(
        json.dumps({"code": 1, "reason": "timeout", "message": "Graceful"})
    )

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.TIMEOUT


def test_job_state_from_path_json_memory(tmp_path):
    """Test JobState.from_path reads JSON format with memory reason"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(json.dumps({"code": 1, "reason": "memory"}))

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.MEMORY


def test_job_state_from_path_json_dependency(tmp_path):
    """Test JobState.from_path reads JSON format with dependency reason"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(json.dumps({"code": 1, "reason": "dependency"}))

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.DEPENDENCY


def test_job_state_from_path_json_failed(tmp_path):
    """Test JobState.from_path reads JSON format with failed reason"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(json.dumps({"code": 1, "reason": "failed"}))

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.FAILED


def test_job_state_from_path_json_unknown_reason(tmp_path):
    """Test JobState.from_path handles unknown reason gracefully"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(json.dumps({"code": 1, "reason": "unknown_reason"}))

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.FAILED  # Falls back to FAILED


def test_job_state_from_path_legacy_integer_nonzero(tmp_path):
    """Test JobState.from_path reads legacy integer format (non-zero = error)"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text("1")

    state = JobStateClass.from_path(tmp_path, "test")
    assert isinstance(state, JobStateError)
    assert state.failure_reason == JobFailureStatus.FAILED


def test_job_state_from_path_legacy_integer_zero(tmp_path):
    """Test JobState.from_path reads legacy integer format (zero = done)"""
    failed_file = tmp_path / "test.failed"
    failed_file.write_text("0")

    state = JobStateClass.from_path(tmp_path, "test")
    assert state == JobState.DONE


def test_job_state_from_path_done_file(tmp_path):
    """Test JobState.from_path reads .done file"""
    done_file = tmp_path / "test.done"
    done_file.touch()

    state = JobStateClass.from_path(tmp_path, "test")
    assert state == JobState.DONE


def test_job_state_from_path_no_file(tmp_path):
    """Test JobState.from_path returns None when no file exists"""
    state = JobStateClass.from_path(tmp_path, "test")
    assert state is None


def test_job_state_from_path_done_takes_precedence(tmp_path):
    """Test JobState.from_path prefers .done over .failed"""
    done_file = tmp_path / "test.done"
    done_file.touch()
    failed_file = tmp_path / "test.failed"
    failed_file.write_text(json.dumps({"code": 1, "reason": "failed"}))

    state = JobStateClass.from_path(tmp_path, "test")
    assert state == JobState.DONE


# =============================================================================
# Tests for GracefulTimeout exception
# =============================================================================


class GracefulTimeoutTask(ResumableTask):
    """Task that raises GracefulTimeout"""

    checkpoint: Param[Path]
    should_timeout: Param[bool] = field(ignore_default=True)

    def execute(self):
        # Count attempts in checkpoint file
        attempt = 1
        if self.checkpoint.exists():
            attempt = int(self.checkpoint.read_text()) + 1
        self.checkpoint.write_text(str(attempt))

        # Raise GracefulTimeout on first attempt if should_timeout is True
        if self.should_timeout and attempt == 1:
            raise GracefulTimeout("Not enough time for another epoch")


# =============================================================================
# Tests for remaining_time() method with mock launcher
# =============================================================================


class MockLauncherWithRemainingTime(DirectLauncher):
    """Mock launcher that provides remaining_time via launcher_info_code()"""

    def __init__(self, remaining_time_value: float | None):
        super().__init__(LocalConnector())
        self.remaining_time_value = remaining_time_value

    def launcher_info_code(self) -> str:
        """Generate code to set up MockLauncherInformation with the specified value"""
        if self.remaining_time_value is None:
            return (
                "    from experimaestro.tests.test_resumable_task import MockLauncherInformation\n"
                "    from experimaestro import taskglobals\n"
                "    taskglobals.Env.instance().launcher_info = MockLauncherInformation(None)\n"
            )
        return (
            "    from experimaestro.tests.test_resumable_task import MockLauncherInformation\n"
            "    from experimaestro import taskglobals\n"
            f"    taskglobals.Env.instance().launcher_info = MockLauncherInformation({self.remaining_time_value})\n"
        )


class MockLauncherInformation:
    """Mock launcher info for testing remaining_time()"""

    def __init__(self, remaining: float | None):
        self._remaining = remaining

    def remaining_time(self) -> float | None:
        return self._remaining


class RemainingTimeTask(ResumableTask):
    """Task that records the remaining_time() value"""

    output_file: Param[Path]

    def execute(self):
        remaining = self.remaining_time()
        self.output_file.write_text(str(remaining) if remaining is not None else "None")


def test_remaining_time_with_mock_launcher():
    """Test remaining_time() works with a mock launcher that provides launcher_info_code()"""
    with TemporaryExperiment("remaining_time", timeout_multiplier=6) as xp:
        output_file = xp.workspace.path / "remaining.txt"
        launcher = MockLauncherWithRemainingTime(remaining_time_value=1234.5)

        task = RemainingTimeTask.C(output_file=output_file).submit(launcher=launcher)

        state = task.__xpm__.job.wait()
        assert state == JobState.DONE

        # Verify the task received the remaining time value
        assert output_file.exists()
        assert output_file.read_text() == "1234.5"


def test_remaining_time_none_with_mock_launcher():
    """Test remaining_time() returns None when launcher has no time limit"""
    with TemporaryExperiment("remaining_time_none", timeout_multiplier=6) as xp:
        output_file = xp.workspace.path / "remaining.txt"
        launcher = MockLauncherWithRemainingTime(remaining_time_value=None)

        task = RemainingTimeTask.C(output_file=output_file).submit(launcher=launcher)

        state = task.__xpm__.job.wait()
        assert state == JobState.DONE

        # Verify the task received None
        assert output_file.exists()
        assert output_file.read_text() == "None"
