"""Tests for ResumableTask with timeout retry logic"""

from pathlib import Path
from experimaestro import ResumableTask, Task, Param
from experimaestro.scheduler.workspace import RunMode
from experimaestro.scheduler import JobState, JobFailureStatus
from experimaestro.scheduler.jobs import JobStateError
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
    with TemporaryExperiment("resumable_flag", maxwait=0):
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
    with TemporaryExperiment("max_retries_default", maxwait=0):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN
        )

        # Default should be 3 (from workspace settings)
        assert task.__xpm__.job.max_retries == 3
        assert task.__xpm__.job.retry_count == 0


def test_max_retries_custom():
    """Test that custom max_retries parameter is respected"""
    with TemporaryExperiment("max_retries_custom", maxwait=0):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN, max_retries=5
        )

        assert task.__xpm__.job.max_retries == 5
        assert task.__xpm__.job.retry_count == 0


def test_max_retries_zero():
    """Test that max_retries=0 is allowed (no retries)"""
    with TemporaryExperiment("max_retries_zero", maxwait=0):
        launcher = DirectLauncher(LocalConnector())

        task = SimpleResumableTask.C().submit(
            launcher=launcher, run_mode=RunMode.DRY_RUN, max_retries=0
        )

        assert task.__xpm__.job.max_retries == 0
        assert task.__xpm__.job.retry_count == 0


def test_resumable_task_succeeds_after_timeouts():
    """Test that a resumable task retries and succeeds after timeouts"""
    with TemporaryExperiment("resumable_timeout_success", maxwait=20) as xp:
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
        with TemporaryExperiment("resumable_timeout_fail", maxwait=20) as xp:
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
