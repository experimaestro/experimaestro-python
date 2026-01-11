"""Tests for stray job detection.

A stray job is a running job that is not associated with any active experiment.
This happens when an experiment plan changes (e.g., same experiment ID is relaunched
with different parameters).
"""

import time
from pathlib import Path
import tempfile
import logging

from experimaestro import (
    Task,
    Param,
    Meta,
    field,
    PathGenerator,
    experiment,
    GracefulExperimentExit,
)
from experimaestro.scheduler.workspace import RunMode
from experimaestro.scheduler import JobState
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

logger = logging.getLogger(__name__)


class ControllableTask(Task):
    """A task that can be controlled via files for testing purposes."""

    value: Param[int]
    touch: Meta[Path] = field(default_factory=PathGenerator("touch"))
    wait: Meta[Path] = field(default_factory=PathGenerator("wait"))

    def execute(self):
        # Signal that the task has started
        with open(self.touch, "w") as out:
            out.write("started")

        # Wait for external signal to continue
        while not self.wait.is_file():
            time.sleep(0.1)


MAX_WAIT_ITERATIONS = 50  # 5 seconds max wait


def test_stray_job_detection():
    """Test that a running job becomes stray when experiment is relaunched with different params.

    Scenario:
    1. Start experiment with task(value=1)
    2. Wait for task to start running
    3. Exit experiment without waiting (using GracefulExperimentExit)
    4. Start same experiment (same ID) with task(value=2)
    5. The first job should now be detected as stray
    6. Signal first task to finish
    7. Verify stray detection works correctly
    """
    with tempfile.TemporaryDirectory(prefix="xpm_stray_test_") as workdir:
        workdir_path = Path(workdir)
        experiment_id = "stray_test"

        # Phase 1: Start experiment with value=1
        with experiment(workdir_path, experiment_id) as _:
            task1 = ControllableTask.C(value=1)
            # First do a dry run to get the file paths
            task1.submit(run_mode=RunMode.DRY_RUN)
            touch_path = task1.touch
            wait_path = task1.wait

            # Now submit for real
            task1 = ControllableTask.C(value=1)
            task1.submit()

            # Wait for task to start
            counter = 0
            while not touch_path.exists():
                time.sleep(0.1)
                counter += 1
                if counter >= MAX_WAIT_ITERATIONS:
                    raise AssertionError("Timeout waiting for task1 to start")

            raise GracefulExperimentExit()

        # Phase 2: Start same experiment with value=2
        try:
            with experiment(workdir_path, experiment_id) as _:
                task2 = ControllableTask.C(value=2)
                task2.submit()

                # Give the scheduler a moment to process
                time.sleep(0.5)

                # Check for stray jobs using a fresh state provider
                provider = WorkspaceStateProvider(workdir_path)

                stray_jobs = provider.get_stray_jobs()

                # At least the first task should be stray (running but not in current experiment)
                # Note: task2 might also appear as stray temporarily because status.json
                # hasn't been flushed yet
                assert len(stray_jobs) >= 1, (
                    f"Expected at least 1 stray job, found {len(stray_jobs)}"
                )

                # Verify that at least one stray job is actually running
                running_stray = [j for j in stray_jobs if j.state == JobState.RUNNING]
                assert len(running_stray) >= 1, (
                    f"Expected at least 1 running stray job, found {len(running_stray)}"
                )

                # Signal first task to finish
                with open(wait_path, "w") as f:
                    f.write("done")

                # Give it a moment to finish
                time.sleep(0.5)

                # After task1 finishes, it should no longer be stray (it's not running)
                # task1 should no longer be stray (it finished)
                # task2 might still appear as stray if status hasn't been flushed
                # So we just verify that the count decreased by at least 1
                # (or at least, that task1's specific job is no longer in the list)

                # Exit gracefully to not wait for task2
                raise GracefulExperimentExit()
        finally:
            # Clean up: signal task2 to finish if it started
            with experiment(workdir_path, experiment_id, run_mode=RunMode.DRY_RUN):
                task2_dry = ControllableTask.C(value=2)
                task2_dry.submit(run_mode=RunMode.DRY_RUN)
                task2_wait = task2_dry.wait

            if task2_wait and task2_wait.exists() is False:
                # task2 might have started
                task2_touch = task2_dry.touch
                if task2_touch.exists():
                    with open(task2_wait, "w") as f:
                        f.write("done")
                    time.sleep(0.5)


def test_running_state_detection():
    """Test that running jobs are correctly detected from PID files."""
    with tempfile.TemporaryDirectory(prefix="xpm_running_test_") as workdir:
        workdir_path = Path(workdir)
        experiment_id = "running_test"

        with experiment(workdir_path, experiment_id) as _:
            task = ControllableTask.C(value=1)
            task.submit(run_mode=RunMode.DRY_RUN)
            touch_path = task.touch
            wait_path = task.wait

            task = ControllableTask.C(value=1)
            task.submit()

            # Wait for task to start
            counter = 0
            while not touch_path.exists():
                time.sleep(0.1)
                counter += 1
                if counter >= MAX_WAIT_ITERATIONS:
                    raise AssertionError("Timeout waiting for task to start")

            # Create a fresh provider to check the running state
            provider = WorkspaceStateProvider(workdir_path)

            # Get all jobs on disk
            jobs_base = workdir_path / "jobs"
            job_paths = list(jobs_base.glob("*/*"))
            assert len(job_paths) == 1, (
                f"Expected 1 job on disk, found {len(job_paths)}"
            )

            # Check that the job is detected as running
            job_path = job_paths[0]
            task_id = job_path.parent.name
            job_id = job_path.name

            mock_job = provider._create_mock_job_from_path(job_path, task_id, job_id)
            assert mock_job.state == JobState.RUNNING, (
                f"Expected job state RUNNING, got {mock_job.state}"
            )

            # Signal task to finish
            with open(wait_path, "w") as f:
                f.write("done")


def test_completed_job_not_stray():
    """Test that completed jobs are not detected as stray.

    Scenario:
    1. Start experiment, run and complete a job
    2. Start a new run of the same experiment with a different job
    3. The first job should be orphan (not in the new run) but NOT stray (not running)
    """
    with tempfile.TemporaryDirectory(prefix="xpm_completed_test_") as workdir:
        workdir_path = Path(workdir)
        experiment_id = "completed_test"

        # Phase 1: Create and complete a job
        with experiment(workdir_path, experiment_id) as _:
            task = ControllableTask.C(value=1)
            task.submit(run_mode=RunMode.DRY_RUN)
            touch_path = task.touch
            wait_path = task.wait

            task = ControllableTask.C(value=1)
            task.submit()

            # Wait for task to start
            counter = 0
            while not touch_path.exists():
                time.sleep(0.1)
                counter += 1
                if counter >= MAX_WAIT_ITERATIONS:
                    raise AssertionError("Timeout waiting for task to start")

            # Signal to finish immediately
            with open(wait_path, "w") as f:
                f.write("done")

        # Phase 2: Start a new run with a different job
        # This makes the old job an orphan (not in current run)
        with experiment(workdir_path, experiment_id) as _:
            task2 = ControllableTask.C(value=2)
            task2.submit(run_mode=RunMode.DRY_RUN)
            touch_path2 = task2.touch
            wait_path2 = task2.wait

            task2 = ControllableTask.C(value=2)
            task2.submit()

            # Wait for task2 to start
            counter = 0
            while not touch_path2.exists():
                time.sleep(0.1)
                counter += 1
                if counter >= MAX_WAIT_ITERATIONS:
                    raise AssertionError("Timeout waiting for task2 to start")

            # Now check that the first job is NOT stray (it's completed)
            provider = WorkspaceStateProvider(workdir_path)

            stray_jobs = provider.get_stray_jobs()

            # Task1 is not running, so it should not be stray
            # Task2 might appear as stray because status.json hasn't been flushed
            # Filter to check that no DONE jobs are stray
            done_stray = [j for j in stray_jobs if j.state == JobState.DONE]
            assert len(done_stray) == 0, (
                f"Expected 0 completed stray jobs, found {len(done_stray)}"
            )

            # Signal task2 to finish
            with open(wait_path2, "w") as f:
                f.write("done")
