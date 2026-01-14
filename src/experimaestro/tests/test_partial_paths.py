"""Integration tests for partial paths and cleanup"""

from pathlib import Path
from experimaestro import (
    Task,
    Param,
    Meta,
    field,
    PathGenerator,
    partial,
    param_group,
)
from experimaestro.scheduler import JobState

from .utils import TemporaryExperiment, TemporaryDirectory


# Define parameter groups
iter_group = param_group("iter")


class TaskWithPartial(Task):
    """Task that uses partial for partial paths"""

    # Define a partial set
    checkpoints = partial(exclude_groups=[iter_group])

    # Parameter in iter_group - excluded from partial identifier
    max_iter: Param[int] = field(groups=[iter_group])

    # Parameter not in any group - included in partial identifier
    learning_rate: Param[float]

    # Path generated using the partial identifier
    checkpoint_path: Meta[Path] = field(
        default_factory=PathGenerator("checkpoint", partial=checkpoints)
    )

    def execute(self):
        # Create the checkpoint directory and a marker file
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_path / "model.pt").write_text("checkpoint data")


def test_partial_path_created():
    """Test that partial paths are correctly created during task execution"""
    with TemporaryDirectory(prefix="xpm", suffix="partial") as workdir:
        with TemporaryExperiment(
            "partial_test", workdir=workdir, timeout_multiplier=12
        ):
            task = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()

        assert task.__xpm__.job.state == JobState.DONE

        # Verify the partial path was created
        assert task.checkpoint_path.exists()
        assert (task.checkpoint_path / "model.pt").exists()

        # Verify the path is in the partials directory
        partials_path = workdir / "partials"
        assert partials_path.exists()

        # The checkpoint_path should be under partials/TASK_ID/checkpoints/PARTIAL_ID/
        # Use resolve() to handle symlinks like /var -> /private/var on macOS
        assert task.checkpoint_path.resolve().is_relative_to(partials_path.resolve())


def test_partial_path_shared_across_tasks():
    """Test that tasks with same non-excluded params share partial paths"""
    with TemporaryDirectory(prefix="xpm", suffix="partial_shared") as workdir:
        with TemporaryExperiment(
            "partial_shared", workdir=workdir, timeout_multiplier=12
        ):
            # Submit two tasks with different max_iter but same learning_rate
            task1 = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()
            task2 = TaskWithPartial.C(max_iter=200, learning_rate=0.1).submit()

        assert task1.__xpm__.job.state == JobState.DONE
        assert task2.__xpm__.job.state == JobState.DONE

        # They should share the same partial path
        assert task1.checkpoint_path == task2.checkpoint_path


def test_partial_path_different_for_different_params():
    """Test that tasks with different non-excluded params have different partial paths"""
    with TemporaryDirectory(prefix="xpm", suffix="partial_diff") as workdir:
        with TemporaryExperiment(
            "partial_diff", workdir=workdir, timeout_multiplier=12
        ):
            # Submit two tasks with different learning_rate
            task1 = TaskWithPartial.C(max_iter=100, learning_rate=0.1).submit()
            task2 = TaskWithPartial.C(max_iter=100, learning_rate=0.2).submit()

        assert task1.__xpm__.job.state == JobState.DONE
        assert task2.__xpm__.job.state == JobState.DONE

        # They should have different partial paths
        assert task1.checkpoint_path != task2.checkpoint_path


def test_partial_concurrent_processes():
    """Test that two processes competing for the same partial are serialized.

    Similar to test_token_reschedule but for partial locking:
    - Two tasks with different x (excluded param) share the same partial
    - They should run sequentially (one after the other)
    """
    import sys
    import subprocess
    import logging
    import time
    import pytest
    from .utils import TemporaryDirectory, timeout, get_times_frompath

    with TemporaryDirectory("partial_reschedule") as workdir:
        lockingpath = workdir / "lockingpath"

        command = [
            sys.executable,
            Path(__file__).parent / "partial_reschedule.py",
            workdir,
        ]

        ready1 = workdir / "ready.1"
        time1 = workdir / "time.1"
        p1 = subprocess.Popen(
            command + ["1", str(lockingpath), str(ready1), str(time1)]
        )

        ready2 = workdir / "ready.2"
        time2 = workdir / "time.2"
        p2 = subprocess.Popen(
            command + ["2", str(lockingpath), str(ready2), str(time2)]
        )

        try:
            with timeout(30):
                logging.info("Waiting for both experiments to be ready")
                # Wait that both processes are ready
                while not ready1.is_file():
                    time.sleep(0.01)
                while not ready2.is_file():
                    time.sleep(0.01)

                # Create the locking path to allow tasks to finish
                logging.info(
                    "Both processes are ready: allowing tasks to finish by writing in %s",
                    lockingpath,
                )
                lockingpath.write_text("Let's go")

                # Waiting for the output
                logging.info("Waiting for XP1 to finish (%s)", time1)
                while not time1.is_file():
                    time.sleep(0.01)
                logging.info("Experiment 1 finished")

                logging.info("Waiting for XP2 to finish")
                while not time2.is_file():
                    time.sleep(0.01)
                logging.info("Experiment 2 finished")

                time1_val = get_times_frompath(time1)
                time2_val = get_times_frompath(time2)

                logging.info("%s vs %s", time1_val, time2_val)
                # One should have finished before the other started
                # (they share the same partial, so only one can run at a time)
                assert time1_val > time2_val or time2_val > time1_val
        except TimeoutError:
            p1.terminate()
            p2.terminate()
            pytest.fail("Timeout")

        except Exception:
            logging.warning("Other exception: killing processes (just in case)")
            p1.terminate()
            p2.terminate()
            raise
