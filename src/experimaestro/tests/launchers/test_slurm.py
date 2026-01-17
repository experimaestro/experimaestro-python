from pathlib import Path
import sys
import asyncio
import threading
import concurrent.futures
from experimaestro.connectors import Redirect
from experimaestro.tests.utils import TemporaryExperiment
from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.slurm import (
    SlurmLauncher,
)
from experimaestro.launchers.slurm.base import SlurmProcessWatcher
from experimaestro import field, ResumableTask, Param
from experimaestro.scheduler import JobState
import shutil
import pytest
from .common import waitFromSpec, takeback

# Mark all tests in this module as launcher tests
pytestmark = pytest.mark.launchers

BINPATH = Path(__file__).parent / "bin"


@pytest.fixture(scope="session")
def slurmlauncher(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("slurm-launcher")
    assert tmpdir.is_dir()

    binpath = tmpdir / "bin"
    (tmpdir / "slurm").mkdir()
    shutil.copytree(BINPATH, binpath)

    launcher = SlurmLauncher(
        connector=LocalConnector.instance(),
        binpath=binpath,
        interval=0.01,
    )
    yield launcher


@pytest.mark.timeout(10)
def test_slurm_ok(slurmlauncher: SlurmLauncher):
    builder = slurmlauncher.processbuilder()
    builder.command = [sys.executable, slurmlauncher.binpath / "test.py"]
    p = builder.start()
    assert p.wait() == 0


@pytest.mark.timeout(10)
def test_slurm_failed(slurmlauncher: SlurmLauncher):
    builder = slurmlauncher.processbuilder()
    builder.command = [sys.executable, slurmlauncher.binpath / "test.py", "--fail"]
    p = builder.start()
    assert p.wait() == 1


@pytest.mark.timeout(10)
def test_slurm_config(tmp_path, slurmlauncher: SlurmLauncher):
    """Test that sbatch is called properly"""
    options = {
        "nodes": 2,
        "gpus_per_node": 3,
        "time": 4,
    }
    launcher = slurmlauncher.config(**options)
    builder = launcher.processbuilder()

    outpath = tmp_path / "out.txt"
    builder.stdout = Redirect.file(outpath)
    builder.command = [sys.executable, slurmlauncher.binpath / "test.py"]
    p = builder.start()
    assert p.wait() == 0

    gotoptions = {}
    for line in outpath.read_text().split("\n"):
        if line != "":
            key, value = line.split("=")
            gotoptions[key] = value

    assert gotoptions == {key: str(value) for key, value in options.items()}


@pytest.mark.timeout(3)
def test_slurm_batchprocess(tmp_path: Path, slurmlauncher: SlurmLauncher):
    waitFromSpec(tmp_path, slurmlauncher)


def test_slurm_takeback(slurmlauncher, tmp_path):
    """Test whether a task can be taken back when running"""
    txp1 = TemporaryExperiment("slurm-takeback-1", workdir=tmp_path / "xp")
    txp2 = TemporaryExperiment("slurm-takeback-2", workdir=tmp_path / "xp")
    datapath = tmp_path / "data"

    takeback(slurmlauncher, datapath, txp1, txp2)


class SlurmResumableTask(ResumableTask):
    """ResumableTask that simulates timeout on first N attempts for SLURM testing"""

    checkpoint: Param[Path]
    timeout_count: Param[int] = field(ignore_default=2)
    slurm_jobs_dir: Param[Path]  # Path to mock SLURM jobs directory
    output_file: Param[Path] = field(ignore_default=None)

    def execute(self):
        import os

        # Read current attempt count from checkpoint
        attempt = 1
        if self.checkpoint.exists():
            attempt = int(self.checkpoint.read_text()) + 1

        print(f"SlurmResumableTask attempt #{attempt}")  # noqa: T201

        # Write updated attempt count
        self.checkpoint.write_text(str(attempt))

        # Simulate timeout for first timeout_count attempts
        if attempt <= self.timeout_count:
            print(f"Simulating SLURM TIMEOUT on attempt {attempt}")  # noqa: T201
            # Create timeout marker file for mock SLURM
            # The marker needs to be named <jobid>.timeout in the SLURM jobs directory
            # Use SLURM_JOB_ID environment variable (set by mock sbatch, like real SLURM)
            job_id = os.environ.get("SLURM_JOB_ID")
            if job_id:
                timeout_marker = self.slurm_jobs_dir / f"{job_id}.timeout"
                timeout_marker.write_text(f"timeout on attempt {attempt}")
            # Exit with error to trigger SLURM timeout handling
            raise RuntimeError(f"Simulated timeout on attempt {attempt}")

        # Success - task completed
        print(f"Task completed successfully on attempt {attempt}")  # noqa: T201
        if self.output_file:
            self.output_file.write_text(f"Completed after {attempt} attempts")


@pytest.mark.timeout(30)
def test_slurm_resumable_task(tmp_path: Path, slurmlauncher: SlurmLauncher):
    """Test that ResumableTask retries and resumes after SLURM timeouts"""
    with TemporaryExperiment("slurm-resumable", workdir=tmp_path / "xp"):
        checkpoint = tmp_path / "checkpoint.txt"
        output_file = tmp_path / "output.txt"

        # Get the SLURM jobs directory from the launcher's binpath
        slurm_jobs_dir = slurmlauncher.binpath.parent / "slurm" / "jobs"

        # Submit task with max_retries to allow multiple timeout retries
        task = SlurmResumableTask.C(
            checkpoint=checkpoint,
            timeout_count=2,  # Timeout on first 2 attempts
            slurm_jobs_dir=slurm_jobs_dir,
            output_file=output_file,
        ).submit(launcher=slurmlauncher, max_retries=5)

        # Wait for the task to complete
        state = task.__xpm__.job.wait()

        # Verify task completed successfully after retries
        assert state == JobState.DONE, f"Task did not complete successfully: {state}"
        assert task.__xpm__.job.retry_count == 2, (
            f"Expected 2 retries, got {task.__xpm__.job.retry_count}"
        )

        # Verify checkpoint shows 3 attempts (2 timeouts + 1 success)
        assert checkpoint.exists(), "Checkpoint file was not created"
        assert int(checkpoint.read_text()) == 3, (
            f"Expected 3 attempts, got {checkpoint.read_text()}"
        )

        # Verify output file was created on success
        assert output_file.exists(), "Output file was not created"
        assert "Completed after 3 attempts" in output_file.read_text()


@pytest.mark.timeout(30)
def test_slurm_watcher_race_condition(slurmlauncher: SlurmLauncher):
    """Stress test for SlurmProcessWatcher race conditions.

    This test verifies that concurrent access to the watcher doesn't cause:
    - Premature watcher shutdown
    - Deadlocks
    - Lost notifications
    """
    num_threads = 20
    iterations_per_thread = 10
    errors: list[Exception] = []
    barrier = threading.Barrier(num_threads)

    def worker(thread_id: int):
        """Worker that repeatedly gets/releases the watcher and submits jobs."""
        try:
            barrier.wait()  # Synchronize all threads to start at once
            for i in range(iterations_per_thread):
                # Rapidly get and release the watcher
                with SlurmProcessWatcher.get(slurmlauncher) as watcher:
                    # Do a quick job check to exercise the watcher
                    _ = watcher.getjob(f"nonexistent-{thread_id}-{i}", timeout=0.01)
        except Exception as e:
            errors.append(e)

    # Run workers concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    # Check for errors
    assert not errors, f"Errors during stress test: {errors}"

    # Verify watcher properly cleaned up (no leaked watchers)
    # Give a moment for cleanup
    import time

    time.sleep(0.1)
    assert slurmlauncher.key not in SlurmProcessWatcher.WATCHERS, (
        "Watcher was not cleaned up after all users exited"
    )


@pytest.mark.timeout(30)
def test_slurm_concurrent_processes(slurmlauncher: SlurmLauncher):
    """Test that multiple processes can be created and waited on concurrently."""
    num_processes = 5

    # Submit multiple processes
    processes = []
    for _ in range(num_processes):
        builder = slurmlauncher.processbuilder()
        builder.command = [sys.executable, slurmlauncher.binpath / "test.py"]
        processes.append(builder.start())

    # Wait for all processes in parallel threads
    results: list[int | None] = [None] * num_processes
    errors: list[Exception] = []

    def wait_for_process(idx: int):
        try:
            results[idx] = processes[idx].wait()
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=wait_for_process, args=(i,))
        for i in range(num_processes)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors waiting for processes: {errors}"
    assert all(r == 0 for r in results), f"Not all processes succeeded: {results}"


@pytest.mark.timeout(30)
def test_slurm_async_concurrent_wait(slurmlauncher: SlurmLauncher):
    """Test that async waiting works correctly with multiple concurrent waiters."""

    async def run_test():
        num_processes = 5

        # Submit multiple processes
        processes = []
        for _ in range(num_processes):
            builder = slurmlauncher.processbuilder()
            builder.command = [sys.executable, slurmlauncher.binpath / "test.py"]
            processes.append(builder.start())

        # Wait for all processes concurrently using asyncio
        results = await asyncio.gather(*[p.aio_wait() for p in processes])

        assert all(r == 0 for r in results), f"Not all processes succeeded: {results}"

    asyncio.run(run_test())


@pytest.mark.timeout(30)
def test_slurm_watcher_reuse_after_cleanup(slurmlauncher: SlurmLauncher):
    """Test that a new watcher is properly created after the previous one cleans up."""
    # First, use and release a watcher
    with SlurmProcessWatcher.get(slurmlauncher) as watcher1:
        _ = watcher1.getjob("test-job", timeout=0.01)

    # Wait for cleanup
    import time

    time.sleep(0.2)

    # Verify cleanup happened
    assert slurmlauncher.key not in SlurmProcessWatcher.WATCHERS

    # Now get a new watcher - should work without issues
    with SlurmProcessWatcher.get(slurmlauncher) as watcher2:
        # Watcher should be running and functional
        assert watcher2.is_alive()
        _ = watcher2.getjob("test-job-2", timeout=0.01)

    # Submit and wait for a real process to confirm everything works
    builder = slurmlauncher.processbuilder()
    builder.command = [sys.executable, slurmlauncher.binpath / "test.py"]
    p = builder.start()
    assert p.wait() == 0
