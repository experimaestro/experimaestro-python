from pathlib import Path
import sys
from experimaestro.connectors import Redirect
from experimaestro.tests.utils import TemporaryExperiment
from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.slurm import (
    SlurmLauncher,
)
from experimaestro import field, ResumableTask, Param
from experimaestro.scheduler import JobState
import shutil
import pytest
from .common import waitFromSpec, takeback

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
