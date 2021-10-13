from pathlib import Path
import sys
import time
from experimaestro.connectors import Process, Redirect
from experimaestro.connectors.local import LocalConnector
import logging
from experimaestro.launchers import slurm
from experimaestro.launchers.slurm import (
    SlurmLauncher,
    BatchSlurmProcess,
    SlurmProcessBuilder,
)
from experimaestro.tests.utils import TemporaryDirectory
import pytest
from .common import waitFromSpec

binpath = Path(__file__).parent / "bin"


@pytest.fixture(scope="session")
def slurmlauncher(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("slurm-launcher")
    env = {"XPM_SLURM_DIR": str(tmpdir)}
    launcher = SlurmLauncher(
        connector=LocalConnector.instance(),
        binpath=binpath,
        launcherenv=env,
        interval=0.1,
    )
    yield launcher


@pytest.mark.timeout(10)
def test_slurm_ok(slurmlauncher: SlurmLauncher):
    builder = slurmlauncher.processbuilder()
    builder.command = [sys.executable, binpath / "test.py"]
    p = builder.start()
    assert p.wait() == 0


@pytest.mark.timeout(10)
def test_slurm_failed(slurmlauncher: SlurmLauncher):
    builder = slurmlauncher.processbuilder()
    builder.command = [sys.executable, binpath / "test.py", "--fail"]
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
    builder.command = [sys.executable, binpath / "test.py"]
    p = builder.start()
    assert p.wait() == 0

    gotoptions = {}
    for line in outpath.read_text().split("\n"):
        if line != "":
            key, value = line.split("=")
            gotoptions[key] = value

    assert gotoptions == {key: str(value) for key, value in options.items()}


@pytest.mark.timeout(timeout=3)
def test_slurm_batchprocess(tmp_path: Path, slurmlauncher: SlurmLauncher):
    waitFromSpec(tmp_path, slurmlauncher)
