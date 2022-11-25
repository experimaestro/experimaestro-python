from pathlib import Path
import pytest
from experimaestro.launchers import Launcher
from experimaestro.tests.utils import TemporaryExperiment
from .common import waitFromSpec, takeback


@pytest.mark.timeout(3)
def test_local_process(tmp_path: Path):
    launcher = Launcher.get(tmp_path)
    waitFromSpec(tmp_path, launcher)


@pytest.mark.timeout(3)
def test_local_takeback(tmp_path):
    """Test whether a task can be taken back when running"""
    txp1 = TemporaryExperiment("slurm-takeback-1", workdir=tmp_path / "xp")
    txp2 = TemporaryExperiment("slurm-takeback-2", workdir=tmp_path / "xp")
    datapath = tmp_path / "data"
    launcher = Launcher.get(tmp_path)

    takeback(launcher, datapath, txp1, txp2)
