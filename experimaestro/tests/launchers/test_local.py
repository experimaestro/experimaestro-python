from pathlib import Path
import pytest
from experimaestro.launchers import Launcher
from .common import waitFromSpec


@pytest.mark.timeout(timeout=3)
def test_local_process(tmp_path: Path):
    launcher = Launcher.get(tmp_path)
    waitFromSpec(tmp_path, launcher)
