import pytest
import logging
import time

from .utils import TemporaryExperiment
from .task_tokens import TokenTask

def test_token():
    """Simple token test"""
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token = xp.workspace.connector.createtoken("test-token", 1)
        path = xp.workspace.path / "test_token.file"
        task1 = TokenTask(path=path, x=1).add_dependencies(token.dependency(1)).submit()
        task2 = TokenTask(path=path, x=2).add_dependencies(token.dependency(1)).submit()

        path.write_text("Hello world")

        xp.wait()

        start1, stop1 = [float(t) for t in task1.stdout().read_text().strip().split("\n")]
        start2, stop2 = [float(t) for t in task2.stdout().read_text().strip().split("\n")]

    assert start1 > stop1 or start2 > stop2



@pytest.mark.skip("TODO: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()
