import pytest
from .utils import TemporaryExperiment
from .task_tokens import TokenTask

def test_token():
    """Simple token test"""
    with TemporaryExperiment("restart", maxwait=10) as ws:
        token = ws.connector.createtoken("test-token", 1)
        path = ws.path / "test_token.file"
        task1 = TokenTask(path=path).tokens(token).submit()
        task2 = TokenTask(path=path).tokens(token).submit()

    path.write_text("Hello world")
    start1, stop1 = [int(t) for t in task1.stdout().read_text().split("\n")]
    start2, stop2 = [int(t) for t in task2.stdout().read_text().split("\n")]

    assert()

@pytest.mark.skip("TODO: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()
