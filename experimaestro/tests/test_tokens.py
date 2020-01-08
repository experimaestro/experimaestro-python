import pytest
import logging
import time

from experimaestro.scheduler import JobState
from .utils import TemporaryExperiment
from .task_tokens import TokenTask

def token_experiment(xp, token):
    path = xp.workspace.path / "test_token.file"
    task1 = TokenTask(path=path, x=1)
    task2 = TokenTask(path=path, x=2)
    for task in [task1, task2]:
        if token:
            task.add_dependencies(token.dependency(1))
        task.submit()

    time.sleep(0.5)
    path.write_text("Hello world")

    xp.wait()

    start1, stop1 = [float(t) for t in task1.stdout().read_text().strip().split("\n")]
    start2, stop2 = [float(t) for t in task2.stdout().read_text().strip().split("\n")]
    mintime = min(start1, stop1, start2, stop2)
    logging.debug("Times: %s to %s, %s to %s [delta %f, %f]", start1-mintime, stop1-mintime, start2-mintime, stop2-mintime, stop2-start1, stop1-start2)

    assert start1 > stop2 or start2 > stop1

@pytest.mark.xfail(strict=True)
def test_token_fail():
    """Simple token test: should fail without token"""
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token_experiment(xp, token)
    
def test_token():
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token = xp.workspace.connector.createtoken("test-token", 1)
        token_experiment(xp, token)


@pytest.mark.skip("TODO: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()
