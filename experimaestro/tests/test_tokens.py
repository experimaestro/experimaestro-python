import pytest
import logging
import time

from experimaestro.scheduler import JobState
from .utils import TemporaryExperiment
from .task_tokens import TokenTask

class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, value):
        return self.start > value.end

    def __str__(self):
        return "%.4f - %.4f" % (self.start, self.end)


def get_times(task):
    return TimeInterval(*(float(t) for t in task.stdout().read_text().strip().split("\n")))


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

    time1 = get_times(task1)
    time2 = get_times(task2)

    assert time1 > time2 or time2 > time1

@pytest.mark.xfail(strict=True)
def test_token_fail():
    """Simple token test: should fail without token"""
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token_experiment(xp, token)
    
def test_token():
    with TemporaryExperiment("tokens", maxwait=10) as xp:
        token = xp.workspace.connector.createtoken("test-token", 1)
        token_experiment(xp, token)



def test_token_reexecute():
    """Two different schedulers
    
    Test the ability of the token to check IPC signals
    """
    def run(xp, x, path):
            token = xp.workspace.connector.createtoken("test-token-reexecute", 1)
            task = TokenTask(path=path, x=x).add_dependencies(token.dependency(1)).submit()
            return task

    with TemporaryExperiment("tokens1", maxwait=10) as xp1, \
        TemporaryExperiment("tokens2", maxwait=10) as xp2:
        path = xp1.workspace.path / "test_token.file"
        task1 = run(xp1, 1, path)
        task2 = run(xp2, 2, path)

        time.sleep(0.5)
        path.write_text("Hello world")

        xp1.wait()
        xp2.wait()
        time1 = get_times(task1)
        time2 = get_times(task2)

        logging.info("%s vs %s", time1, time2)
        assert time1 > time2 or time2 > time1


@pytest.mark.skip("TODO: not implemented")
def test_token_process():
    """Test tokens with two processes"""
    raise NotImplementedError()
